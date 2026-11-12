"""全文抓取与节选：arXiv HTML/PDF、普通网页、PDF。

安全：只允许 http(s)；每一跳（含重定向）都会解析主机并拒绝内网/回环/链路本地地址（防 SSRF）；
下载大小有上限；全部失败时返回空串，不影响主流程。
"""

from __future__ import annotations

import asyncio
import io
import ipaddress
import logging
import os
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

from ..state import Source

logger = logging.getLogger(__name__)

MAX_BYTES = 4 * 1024 * 1024
MAX_REDIRECTS = 4
MAX_PDF_PAGES = 14
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DeepResearchAgent/0.1)", "Accept": "text/html,application/pdf,*/*;q=0.8"}


# ---------------------------------------------------------------------------
# 抓取
# ---------------------------------------------------------------------------


def _allowed_networks() -> list[ipaddress._BaseNetwork]:
    """即使落在保留网段也放行的网络。

    默认放行 198.18.0.0/15（RFC 2544 基准测试段）：Clash / Surge 等代理的 fake-ip DNS 模式会把
    公网域名解析到该网段，真实请求由代理转发。其他代理方案可通过 FULLTEXT_ALLOWED_NETWORKS
    （逗号分隔的 CIDR）追加。
    """
    nets = ["198.18.0.0/15"] + [x.strip() for x in os.getenv("FULLTEXT_ALLOWED_NETWORKS", "").split(",") if x.strip()]
    out = []
    for n in nets:
        try:
            out.append(ipaddress.ip_network(n, strict=False))
        except ValueError:
            logger.warning("忽略无效的 FULLTEXT_ALLOWED_NETWORKS 项: %s", n)
    return out


def _ip_is_public(ip: ipaddress._BaseAddress) -> bool:
    if any(ip in net for net in _allowed_networks() if net.version == ip.version):
        return True
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified)


def _host_is_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    return bool(infos) and all(_ip_is_public(ipaddress.ip_address(i[4][0].split("%")[0])) for i in infos)


async def is_public_url(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    return await asyncio.to_thread(_host_is_public, p.hostname)


async def _download(client: httpx.AsyncClient, url: str) -> tuple[bytes, str, str] | None:
    """返回 (内容, content-type, 最终 URL)。"""
    for _ in range(MAX_REDIRECTS + 1):
        if not await is_public_url(url):
            logger.info("fulltext: 拒绝非公网地址 %s", url)
            return None
        async with client.stream("GET", url, headers=HEADERS) as resp:
            if resp.is_redirect:
                loc = resp.headers.get("location")
                if not loc:
                    return None
                url = urljoin(url, loc)
                continue
            if resp.status_code >= 400:
                return None
            ctype = resp.headers.get("content-type", "").lower()
            buf = bytearray()
            async for chunk in resp.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > MAX_BYTES:
                    break
            return bytes(buf), ctype, url
    return None


# ---------------------------------------------------------------------------
# 文本提取
# ---------------------------------------------------------------------------

_SKIP = {"script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form", "button", "iframe"}
_BLOCK = {"p", "div", "br", "li", "tr", "section", "article", "table", "ul", "ol", "blockquote", "figure", "pre"}
_HEADING = {"h1", "h2", "h3", "h4"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self._in_heading = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
        elif tag in _HEADING:
            self.parts.append("\n## ")
            self._in_heading = True
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in _HEADING:
            self.parts.append("\n")
            self._in_heading = False
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:  # noqa: BLE001  畸形 HTML
        pass
    return "".join(p.parts)


def pdf_to_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = reader.pages[:MAX_PDF_PAGES]
        return "\n".join((pg.extract_text() or "") for pg in pages)
    except Exception as e:  # noqa: BLE001
        logger.info("pdf parse failed: %s", e)
        return ""


_DROP_HEADING = re.compile(r"references|bibliography|acknowledg|appendix|参考文献|致谢|附录|related work", re.I)
_KEY_HEADING = re.compile(
    r"abstract|introduction|method|approach|model|architecture|experiment|result|evaluation|"
    r"analysis|discussion|conclusion|摘要|引言|方法|实验|结果|评测|结论|overview|summary|features|benchmark",
    re.I,
)
_REF_TAIL = re.compile(r"\n\s*(References|REFERENCES|Bibliography)\s*\n")


_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def _is_substantive(line: str) -> bool:
    """过滤导航/菜单等短行。中日韩文字信息密度高，阈值比拉丁文低。"""
    cjk_ratio = len(_CJK.findall(line)) / max(1, len(line))
    return len(line) >= (14 if cjk_ratio > 0.3 else 40)


_BOILERPLATE = re.compile(r"^(License:|Provided proper attribution|Permission to make digital|©|Copyright)", re.I)


def _clean_lines(text: str) -> list[str]:
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if _BOILERPLATE.match(ln):
            continue
        if ln.startswith("## ") or _is_substantive(ln):
            lines.append(ln)
    return lines


def select_excerpt(text: str, max_chars: int) -> str:
    """节选：开头 + 方法/实验/结论等关键章节，丢弃参考文献/附录。"""
    if not text.strip():
        return ""
    m = _REF_TAIL.search(text)
    if m and m.start() > len(text) * 0.3:
        text = text[: m.start()]

    sections: list[tuple[str, str]] = []
    head, body = "", []
    for ln in _clean_lines(text):
        if ln.startswith("## "):
            sections.append((head, "\n".join(body)))
            head, body = ln[3:].strip(), []
        else:
            body.append(ln)
    sections.append((head, "\n".join(body)))
    sections = [(h, b) for h, b in sections if b.strip() and not (h and _DROP_HEADING.search(h))]
    if not sections:
        return ""

    if len(sections) == 1:  # 没有章节结构（普通网页 / PDF）
        return sections[0][1][:max_chars].strip()

    first_budget = max_chars // 3
    out = [sections[0][1][:first_budget]]
    used = len(out[0])
    rest = sections[1:]
    prioritized = [s for s in rest if _KEY_HEADING.search(s[0])] + [s for s in rest if not _KEY_HEADING.search(s[0])]
    for h, b in prioritized:
        if used >= max_chars:
            break
        room = max_chars - used
        chunk = f"[{h}] " + b[: min(room, max(600, max_chars // 3))]
        out.append(chunk)
        used += len(chunk)
    return "\n".join(out)[: max_chars + 400].strip()


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------


def candidate_urls(src: Source) -> list[str]:
    key = src.get("key", "")
    if key.startswith("arxiv:"):
        aid = key.split(":", 1)[1]
        return [f"https://arxiv.org/html/{aid}", f"https://ar5iv.labs.arxiv.org/html/{aid}", f"https://arxiv.org/pdf/{aid}"]
    url = src.get("url", "")
    return [url] if url else []


async def fetch_fulltext(src: Source, max_chars: int = 3500, timeout: float = 25.0) -> str:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for url in candidate_urls(src):
            try:
                got = await asyncio.wait_for(_download(client, url), timeout=timeout + 5)
            except Exception as e:  # noqa: BLE001
                logger.info("fulltext fetch failed %s: %r", url, e)
                continue
            if not got:
                continue
            data, ctype, final = got
            if "pdf" in ctype or final.lower().split("?")[0].endswith(".pdf"):
                text = await asyncio.to_thread(pdf_to_text, data)
            elif "html" in ctype or "xml" in ctype or "text" in ctype:
                text = html_to_text(data.decode("utf-8", errors="ignore"))
            else:
                continue
            excerpt = select_excerpt(text, max_chars)
            if len(excerpt) >= 300:
                return excerpt
    return ""
