# 贡献指南 (Contributing Guide)

感谢你对 **Deep Research Agent** 的关注与贡献！无论是修复 Bug、增加新的学术/行业检索数据源，还是打磨 Web 智能研究工作台，都非常欢迎提交 Issue 或 Pull Request。

---

## 1. 本地开发环境搭建

```bash
# 1. 克隆仓库并创建虚拟环境
git clone https://github.com/ruantao1207/deepResearchAgent.git
cd deepResearchAgent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. 以可编辑模式安装全量依赖（含测试与 Web）
pip install -e ".[ddg,web,dev]"

# 3. 运行离线单元测试（无需配置任何真实 API Key）
pytest -q
```

---

## 2. 如何新增一个检索数据源（Tool）

所有检索工具位于 `deep_research/tools/` 下，返回统一的 `Source` 字典结构（定义于 `deep_research/state.py`）：

1. 在 `deep_research/tools/` 下新建模块（例如 `pubmed.py`），实现异步函数：
   ```python
   async def search_pubmed(query: str, max_results: int = 5) -> list[Source]:
       ...
   ```
2. 在 `deep_research/state.py` 的 `SourceName` Literal 类型中注册新数据源名称。
3. 在 `deep_research/nodes.py` 的 `_DISPATCH` 字典中挂载该检索函数。
4. 在 `tests/test_tools.py` 中补充对应的离线 Mock 单元测试。

---

## 3. 提交规范与安全检查

- **严禁提交真实密钥**：请确保 `.env` 与 `.deep_research_settings.json` 未被加入版本控制（已在 `.gitignore` 中排除）。
- **保持测试全绿**：提交 PR 前请运行 `pytest -q`，确保全部测试通过。
- **前端脚本语法检查**：如修改了 `deep_research/web/static/index.html`，请确保内联 JS 通过 `node --check` 校验。
