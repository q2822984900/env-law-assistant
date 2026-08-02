# 环境法规问答助手

基于本地法规库的 RAG 中文问答系统：用户提问 → 检索法规条文 → DeepSeek 生成**带出处、可核验原文**的回答。全部本地运行，唯一外部依赖为 DeepSeek API。

## 运行命令

- 建库：`python ingest.py`（解析 `data/raw/` 下的 .docx/.pdf，重建检索索引，可重复运行）
- 问答：`streamlit run app.py`（浏览器访问 http://localhost:8501）

## 技术栈

- Python 3.14 · PyMuPDF(fitz) · python-docx · langchain-text-splitters 1.x
- BAAI/bge-small-zh-v1.5（本地 CPU）· Chroma · DeepSeek（deepseek-v4-flash）· Streamlit

## 架构

- `ingest.py`：解析文件 → 条文切分 → bge 向量化 → 写入 Chroma（collection `env_law`，目录 `data/vector_db/`）
- `app.py`：问题向量化 → Chroma 检索 top-8 → 构造提示词 → DeepSeek 生成 → 界面展示（含引用原文）

## 关键实现决策（改动前必读）

- **切分策略**：以"第X条"为语义单位，每条条文一个块；"第X章"标题拼入该章每条条文作上下文；超长条文（>500 字）内部二次切分（chunk_size=500, overlap=100）。勿改回纯字符硬切。
- **向量维度**：bge-small-zh-v1.5 实际为 **512 维**（早期文档曾误写 384，已纠正）。
- **检索参数**：`TOP_K=8`（曾为 4，会漏上位法）；查询端加 bge 指令前缀 `为这个句子生成表示以用于检索相关文章：`；提示词要求 LLM"忽略与本问题无关的条文"。
- **引用呈现**：回答正文须写明 `[编号]《法律名》第X条`；"引用出处"展开器展示**条文原文**，且只显示回答实际引用的条文（解析回答中的 `[n]` 编号，解析失败回退显示全部）。
- **PDF 支持**：`ingest.py` 已按扩展名分流，`.pdf` 用 fitz 解析（代码就绪）；当前数据全为 `.docx`，`.pdf` 分支未触发。
- **幂等建库**：每次跑 ingest.py 会清空重建 collection。

## 数据

- `data/raw/`：现含 10 部环境法规（.docx，文件名带日期后缀）。新增/替换法规后重跑 `python ingest.py` 即可。
- `data/vector_db/`：检索索引，可随时重建，未纳入 git（.gitignore 排除）。

## 注意事项

- `.env` 存有 `DEEPSEEK_API_KEY`，已被 .gitignore 排除，**切勿提交或泄露**。
- DeepSeek 模型 ID 为 `deepseek-v4-flash`；如平台变更需按平台文档校正。
- bge 模型首次运行需联网下载；国内直连失败时设置 `HF_ENDPOINT=https://hf-mirror.com` 重跑。
- 访问 GitHub 等境外网络需走代理：`HTTPS_PROXY=http://127.0.0.1:7897`（本机 Clash 端口）。
- langchain 主版本为 1.x，代码必须用 1.x API，勿照抄 0.x 教程。
