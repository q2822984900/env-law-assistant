# 环境法规问答助手

基于本地法规库的智能问答系统：用自然语言提问，系统从 10 部环境法规中检索相关条文，由大模型生成**带出处、可核验**的回答。全部本地运行，唯一外部依赖为 DeepSeek API。

## 功能

- 覆盖 10 部环境法规，支持问任意相关内容（审批手续、处罚标准、责任主体等）
- 回答逐条引用来源，可展开查看**条文原文**，随时核验模型是否忠实于法条
- 多轮对话，历史记录保留在界面
- 资料库未覆盖的问题，会明确提示"资料中未找到相关依据"

## 快速开始

### 1. 环境要求

- Python 3.14+
- 一个 DeepSeek API Key（在 [platform.deepseek.com](https://platform.deepseek.com) 获取）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 Key：

```
DEEPSEEK_API_KEY=sk-你的Key
```

> `.env` 已被 `.gitignore` 排除，不会进入仓库，注意保密。

### 4. 构建向量库（首次运行）

```bash
python ingest.py
```

脚本会解析 `data/raw/` 下的法规文件（`.docx` / `.pdf`），按条文切块、向量化后存入本地索引。**重复运行会重建索引，可安全重跑**。

### 5. 启动问答界面

```bash
streamlit run app.py
```

浏览器访问 **http://localhost:8501** 即可开始提问。

## 收录法规

| 法规名称 | 施行/修订日期 |
|---|---|
| 中华人民共和国环境保护法 | 2014-04-24 |
| 中华人民共和国水污染防治法 | 2017-06-27 |
| 中华人民共和国大气污染防治法 | 2018-10-26 |
| 中华人民共和国环境影响评价法 | 2018-12-29 |
| 中华人民共和国土壤污染防治法 | 2018-08-31 |
| 中华人民共和国固体废物污染环境防治法 | 2020-04-29 |
| 中华人民共和国噪声污染防治法 | 2021-12-24 |
| 建设项目环境保护管理条例 | 2017-07-16 |
| 排污许可管理条例 | 2021-01-24 |
| 碳排放权交易管理暂行条例 | 2024-01-25 |

## 项目结构

```
env-law-assistant/
├── data/
│   ├── raw/          # 法规源文件（可自行增删）
│   └── vector_db/    # 检索索引（ingest.py 自动生成）
├── ingest.py         # 建库脚本（解析 + 切分 + 向量化 + 入库）
├── app.py            # Streamlit 问答应用
├── requirements.txt  # 依赖清单
└── .env.example      # 环境变量模板
```

## 技术栈

PyMuPDF · python-docx · langchain-text-splitters · BAAI/bge-small-zh-v1.5 · Chroma · DeepSeek · Streamlit
