# 📚 DeepSearch-RAG 深度检索增强问答系统

> 一个基于本地大模型、多路召回、知识图谱、深度搜索的离线 RAG 框架  
> 面向超长文本、多级摘要、渐进式知识发现与问答

---

## 1. 项目简介

| 特性 | 说明 |
|---|---|
| **离线运行** | 所有模型、索引、知识库均可本地部署，无需外网 |
| **超长文本** | 支持单文件百万级 token，自动分句、分块、摘要 |
| **多路召回** | 向量 + 倒排 + 知识图谱三元组联合召回 |
| **重排序** | Cross-Encoder 精排 + MMR 多样性 |
| **深度搜索** | 自动生成子问题 → 多轮检索 → 信息增益早停 |
| **知识图谱** | 自动抽取 <人、地、事件> 三元组并建立倒排 |
| **一键脚本** | 5 条命令即可完成「文本 → 索引 → 问答」 |

---

## 2. 项目结构

```
DeepSearch-RAG
├─ README.md
├─ chat.py              # 本地 LLM 推理封装
├─ deepsearch.py        # 深度检索主流程
├─ recall_rank.py       # 召回 + 重排 + 问答
├─ split_chunk.py       # 文本分句、分块、摘要、父块生成
├─ insert_vector_index.py
├─ insert_tfidf.py
├─ insert_kg_index.py   # 知识图谱三元组抽取
├─ extend_query.py      # 子问题生成 & 合并
├─ get_vector.py        # 句向量编码
├─ document_chunk.py    # Chunk 数据结构
├─ index_data/          # 所有索引落盘目录
│  ├─ chunk_index/
│  ├─ tfidf/
│  └─ kg_index/
└─ models/              # 本地模型目录
   ├─ qwen2.5-14B
   ├─ Qwen3-8B
   ├─ Qwen3-Eb-06.B
   └─ mxbai-rerank
```

---

## 3. 环境准备

### 3.1 硬件建议
| 组件 | 最低 | 推荐 |
|---|---|---|
| GPU | RTX 3090 24G | RTX 4090 / A100 |
| 内存 | 32 GB | 64 GB |
| 硬盘 | 50 GB | 200 GB+（大模型 + 索引）

### 3.2 软件依赖
```bash
conda create -n rag python=3.10
conda activate rag
pip install torch==2.2.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.40.2 sentence-transformers==2.7.0 faiss-cpu==1.8.0 whoosh==2.7.4 jieba==0.42.1 tqdm scikit-learn
```

---

## 4. 运行流程（5 步走）

> 以下命令均在项目根目录执行

| 步骤 | 目的 | 命令 |
|---|---|---|
| ① | 将原始文本拆分为 Chunk 并生成摘要 | `python split_chunk.py` |
| ② | 建立向量索引（FAISS） | `python insert_vector_index.py` |
| ③ | 建立倒排索引（Whoosh TF-IDF） | `python insert_tfidf.py` |
| ④ | 抽取知识图谱并建立三元组索引 | `python insert_kg_index.py` |
| ⑤ | 启动深度检索问答 | `python deepsearch.py` |

### 4.1 放置原始文本
把需要问答的 txt 文件（如《明朝那些事儿.txt》）放到项目根目录，并修改 `split_chunk.py` 最后一行的文件名即可。

### 4.2 模型路径
检查 `chat.py / recall_rank.py / get_vector.py` 里的 `model_name = "E:\\code\\qwen2.5-14B"` 等路径，指向你本地实际目录。

---

## 5. 各模块详解

### 5.1 split_chunk.py
- **输入**：原始 txt
- **输出**：
  - `index_chunk`：字典 `{chunk_id: Chunk}`
  - `id_data`：句子级原始文本 `{id: sentence}`
- **关键参数**：
  - `chunk_size=256`：每个 chunk 的目标字符数
  - `common_token=30`：相邻 chunk 重叠字符数
  - `abstract`：由 LLM 生成的 50 字摘要
  - `parent`：上下文窗口（±2 个 chunk）拼接的父块

### 5.2 索引构建
| 索引 | 技术 | 文件 | 用途 |
|---|---|---|---|
| 向量 | FAISS-IP | `chunk_vector` | 语义召回 |
| 倒排 | Whoosh TF-IDF | `tfidf/` | 关键词召回 |
| 图谱 | 三元组 → 倒排 | `kg_index/entity` | 实体关系召回 |

### 5.3 recall_rank.py
- `search(query, topK)`：标准 RAG 流程
  1. `get_faiss_candidate` → 向量召回 100
  2. `get_inverted_candidate` → 关键词召回 100
  3. `reranker` → Cross-Encoder 精排 + MMR 去重 → topK
- `chat(query, search_topK)`：召回后交给本地 LLM 生成答案

### 5.4 deepsearch.py
- **渐进式信息发现**
  - 每轮根据「信息增长率」早停
  - `extend_query` 利用 LLM 生成下一轮子问题
  - 最终聚合所有 chunk，生成全面答案

---

## 6. 一键脚本（可选）

新建 `run.sh`：

```bash
#!/bin/bash
echo ">>> Step1 文本分块..."
python split_chunk.py
echo ">>> Step2 向量索引..."
python insert_vector_index.py
echo ">>> Step3 倒排索引..."
python insert_tfidf.py
echo ">>> Step4 知识图谱..."
python insert_kg_index.py
echo ">>> Step5 深度问答..."
python deepsearch.py
chmod +x run.sh && ./run.sh
```

---

## 7. 自定义问答

修改 `deepsearch.py` 最后一行：

```python
if __name__ == "__main__":
    query = "明朝内阁首辅的权力如何演变？"
    result = deepsearch(query)
    print(result)
```

直接运行即可得到深度研究报告式答案。

---


---


MIT © 2025 DeepSearch-RAG Contributors
