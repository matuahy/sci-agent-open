# writing.py 检索增强生成
import json
import os
import logging
import multiprocessing
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores import Chroma
from sciengine import bioembedding, llm_models
from rank_bm25 import BM25Okapi
from sciengine import bge_reranker

# ==============================
# 日志配置（终端 + 文件）
# ==============================
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = f"{log_dir}/writing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def info(msg): print(f"[INFO] {msg}"); logging.info(msg)
def debug(msg): print(f"[DEBUG] {msg}"); logging.debug(msg)
def error(msg): print(f"[ERROR] {msg}"); logging.error(msg)

info("=== 综述生成系统启动 ===")

# ==============================
# LLM & 嵌入
# ==============================
info("正在加载 LLM...")
try:
    llm = llm_models.get_chat_model()
    info("LLM 加载成功")
except Exception as e:
    error(f"LLM 加载失败: {e}")
    llm = None

info("正在加载 BioBERT 嵌入...")
embedding_model = bioembedding.BioBERTEmbeddings("/root/autodl-tmp/backend/biobert-embeddings")
info("BioBERT 嵌入加载成功")

info("正在加载 Chroma 向量库...")
vector_store = Chroma(persist_directory="./chroma_papers", embedding_function=embedding_model)
docs = vector_store.get()['documents']
info(f"向量库加载完成，共 {len(docs)} 篇文档")

# 读取大纲
info("正在读取大纲文件...")
with open("./full_state_20251110_183552.json", "r", encoding="utf-8") as f:
    overallstate = json.load(f)
report_outline = overallstate['planner_output'].get("report_outline", {})
info(f"大纲加载成功，报告标题: {report_outline.get('title', '未知')}")

# ==============================
# BM25 索引
# ==============================
info("正在构建 BM25 索引...")
bm25 = BM25Okapi([d.split() for d in docs if isinstance(d, str) and d.strip()])
info("BM25 索引构建完成")

# ==============================
# 检索函数（扩大检索量）
# ==============================
def hybrid_retrieve(q: str, k=30):
    sem = vector_store.similarity_search_with_score(q, k=k)
    bm25_scores = bm25.get_scores(q.split())
    bm25_idx = bm25_scores.argsort()[-k:][::-1]
    scores = {}
    for doc, sc in sem:
        text = doc.page_content if isinstance(doc, Document) else str(doc)
        scores[text] = scores.get(text, 0) + sc
    for i in bm25_idx:
        text = docs[i]
        norm = bm25_scores[i] / (bm25_scores.max() + 1e-8)
        scores[text] = scores.get(text, 0) + norm * 0.5
    return [Document(page_content=t) for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:k]

# ==============================
# 全局 reranker
# ==============================
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        info("正在加载本地 bge-reranker-large...")
        try:
            _reranker = bge_reranker.BgeReranker("/root/autodl-tmp/backend/bge-reranker-large")
            info("reranker 加载成功")
        except Exception as e:
            error(f"reranker 加载失败: {e}，将跳过精排")
            _reranker = None
    return _reranker

def strongest_retrieve(q: str) -> List[Document]:
    try:
        hybrid = hybrid_retrieve(q, 30)
        mmr = vector_store.max_marginal_relevance_search(q, k=15, fetch_k=30, lambda_mult=0.5)
        reranker = get_reranker()
        if reranker:
            pairs = [(q, d.page_content) for d in mmr]
            scores = reranker.compute_score(pairs)
            ranked = [d for _, d in sorted(zip(scores, mmr), key=lambda x: x[0], reverse=True)]
            return ranked[:10]
        else:
            return mmr[:10]
    except Exception as e:
        debug(f"strongest_retrieve 异常: {e}")
        return mmr[:10]

# ==============================
# Agent Prompt（修改引用格式）
# ==============================
QUESTION_SYSTEM_PROMPT = """
You are a research assistant specialized in turning a chapter/section description into **concrete, searchable scientific questions** for retrieval-augmented generation.
### INPUT (JSON)
{
  "title": "<section title>",
  "content": "<section description>"
}
### OUTPUT (strict JSON only)
{
  "query": "<concise search query: title + 1-2 sentence summary>",
  "questions": ["question 1", "question 2", "..."]
}
"""

GENERATE_SYSTEM_PROMPT = """
You are an expert academic writer. Given a section query, a list of retrieval questions, the retrieved evidence snippets (with metadata), and the full report outline, write a **concise, rigorous, and well-structured review paragraph**.

### INPUT
- Query: <query string>
- Questions: <JSON list>
- Context Snippets: <JSON list of short evidence excerpts with metadata>
- Full Outline: <complete JSON outline>

### OUTPUT (strict JSON only)
{
  "section_title": "<title>",
  "content": "<Markdown text with inline citations [Title](PubMed_URL)>",
  "subsections": [ ... ]
}

### RULES
1. Style: formal, objective, academic.
2. Structure: introduce → synthesize evidence → conclude/transition.
3. Citations:
   - Each snippet includes `title` and `pubmed_url`.
   - Cite inline as [<title>](<pubmed_url>).
   - If multiple, cite as ([T1](url1); [T2](url2)).
4. General: field-agnostic writing.
5. Always output **valid JSON only**.
"""

question_agent = create_react_agent(model=llm, tools=[], prompt=SystemMessage(content=QUESTION_SYSTEM_PROMPT))
generate_agent = create_react_agent(model=llm, tools=[], prompt=SystemMessage(content=GENERATE_SYSTEM_PROMPT))
info("Agent 创建完成")

# ==============================
# 单章节写作
# ==============================
def run_writing_node(section: Dict[str, Any], overallstate: Dict[str, Any]) -> Dict[str, Any]:
    section_title = section.get("title", "未知")
    info(f"开始处理章节: {section_title}")

    def process(sec: Dict[str, Any], depth=0) -> Dict[str, Any]:
        indent = "  " * depth
        title = sec.get("title", "Untitled")
        content = sec.get("content", "")
        print(f"{indent}处理: {title}")

        # Step 1: 生成问题
        input_json = json.dumps({"title": title, "content": content}, ensure_ascii=False)
        try:
            q_resp = question_agent.invoke({"messages": [HumanMessage(content=input_json)]})
            q_data = json.loads(q_resp['messages'][-1].content)
            query = q_data.get("query", title)
            questions = q_data.get("questions", [])
        except Exception as e:
            error(f"{indent}提问失败: {e}")
            query = title
            questions = [f"What is known about {title}?"]

        # Step 2: 检索
        context = []
        for q in questions:
            context.extend(strongest_retrieve(q))
        seen = set()
        context = [d for d in context if d.page_content not in seen and not seen.add(d.page_content)]

        # Step 3: 构造 snippets（含 metadata）
        snippets = [
            {
                "text": d.page_content[:280] + "...",
                "title": d.metadata.get("title", "Unknown Title"),
                "pubmed_url": d.metadata.get("pubmed_url", "")
            }
            for d in context[:5]
        ]

        # Step 4: 调用生成
        input_msg = (
            f"Query: {query}\n"
            f"Questions: {json.dumps(questions, ensure_ascii=False)}\n"
            f"Context: {json.dumps(snippets, ensure_ascii=False)}\n"
            f"Outline: {json.dumps(report_outline, ensure_ascii=False)}"
        )
        try:
            g_resp = generate_agent.invoke({"messages": [HumanMessage(content=input_msg)]})
            result = json.loads(g_resp['messages'][-1].content)
        except Exception as e:
            error(f"{indent}生成失败: {e}")
            result = {"section_title": title, "content": f"**生成失败**: {e}", "subsections": []}

        # Step 5: 子章节递归
        subs = sec.get("subsections", [])
        if subs:
            result["subsections"] = [process(s, depth + 1) for s in subs]

        return result

    result = process(section)
    info(f"章节完成: {section_title}")
    return result

# ==============================
# 并发主函数
# ==============================
async def run_con_writing_node(overallstate: Dict[str, Any], output_path="./final_report.json"):
    sections = report_outline.get("sections", [])
    top_sections = [s for s in sections if str(s.get("section_number", "")).isdigit() and len(str(s.get("section_number"))) == 1]
    total = len(top_sections)
    max_workers = min(multiprocessing.cpu_count(), total, 6)
    info(f"开始并发处理 {total} 个顶级章节，最大线程 {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, run_writing_node, sec, overallstate) for sec in top_sections]
        results = []
        for i, task in enumerate(tasks, 1):
            res = await task
            results.append(res)
            print(f"已完成 {i}/{total}: {top_sections[i-1].get('title', '未知')}")

    report = {"title": report_outline.get("title", "Generated Report"), "sections": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    info(f"报告生成成功，保存至: {os.path.abspath(output_path)}")
    return report

# ==============================
# 主入口
# ==============================
if __name__ == "__main__":
    asyncio.run(run_con_writing_node(overallstate, "./final_report_20251113.json"))
