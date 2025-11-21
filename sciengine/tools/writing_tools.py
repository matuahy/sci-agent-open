# sciengine/agent/writing_tools.py
"""
writing_node的 retriever tools
"""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sciengine.agent.utils import info, error
from sciengine.tools.bge_reranker import BgeReranker
import os
from langchain_community.vectorstores import Chroma
from sciengine.model.bioembedding_model import BioBERTEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# ==============================
# 全局 reranker（这个可以保留全局，加载一次）
# ==============================
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            _reranker = BgeReranker("/root/autodl-tmp/backend/bge-reranker-large")
            info("BGE Reranker 加载成功")
        except Exception as e:
            error(f"reranker 加载失败: {e}，将跳过精排")
            _reranker = None
    return _reranker


# ==============================
# 核心：根据 state 实时构建检索器（关键函数！）
# ==============================
def build_retriever_from_state(state: Dict[str, Any], k: int = 20) -> Optional[Any]:
    """
    每次调用都根据 state 中的最新向量库路径，实时构建混合检索器
    """
    db_path = state.get("vector_db_path") or state.get("chroma_dir", "").strip()

    if not db_path or not os.path.exists(db_path):
        debug(f"[Retriever] 无有效向量库路径: {db_path}，返回空检索器")
        return None

    if not os.path.isdir(db_path):
        debug(f"[Retriever] 路径不是目录: {db_path}")
        return None

    try:
        info(f"正在从实时向量库加载检索器: {db_path}")

        # 1. 加载 Chroma 向量库（只加载一次，后面复用）
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=BioBERTEmbeddings("/root/autodl-tmp/backend/biobert-embeddings")
        )

        # 2. 获取所有文档文本，用于 BM25
        result = vectorstore.get(include=["documents", "metadatas"])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        if not documents:
            info("[Retriever] 向量库为空")
            return vectorstore.as_retriever(search_kwargs={"k": k})

        # 3. 构建 BM25 检索器
        bm25_retriever = BM25Retriever.from_texts(
            texts=documents,
            metadatas=metadatas,
            k=k
        )
        bm25_retriever.k = k

        # 4. 构建 Chroma 检索器
        chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        # 5. 混合检索（推荐权重）
        ensemble_retriever = EnsembleRetriever(
            retrievers=[chroma_retriever, bm25_retriever],
            weights=[0.7, 0.3]  # 语义 > 关键词
        )

        info(f"[Retriever] 混合检索器构建完成，共 {len(documents)} 篇候选文档")
        return ensemble_retriever

    except Exception as e:
        error(f"[Retriever] 构建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================
# 最强检索入口（对外接口）
# ==============================
def strongest_retrieve(query: str, state: Dict[str, Any]) -> List[Document]:
    """
    最终对外调用的检索函数，必须传入 state！
    """
    retriever = build_retriever_from_state(state, k=30)

    if not retriever:
        info("[strongest_retrieve] 无可用检索器，返回空结果")
        return []

    try:
        # 第一步：混合检索召回
        candidate_docs = retriever.invoke(query)
        info(f"[strongest_retrieve] 混合检索召回 {len(candidate_docs)} 篇")

        # 第二步：MMR 多样性排序（可选）
        db_path = state.get("vector_db_path") or state.get("chroma_dir", "").strip()
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=BioBERTEmbeddings("/root/autodl-tmp/backend/biobert-embeddings")
        )

        mmr_docs = vectorstore.max_marginal_relevance_search(query, k=15, fetch_k=30)

        # 第三步：BGE Reranker 精排（最强）
        reranker = get_reranker()
        if reranker and mmr_docs:
            pairs = [(query, doc.page_content) for doc in mmr_docs]
            scores = reranker.compute_score(pairs)
            # 合并分数
            for doc, score in zip(mmr_docs, scores):
                doc.metadata["rerank_score"] = score
            # 按精排分数排序
            ranked = sorted(mmr_docs, key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
            final_docs = ranked[:10]
        else:
            final_docs = mmr_docs[:10] if mmr_docs else candidate_docs[:10]

        info(f"[strongest_retrieve] 最终返回 {len(final_docs)} 篇精选文献")
        return final_docs

    except Exception as e:
        error(f"strongest_retrieve 异常: {e}")
        import traceback
        traceback.print_exc()
        return []
