from sciengine.agent.agent_prompts import PLAN_SYSTEM_PROMPT
from typing_extensions import TypedDict
import json
from typing import Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from operator import add
import traceback
from sciengine.utils import debug_log
from langgraph.graph import START, END
import asyncio
from sciengine.con_sci_RAG import Pubmed_RAG
import os
from sciengine.agent.overallstate import OverallState

#————————————维护状态————————————————————

# ──────────────────────────────────────────────────────────────
# ③ RAG Node
# ──────────────────────────────────────────────────────────────
def run_RAG_node(state: OverallState) -> OverallState:
    """
    1. 从 search_results 中提取所有 PubMed URL
    2. 下载 PMC 全文 → 切块 → 写入 Chroma（逐篇写入）
    3. 把 paper_content 与 chroma_dir 写回 state，供后续节点使用
    """
    debug_log("Starting RAG node (download + vector DB)")

    try:
        # 直接调用 Pubmed_RAG.run_RAG（已封装好全部流程）
        rag = Pubmed_RAG()
        rag_result = rag.run_RAG(state)

        # 更新 state
        state.update(rag_result)

        debug_log(
            f"RAG node completed – "
            f"{len(state.get('paper_content', []))} papers stored, "
            f"vector DB at {state.get('chroma_dir')}"
        )
    except Exception as e:
        debug_log(f"RAG node exception: {str(e)}")
        traceback.print_exc()
        # 即使出错也把空结果写回，防止流程卡死
        state["paper_content"] = []
        state["chroma_dir"] = rag.persist_directory

    return state
