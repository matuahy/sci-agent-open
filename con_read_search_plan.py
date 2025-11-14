#con_read_search_plan.py 并发
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from operator import add
import requests
from sciengine.llm_models import get_chat_model
import os
from concurrent.futures import ThreadPoolExecutor
from sciengine.agent.agent_prompts import PLAN_SYSTEM_PROMPT, GENERATE_SYSTEM_PROMPT,SEARCH_SYSTEM_PROMPT,QUESTION_SYSTEM_PROMPT
from sciengine.agent.search_tools import search_tools
from sciengine.node.planner_node import run_planner_node
from sciengine.node.search_node import run_search_node
from sciengine.node.RAG_node import run_RAG_node
from sciengine.edge.should_search import should_run_search
from sciengine.edge.should_report import should_run_report
from langchain_core.messages import SystemMessage
from sciengine.utils import debug_log,test_api_connectivity

# Planner Agent

# --- Search Agent Tools ---

# --- Search Agent ---

#————————————维护状态————————————————————
class OverallState(TypedDict):
    """LangGraph 维护的全局状态。"""
    query: str  # 用户的原始查询 (input.query)
    planner_output: Dict[str, Any] # 规划器生成的大纲
    tasks: List[str] # 任务列表 (未使用但保留)
    search_results: List[Dict[str, Any]] # 搜索结果 (PubMed URL列表)
    paper_content: List[Dict[str, Any]]     # 下载的全文（含 content）
    chroma_dir: str                 # Chroma 持久化目录
    messages: Annotated[List[BaseMessage], add] # 对话历史或 agent 间的消息

# Planner Node

# Search Node

# ③ RAG Node

# ──────────────────────────────────────────────────────────────
# ④ 决策函数
# ──────────────────────────────────────────────────────────────

# 条件边：只有无澄清问题时才执行search


# 条件边：RAG 完成后检查是否有内容

# -------------------------------------------------
# ② 工作流：Planner → Search → RAG → Condition → END
# -------------------------------------------------
workflow = StateGraph(OverallState)

# 节点注册
workflow.add_node("planner_node", run_planner_node)
workflow.add_node("search_node", run_search_node)
workflow.add_node("rag_node", run_RAG_node)

# 边
workflow.add_edge(START, "planner_node")

workflow.add_conditional_edges(
    "planner_node",
    should_run_search,
    {
        "search_node": "search_node",
        END: END
    }
)

# Search → RAG
workflow.add_edge("search_node", "rag_node")   

# RAG → Conditional END (修复 #2: 确保只有在有数据时才继续)
workflow.add_conditional_edges(
    "rag_node",
    should_run_report,
    {
        "SUCCESS": END,  # 流程成功，暂时终止（未来应导向报告生成节点）
        "FAIL": END      # 流程失败，终止
    }
)

app = workflow.compile()