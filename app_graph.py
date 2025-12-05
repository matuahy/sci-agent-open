# ./app_graph.py
"""
构建图
"""

from fastapi import FastAPI, Request, HTTPException
from langgraph.graph import StateGraph, START, END
from sciengine.node.planner_node import run_planner_node
from sciengine.node.search_node import run_search_node
from sciengine.node.RAG_node import run_RAG_node
from sciengine.edge.should_search import should_run_search
from sciengine.edge.should_report import should_run_report
from sciengine.agent.overallstate import OverallState
from sciengine.node.writing_node import run_writing_node

# -------------------------------------------------
# ② 工作流
# -------------------------------------------------
workflow = StateGraph(OverallState)

# 节点注册
workflow.add_node("planner_node", run_planner_node)
workflow.add_node("search_node", run_search_node)
workflow.add_node("rag_node", run_RAG_node)
workflow.add_node("writing_node",run_writing_node)
# 边
workflow.add_edge(START, "planner_node")

workflow.add_conditional_edges(
    "planner_node",
    should_run_search,
    {
        "SUCCESS" : "search_node",
        "FAIL": END
    }
)

# Search → RAG
workflow.add_edge("search_node", "rag_node")

# RAG → Conditional END
workflow.add_conditional_edges(
    "rag_node",
    should_run_report,
    {
        "SUCCESS": "writing_node",  # 流程成功，暂时终止（未来应导向报告生成节点）
        "FAIL": END      # 流程失败，终止
    }
)

# Writing → END
workflow.add_edge("writing_node", END)

app_graph = workflow.compile()