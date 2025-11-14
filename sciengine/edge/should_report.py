from sciengine.agent.agent_prompts import PLAN_SYSTEM_PROMPT
from typing_extensions import TypedDict
import json
from typing import Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from operator import add
from sciengine.utils import debug_log
from langgraph.graph import START, END
import os
from sciengine.agent.overallstate import OverallState

#————————————维护状态————————————————————


def should_run_report(state: OverallState) -> str:
    """
    检查 RAG 节点是否成功下载并存储了论文内容。
    如果获取到内容，则流程继续。否则终止。
    """
    papers = state.get("paper_content", [])
    if len(papers) > 0:
        debug_log(f"✅ RAG found {len(papers)} papers. Continuing to next step.")
        # 由于后续的报告生成节点尚未定义，这里暂时指向 END，但标记为 'SUCCESS'
        return "SUCCESS" 
    else:
        debug_log("❌ RAG found 0 papers. Terminating flow to prevent generating empty report.")
        return "FAIL" # 流程终止