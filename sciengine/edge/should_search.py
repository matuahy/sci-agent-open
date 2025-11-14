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


# 条件边：只有无澄清问题时才执行search
def should_run_search(state: OverallState) -> str:
    questions = state["planner_output"].get("clarifying_questions", [])
    if questions:
        debug_log(f"⏭️ Skipping search: {len(questions)} clarifying questions")
        return END
    debug_log("✅ No clarifying questions, running search")
    return "search_node"