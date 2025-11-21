# sciengine/edge/should_search.py
"""
条件边，检查plan agent节点，用户是否clarifying questions
"""
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from sciengine.agent.utils import debug_log
from langgraph.graph import START, END
from sciengine.agent.overallstate import OverallState

#————————————维护状态————————————————————


# 条件边：只有无澄清问题时才执行search
def should_run_search(state: OverallState) -> str:
    questions = state["planner_output"].get("clarifying_questions", [])
    if questions:
        debug_log(f"⏭️ Skipping search: {len(questions)} clarifying questions")
        return "FAIL"
    debug_log("✅ No clarifying questions, running search")
    return "SUCCESS"
