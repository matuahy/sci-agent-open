# sciengine/edge/should_search.py
"""
条件边，检查plan agent节点，用户是否clarifying questions
"""

from sciengine.agent.utils import debug_log
from sciengine.agent.overallstate import OverallState




# 条件边：只有无澄清问题时才执行search
def should_run_search(state: OverallState) -> str:
    questions = state["planner_output"].get("clarifying_questions", [])
    if questions:
        debug_log(f"⏭️ Skipping search: {len(questions)} clarifying questions")
        return "FAIL"
    debug_log("✅ No clarifying questions, running search")
    return "SUCCESS"
