# sciengine/edge/should_report.py
"""
条件边，检查RAG节点是否获取了paper_content
"""


from sciengine.agent.utils import debug_log
from sciengine.agent.overallstate import OverallState

def should_run_report(state: OverallState) -> str:
    """
    检查 RAG 节点是否成功下载并存储了论文内容。
    如果获取到内容，则流程继续。否则终止。
    """
    papers = state.get("paper_content", [])
    if len(papers) > 0:
        debug_log(f"✅ RAG found {len(papers)} papers. Continuing to next step.")
        return "SUCCESS" 
    else:
        debug_log("❌ RAG found 0 papers. Terminating flow to prevent generating empty report.")
        return "FAIL" # 流程终止