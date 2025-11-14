from typing_extensions import TypedDict
import json
from typing import Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from operator import add

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