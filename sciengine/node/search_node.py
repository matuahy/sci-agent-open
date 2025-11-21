# sciengine/node/con_search_node.py
"""
单线程顺序查询文献
"""

import json
import traceback
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from sciengine.model.llm_models import get_chat_model
from sciengine.agent.agent_prompts import SEARCH_SYSTEM_PROMPT
from sciengine.tools.search_tools import search_tools
from sciengine.agent.utils import debug_log
from sciengine.agent.overallstate import OverallState


# 创建本地 search_agent
llm = get_chat_model()



search_agent = create_react_agent(
    model=llm,
    tools=search_tools,
    prompt=SystemMessage(content=SEARCH_SYSTEM_PROMPT),
)

# Search Node
def run_search_node(state: OverallState) -> OverallState:
    debug_log("Starting Search Agent node")
    try:
        config = {
            "configurable": {
                "recursion_limit": 5,  # 严格限制5次
                "max_iterations": 6  # 最多6个任务
            }
        }
        search_results = []
        all_tasks = state.get("tasks", [])
        search_tasks = [task for task in all_tasks if task.get('agent') == 'Search Agent']
        debug_log(f"Found {len(search_tasks)} Search Agent tasks")

        for task in search_tasks:
            instruction = task['instruction']
            task_id = task['task_id']
            debug_log(f"Processing task {task_id}: {instruction[:100]}...")
            search_state = {
                "question": instruction,
                "pmids": [],
                "results": [],
                "messages": [HumanMessage(content=instruction)]
            }
            search_result = search_agent.invoke(search_state)
            debug_log(f"Search Agent result for task {task_id}: {search_result}")

            if isinstance(search_result, dict) and "messages" in search_result:
                last_search_msg = search_result["messages"][-1]
                if isinstance(last_search_msg, AIMessage):
                    try:
                        result_json = json.loads(last_search_msg.content)
                        search_results.append(result_json)
                        debug_log(f"Task {task_id} result parsed")
                    except json.JSONDecodeError:
                        search_results.append({
                            "task_id": task_id,
                            "task": task,
                            "result": last_search_msg.content
                        })
                        debug_log(f"Task {task_id} result not JSON, stored as string")
                else:
                    debug_log(f"Task {task_id} returned non-AIMessage")
            else:
                debug_log(f"Task {task_id} returned unexpected result format")
        state["search_results"] = search_results
        debug_log("Search Agent node completed")
        return state
    except Exception as e:
        debug_log(f"Search node exception: {str(e)}")
        traceback.print_exc()
        return state