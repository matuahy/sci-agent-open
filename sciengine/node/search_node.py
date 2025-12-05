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


# 创建本地 planner_agent
llm = get_chat_model()

# Search Agent
search_agent = create_react_agent(
    model=llm,
    tools=search_tools,
    prompt=SystemMessage(content=SEARCH_SYSTEM_PROMPT),
)

# Search Node
def run_search_node(state: OverallState) -> OverallState:
    debug_log("Starting Search Agent node")
    try:
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

            # --------------------------
            # MUST find the final AIMessage
            # --------------------------
            if isinstance(search_result, dict) and "messages" in search_result:
                last_msg = search_result["messages"][-1]

                if isinstance(last_msg, AIMessage):
                    raw_content = last_msg.content

                    # 💡 尝试解析 JSON
                    try:
                        parsed = json.loads(raw_content)

                        # 如果成功解析但结构不完整，也自动补全
                        if not isinstance(parsed, dict):
                            raise ValueError("Parsed JSON is not a dict")

                        if "result" not in parsed:
                            parsed["result"] = {
                                "papers": [],
                                "datasets": [],
                                "explanation": "Missing result field, auto-filled"
                            }
                        else:
                            # Fill missing keys inside result
                            parsed["result"].setdefault("papers", [])
                            parsed["result"].setdefault("datasets", [])
                            parsed["result"].setdefault("explanation", "")

                        search_results.append(parsed)
                        debug_log(f"Task {task_id} JSON parsed OK")
                        continue

                    except Exception as e:
                        debug_log(f"Task {task_id} JSON parse failed: {e}")

                        # ❗ ALWAYS return valid JSON
                        fallback_json = {
                            "task_id": task_id,
                            "task": {
                                "agent": "Search Agent",
                                "instruction": instruction
                            },
                            "result": {
                                "papers": [],
                                "datasets": [],
                                "explanation": f"Search Agent returned non-JSON output: {raw_content}"
                            }
                        }

                        search_results.append(fallback_json)
                        debug_log(f"Task {task_id} stored as fallback JSON")
                        continue

                else:
                    debug_log(f"Task {task_id} last message is not AIMessage")
            else:
                debug_log(f"Task {task_id} returned unexpected result format")

            # Ultimate fallback (should rarely happen)
            fallback_json = {
                "task_id": task_id,
                "task": {
                    "agent": "Search Agent",
                    "instruction": instruction
                },
                "result": {
                    "papers": [],
                    "datasets": [],
                    "explanation": "Search Agent returned invalid format"
                }
            }
            search_results.append(fallback_json)

        # Save into state
        state["search_results"] = search_results
        debug_log("Search Agent node completed")
        return state

    except Exception as e:
        debug_log(f"Search node exception: {str(e)}")
        traceback.print_exc()
        return state
