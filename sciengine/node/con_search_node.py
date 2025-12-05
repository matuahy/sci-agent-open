# sciengine/node/con_search_node.py
"""
多线程并发查询文献
"""
import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import traceback
from sciengine.model.llm_models import get_chat_model
from sciengine.agent.agent_prompts import SEARCH_SYSTEM_PROMPT
from sciengine.tools.search_tools import search_tools
from sciengine.agent.utils import debug_log
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from sciengine.agent.overallstate import OverallState
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 创建本地 planner_agent
llm = get_chat_model()

# Search Agent
checkpointer = MemorySaver()

search_agent = create_react_agent(
    model=llm,
    tools=search_tools,
    prompt=SystemMessage(content=SEARCH_SYSTEM_PROMPT),
    checkpointer=checkpointer,  # 必须传入
    # 直接在这里设置 recursion_limit
).with_config({"recursion_limit": 50})


# Search Node
# -------------------------------------------------
# 单个 Search 任务的同步执行函数
# -------------------------------------------------
def _run_one_search_task(
        task: Dict[str, Any],
        search_agent,
        config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    同步执行一个 Search Agent 任务，返回解析好的 dict。
    """
    task_id = task.get("task_id", "UNKNOWN")
    instruction = task.get("instruction", "")
    debug_log(f"[Thread] Starting task {task_id}")

    try:
        # 1. 构造 search_agent 的初始 state
        search_state = {
            "question": instruction,
            "pmids": [],
            "results": [],
            "messages": [HumanMessage(content=instruction)]
        }

        # 2. 调用 ReAct 代理（同步）
        search_result = search_agent.invoke(search_state, config=config)

        # 3. 提取最后一条 AI 消息
        if not (isinstance(search_result, dict) and "messages" in search_result):
            raise ValueError("Unexpected result format")

        last_msg = search_result["messages"][-1]
        if not isinstance(last_msg, AIMessage):
            raise ValueError("Last message is not AIMessage")

        raw = last_msg.content.strip()

        # 4. 清理 markdown fences
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()

        # 5. 解析 JSON（可能嵌套）
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            pass

        if isinstance(parsed, dict) and "result" in parsed and isinstance(parsed["result"], str):
            try:
                parsed["result"] = json.loads(parsed["result"])
            except Exception:
                pass

        # 6. 统一返回结构
        if isinstance(parsed, dict):
            parsed.setdefault("task_id", task_id)
            parsed.setdefault("task", task)
            debug_log(f"[Thread] Task {task_id} parsed successfully")
            return parsed
        else:
            # fallback：保持 dict 结构防止 downstream 崩溃
            fallback = {
                "task_id": task_id,
                "task": task,
                "result": raw
            }
            debug_log(f"[Thread] Task {task_id} fallback (raw content)")
            return fallback

    except Exception as e:
        debug_log(f"[Thread] Task {task_id} error: {str(e)}")
        traceback.print_exc()
        return {
            "task_id": task_id,
            "task": task,
            "error": str(e),
            "result": None
        }


# -------------------------------------------------
# 并发版 run_search_node（替换原来的同步实现）
# -------------------------------------------------
async def run_search_node(state: OverallState) -> OverallState:
    debug_log("Starting concurrent Search Agent node")
    try:
        config = {
            "configurable": {
                "recursion_limit": 15,
                "max_iterations": 16
            }
        }

        # ---- 筛选 Search 任务 ----
        search_tasks = [
            t for t in state.get("tasks", [])
            if t.get("agent") == "Search Agent"
        ]

        total = len(search_tasks)
        debug_log(f"Found {total} Search tasks")

        if total == 0:
            state["search_results"] = []
            return state

        # ---- 并发执行 ----
        loop = asyncio.get_running_loop()
        max_workers = min(6, total, os.cpu_count() or 1)
        debug_log(f"Using ThreadPoolExecutor({max_workers})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 生成 future 列表
            futures = [
                loop.run_in_executor(
                    executor,
                    _run_one_search_task,
                    task,
                    search_agent,
                    config
                )
                for task in search_tasks
            ]

            # gather 等待所有完成
            results = await asyncio.gather(*futures, return_exceptions=False)

        state["search_results"] = results
        debug_log(f"Concurrent search finished ({len(results)} results)")

    except Exception as e:
        debug_log(f"Search node error: {str(e)}")
        traceback.print_exc()
        state["search_results"] = []

    return state


