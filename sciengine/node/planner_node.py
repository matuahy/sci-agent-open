from sciengine.agent.agent_prompts import PLAN_SYSTEM_PROMPT
from typing_extensions import TypedDict
import json
from typing import Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from operator import add
import traceback
from sciengine.llm_models import get_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from sciengine.utils import debug_log
from langgraph.graph import START, END
import os
from sciengine.agent.overallstate import OverallState

# 创建本地 planner_agent
llm = get_chat_model()

#————————————维护状态————————————————————

planner_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt=SystemMessage(content=PLAN_SYSTEM_PROMPT),
    name="planner_agent",
)

# Planner Node
def run_planner_node(state: OverallState) -> OverallState:
    debug_log("Starting Planner Agent node")
    try:
        result = planner_agent.invoke({
            "messages": [HumanMessage(content=state["query"])]
        })
        debug_log(f"Planner Agent result: {result}")
        if isinstance(result, dict) and "messages" in result:
            new_messages = result["messages"]
            last_message = new_messages[-1]
            if isinstance(last_message, AIMessage):
                try:
                    planner_json = json.loads(last_message.content)
                    state["planner_output"] = planner_json
                    state["tasks"] = planner_json.get("task_decomposition", [])
                    debug_log(f"Planner output parsed: {len(state['tasks'])} tasks")
                except json.JSONDecodeError as e:
                    debug_log(f"Planner output JSON decode error: {str(e)}")
                    state["planner_output"] = {"error": "Invalid JSON"}
        else:
            debug_log("Planner Agent returned unexpected result format")
        debug_log("Planner Agent node completed")
        return state
    except Exception as e:
        debug_log(f"Planner node exception: {str(e)}")
        traceback.print_exc()
        return state

    
