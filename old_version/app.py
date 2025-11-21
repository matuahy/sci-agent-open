from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Any
import os
from dotenv import load_dotenv
import logging
import traceback
from utils import save_state_for_reading_agent
from operator import add
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Import the workflow from search_plan.py
# from search_plan import app_graph as workflow_app
# from search_plan import OverallState
from read_search_plan import app as workflow_app
from read_search_plan import OverallState

# -----------------------------------------------------
# Initialize FastAPI app_graph
api_app = FastAPI(
    title="Multi-Agent Scientific Research Backend",
    description="API for running multi-agent workflow on scientific queries",
    version="1.0.0"
)

# Get absolute path to static directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")

# Mount static files
api_app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Define input model with validation
class QueryInput(BaseModel):
    query: str

    @classmethod
    def model_validate(cls, values):
        query = values.get("query", "")
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if len(query) > 500:
            raise ValueError("Query length exceeds 500 characters")
        return values


# ---------------------------------------------
# Define output model

class Task(BaseModel):
    task_id: str
    agent: str
    instruction: str


class ResponseOutput(BaseModel):
    query: str  # 用户的原始查询 (input.query)
    planner_output: Dict[str, Any]  # 规划器生成的大纲
    tasks: List[Task]  # 任务列表 (未使用但保留)
    search_results: List[Dict[str, Any]]  # 搜索结果 (PubMed URL列表)
    paper_content: List[Dict[str, Any]]  # 下载的全文（含 content）
    chroma_dir: str  # Chroma 持久化目录
    messages: Annotated[List[BaseMessage], add]  # 对话历史或 agent 间的消息


# ------------------------------------------------

# Health check endpoint
@api_app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running"}


# Favicon endpoint with absolute path
@api_app.get("/favicon.ico")
async def favicon():
    favicon_path = os.path.join(static_dir, "favicon.ico")
    logger.info(f"Checking favicon at: {favicon_path}")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return {"message": "Favicon not found"}


# GET /query for guidance
@api_app.get("/query")
async def get_query_guidance():
    return {
        "message": "Please use POST /query with a JSON body containing 'query' field.",
        "example": {"query": "how does Single-cell-RNA-sequence is used in Drosophila?"},
        "docs": "Visit /docs for interactive API documentation"
    }


# Main query endpoint
@api_app.post("/query", response_model=ResponseOutput)
async def process_query(input: QueryInput):
    try:
        logger.info(f"Processing query from {input.query}")

        initial_state: OverallState = {
            "query": input.query,
            "planner_output": {},
            "tasks": [],
            "search_results": [],
            "paper_content": [],
            "chroma_dir": "",
            "messages": []
        }

        logger.info("Running workflow")
        final_state = None
        for output in workflow_app.stream(initial_state):
            for key, value in output.items():
                logger.debug(f"Node {key} output: {value}")
                final_state = value
        if not final_state:
            raise ValueError("Workflow did not produce a final state")
        logger.info("Workflow completed successfully")

        save_state_for_reading_agent(final_state, filename_prefix="full_state")
        logger.info("✅ Saved full_state.json successfully.")

        return ResponseOutput(
            query=input.query,
            planner_output=final_state.get("planner_output", {}),
            tasks=final_state.get("tasks", []),
            search_results=final_state.get("search_results", []),
            paper_content=final_state.get("paper_content", []),
            chroma_dir=final_state.get("chroma_dir", ""),  # 正确位置
            messages=final_state.get("messages", []),
        )


    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api_app, host="0.0.0.0", port=8000)