# 运行后端API
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any
import os
import json
import asyncio
from dotenv import load_dotenv
import logging
import traceback
from sciengine.agent.utils import save_state_for_reading_agent
from langchain_core.messages import BaseMessage

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Import the workflow from app_graph.py
from app_graph import app_graph as workflow_app
from app_graph import OverallState

# -----------------------------------------------------
# Initialize FastAPI app_graph
api_app = FastAPI(
    title="Multi-Agent Scientific Research Backend",
    description="API for running multi-agent workflow on scientific queries",
    version="1.0.0"
)

# CORS configuration
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get absolute path to static directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
api_app.mount("/static", StaticFiles(directory=static_dir), name="static")


class QueryInput(BaseModel):
    query: str


# ------------------------------------------------
# Health check & Favicon
@api_app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running"}


@api_app.get("/favicon.ico")
async def favicon():
    favicon_path = os.path.join(static_dir, "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return {"message": "Favicon not found"}


# ------------------------------------------------
#  Core Workflow Stream (带心跳保活)
# ------------------------------------------------
async def run_workflow_stream(input_query: str):
    q = asyncio.Queue()

    async def producer():
        try:
            initial_state: OverallState = {
                "query": input_query,
                "planner_output": {},
                "tasks": [],
                "search_results": [],
                "paper_content": [],
                "chroma_dir": "",
                "messages": []
            }

            await q.put(json.dumps({"type": "log", "content": "🚀 Workflow started (Keep-Alive enabled)..."}) + "\n")

            final_state = None

            async for output in workflow_app.astream(initial_state):
                for key, value in output.items():
                    final_state = value
                    log_msg = f"✅ Finished step: {key}"
                    await q.put(json.dumps({"type": "log", "content": log_msg, "node": key}) + "\n")

                    if key == "planner" and "planner_output" in value:
                        plan = value["planner_output"]
                        if "clarifying_questions" in plan and plan["clarifying_questions"]:
                            await q.put(
                                json.dumps({"type": "log", "content": "❓ Generated clarifying questions"}) + "\n")
                        else:
                            await q.put(json.dumps({"type": "log", "content": "正在进行search..."}) + "\n")

            if not final_state:
                await q.put(json.dumps({"type": "error", "content": "Workflow did not produce a final state"}) + "\n")
                return

            await q.put(json.dumps({"type": "log", "content": "💾 Saving state..."}) + "\n")

            for key in ["dynamic_bm25", "dynamic_docs", "dynamic_vectorstore", "retriever", "bm25"]:
                final_state.pop(key, None)

            save_state_for_reading_agent(final_state, filename_prefix="full_state")

            messages_serialized = [
                {"type": m.type, "content": m.content} if isinstance(m, BaseMessage) else m
                for m in final_state.get("messages", [])
            ]

            result_data = {
                "query": input_query,
                "planner_output": final_state.get("planner_output", {}),
                "tasks": final_state.get("tasks", []),
                "search_results": final_state.get("search_results", []),
                "chroma_dir": final_state.get("chroma_dir", ""),
                "messages": messages_serialized,
                "final_report": final_state.get("final_report", {})
            }

            await q.put(json.dumps({"type": "result", "data": result_data}) + "\n")

        except Exception as e:
            logger.error(f"Workflow error: {str(e)}")
            traceback.print_exc()
            await q.put(json.dumps({"type": "error", "content": str(e)}) + "\n")
        finally:
            await q.put(None)

    task = asyncio.create_task(producer())

    while True:
        try:
            data = await asyncio.wait_for(q.get(), timeout=10.0)
            if data is None: break
            yield data
        except asyncio.TimeoutError:
            yield json.dumps({"type": "ping"}) + "\n"

    await task


@api_app.post("/query")
async def process_query(input: QueryInput):
    logger.info(f"Processing query: {input.query}")
    return StreamingResponse(
        run_workflow_stream(input.query),
        media_type="application/x-ndjson"
    )


# --------------------- 文件下载路由 ---------------------

# 定义所有可能的输出目录路径
SEARCH_DIRS = [
    # 1. 您指定的绝对路径 (AutoDL 标准路径)
    # "/root/autodl-tmp/agent_test/outputs",
    "./outputs",
    # 2. 相对于 app.py 的 outputs 目录
    os.path.join(BASE_DIR, "outputs"),
    # 3. 相对于当前运行目录(CWD)的 outputs 目录
    os.path.join(os.getcwd(), "outputs")
]

ALLOWED_FILES = {
    "final_report_styled.docx",
    "final_report.md",
    "final_report.json"
}


@api_app.get("/download/{filename}")
async def download_report(filename: str):
    logger.info(f"📥 Received download request for: {filename}")

    if filename not in ALLOWED_FILES:
        logger.warning(f"⛔ Access denied for file: {filename}")
        raise HTTPException(status_code=403, detail="Access denied: File not allowed")

    # 遍历所有可能的目录寻找文件
    found_path = None
    checked_paths = []

    for directory in SEARCH_DIRS:
        possible_path = os.path.join(directory, filename)
        checked_paths.append(possible_path)
        if os.path.exists(possible_path):
            found_path = possible_path
            break

    if found_path:
        logger.info(f"✅ File found at: {found_path}")
        return FileResponse(
            path=found_path,
            filename=filename,
            media_type='application/octet-stream',
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    else:
        error_msg = f"File not found. Checked locations: {checked_paths}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=404, detail=f"File not found on server.")


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.getenv("PORT", 6006))
    logger.info(f"Starting server on port {port}...")
    uvicorn.run(api_app, host="0.0.0.0", port=port)