# sciengine/utils.py
"""
后端打印调试信息，保存最终state文件
"""
import json
import os
from datetime import datetime
import requests


# =====================================================
# 标准化日志函数
# =====================================================

def info(message: str):
    print(f"[INFO] {message}")


def debug(message: str):
    print(f"[DEBUG] {message}")


def warn(message: str):
    print(f"[WARN] {message}")


def error(message: str):
    print(f"[ERROR] {message}")


# 兼容旧代码（仍然有人会调用 debug_log）
def debug_log(message: str):
    debug(message)


# =====================================================
# 安全保存 state（重点：保证 final_report 一定在！）
# =====================================================
def save_state_for_reading_agent(state, filename_prefix="full_state"):
    """
    安全保存整个 workflow 的最终 state
    - 主动清理不能序列化的对象（BM25、Chroma、Document 等）
    - 手动保证 final_report、paper_content 等关键字段完整保存
    """
    from langchain_core.messages import BaseMessage

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    output_path = os.path.join(os.getcwd(), filename)

    # 构建一个绝对安全的、可序列化的副本
    safe_state = {
        "saved_at": datetime.now().isoformat(),
        "query": state.get("query", ""),
        "planner_output": state.get("planner_output", {}),
        "search_results": state.get("search_results", []),
        "paper_content": state.get("paper_content", []),
        "chroma_dir": state.get("chroma_dir", ""),
        "session_id": state.get("session_id", ""),
        "final_report": state.get("final_report", {}),  # 重点保护！
    }

    # 处理 messages（可能是 BaseMessage 对象）
    messages = state.get("messages", [])
    serialized_messages = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            serialized_messages.append({
                "type": msg.type,
                "content": msg.content
            })
        elif isinstance(msg, dict):
            serialized_messages.append(msg)
        else:
            serialized_messages.append({
                "type": "unknown",
                "content": str(msg)
            })
    safe_state["messages"] = serialized_messages

    # 主动删除所有可能导致 json.dump 崩溃的字段（防止静默丢数据）
    dangerous_keys = [
        "dynamic_bm25", "dynamic_docs", "dynamic_vectorstore",
        "retriever", "bm25", "vectorstore", "docs"
    ]
    for key in dangerous_keys:
        state.pop(key, None)

    # 最终写入磁盘（现在绝对不会炸了）
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(safe_state, f, ensure_ascii=False, indent=2)
        info(f"已安全保存完整状态（含 final_report）→ {output_path}")
    except Exception as e:
        error(f"保存状态失败: {e}")
        # 即使出错了，也尝试保存一个最小版本
        minimal = {"error": "save_failed", "final_report": safe_state.get("final_report", {})}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(minimal, f, ensure_ascii=False, indent=2)
        info(f"已保存最小版本（仅含 final_report）→ {output_path}")

    return output_path


# =====================================================
# 测试 PubMed API
# =====================================================

def test_api_connectivity():
    debug("Testing PubMed API connectivity...")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": "test",
        "retmode": "xml",
        "retmax": 1,
        "api_key": os.environ.get("NCBI_API_KEY")
    }

    try:
        response = requests.get(base_url, params=params)
        debug(f"PubMed API status: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        error(f"PubMed API test failed: {str(e)}")
        return False
