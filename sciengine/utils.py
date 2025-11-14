# sciengine/utils.py
import json
import os
from datetime import datetime
def debug_log(message: str):
    print(f"[DEBUG] {message}")

def save_state_for_reading_agent(state, filename_prefix="searchplan_output"):
    """保存 search_plan 阶段的 overall state 为 JSON 文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    output_path = os.path.join(os.getcwd(), filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved searchplan output to {output_path}")
    return output_path

# 测试 API 可用性
def test_api_connectivity():
    debug_log("Testing PubMed API connectivity...")
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
        debug_log(f"PubMed API status: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        debug_log(f"PubMed API test failed: {str(e)}")
        return False