import json
import os
from datetime import datetime

def save_state_for_reading_agent(state, filename_prefix="searchplan_output"):
    """保存 search_plan 阶段的 overall state 为 JSON 文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    output_path = os.path.join(os.getcwd(), filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved searchplan output to {output_path}")
    return output_path