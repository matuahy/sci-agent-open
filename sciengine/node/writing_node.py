# sciengine/node/writing_node.py
"""
根据plan agent的大纲<title & content>，进行写作;有两个agent，question_agent对大纲进行提问，generate_agent进行撰写
- 顺序写作
- 多章节并发写作（Transformer 易报错）
"""
import json
import multiprocessing
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from sciengine.model.llm_models import get_chat_model
from sciengine.tools.writing_tools import strongest_retrieve
from sciengine.agent.agent_prompts import QUESTION_SYSTEM_PROMPT, GENERATE_SYSTEM_PROMPT
import os
from sciengine.model.bioembedding_model import BioBERTEmbeddings
from sciengine.agent.utils import info, error

# 初始化model及agent
embedding = BioBERTEmbeddings("/root/autodl-tmp/backend/biobert-embeddings")

llm = get_chat_model()

question_agent = create_react_agent(
    model=llm, tools=[], prompt=SystemMessage(content=QUESTION_SYSTEM_PROMPT)
)
generate_agent = create_react_agent(
    model=llm, tools=[], prompt=SystemMessage(content=GENERATE_SYSTEM_PROMPT)
)


# ==============================
# 单章节写作
# ==============================
def _write_one_section(section: Dict[str, Any], overallstate: Dict[str, Any]) -> Dict[str, Any]:
    report_outline = overallstate["planner_output"]["report_outline"]
    sections = report_outline.get("sections", [])
    section_title = section.get("title", "未知")
    info(f"开始处理章节: {section_title}")

    def process(sec: Dict[str, Any], depth=0) -> Dict[str, Any]:
        indent = "  " * depth
        title = sec.get("title", "Untitled")
        content = sec.get("content", "")
        print(f"{indent}处理: {title}")

        # Step 1: 生成问题

        input_json = json.dumps({"title": title, "content": content}, ensure_ascii=False)
        try:
            q_resp = question_agent.invoke({"messages": [HumanMessage(content=input_json)]})
            q_data = json.loads(q_resp['messages'][-1].content)
            query = q_data.get("query", title)
            questions = q_data.get("questions", [])
        except Exception as e:
            error(f"{indent}提问失败: {e}")
            query = title
            questions = [f"What is known about {title}?"]

        # Step 2: 检索
        context = []
        for q in questions:
            context.extend(strongest_retrieve(q, overallstate))
        seen = set()
        context = [d for d in context if d.page_content not in seen and not seen.add(d.page_content)]

        # Step 3: 构造 snippets（含 metadata）
        snippets = [
            {
                "text": d.page_content[:280] + "...",
                "title": d.metadata.get("title", "Unknown Title"),
                "pubmed_url": d.metadata.get("pubmed_url", "")
            }
            for d in context[:5]
        ]

        # Step 4: 调用生成
        input_msg = (
            f"Query: {query}\n"
            f"Questions: {json.dumps(questions, ensure_ascii=False)}\n"
            f"Context: {json.dumps(snippets, ensure_ascii=False)}\n"
            f"Outline: {json.dumps(report_outline, ensure_ascii=False)}"
        )
        try:
            g_resp = generate_agent.invoke({"messages": [HumanMessage(content=input_msg)]})
            result = json.loads(g_resp['messages'][-1].content)
        except Exception as e:
            error(f"{indent}生成失败: {e}")
            result = {"section_title": title, "content": f"**生成失败**: {e}", "subsections": []}

        # Step 5: 子章节递归
        subs = sec.get("subsections", [])
        if subs:
            result["subsections"] = [process(s, depth + 1) for s in subs]

        return result

    result = process(section)
    info(f"章节完成: {section_title}")
    return result


# ============================
# 并发写作入口
# ============================
async def run_con_writing_node(overallstate: Dict[str, Any], output_path="./final_report.json"):
    report_outline = overallstate["planner_output"].get("report_outline", {})
    sections = report_outline.get("sections", [])

    top_sections = [
        s for s in sections if str(s.get("section_number", "")).isdigit()
                               and len(str(s.get("section_number"))) == 1
    ]

    max_workers = min(multiprocessing.cpu_count(), len(top_sections), 6)
    info(f"开始并发写作: {len(top_sections)} 个章节，线程={max_workers}")

    results = []
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        tasks = [
            loop.run_in_executor(pool, _write_one_section, sec, overallstate)
            for sec in top_sections
        ]

        for i, t in enumerate(tasks, 1):
            r = await t
            results.append(r)
            info(f"完成章节 {i}/{len(top_sections)}")

        final_report = {"title": report_outline.get("title", "Generated Report"), "sections": results}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)

        info(f"报告生成成功，保存至: {os.path.abspath(output_path)}")
        # 直接写回 state，供前端读取，不再写磁盘
        overallstate["final_report"] = final_report

        info("写作完成，报告已写入 state")
        return overallstate


# ============================
# 顺序写作入口
# ============================

async def run_writing_node(overallstate: Dict[str, Any], output_path: str = "./final_report.json"):
    """
    顺序写作：按顶级章节顺序写作，子章节由 _write_one_section 内部递归
    """
    report_outline = overallstate["planner_output"].get("report_outline", {})
    sections = report_outline.get("sections", [])

    # 只处理顶级章节
    top_sections = [
        s for s in sections
        if str(s.get("section_number", "")).isdigit()
           and len(str(s.get("section_number")).split(".")) == 1
    ]

    info(f"开始顺序写作，共 {len(top_sections)} 个顶级章节")
    final_sections = []

    loop = asyncio.get_event_loop()

    for i, sec in enumerate(top_sections, 1):
        section_title = sec.get("title", "未知章节")
        info(f"正在写作第 {i}/{len(top_sections)} 章: {section_title}")

        try:
            # 用线程池调用同步函数 _write_one_section
            result = await loop.run_in_executor(None, _write_one_section, sec, overallstate)
            final_sections.append(result)
            info(f"第 {i} 章写作完成: {section_title}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            error(f"第 {i} 章写作失败: {e}")
            final_sections.append({
                "section_title": section_title,
                "content": f"**写作失败**: {e}",
                "subsections": []
            })

    # 组装最终报告
    final_report = {
        "title": report_outline.get("title", "Scientific Review Report"),
        "sections": final_sections
    }

    # 保存到文件
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            import json
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        info(f"报告已保存到: {os.path.abspath(output_path)}")
    except Exception as e:
        error(f"保存报告失败: {e}")

    # 写回 state
    overallstate["final_report"] = final_report
    info("顺序写作全部完成！报告已写入 state['final_report']")

    return overallstate

