# sciengine/tools/generate_report.py
import json
import os
import subprocess
from typing import Dict, Any


# =========================================================================
# 辅助函数 1: JSON 到 Markdown 转换
# =========================================================================

def json_to_markdown(report_data: Dict[str, Any], output_filename: str):
    """
    将报告的 JSON 结构转换为 Markdown 文本。
    """
    markdown_lines = []

    # H1 标题
    report_title = report_data.get("title", "未命名报告")
    markdown_lines.append(f"# {report_title}\n")

    # 遍历主章节 (H2)
    for section in report_data.get("sections", []):
        section_title = section.get("section_title")
        section_content = section.get("content")

        if section_title:
            markdown_lines.append(f"## {section_title}\n")

        # 写入主章节内容
        if section_content:
            markdown_lines.append(section_content)
            markdown_lines.append("")  # 添加空行用于段落分隔

        # 遍历子章节 (H3)
        for subsection in section.get("subsections", []):
            subsection_title = subsection.get("section_title")
            subsection_content = subsection.get("content")

            if subsection_title:
                markdown_lines.append(f"### {subsection_title}\n")

            # 写入子章节内容
            if subsection_content:
                markdown_lines.append(subsection_content)
                markdown_lines.append("")

    markdown_content = "\n".join(markdown_lines)

    # 保存 .md 文件
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"✅ Markdown 文件已保存: {output_filename}")

    except Exception as e:
        print(f"❌ 写入 {output_filename} 文件时出错: {e}")
        raise  # 抛出异常，阻止后续 Word 转换


# =========================================================================
# 辅助函数 2: Markdown 到 Word 转换 (调用 Pandoc)
# =========================================================================

def convert_markdown_to_word(markdown_file: str, reference_docx: str, output_docx: str):
    """
    使用 Pandoc 执行 Markdown 到 Word 的转换，并将结果保存为 .docx。
    """

    # Pandoc 命令的基础结构
    command = ["pandoc", markdown_file]

    # 检查参考文档，如果存在则添加样式参数
    if os.path.exists(reference_docx):
        command.extend(["--reference-doc", reference_docx])
    else:
        print(f"⚠️ 警告：参考文档未找到: {reference_docx}。将使用 Pandoc 默认样式。")

    # 添加输出文件名
    command.extend(["-o", output_docx])

    try:
        print(f"🚀 正在执行 Pandoc 转换...")

        # 执行命令
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True  # 失败时抛出 CalledProcessError
        )

        print(f"✅ Word 文件已保存: {output_docx}")

        # 打印 Pandoc 的任何输出信息
        if result.stderr:
            print("📢 Pandoc 报告的警告/错误：")
            print(result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"❌ Pandoc 转换失败:")
        print(f"Pandoc 错误信息: \n{e.stderr}")
    except FileNotFoundError:
        print("❌ 错误：找不到 'pandoc' 命令。请确保 Pandoc 已安装并配置到系统 PATH 中。")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

