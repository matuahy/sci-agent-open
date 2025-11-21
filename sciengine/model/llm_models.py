# sciengine/llm_model.py
"""
封装模型，阿里百炼
"""
def get_chat_model():
    from langchain_openai import ChatOpenAI  # 延迟导入
    import os
    llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus"
    )
    return llm