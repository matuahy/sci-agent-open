# sciengine/bioembedding_model.py
"""
BioBERTEmbeddings 是一个基于 BioBERT 的文本向量化组件，用于 生物医学领域 RAG（Retrieval-Augmented Generation）系统
它实现了 LangChain 的 Embeddings 接口，可直接作为 LLM 检索模块的向量编码器，
用于构建向量数据库（如 Chroma、FAISS）或在查询时生成 Query Embeddings
"""
from langchain.embeddings.base import Embeddings
import torch
from transformers import AutoTokenizer, AutoModel
import os

# ---------------------------------------
# 本地模型路径
# ---------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(CURRENT_DIR, "biobert-embeddings")
model_path = os.path.abspath(model_path)

print("[BioBERT] MODEL PATH =", model_path)


class BioBERTEmbeddings(Embeddings):
    def __init__(self, model_path=model_path):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BioBERT 模型不存在: {model_path}")

        print(f"[BioBERT] 加载 tokenizer + model (local only) ...")

        # 强制本地加载，不访问HF
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.torch = torch

    # ---- langchain 接口实现 ----
    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    # ---- 嵌入函数 ----
    def _embed(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        # 使用 CLS token 向量
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().tolist()
