# sciengine/bioembedding_model.py
"""
BioBERTEmbeddings 是一个基于 BioBERT 的文本向量化组件，用于 生物医学领域 RAG（Retrieval-Augmented Generation）系统
它实现了 LangChain 的 Embeddings 接口，可直接作为 LLM 检索模块的向量编码器，
用于构建向量数据库（如 Chroma、FAISS）或在查询时生成 Query Embeddings
"""
from langchain.embeddings.base import Embeddings
import torch
from transformers import AutoTokenizer, AutoModel

class BioBERTEmbeddings(Embeddings):
    def __init__(self, model_path="/root/autodl-tmp/backend/biobert-embeddings"):
        # 只在初始化加载 CPU 权重（最安全）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BioBERT] 加载 tokenizer + model 到 CPU...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 强制在 CPU 上加载，不会触发 meta tensor
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
        self.model.eval()

        import torch as torch_module
        self.torch = torch_module

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        # 每次调用时，动态把模型搬到 GPU，仅一次
        self.model = self.model.to(self.device)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().tolist()
