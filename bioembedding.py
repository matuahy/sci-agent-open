from langchain.embeddings.base import Embeddings

class BioBERTEmbeddings(Embeddings):
    def __init__(self, model_path="/root/autodl-tmp/backend/biobert-embeddings"):
        from transformers import AutoTokenizer, AutoModel
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.torch = torch

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy().tolist()

# 使用
embedder = BioBERTEmbeddings("/root/autodl-tmp/backend/biobert-embeddings")
