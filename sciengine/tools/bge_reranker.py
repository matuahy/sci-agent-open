# packages/agent/bge_reranker.py
'''
BgeReranker 是一个基于 BGE ReRanker（bge-reranker-large） 的句对重排序模块，
用于在 RAG 或信息检索系统中对候选文档进行相关性重排序（Re-ranking）
该模块依赖本地加载的 BGE 模型，在离线环境下运行，并输出 query–passage 的语义匹配分数。
'''

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 强制离线
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class BgeReranker:
    def __init__(self, model_path="/root/autodl-tmp/backend/bge-reranker-large"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"bge-reranker-large not found at {model_path}")

        print(f"[Reranker] Loading from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        # 关键：使用 ForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print("[Reranker] Loaded successfully!")

    def compute_score(self, sentence_pairs):
        """
        输入: [("query", "passage"), ...]
        输出: [score1, score2, ...] (higher = more relevant)
        """
        # 构造 input: query [SEP] passage
        texts = [f"{q} [SEP] {p}" for q, p in sentence_pairs]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # 正确获取 logits
            scores = outputs.logits.view(-1).float()
        return scores.cpu().numpy().tolist()