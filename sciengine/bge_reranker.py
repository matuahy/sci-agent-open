# sciengine/bge_reranker.py
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