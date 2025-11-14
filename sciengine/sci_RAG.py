import os
import json
from typing import List, Dict, Any
import trafilatura
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from pubmed_to_pmc import extract_pmc_link_from_pubmed
from bioembedding import BioBERTEmbeddings
from llm_models import get_chat_model


class Pubmed_RAG:
    """
    - PubMed → PMC 解析
    - 下载全文
    - 创建向量库
    - 对 outline 执行 RAG
    - 使用 LLM 生成响应
    """

    def __init__(self, embedding_path="/root/autodl-tmp/backend/biobert-embeddings"):
        self.embedding = BioBERTEmbeddings(embedding_path)
        self.llm = get_chat_model()
        self.persist_directory = "./chroma_papers"

        
    # =====================================================================
    # ⑧ 从 search_results 中提取 PubMed URL（你需要的功能）
    # =====================================================================
    def extract_pubmed_urls_from_tasks(self, search_results: List[Dict[str, Any]]) -> List[str]:
        """
        从 state['search_results'] 中提取所有 PubMed URL。
        search_results 的结构一般为：
            [
                {
                    "result": {
                        "papers": [
                            {"url": "..."},
                            ...
                        ]
                    }
                },
                ...
            ]
        """
        urls = []

        for task in search_results:
            result = task.get("result", {})
            papers = result.get("papers", [])

            for paper in papers:
                url = paper.get("url", "")
                if isinstance(url, str) and "pubmed" in url.lower():
                    urls.append(url)

        # 去重
        urls = list(set(urls))

        print(f"✅ 共提取到 {len(urls)} 条 PubMed 链接")
        return urls
        
        
    # =====================================================================
    # ① PubMed → PMC
    # =====================================================================
    def batch_get_pmcid(self, urls: List[str]) -> List[Dict[str, Any]]:
        results = []
        for url in urls:
            try:
                res = extract_pmc_link_from_pubmed(url)
                if res:
                    results.append(res)
                else:
                    print(f"[WARN] 无法解析 {url}")
            except Exception as e:
                print(f"[ERROR] {url}: {e}")
        return results

    # =====================================================================
    # ② 下载 PMC 全文
    # =====================================================================
    def get_paper_content(self, pmcid_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for item in pmcid_items:
            pmc_url = item.get("pmc_url")
            print(f"Downloading: {pmc_url}")

            downloaded = trafilatura.fetch_url(pmc_url)
            text = trafilatura.extract(downloaded) if downloaded else None

            results.append({
                "pubmed_url": item.get("pubmed_url"),
                "pmcid": pmc_url,
                "title": item.get("title"),
                "content": text
            })

        # 保存到 json（可选）
        with open("../../../../../../../EdgeX/paper_content.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results

    # =====================================================================
    # ③ 创建向量数据库
    # =====================================================================
    def create_VDB_fixed(self,papers):
        """
        从 paper_content.json 中读取内容，
        每篇文章单独切片并写入向量库（在 for 循环内部添加 Chroma.from_texts）
        """

        # BERT chunk 建议 <= 300 字符
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

        # 创建向量库目录
        os.makedirs(self.persist_directory, exist_ok=True)

        # ============================
        # ✅ 遍历每篇文章 单独写入 Chroma
        # ============================
        for idx, paper in enumerate(papers):
            content = paper.get("content")
            title = paper.get("title")

            if not content or not isinstance(content, str):
                print(f"[跳过] 第 {idx+1} 篇文章（无 content）")
                continue

            print(f"📘 正在处理: {title}")

            # 切片
            chunks = splitter.split_text(content)

            if not chunks:
                print("⚠️ 切片为空，跳过")
                continue

            # metadata 对应每个 chunk
            metas = [
                {
                    "pmcid": paper.get("pmcid"),
                    "title": paper.get("title"),
                    "pubmed_url": paper.get("pubmed_url")
                }
                for _ in chunks
            ]

            # ✅ 将 chunks 写入 Chroma（逐篇写入）
            Chroma.from_texts(
                texts=chunks,
                metadatas=metas,
                embedding=self.embedding,
                persist_directory=self.persist_directory
            )

            print(f"✅ 已写入 {len(chunks)} 个 chunk 到向量库")

        print("🎉 向量库构建完成（逐篇写入模式）!")


    # =====================================================================
    # ③ 创建向量数据库 (已修改为优先按段落切块)
    # =====================================================================
    def create_VDB_par(self, papers):
        """
        从 paper_content.json 中读取内容，
        每篇文章单独切片并写入向量库。
        
        修改：使用 RecursiveCharacterTextSplitter，优先按段落分隔符切块。
        """

        # BERT chunk 建议 <= 300 字符。使用递归切块，优先按段落切分。
        # separators 顺序：双换行符 (段落)、单换行符、空格、字符
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""], 
            chunk_size=500,  # 仍然保留，用于处理超长段落
            chunk_overlap=75 # 仍然保留，用于处理超长段落的 overlap
        )

        # 创建向量库目录
        os.makedirs(self.persist_directory, exist_ok=True)

        # ============================
        # ✅ 遍历每篇文章 单独写入 Chroma
        # ============================
        for idx, paper in enumerate(papers):
            content = paper.get("content")
            title = paper.get("title")

            if not content or not isinstance(content, str):
                print(f"[跳过] 第 {idx+1} 篇文章（无 content）")
                continue

            print(f"📘 正在处理: {title}")

            # 切片 (现在它会优先按段落切片)
            chunks = splitter.split_text(content)

            if not chunks:
                print("⚠️ 切片为空，跳过")
                continue

            # metadata 对应每个 chunk
            metas = [
                {
                    "pmcid": paper.get("pmcid"),
                    "title": paper.get("title"),
                    "pubmed_url": paper.get("pubmed_url")
                }
                for _ in chunks
            ]

            # ✅ 将 chunks 写入 Chroma（逐篇写入）
            Chroma.from_texts(
                texts=chunks,
                metadatas=metas,
                embedding=self.embedding,
                persist_directory=self.persist_directory
            )

            print(f"✅ 已写入 {len(chunks)} 个 chunk 到向量库")

        print("🎉 向量库构建完成（逐篇写入模式）!")

        
        
    # =====================================================================
    # ④ 一个文本执行 RAG 查询
    # =====================================================================
    def rag_query(self, text: str, k=3):
        if not text.strip():
            return []
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding
        )
        results = db.similarity_search_with_score(text, k=k)

        formatted = []
        for doc, score in results:
            formatted.append({
                "text": doc.page_content,
                "score": score,
                "metadata": doc.metadata
            })
        return formatted

    # =====================================================================
    # ⑤ 对 outline 执行 RAG 查询
    # =====================================================================
    def query_outline(self, state: Dict[str, Any]) -> Dict[str, Any]:
        outline = state["planner_output"]["report_outline"]["section"]
        outline_obj = outline.get("report_outline", outline)

        for section in outline_obj.get("sections", []):
            section["paper_response_title"] = self.rag_query(section.get("title", ""))
            section["paper_response_content"] = self.rag_query(section.get("content", ""))

            for sub in section.get("subsections", []):
                sub["paper_response_title"] = self.rag_query(sub.get("title", ""))
                sub["paper_response_content"] = self.rag_query(sub.get("content", ""))

        outline["report_outline"] = outline_obj
        return outline

    # =====================================================================
    # ⑥ 用 LLM 生成文本
    # =====================================================================
    def llm_generate(self, query, context):
        prompt = (
            "You are a scientific summarizer.\n\n"
            f"Query: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Write a concise scientific explanation."
        )
        response = self.llm.invoke([HumanMessage(prompt)])
        return response.content

    def _compose_context(self, items):
        out = []
        for it in items:
            meta = it.get("metadata", {})
            src = meta.get("title") or meta.get("pmcid") or meta.get("pubmed_url")
            out.append(f"[{src}]\n{it.get('text','')}")
        return "\n---\n".join(out)

    # =====================================================================
    # ⑦ 对 outline 的每个条目生成 LLM 内容
    # =====================================================================
    def generate_responses(self, outline: Dict[str, Any]) -> Dict[str, Any]:
        outline_obj = outline.get("report_outline", outline)

        for section in outline_obj.get("sections", []):

            # section-title
            ctx = self._compose_context(section.get("paper_response_title", []))
            section["generate_response_title"] = self.llm_generate(section["title"], ctx)

            # section-content
            ctx = self._compose_context(section.get("paper_response_content", []))
            section["generate_response_content"] = self.llm_generate(
                section.get("content", section["title"]), ctx
            )

            for sub in section.get("subsections", []):
                ctx1 = self._compose_context(sub.get("paper_response_title", []))
                ctx2 = self._compose_context(sub.get("paper_response_content", []))

                sub["generate_response_title"] = self.llm_generate(sub["title"], ctx1)
                sub["generate_response_content"] = self.llm_generate(
                    sub.get("content", sub["title"]), ctx2
                )

        outline["report_outline"] = outline_obj
        return outline
    
    def run_RAG(self, state):
        pubmed_urls = self.extract_pubmed_urls_from_tasks(state["search_results"])
        print("已提取 pubmed 链接")

        pmcid_urls = self.batch_get_pmcid(pubmed_urls)
        print("已提取 pmcid 链接")

        paper_content = self.get_paper_content(pmcid_urls)
        print("已获取 paper content")

        self.create_VDB_par(paper_content)
        print("已构建向量数据库")

        state["paper_content"] = paper_content
        state["chroma_dir"] = self.persist_directory
        print("已更新state")
        
        return {"paper_content": state["paper_content"],
               "chroma_dir":state["chroma_dir"]}

    
    
    
