import os
import json
from typing import List, Dict, Any
import trafilatura
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from sciengine import llm_models,bioembedding,pubmed_to_pmc
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import asyncio

class Pubmed_RAG:
    """
    - PubMed → PMC 解析
    - 下载全文
    - 创建向量库
    - 对 outline 执行 RAG
    - 使用 LLM 生成响应
    """

    def __init__(self, embedding_path="/root/autodl-tmp/backend/biobert-embeddings"):
        self.embedding = bioembedding.BioBERTEmbeddings(embedding_path)
        self.llm = llm_models.get_chat_model()
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
                res = pubmed_to_pmc.extract_pmc_link_from_pubmed(url)
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
            
        
        
    # ==============================================================
    # ① 单个文章处理函数（同步，线程安全）
    # ==============================================================
    def _process_one_paper(
        paper: Dict[str, Any],
        splitter: RecursiveCharacterTextSplitter,
        embedding,
        temp_dir: str,
        method_name: str  # "fixed" or "par"
    ) -> Dict[str, Any]:
        """
        处理一篇论文：切块 → 写入临时 Chroma → 返回临时路径
        """
        idx = paper.get("idx", -1)
        content = paper.get("content")
        title = paper.get("title", "Unknown Title")

        if not content or not isinstance(content, str):
            return {"idx": idx, "status": "skipped", "reason": "no content"}

        print(f"[{method_name}] 正在处理 #{idx+1}: {title}")

        try:
            # 切片
            chunks = splitter.split_text(content)
            if not chunks:
                return {"idx": idx, "status": "skipped", "reason": "empty chunks"}

            # metadata
            metas = [
                {
                    "pmcid": paper.get("pmcid"),
                    "title": title,
                    "pubmed_url": paper.get("pubmed_url")
                }
                for _ in chunks
            ]

            # 临时向量库路径（每篇一个）
            temp_db_path = os.path.join(temp_dir, f"temp_db_{idx}_{os.getpid()}_{id(paper)}")
            os.makedirs(temp_db_path, exist_ok=True)

            # 写入临时库
            Chroma.from_texts(
                texts=chunks,
                metadatas=metas,
                embedding=embedding,
                persist_directory=temp_db_path
            )

            print(f"[{method_name}] 已写入 {len(chunks)} 个 chunk → {temp_db_path}")
            return {
                "idx": idx,
                "status": "success",
                "temp_path": temp_db_path,
                "chunk_count": len(chunks)
            }

        except Exception as e:
            print(f"[{method_name}] 处理失败 #{idx+1}: {e}")
            return {"idx": idx, "status": "error", "reason": str(e)}    

    # ==============================================================
    # ① 并发，构建向量数据库-固定长度
    # ==============================================================
    async def create_VDB_fixed(self, papers: List[Dict[str, Any]]):
        """
        并发版：每篇文章并发处理，固定长度切块
        """
        print("Starting 并发向量库构建 (create_VDB_fixed)")
        os.makedirs(self.persist_directory, exist_ok=True)

        # 固定长度切块
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

        # 临时目录
        with tempfile.TemporaryDirectory() as temp_root:
            temp_dir = os.path.join(temp_root, "chroma_temp")
            os.makedirs(temp_dir, exist_ok=True)

            papers_with_idx = [{**p, "idx": i} for i, p in enumerate(papers)]
            total = len(papers_with_idx)
            max_workers = min(6, total, os.cpu_count() or 1)
            print(f"Using {max_workers} threads for {total} papers")

            results = []
            loop = asyncio.get_event_loop()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    loop.run_in_executor(
                        executor,
                        _process_one_paper,
                        paper,
                        splitter,
                        self.embedding,
                        temp_dir,
                        "fixed"
                    )
                    for paper in papers_with_idx
                ]

                for i, future in enumerate(futures, 1):
                    res = await future
                    results.append(res)
                    if res["status"] == "success":
                        print(f"Completed {i}/{total}: {res['chunk_count']} chunks")
                    else:
                        print(f"Failed {i}/{total}: {res.get('reason', 'unknown')}")

            # ========== 合并所有临时库 ==========
            print("Merging all temporary DBs...")
            main_db = None
            merged_count = 0

            for res in results:
                if res["status"] != "success":
                    continue
                temp_path = res["temp_path"]
                try:
                    temp_db = Chroma(persist_directory=temp_path, embedding_function=self.embedding)
                    if main_db is None:
                        import shutil
                        shutil.rmtree(self.persist_directory, ignore_errors=True)
                        shutil.copytree(temp_path, self.persist_directory)
                        main_db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
                        merged_count += res["chunk_count"]
                    else:
                        docs = temp_db._collection.get(include=["documents", "metadatas", "embeddings"])
                        if docs["ids"]:
                            main_db._collection.add(
                                ids=docs["ids"],
                                documents=docs["documents"],
                                metadatas=docs["metadatas"],
                                embeddings=docs["embeddings"]
                            )
                            merged_count += len(docs["ids"])
                    import shutil
                    shutil.rmtree(temp_path, ignore_errors=True)
                except Exception as e:
                    print(f"Merge failed: {e}")

            print(f"Vector DB 合并完成！总计 {merged_count} 个 chunks")
            print(f"Main DB: {self.persist_directory}")
        
    # ==============================================================
    # ① 并发，构建向量数据库-段落
    # ==============================================================    
        
    async def create_VDB_par(self, papers: List[Dict[str, Any]]):
        """
        并发版：每篇文章并发处理，优先按段落切块
        """
        print("Starting 并发向量库构建 (create_VDB_par)")
        os.makedirs(self.persist_directory, exist_ok=True)

        # 优先按段落切块
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=500,
            chunk_overlap=75
        )

        # 临时目录（所有临时向量库）
        with tempfile.TemporaryDirectory() as temp_root:
            temp_dir = os.path.join(temp_root, "chroma_temp")
            os.makedirs(temp_dir, exist_ok=True)

            # 添加索引
            papers_with_idx = [{**p, "idx": i} for i, p in enumerate(papers)]

            total = len(papers_with_idx)
            max_workers = min(6, total, os.cpu_count() or 1)
            print(f"Using {max_workers} threads for {total} papers")

            results = []
            loop = asyncio.get_event_loop()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    loop.run_in_executor(
                        executor,
                        _process_one_paper,
                        paper,
                        splitter,
                        self.embedding,
                        temp_dir,
                        "par"
                    )
                    for paper in papers_with_idx
                ]

                for i, future in enumerate(futures, 1):
                    res = await future
                    results.append(res)
                    if res["status"] == "success":
                        print(f"Completed {i}/{total}: {res['chunk_count']} chunks")
                    else:
                        print(f"Failed {i}/{total}: {res.get('reason', 'unknown')}")

            # ========== 合并所有临时库到主库 ==========
            print("Merging all temporary DBs into main DB...")
            main_db = None
            merged_count = 0

            for res in results:
                if res["status"] != "success":
                    continue
                temp_path = res["temp_path"]
                try:
                    temp_db = Chroma(persist_directory=temp_path, embedding_function=self.embedding)
                    if main_db is None:
                        # 第一个成功的临时库 → 直接迁移为主库
                        import shutil
                        shutil.rmtree(self.persist_directory, ignore_errors=True)
                        shutil.copytree(temp_path, self.persist_directory)
                        main_db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
                        merged_count += res["chunk_count"]
                    else:
                        # 后续库：使用 merge_from
                        docs = temp_db._collection.get(include=["documents", "metadatas", "embeddings"])
                        if docs["ids"]:
                            main_db._collection.add(
                                ids=docs["ids"],
                                documents=docs["documents"],
                                metadatas=docs["metadatas"],
                                embeddings=docs["embeddings"]
                            )
                            merged_count += len(docs["ids"])
                    # 清理临时
                    import shutil
                    shutil.rmtree(temp_path, ignore_errors=True)
                except Exception as e:
                    print(f"Merge failed for temp {temp_path}: {e}")

            print(f"Vector DB 合并完成！总计 {merged_count} 个 chunks 写入主库")
            print(f"Main DB: {self.persist_directory}")
        
        
    # =====================================================================
    # 执行全部程序
    # =====================================================================
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

    
    
    
