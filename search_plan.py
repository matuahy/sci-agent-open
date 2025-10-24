import json
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from operator import add
import traceback
import os
import requests
import xmltodict
from langchain_core.tools import tool

# 调试日志函数
def debug_log(message: str):
    print(f"[DEBUG] {message}")

# 测试 API 可用性
def test_api_connectivity():
    debug_log("Testing PubMed API connectivity...")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": "test",
        "retmode": "xml",
        "retmax": 1,
        "api_key": os.environ.get("NCBI_API_KEY")
    }
    try:
        response = requests.get(base_url, params=params)
        debug_log(f"PubMed API status: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        debug_log(f"PubMed API test failed: {str(e)}")
        return False

# 导入 LLM 模型
try:
    from llm_models import get_chat_model
    llm = get_chat_model()
    debug_log("LLM initialized successfully")
except ImportError as e:
    debug_log(f"Failed to import llm_models: {str(e)}")
    llm = None

# Planner Agent
from langgraph.prebuilt import create_react_agent

planner_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt=(
        """

        You are the **Planner Agent** in a multi-agent scientific research system.  
        Your responsibility is to transform a user's request or question into a structured research plan.

        **CRITICAL TWO-STAGE LOGIC**:
        1. If query does NOT contain "Answers to clarifying questions:":
           - Generate clarifying_questions to resolve ambiguities
           - Set report_outline = {}, task_decomposition = [], assumptions = []
        2. If query CONTAINS "Answers to clarifying questions:":
           - Use the answers to generate COMPLETE plan
           - Set clarifying_questions = []

        **🚨 TASK DECOMPOSITION RULES (CRITICAL)**:
        - Maximum 4 search tasks (T1-T4)
        - Each search task MUST succeed independently
        - Use **OR logic** instead of **AND logic** for comparisons
        - Always include fallback: "Drosophila (any tissue)" if specific tissue has limited data
        - Each task instruction must contain: "retmax: 30" and "2015-2025"

        --- OUTPUT FORMAT (must be valid JSON) ---
        {
          "report_outline": {
            "title": "",
            "sections": [
              {
                "section_number": "",
                "title": "",
                "content": ""
              },
              {
                "section_number": "",
                "title": "",
                "subsections": [
                  {
                    "subsection_number": "",
                    "title": "",
                    "content": ""
                  }
                ]
              }
            ]
          },
          "task_decomposition": [
            {"task_id": "T1", "agent": "Search Agent", "instruction": ""},
            {"task_id": "T2", "agent": "Reading Agent", "instruction": ""}
          ],
          "clarifying_questions": [""],
        }
        """
    ),
    name="planner_agent",
)

# --- Search Agent Tools ---
@tool
def search_pubmed(query: str, retmax: int = 50) -> List[str]:
    """
    使用 Entrez eSearch API 搜索 PubMed 数据库，返回相关的文章 ID (PMID) 列表。
    """
    debug_log(f"Executing PubMed search with query: {query}, retmax: {retmax}")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "xml",
        "retmax": retmax,
        "mindate": "2018",
        "maxdate": "2025",
        "api_key": os.environ.get("NCBI_API_KEY")
    }
    try:
        response = requests.get(base_url, params=params)
        debug_log(f"PubMed API response status: {response.status_code}")
        if response.status_code == 200:
            data = xmltodict.parse(response.text)
            id_list_container = data['eSearchResult'].get('IdList')
            if id_list_container and 'Id' in id_list_container:
                ids = id_list_container['Id']
                ids = [ids] if isinstance(ids, str) else ids
                debug_log(f"Retrieved {len(ids)} PMIDs")
                return ids
            debug_log("No PMIDs found in response")
            return []
        else:
            debug_log(f"PubMed API search failed with status: {response.status_code}")
            return []
    except Exception as e:
        debug_log(f"Error in PubMed search: {str(e)}")
        return []

@tool
def fetch_pubmed_details(pmids: List[str]) -> List[Dict[str, Any]]:
    """
    使用 Entrez eFetch API 根据 PMID 列表获取论文的详细信息（标题、摘要、作者等）。
    """
    if not pmids:
        debug_log("No PMIDs provided for fetch_pubmed_details")
        return []
    pmid_str = ",".join(pmids[:50])
    debug_log(f"Fetching details for {len(pmids[:50])} PMIDs")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid_str,
        "retmode": "xml",
        "rettype": "abstract",
        "api_key": os.environ.get("NCBI_API_KEY")
    }
    try:
        response = requests.get(base_url, params=params)
        debug_log(f"PubMed fetch response status: {response.status_code}")
        if response.status_code == 200:
            data = xmltodict.parse(response.text)
            articles = []
            pubmed_article_set = data.get('PubmedArticleSet', {})
            raw_articles = pubmed_article_set.get('PubmedArticle')
            if not raw_articles:
                debug_log("No articles found in PubMed fetch response")
                return []
            if not isinstance(raw_articles, list):
                raw_articles = [raw_articles]
            for article in raw_articles:
                medline_citation = article.get('MedlineCitation', {})
                article_info = medline_citation.get('Article', {})
                pmid_data = medline_citation.get('PMID', '#N/A')
                title = article_info.get('ArticleTitle', 'No Title')
                abstract_text = article_info.get('Abstract', {}).get('AbstractText', 'No Abstract')
                if isinstance(abstract_text, list):
                    abstract_text = ' '.join([t.get('#text', '') if isinstance(t, dict) else str(t) for t in abstract_text])
                journal = article_info.get('Journal', {}).get('Title', 'No Journal')
                pub_date = article_info.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
                year = pub_date.get('Year', 'Unknown')
                author_list = article_info.get('AuthorList', {}).get('Author', [])
                if not isinstance(author_list, list):
                    author_list = [author_list] if author_list else []
                authors = []
                for author in author_list:
                    if isinstance(author, dict):
                        last_name = author.get('LastName', '')
                        fore_name = author.get('ForeName', '')
                        initials = author.get('Initials', '')
                        author_name = f"{last_name} {fore_name}" if fore_name else f"{last_name} {initials}"
                        authors.append(author_name.strip())
                authors_str = ", ".join(authors)
                doi_container = article.get('PubmedData', {}).get('ArticleIdList', {}).get('ArticleId', [])
                if not isinstance(doi_container, list):
                    doi_container = [doi_container] if doi_container else []
                doi = next((id.get('#text') for id in doi_container if id.get('@IdType') == 'doi'), 'No DOI')
                articles.append({
                    "pmid": pmid_data.get('#text', pmid_data) if isinstance(pmid_data, dict) else pmid_data,
                    "title": title,
                    "journal": journal,
                    "year": year,
                    "authors": authors_str,
                    "abstract": abstract_text,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_data}/",
                    "doi": doi
                })
            debug_log(f"Fetched details for {len(articles)} articles")
            return articles
        else:
            debug_log(f"PubMed fetch failed with status: {response.status_code}")
            return []
    except Exception as e:
        debug_log(f"Error in PubMed fetch: {str(e)}")
        return []


@tool
def search_geo(query: str, retmax: int = 20) -> List[str]:
    """
    使用 Entrez ESearch API 搜索 GEO，返回 GSE 访问号列表。
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "gds",
        "term": query,
        "retmode": "xml",
        "retmax": retmax,
        "api_key": os.environ.get("NCBI_API_KEY")
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        try:
            data = xmltodict.parse(response.text)
            id_list_container = data['eSearchResult'].get('IdList')
            if id_list_container and 'Id' in id_list_container:
                ids = id_list_container['Id']
                return [ids] if isinstance(ids, str) else ids
            return []
        except Exception as e:
            print(f"Error parsing GEO search: {e}")
            return []
    else:
        print(f"GEO search failed: {response.status_code}")
        return []

@tool
def fetch_geo_details(gse_ids: List[str]) -> List[Dict[str, Any]]:
    """
    使用 ESummary API 获取 GEO 数据集详情（标题、摘要、样本数等）。
    """
    if not gse_ids:
        return []
    gse_str = ",".join(gse_ids[:20])  # 限制最多 20 个
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "gds",
        "id": gse_str,
        "retmode": "xml",
        "api_key": os.environ.get("NCBI_API_KEY")
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        try:
            data = xmltodict.parse(response.text)
            doc_sum_set = data.get('eSummaryResult', {}).get('DocSum', [])
            datasets = []
            if not isinstance(doc_sum_set, list):
                doc_sum_set = [doc_sum_set]
            for doc in doc_sum_set:
                gse_id = doc.get('Id', 'No ID')
                summary_items = doc.get('Item', [])
                if not isinstance(summary_items, list):
                    summary_items = [summary_items]
                title = 'No Title'
                summary = 'No Summary'
                samples = 'Unknown'
                for item in summary_items:
                    if item.get('@Name') == 'title':
                        title = item.get('#text', 'No Title')
                    elif item.get('@Name') == 'summary':
                        summary = item.get('#text', 'No Summary')
                    elif item.get('@Name') == 'Samples':
                        samples = item.get('#text', 'Unknown')
                datasets.append({
                    "gse": gse_id,
                    "title": title,
                    "summary": summary,
                    "samples": samples,
                    "url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE{gse_id}"
                })
            return datasets
        except Exception as e:
            print(f"Error parsing GEO details: {e}")
            return []
    else:
        print(f"GEO API fetch failed with status code: {response.status_code}")
        return []

search_tools = [search_pubmed, fetch_pubmed_details, search_geo, fetch_geo_details]

# --- Search Agent ---
search_agent = create_react_agent(
    model=llm,
    tools=search_tools,
    prompt=(
        """
        You are an expert scientific Search Agent. Your goal is to **STRICTLY follow the given instruction** to find relevant academic papers on PubMed and datasets on GEO. The instruction is found in state['question']. Do NOT use the original user query; base everything on the instruction.

        PROCEDURE:
        1. **Analyze**: Carefully read the instruction (state['question']). Construct search queries using advanced PubMed/GEO syntax, e.g., for PubMed: "(single-cell RNA sequencing OR scRNA-seq) AND (Drosophila OR fruit fly) AND (atlas OR neural OR immunity OR aging) AND (2018/01/01[Date - Publication] : 2025/12/31[Date - Publication])".
        2. **Search (Tool 1)**: Use **up to three attempts** to call `search_pubmed(query, retmax=50)` to get PMIDs. If fewer than 20 PMIDs are found, broaden the query (e.g., use synonyms, remove time restrictions). Ensure queries match the instruction's focus (e.g., time range, tissues).
        3. **Fetch Details (Tool 2)**: If PMIDs are found, **MUST** call `fetch_pubmed_details(pmids)` for the top 50 papers to ensure at least 20 detailed entries.
        4. **Output**: Summarize findings clearly, listing at least 20 papers if available, and relevant datasets. Include DOI for papers if available.
        6. **Final Answer**: Provide a detailed list of papers and datasets (or explanation if none found). Do NOT deviate from the instruction.

        --- OUTPUT FORMAT (must be valid JSON) ---
        {
          "task_id": "",
          "task": {
            "agent": "Search Agent",
            "instruction": ""
          },
          "result": {
            "papers": [
              {
                "pmid": "",
                "title": "",
                "authors": "",
                "journal": "",
                "year": "",
                "abstract": "",
                "url": "",
                "doi": ""
              }
            ],
            "datasets": [
              {
                "gse": "",
                "title": "",
                "summary": "",
                "samples": "",
                "platform": "",
                "pub_date": "",
                "url": ""
              }
            ],
            "explanation": ""
          }
        }
        """
    ),
    name="search_agent",
)

# 定义状态类型
class OverallState(TypedDict):
    query: str
    planner_output: Dict[str, Any]
    tasks: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    messages: List[Any]

# Planner Node
def run_planner_node(state: OverallState) -> OverallState:
    debug_log("Starting Planner Agent node")
    try:
        result = planner_agent.invoke({
            "messages": [HumanMessage(content=state["query"])]
        })
        debug_log(f"Planner Agent result: {result}")
        if isinstance(result, dict) and "messages" in result:
            new_messages = result["messages"]
            last_message = new_messages[-1]
            if isinstance(last_message, AIMessage):
                try:
                    planner_json = json.loads(last_message.content)
                    state["planner_output"] = planner_json
                    state["tasks"] = planner_json.get("task_decomposition", [])
                    debug_log(f"Planner output parsed: {len(state['tasks'])} tasks")
                except json.JSONDecodeError as e:
                    debug_log(f"Planner output JSON decode error: {str(e)}")
                    state["planner_output"] = {"error": "Invalid JSON"}
        else:
            debug_log("Planner Agent returned unexpected result format")
        debug_log("Planner Agent node completed")
        return state
    except Exception as e:
        debug_log(f"Planner node exception: {str(e)}")
        traceback.print_exc()
        return state

# Search Node
def run_search_node(state: OverallState) -> OverallState:
    debug_log("Starting Search Agent node")
    try:
        config = {
        "configurable": {
            "recursion_limit": 5,    # 严格限制5次
            "max_iterations": 6       # 最多6个任务
            }
        }
        search_results = []
        all_tasks = state.get("tasks", [])
        search_tasks = [task for task in all_tasks if task.get('agent') == 'Search Agent']
        debug_log(f"Found {len(search_tasks)} Search Agent tasks")

        for task in search_tasks:
            instruction = task['instruction']
            task_id = task['task_id']
            debug_log(f"Processing task {task_id}: {instruction[:100]}...")
            search_state = {
                "question": instruction,
                "pmids": [],
                "results": [],
                "messages": [HumanMessage(content=instruction)]
            }
            search_result = search_agent.invoke(search_state)
            debug_log(f"Search Agent result for task {task_id}: {search_result}")

            if isinstance(search_result, dict) and "messages" in search_result:
                last_search_msg = search_result["messages"][-1]
                if isinstance(last_search_msg, AIMessage):
                    try:
                        result_json = json.loads(last_search_msg.content)
                        search_results.append(result_json)
                        debug_log(f"Task {task_id} result parsed")
                    except json.JSONDecodeError:
                        search_results.append({
                            "task_id": task_id,
                            "task": task,
                            "result": last_search_msg.content
                        })
                        debug_log(f"Task {task_id} result not JSON, stored as string")
                else:
                    debug_log(f"Task {task_id} returned non-AIMessage")
            else:
                debug_log(f"Task {task_id} returned unexpected result format")
        state["search_results"] = search_results
        debug_log("Search Agent node completed")
        return state
    except Exception as e:
        debug_log(f"Search node exception: {str(e)}")
        traceback.print_exc()
        return state

# 条件边：只有无澄清问题时才执行search
def should_run_search(state: OverallState) -> str:
    questions = state["planner_output"].get("clarifying_questions", [])
    if questions:
        debug_log(f"⏭️ Skipping search: {len(questions)} clarifying questions")
        return END
    debug_log("✅ No clarifying questions, running search")
    return "search_node"

# 构建 LangGraph 流程 - 添加条件边
workflow = StateGraph(OverallState)
workflow.add_node("planner_node", run_planner_node)
workflow.add_node("search_node", run_search_node)
workflow.add_edge(START, "planner_node")
workflow.add_conditional_edges(
    "planner_node",
    should_run_search,
    {"search_node": "search_node", END: END}
)
workflow.add_edge("search_node", END)

app = workflow.compile()

