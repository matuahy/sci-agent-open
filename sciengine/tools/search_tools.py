# sciengine/agent/search_tools.py
"""
search_agent的tools
"""
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
import requests
import xmltodict
from langchain_core.tools import tool
import os
from sciengine.agent.utils import debug_log


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

