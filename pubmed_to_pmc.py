import requests
from bs4 import BeautifulSoup
import re

def extract_pmc_link_from_pubmed(pubmed_url):
    """
    从 PubMed 页面提取 PMC 文章链接

    参数:
        pubmed_url: PubMed 文章 URL，例如 https://pubmed.ncbi.nlm.nih.gov/29844090/

    返回:
        dict: 包含 PMC 链接和相关信息
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        print(f"正在访问 PubMed 页面: {pubmed_url}")
        response = requests.get(pubmed_url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        result = {
            'pubmed_url': pubmed_url,
            'pmc_url': None,
            'pmcid': None,
            'pmid': None,
            'title': None,
            'doi': None
        }

        # 提取 PMID（从 URL 或页面中）
        pmid_match = re.search(r'/(\d+)/?$', pubmed_url)
        if pmid_match:
            result['pmid'] = pmid_match.group(1)

        # 提取标题
        title_tag = soup.find('h1', class_='heading-title')
        if title_tag:
            result['title'] = title_tag.get_text(strip=True)

        # 方法1: 从侧边栏的 PMCID 链接提取
        pmc_link = soup.find('a', {'data-ga-action': 'PMCID'})
        if pmc_link and pmc_link.get('href'):
            result['pmc_url'] = pmc_link['href']
            # 提取 PMCID
            pmcid_match = re.search(r'PMC(\d+)', pmc_link['href'])
            if pmcid_match:
                result['pmcid'] = f"PMC{pmcid_match.group(1)}"

        # 方法2: 从 "Full text links" 部分查找
        if not result['pmc_url']:
            full_text_section = soup.find('div', class_='full-text-links')
            if full_text_section:
                pmc_link = full_text_section.find('a', href=re.compile(r'pmc\.ncbi\.nlm\.nih\.gov'))
                if pmc_link:
                    result['pmc_url'] = pmc_link['href']
                    pmcid_match = re.search(r'PMC(\d+)', pmc_link['href'])
                    if pmcid_match:
                        result['pmcid'] = f"PMC{pmcid_match.group(1)}"

        # 方法3: 查找所有包含 PMC 的链接
        if not result['pmc_url']:
            all_links = soup.find_all('a', href=re.compile(r'pmc\.ncbi\.nlm\.nih\.gov/articles/PMC\d+'))
            if all_links:
                result['pmc_url'] = all_links[0]['href']
                pmcid_match = re.search(r'PMC(\d+)', all_links[0]['href'])
                if pmcid_match:
                    result['pmcid'] = f"PMC{pmcid_match.group(1)}"

        # 提取 DOI
        doi_tag = soup.find('span', class_='identifier doi')
        if doi_tag:
            doi_link = doi_tag.find('a', class_='id-link')
            if doi_link:
                result['doi'] = doi_link.get_text(strip=True)

        # 确保 PMC URL 是完整的
        if result['pmc_url'] and not result['pmc_url'].startswith('http'):
            result['pmc_url'] = 'https://pmc.ncbi.nlm.nih.gov' + result['pmc_url']

        return result

    except requests.RequestException as e:
        print(f"请求错误: {e}")
        return None
    except Exception as e:
        print(f"解析错误: {e}")
        return None

