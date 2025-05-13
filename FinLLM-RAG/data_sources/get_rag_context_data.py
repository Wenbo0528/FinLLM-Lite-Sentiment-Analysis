import requests
from bs4 import BeautifulSoup
import urllib.parse
import json
from pathlib import Path
import time
import random
from fake_useragent import UserAgent
import logging

# ========== 日志设置 ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== 相似度函数 ==========
def similarity_score(a, b):
    a_words, b_words = set(a.lower().split()), set(b.lower().split())
    return len(a_words & b_words) / (min(len(a_words), len(b_words)) + 1e-6)

# ========== 生成随机 headers ==========
def get_random_headers():
    ua = UserAgent()
    return {
        "User-Agent": ua.random,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0"
    }

# ========== 请求网页内容 ==========
def make_request(url, max_retries=3):
    for i in range(max_retries):
        try:
            headers = get_random_headers()
            logging.info(f"尝试请求 URL: {url}")
            logging.info(f"使用 User-Agent: {headers['User-Agent']}")
            
            # 减少随机延迟
            delay = random.uniform(3, 5)
            logging.info(f"等待 {delay:.1f} 秒...")
            time.sleep(delay)
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            logging.info(f"请求成功，状态码: {response.status_code}")
            return response
        except requests.exceptions.RequestException as e:
            logging.warning(f"请求失败 (尝试 {i+1}/{max_retries}): {str(e)}")
            if i < max_retries - 1:
                time.sleep(random.uniform(5, 8))  # 失败后等待时间也减少
            continue
    return None

# ========== 提取文章正文 ==========
def scrape_article_text(url):
    try:
        response = make_request(url)
        if not response:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")
        if "seekingalpha.com" in url:
            paras = soup.select("div.article-content p")
        elif "marketwatch.com" in url:
            paras = soup.select("p.article__body")
        else:
            paras = soup.find_all("p")

        if not paras:
            paras = soup.find_all("p")  # fallback

        text = " ".join(p.text.strip() for p in paras)
        return text[:1000] if text else ""
    except Exception as e:
        logging.error(f"文章内容抓取错误: {str(e)}")
        return ""

# ========== Seeking Alpha 抓取 ==========
def scrape_seeking_alpha(query):
    try:
        # 使用 Seeking Alpha 搜索
        url = f"https://seekingalpha.com/search?q={urllib.parse.quote(query)}"
        response = make_request(url)
        if not response:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("div.article-list article")
        logging.info(f"找到 {len(articles)} 个 Seeking Alpha 文章")

        for article in articles:
            try:
                title_tag = article.select_one("h3.title a")
                if not title_tag:
                    continue
                title = title_tag.text.strip()
                link = title_tag["href"]
                if not link.startswith("http"):
                    link = "https://seekingalpha.com" + link
                score = similarity_score(query, title)
                logging.info(f"标题: {title}, 相似度: {score:.2f}")
                if score > 0.2:
                    context = scrape_article_text(link)
                    if context:
                        return {
                            "source": "Seeking Alpha",
                            "title": title,
                            "url": link,
                            "context": context
                        }
            except Exception as e:
                logging.warning(f"处理文章时出错: {str(e)}")
                continue
    except Exception as e:
        logging.error(f"Seeking Alpha 抓取错误: {str(e)}")
    return None

# ========== MarketWatch 抓取 ==========
def scrape_marketwatch(query):
    try:
        url = f"https://www.marketwatch.com/search?q={urllib.parse.quote(query)}&m=Keyword&rpp=15&mp=806&bd=false&rs=true"
        response = make_request(url)
        if not response:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("div.article__content")
        logging.info(f"找到 {len(articles)} 个 MarketWatch 文章")

        for article in articles:
            try:
                title_tag = article.select_one("h3.article__headline a")
                if not title_tag:
                    continue
                title = title_tag.text.strip()
                link = title_tag["href"]
                if not link.startswith("http"):
                    link = "https://www.marketwatch.com" + link
                score = similarity_score(query, title)
                logging.info(f"标题: {title}, 相似度: {score:.2f}")
                if score > 0.2:
                    # 检查是否是付费内容
                    if "barrons.com" in link:
                        logging.info("跳过付费内容")
                        continue
                    context = scrape_article_text(link)
                    if context:
                        return {
                            "source": "MarketWatch",
                            "title": title,
                            "url": link,
                            "context": context
                        }
            except Exception as e:
                logging.warning(f"处理文章时出错: {str(e)}")
                continue
    except Exception as e:
        logging.error(f"MarketWatch 抓取错误: {str(e)}")
    return None

# ========== 主函数 ==========
def main():
    queries = [
        "Tesla stock price movement",
        "JPMorgan stock analysis",
        "Apple earnings report",
        "Microsoft cloud growth",
        "Amazon quarterly results",
        "Goldman Sachs market outlook",
        "Bank of America financial results",
        "Meta AI development",
        "Federal Reserve interest rates",
        "Oil prices market update"
    ]

    results = []
    for query in queries:
        logging.info(f"\n🔎 正在查询: {query}")

        result = scrape_seeking_alpha(query) or scrape_marketwatch(query)

        if result:
            logging.info(f"✅ [{result['source']}] 成功获取文章: {result['title']}")
            results.append({
                "query": query,
                "retrieved": result
            })
        else:
            logging.warning("❌ 未找到相关文章")

        # 减少查询间隔
        delay = random.uniform(2, 4)
        logging.info(f"⏳ 等待 {delay:.1f} 秒...\n")
        time.sleep(delay)

    # 保存结果到正确的路径
    output_path = Path(__file__).parent / "rag_context_data.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    logging.info(f"\n📁 已保存 {len(results)} 条检索结果到 {output_path}")

if __name__ == "__main__":
    main()
