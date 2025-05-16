import requests
from bs4 import BeautifulSoup
import urllib.parse
import json
from pathlib import Path
import time
import random
from fake_useragent import UserAgent
import logging
import re
from datetime import datetime, timedelta

# ========== Logging Setup ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== Query Cleaning Functions ==========
def clean_ticker_symbols(text):
    # Remove common ticker patterns like $AAPL, AAPL, etc.
    text = re.sub(r'\$[A-Z]{1,5}', '', text)
    text = re.sub(r'\b[A-Z]{1,5}\b', '', text)
    return text

def clean_special_chars(text):
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text

def extract_time_window(text):
    # Extract time information from text
    time_patterns = {
        r'(\d+)\s*(?:days?|d)\s*ago': lambda x: timedelta(days=int(x)),
        r'(\d+)\s*(?:weeks?|w)\s*ago': lambda x: timedelta(weeks=int(x)),
        r'(\d+)\s*(?:months?|m)\s*ago': lambda x: timedelta(days=int(x)*30),
        r'(\d+)\s*(?:years?|y)\s*ago': lambda x: timedelta(days=int(x)*365),
        r'today': lambda x: timedelta(days=0),
        r'yesterday': lambda x: timedelta(days=1),
        r'last\s+week': lambda x: timedelta(weeks=1),
        r'last\s+month': lambda x: timedelta(days=30),
        r'last\s+year': lambda x: timedelta(days=365)
    }
    
    for pattern, delta_func in time_patterns.items():
        match = re.search(pattern, text.lower())
        if match:
            if pattern in ['today', 'yesterday', 'last week', 'last month', 'last year']:
                return delta_func(None)
            return delta_func(match.group(1))
    return None

def clean_query(query):
    # Clean the query text
    cleaned = query.lower()
    cleaned = clean_ticker_symbols(cleaned)
    cleaned = clean_special_chars(cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Extract time window
    time_delta = extract_time_window(cleaned)
    
    return cleaned, time_delta

# ========== Similarity Function ==========
def similarity_score(a, b):
    a_words, b_words = set(a.lower().split()), set(b.lower().split())
    return len(a_words & b_words) / (min(len(a_words), len(b_words)) + 1e-6)

# ========== Generate Random Headers ==========
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

# ========== Request Web Content ==========
def make_request(url, max_retries=3):
    for i in range(max_retries):
        try:
            headers = get_random_headers()
            delay = random.uniform(3, 5)
            time.sleep(delay)
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if i < max_retries - 1:
                time.sleep(random.uniform(5, 8))
            continue
    return None

# ========== Extract Article Content ==========
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
        return ""

# ========== Seeking Alpha Scraping ==========
def scrape_seeking_alpha(query):
    results = []
    try:
        url = f"https://seekingalpha.com/search?q={urllib.parse.quote(query)}"
        response = make_request(url)
        if not response:
            return results

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("div.article-list article")

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
                if score > 0.2:
                    context = scrape_article_text(link)
                    if context:
                        results.append({
                            "source": "Seeking Alpha",
                            "title": title,
                            "url": link,
                            "context": context
                        })
            except Exception:
                continue
    except Exception:
        pass
    return results

# ========== MarketWatch Scraping ==========
def scrape_marketwatch(query, time_delta=None):
    results = []
    try:
        # Add time window to search query if available
        search_query = query
        if time_delta:
            date_str = (datetime.now() - time_delta).strftime("%Y-%m-%d")
            search_query = f"{query} after:{date_str}"
            
        url = f"https://www.marketwatch.com/search?q={urllib.parse.quote(search_query)}&m=Keyword&rpp=15&mp=806&bd=false&rs=true"
        response = make_request(url)
        if not response:
            return results

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("div.article__content")

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
                if score > 0.8:  # Changed threshold to 0.8
                    if "barrons.com" in link:
                        continue
                    context = scrape_article_text(link)
                    if context:
                        results.append({
                            "source": "MarketWatch",
                            "title": title,
                            "url": link,
                            "context": context
                        })
            except Exception:
                continue
    except Exception:
        pass
    return results

# ========== Bloomberg Scraping ==========
def scrape_bloomberg(query):
    results = []
    try:
        url = f"https://www.bloomberg.com/search?query={urllib.parse.quote(query)}"
        response = make_request(url)
        if not response:
            return results

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("div.story-list-story")

        for article in articles:
            try:
                title_tag = article.select_one("h3.headline__text")
                if not title_tag:
                    continue
                title = title_tag.text.strip()
                link = article.select_one("a")["href"]
                if not link.startswith("http"):
                    link = "https://www.bloomberg.com" + link
                score = similarity_score(query, title)
                if score > 0.2:
                    context = scrape_article_text(link)
                    if context:
                        results.append({
                            "source": "Bloomberg",
                            "title": title,
                            "url": link,
                            "context": context
                        })
            except Exception:
                continue
    except Exception:
        pass
    return results

# ========== CNBC Scraping ==========
def scrape_cnbc(query):
    results = []
    try:
        url = f"https://www.cnbc.com/search/?query={urllib.parse.quote(query)}&qsearchterm={urllib.parse.quote(query)}"
        response = make_request(url)
        if not response:
            return results

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("div.Card-titleContainer")

        for article in articles:
            try:
                title_tag = article.select_one("a.Card-title")
                if not title_tag:
                    continue
                title = title_tag.text.strip()
                link = title_tag["href"]
                if not link.startswith("http"):
                    link = "https://www.cnbc.com" + link
                score = similarity_score(query, title)
                if score > 0.2:
                    context = scrape_article_text(link)
                    if context:
                        results.append({
                            "source": "CNBC",
                            "title": title,
                            "url": link,
                            "context": context
                        })
            except Exception:
                continue
    except Exception:
        pass
    return results

# ========== Yahoo Finance Scraping ==========
def scrape_yahoo_finance(query, time_delta=None):
    results = []
    try:
        search_query = query
        if time_delta:
            date_str = (datetime.now() - time_delta).strftime("%Y-%m-%d")
            search_query = f"{query} after:{date_str}"
            
        url = f"https://finance.yahoo.com/news/search?q={urllib.parse.quote(search_query)}"
        response = make_request(url)
        if not response:
            return results

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("div.js-content-viewer")

        for article in articles:
            try:
                title_tag = article.select_one("h3.Mb\\(5px\\)")
                if not title_tag:
                    continue
                title = title_tag.text.strip()
                link = article.select_one("a")["href"]
                if not link.startswith("http"):
                    link = "https://finance.yahoo.com" + link
                score = similarity_score(query, title)
                if score > 0.8:
                    context = scrape_article_text(link)
                    if context:
                        results.append({
                            "source": "Yahoo Finance",
                            "title": title,
                            "url": link,
                            "context": context
                        })
            except Exception:
                continue
    except Exception:
        pass
    return results

# ========== Reuters Scraping ==========
def scrape_reuters(query, time_delta=None):
    results = []
    try:
        search_query = query
        if time_delta:
            date_str = (datetime.now() - time_delta).strftime("%Y-%m-%d")
            search_query = f"{query} after:{date_str}"
            
        url = f"https://www.reuters.com/search/news?blob={urllib.parse.quote(search_query)}"
        response = make_request(url)
        if not response:
            return results

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("div.search-result-content")

        for article in articles:
            try:
                title_tag = article.select_one("h3.search-result-title")
                if not title_tag:
                    continue
                title = title_tag.text.strip()
                link = article.select_one("a")["href"]
                if not link.startswith("http"):
                    link = "https://www.reuters.com" + link
                score = similarity_score(query, title)
                if score > 0.8:
                    context = scrape_article_text(link)
                    if context:
                        results.append({
                            "source": "Reuters",
                            "title": title,
                            "url": link,
                            "context": context
                        })
            except Exception:
                continue
    except Exception:
        pass
    return results

# ========== Market Screener Scraping ==========
def scrape_market_screener(query, time_delta=None):
    results = []
    try:
        search_query = query
        if time_delta:
            date_str = (datetime.now() - time_delta).strftime("%Y-%m-%d")
            search_query = f"{query} after:{date_str}"
            
        url = f"https://www.marketscreener.com/search/?q={urllib.parse.quote(search_query)}"
        response = make_request(url)
        if not response:
            return results

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("div.news-item")

        for article in articles:
            try:
                title_tag = article.select_one("h2.news-title")
                if not title_tag:
                    continue
                title = title_tag.text.strip()
                link = article.select_one("a")["href"]
                if not link.startswith("http"):
                    link = "https://www.marketscreener.com" + link
                score = similarity_score(query, title)
                if score > 0.8:
                    context = scrape_article_text(link)
                    if context:
                        results.append({
                            "source": "Market Screener",
                            "title": title,
                            "url": link,
                            "context": context
                        })
            except Exception:
                continue
    except Exception:
        pass
    return results

# ========== Twitter Scraping ==========
def scrape_twitter(query, time_delta=None):
    results = []
    try:
        search_query = query
        if time_delta:
            date_str = (datetime.now() - time_delta).strftime("%Y-%m-%d")
            search_query = f"{query} since:{date_str}"
            
        url = f"https://twitter.com/search?q={urllib.parse.quote(search_query)}&src=typed_query&f=live"
        response = make_request(url)
        if not response:
            return results

        soup = BeautifulSoup(response.text, "html.parser")
        tweets = soup.select("article[data-testid='tweet']")

        for tweet in tweets:
            try:
                text_tag = tweet.select_one("div[data-testid='tweetText']")
                if not text_tag:
                    continue
                text = text_tag.text.strip()
                link = tweet.select_one("a[href*='/status/']")["href"]
                if not link.startswith("http"):
                    link = "https://twitter.com" + link
                score = similarity_score(query, text)
                if score > 0.8:
                    results.append({
                        "source": "Twitter",
                        "title": text[:100] + "...",  # Use first 100 chars as title
                        "url": link,
                        "context": text
                    })
            except Exception:
                continue
    except Exception:
        pass
    return results

# ========== Reddit Scraping ==========
def scrape_reddit(query, time_delta=None):
    results = []
    try:
        search_query = query
        if time_delta:
            date_str = (datetime.now() - time_delta).strftime("%Y-%m-%d")
            search_query = f"{query} after:{date_str}"
            
        url = f"https://www.reddit.com/search/?q={urllib.parse.quote(search_query)}&t=all"
        response = make_request(url)
        if not response:
            return results

        soup = BeautifulSoup(response.text, "html.parser")
        posts = soup.select("div[data-testid='post-container']")

        for post in posts:
            try:
                title_tag = post.select_one("h3")
                if not title_tag:
                    continue
                title = title_tag.text.strip()
                link = post.select_one("a[data-click-id='body']")["href"]
                if not link.startswith("http"):
                    link = "https://www.reddit.com" + link
                score = similarity_score(query, title)
                if score > 0.8:
                    context = scrape_article_text(link)
                    if context:
                        results.append({
                            "source": "Reddit",
                            "title": title,
                            "url": link,
                            "context": context
                        })
            except Exception:
                continue
    except Exception:
        pass
    return results

# ========== Main Function ==========
def main():
    # Load validation dataset
    validation_data_path = Path(__file__).parent.parent.parent / "FinLLM-Instruction-tuning" / "data" / "validation_data.jsonl"
    queries = []
    
    try:
        with open(validation_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                instruction = data['instruction']
                text = instruction.replace("What is the sentiment of the financial text? Please choose an answer from {negative/neutral/positive}: ", "")
                queries.append(text)
    except Exception as e:
        logging.error(f"Error loading validation dataset: {str(e)}")
        return

    # Randomly select 10 queries
    random.seed(42)  # Set random seed for reproducibility
    selected_queries = random.sample(queries, min(10, len(queries)))
    logging.info(f"Selected {len(selected_queries)} queries from {len(queries)} total queries")

    results = []
    source_stats = {
        "Seeking Alpha": 0,
        "MarketWatch": 0,
        "Bloomberg": 0,
        "CNBC": 0,
        "Yahoo Finance": 0,
        "Reuters": 0,
        "Market Screener": 0,
        "Twitter": 0,
        "Reddit": 0
    }
    
    total_queries = len(selected_queries)
    start_time = time.time()
    
    logging.info(f"Starting to process {total_queries} queries...")
    
    for i, query in enumerate(selected_queries, 1):
        logging.info(f"Processing query {i}/{total_queries}")
        
        # Clean query and get time window
        cleaned_query, time_delta = clean_query(query)
        logging.info(f"Cleaned query: {cleaned_query}")
        if time_delta:
            logging.info(f"Time window: {time_delta}")
        
        # Get results from all sources
        marketwatch_results = scrape_marketwatch(cleaned_query, time_delta)
        seeking_alpha_results = scrape_seeking_alpha(cleaned_query)
        bloomberg_results = scrape_bloomberg(cleaned_query)
        cnbc_results = scrape_cnbc(cleaned_query)
        yahoo_results = scrape_yahoo_finance(cleaned_query, time_delta)
        reuters_results = scrape_reuters(cleaned_query, time_delta)
        market_screener_results = scrape_market_screener(cleaned_query, time_delta)
        twitter_results = scrape_twitter(cleaned_query, time_delta)
        reddit_results = scrape_reddit(cleaned_query, time_delta)
        
        # Update source statistics
        source_stats["Seeking Alpha"] += len(seeking_alpha_results)
        source_stats["MarketWatch"] += len(marketwatch_results)
        source_stats["Bloomberg"] += len(bloomberg_results)
        source_stats["CNBC"] += len(cnbc_results)
        source_stats["Yahoo Finance"] += len(yahoo_results)
        source_stats["Reuters"] += len(reuters_results)
        source_stats["Market Screener"] += len(market_screener_results)
        source_stats["Twitter"] += len(twitter_results)
        source_stats["Reddit"] += len(reddit_results)
        
        # Combine all results
        all_results = (
            marketwatch_results + 
            seeking_alpha_results + 
            bloomberg_results + 
            cnbc_results +
            yahoo_results +
            reuters_results +
            market_screener_results +
            twitter_results +
            reddit_results
        )
        
        if all_results:
            # Save each article separately in the results
            for article in all_results:
                results.append({
                    "query": query,
                    "cleaned_query": cleaned_query,
                    "retrieved": article
                })

        delay = random.uniform(2, 4)
        time.sleep(delay)

    total_time = time.time() - start_time
    total_minutes = int(total_time) // 60
    total_seconds = int(total_time) % 60
    
    logging.info("\n=== Results Summary ===")
    for source, count in source_stats.items():
        logging.info(f"{source}: {count} articles")
    logging.info(f"Time taken: {total_minutes} minutes {total_seconds} seconds")

    output_path = Path(__file__).parent / "rag_context_data.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    logging.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()