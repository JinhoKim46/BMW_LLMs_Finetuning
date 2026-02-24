import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

from logger import get_logger
from utils import retrieve_config

LOGGER = get_logger("article_crawler")

URL = "https://www.press.bmwgroup.com/global/article"
CONFIG_PATH = Path(__file__).parent.parent / "config/config.yaml"
CONFIG = retrieve_config(CONFIG_PATH, "data")
CONFIG_CRAWLER = CONFIG.get("crawler", {})

def expand_all_articles(driver: webdriver, wait: WebDriverWait):
    LOGGER.info("Expanding all articles by clicking 'Show more' button...")
    
    # One-time click on "Show more" button to load more articles
    try:
        show_more_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button#lazy-load-button.remove-on-search.clear")))

        # driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", show_more_btn)
        driver.execute_script("arguments[0].click();", show_more_btn)
        time.sleep(CONFIG_CRAWLER.get("sleep_time", 0.5))
    except TimeoutException:
        pass

    # After clicking "Show more," scroll to load more articles until the configured number of scrolls as scroll_n
    LOGGER.info(f"Scrolling {CONFIG_CRAWLER.get('scroll_n', 10)} times to load articles...", level=1)
    pbar = tqdm(range(CONFIG_CRAWLER.get("scroll_n", 10)), desc="Scrolling to load articles")
    start_time = time.time()
    for _ in pbar:        
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(CONFIG_CRAWLER.get("sleep_time", 0.5))
    elapsed_time = time.time() - start_time
    LOGGER.info(f"Finished scrolling after {elapsed_time:.2f} seconds.", level=1)


def crawl_articles(soup: BeautifulSoup, driver: webdriver, wait: WebDriverWait):
    LOGGER.info("Start crawling articles...")
    article_items = []

    fpath = f"{CONFIG.get('db_root','database')}/{CONFIG_CRAWLER.get('raw_data_fname','articles_raw.jsonl')}"

    # TODO: store hash index (Hash(article_id): idx) to quickly check if article_id is already crawled. This is needed when the crawler needs to be stopped and restarted, as the "Show more" button may not load all articles at once.

    # TODO: store newest_article_id in state file after crawling to stop early when the crawler reaches to the newest article in DB. Also, state file can manage the databaes lifecycle.

    with open(fpath, "a", encoding="utf-8") as f:        
        pbar = tqdm(soup.select("article.newsfeed"))
        start_time = time.time()
        for i, article in enumerate(pbar):
            if i % CONFIG_CRAWLER.get("max_article_N", 1000) == 0 and i > 0:
                break            

            pbar.set_description(f"Crawling article {article.get('data-id')}")
            pbar.update()

            article_id = article.get("data-id")            
            link_tag = article.select_one("div.text h3 a[href]")
            if not link_tag:
                continue

            article_item = {
                    "idx": i, # Database index (not article_id)
                    "title": link_tag.get_text(strip=True),
                    "article_id": article_id,
                    "url": urljoin(URL, link_tag["href"]),
                }

            article_item = extract_details(article_item, driver, wait)
            article_items.append(article_item)

            json.dump(article_item, f, ensure_ascii=False)
            f.write("\n")  # newline for readability

        elapsed_time = time.time() - start_time

    LOGGER.info(f"Total loaded articles: {len(article_items)}")
    LOGGER.info(f"Elapsed time for crawling articles: {elapsed_time:.2f} seconds")


def extract_details(article, driver: webdriver, wait: WebDriverWait):
    detail_url = article["url"]
    driver.get(detail_url)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))

    # Expand full body if article has "Show entire text"
    try:
        readmore_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.readmore")))

        # driver.execute_script("arguments[0].scrollIntoView({block:'center'});", readmore_btn)
        driver.execute_script("arguments[0].click();", readmore_btn)
        time.sleep(CONFIG_CRAWLER.get("sleep_time", 0.5))
    except TimeoutException:
        pass

    soup = BeautifulSoup(driver.page_source, "lxml")

    date_node = soup.select_one(".article-info .date, .article-info time, time")
    date_raw = date_node.get_text(" ", strip=True) if date_node else None
    if date_raw is not None:
        date_raw = datetime.strptime(date_raw, "%d.%m.%Y").strftime("%Y%m%d")

    category_node = soup.select_one(".article-info .category")
    category = category_node.get_text(" ", strip=True) if category_node else None

    teaser_node = soup.select_one(".teaser.clear, .teaser")
    teaser = teaser_node.get_text(" ", strip=True) if teaser_node else None

    body_container = soup.select_one("#article-text")
    body_text = None
    if body_container:
        paragraphs = body_container.select("p")
        if paragraphs:
            body_text = [p.get_text(" ", strip=True) for p in paragraphs if p.get_text(strip=True)]
        else:
            body_text = body_container.get_text("\n", strip=True)

    article["date_raw"] = date_raw
    article["category"] = category
    article["teaser"] = teaser
    article["text"] = body_text
    return article


def run_crawler():
    options = Options()
    options.add_argument("--headless=new") # No GUI

    driver = webdriver.Chrome(options=options)  # let Selenium Manager resolve driver
    driver.get(URL)
    wait = WebDriverWait(driver, CONFIG_CRAWLER.get("wait_time", 10))
    expand_all_articles(driver, wait)    
    
    soup = BeautifulSoup(driver.page_source, "lxml")

    db_root = Path(CONFIG.get("db_root"))
    if not db_root.exists():
        LOGGER.info(f"Creating database directory at {db_root}...")
        db_root.mkdir(parents=True, exist_ok=True)
    
    crawl_articles(soup, driver, wait)    


if __name__ == "__main__":
    run_crawler()
