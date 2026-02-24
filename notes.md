# Development env
- python 3.10
- pytorch 2.2

# Web crawling & scraping
- requests
- beautifulsoup4

Source: https://www.press.bmwgroup.com/global/article
=> Crawl only articles (text data)

# Workflow
1. Crawl articles and save raw data (src/01_article_crawler.py)
2. Preprocess raw data and save processed data (src/02_data_prepare.py)
3. Fine-tuning LLMs using processed data (src/03_model_finetuning.py)
4. Fine-tuning modified LLMs using processed data (src/04_model_finetuning_modified.py)
5. Evaluation (src/05_evaluation.py)
6. Build the end-to-end pipeline (src/06_pipeline.py)