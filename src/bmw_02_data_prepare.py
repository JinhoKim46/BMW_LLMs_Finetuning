import json
import re
import time

from .bmx_opp_base import Base
from utils import load_jsonl, retrieve_config


class DataPreprocessor(Base):
    def __init__(self, config_path, log_name="DataPreprocessor", log_lv=0):
        super().__init__(config_path, log_name, log_lv)
        self.config = retrieve_config(config_path, "data")
        self.config_prep = self.config.get("prepare", {})

    def run(self, **kwargs):
        '''
        Preprocess raw article data and save it in the format below for fine-tuning.
        ```json
            {
                "idx": "dataset_index",
                "article_id": "unique_article_id",
                "x": """
                        <TITLE>: 'title'\n
                        <TEASER>: 'teaser'\n
                        <TEXT>: 'full_text'\n
                    """
            }
            ```
        '''
        self.logger.info("Starting preprocessing of raw articles...", level=self.log_lv)

        fpath_raw = f"{self.config.get('db_root','database')}/{self.config.get('crawler', {}).get('raw_data_fname','articles_raw.jsonl')}"
        fpath_prep = f"{self.config.get('db_root','database')}/{self.config_prep.get('processed_data_fname','articles_raw.jsonl')}"

        articles_raw = load_jsonl(fpath_raw)

        omit_count = 0

        start_time = time.time()
        with open(fpath_prep, "a", encoding="utf-8") as f:
            for article in articles_raw:
                input_data_sample = {}

                data_idx = article.get("idx", None)
                article_id = article.get("article_id", None)
                title = article.get("title", "")
                teaser = self.clean_teaser(article.get("teaser", ""))  # can be None
                text = self.clean_text(article.get("text", ""))

                if not data_idx or not article_id or not title or not text:
                    # TODO: More elaborate handling for missing fields. When text contain a single element in the list, it is easily filtered out during cleaning.
                    self.logger.warning(
                        f"Omitting article with missing fields. idx: {data_idx}, article_id: {article_id}",
                        level=self.log_lv + 1,
                    )
                    omit_count += 1
                    continue

                input_data_sample["idx"] = data_idx
                input_data_sample["article_id"] = article_id
                input_data_sample["x"] = f"<TITLE>: {title}\n<TEASER>: {teaser}\n<TEXT>: {text}"

                json.dump(input_data_sample, f, ensure_ascii=False)
                f.write("\n")  # newline for readability

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Finished preprocessing articles. Total time: {elapsed_time:.2f} seconds. Omitted {omit_count} articles with empty text.",
            level=self.log_lv,
        )

    # Clean text
    @staticmethod
    def clean_text(text):
        # remove elements first then merge paragraphs

        # remove URLs
        text = [i for i in text if not re.search(r"http[s]?://|www\.\S+", i)]

        # remove emails
        text = [i for i in text if not re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", i)]

        # remove phone numbers
        text = [i for i in text if not re.search(r"\+\d{1,3}", i)]

        # remove special characters (except basic punctuation)
        text = [i.replace("\n", "").replace("\r", "").replace("  ", " ").strip() for i in text]

        # merge paragraphs
        text = " ".join(i for i in text)

        return text

    @staticmethod
    def clean_teaser(teaser):
        if teaser:
            teaser = teaser.replace("+++", "").replace("  ", " ").strip()

        return teaser.strip() if teaser else teaser


if __name__ == "__main__":#
    prep = DataPreprocessor(config_path="./config")
    prep.run()
