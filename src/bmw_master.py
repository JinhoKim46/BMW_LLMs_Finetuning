from .bmx_opp_base import Base
from src.utils import prep_eval_wo_ft, retrieve_config

from src.bmw_01_article_crawler import ArticleCrawler
from src.bmw_02_data_prepare import DataPreprocessor
from src.bmw_03_llms_FT import FineTuner
from src.bmw_04_llms_eval import Evaluation


WORKFLOW_OBJS = {
    "crawl": ArticleCrawler,
    "prepare": DataPreprocessor,
    "ft": FineTuner,
    "eval": Evaluation,
}
WORKFLOW_NAME_MAP = {
    "crawl": "Article Crawling",
    "prepare": "Data Preparation",
    "ft": "LLM Fine-tuning",
    "eval": "LLM Evaluation",
}
class Master(Base):
    def __init__(self, config_path, log_name="Master", log_lv=0):
        super().__init__(config_path, log_name, log_lv)
        self.workflow = retrieve_config(config_path, "workflow")
        
        self.workflow_objs = {}
        for step in self.workflow:
            if step in WORKFLOW_OBJS:
                self.workflow_objs[step] = WORKFLOW_OBJS[step](config_path, log_name=WORKFLOW_NAME_MAP[step], log_lv=log_lv+2)
            else:
                self.logger.warning(f"Unknown workflow step: {step}. Skipping.", level=log_lv+1)
    
    def run(self, **kwargs):
        to_eval = None
        for step, obj in self.workflow_objs.items():
            self.logger.info(f"Running step: {step}", level=self.log_lv+1)
            if step == "ft":
                to_eval = obj.run()
            elif step == "eval":
                if to_eval is None:
                    to_eval = prep_eval_wo_ft(self.config_path)
                obj.run(**to_eval)
            else:
                obj.run()
                