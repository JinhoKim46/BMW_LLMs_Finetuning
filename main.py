from pathlib import Path

from transformers import set_seed

from src.bmw_01_article_crawler import run_crawler
from src.bmw_02_data_prepare import preprocess_article
from src.bmw_03_llms_FT import fine_tuning
from src.bmw_04_llms_eval import evaluation
from src.logger import get_logger
from src.utils import prep_eval_wo_ft, retrieve_config

LOGGER = get_logger("E2E workflow")
LOG_LV = 0

CONFIG_PATH = "config/config.yaml"

set_seed(retrieve_config(CONFIG_PATH, "seed"))  # Set random seed for reproducibility

WORKFLOW_FNS = {
    "crawl": run_crawler,
    "prepare": preprocess_article,
    "ft": fine_tuning,
    "eval": evaluation,
}

def run():
    LOGGER.info("Starting end-to-end workflow for BMW-related LLM fine-tuning and evaluation!", level=LOG_LV)

    # TODO: add argparser for configurations so that users can easily modify the configs by passing arguments in command line.

    workflow = retrieve_config(CONFIG_PATH, "workflow")
    LOGGER.info(f"Workflow steps to execute: {workflow}", level=LOG_LV+1)

    to_eval = None
    for step in workflow:
        if step in WORKFLOW_FNS:
            LOGGER.info(f"Running step: {step}", level=LOG_LV+1)
            if step == "ft":
                to_eval = WORKFLOW_FNS[step](log_lv=LOG_LV+2)
            elif step == "eval":
                if to_eval is None:
                    to_eval = prep_eval_wo_ft(CONFIG_PATH)
                WORKFLOW_FNS[step](log_lv=LOG_LV+2, **to_eval)
            else:
                WORKFLOW_FNS[step](log_lv=LOG_LV+2)
        else:
            LOGGER.warning(f"Unknown workflow step: {step}. Skipping.", level=LOG_LV+1)


if __name__ == "__main__":
    run()
