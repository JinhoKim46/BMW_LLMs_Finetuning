from pathlib import Path

from transformers import set_seed

from src.bmw_01_article_crawler import run_crawler
from src.bmw_02_data_prepare import preprocess_article
from src.bmw_03_llms_FT import fine_tuning
from src.bmw_04_llms_eval import evaluation
from src.logger import get_logger
from src.utils import prep_eval_wo_ft, retrieve_config

LOGGER = get_logger("E2E workflow")


CONFIG_PATH = "config/config.yaml"
CONFIG_DATA = retrieve_config(CONFIG_PATH, "data")
CONFIG_LLMS = retrieve_config(CONFIG_PATH, "llms")
CONFIG_TRAIN = retrieve_config(CONFIG_PATH, "train")
CONFIG_GEN = retrieve_config(CONFIG_PATH, "generation")

set_seed(retrieve_config(CONFIG_PATH, "seed"))  # Set random seed for reproducibility

WORKFLOW_FNS = {
    "crawl": run_crawler,
    "prepare": preprocess_article,
    "ft": fine_tuning,
    "eval": evaluation,
}

def run():
    LOGGER.info("Starting end-to-end workflow for BMW-related LLM fine-tuning and evaluation!", level=0)

    # TODO: add argparser for configurations so that users can easily modify the configs by passing arguments in command line.

    workflow = retrieve_config(CONFIG_PATH, "workflow")
    LOGGER.info(f"Workflow steps to execute: {workflow}", level=1)

    to_eval = None
    for step in workflow:
        if step in WORKFLOW_FNS:
            LOGGER.info(f"Running step: {step}", level=1)
            if step == "ft":
                to_eval = WORKFLOW_FNS[step]()
            elif step == "eval":
                if to_eval is None:
                    to_eval = prep_eval_wo_ft(CONFIG_PATH)
                WORKFLOW_FNS[step](**to_eval)
            else:
                WORKFLOW_FNS[step]()
        else:
            LOGGER.warning(f"Unknown workflow step: {step}. Skipping.", level=1)


if __name__ == "__main__":
    run()
