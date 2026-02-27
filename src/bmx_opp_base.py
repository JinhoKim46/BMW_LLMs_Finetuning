from pathlib import Path

from transformers import set_seed

from logger import get_logger
from utils import retrieve_config


class Base:
    def __init__(self, config_path:Path, log_name:str, log_lv:int=0):
        self.config_path = Path(config_path) 
        self.logger = get_logger(log_name)
        self.log_lv = log_lv
        self.logger.info(f"Initializing {log_name}", level=self.log_lv)
        
        self.seed = retrieve_config(self.config_path, "seed")
        set_seed(self.seed)  # Set random seed for reproducibility
        
    def run(self, **kwargs):
        raise NotImplementedError("Subclasses must implement the run method.")