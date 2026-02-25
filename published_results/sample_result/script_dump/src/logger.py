# crawler/shared/logger.py

import logging
import os
from datetime import datetime

class StructuredLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def info(self, message: str, level: int = 0):
        indent = "  " * level
        self.logger.info(f"{indent}{message}")

    def debug(self, message: str, level: int = 0):
        indent = "  " * level
        self.logger.debug(f"{indent}{message}")

    def warning(self, message: str, level: int = 0):
        indent = "  " * level
        self.logger.warning(f"{indent}{message}")

    def error(self, message: str, level: int = 0):
        indent = "  " * level
        self.logger.error(f"{indent}{message}")


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file_path = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y%m%d')}.log")

formatter = logging.Formatter(
    fmt="[{asctime}] [{levelname:^7}] [{name:^20}] {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{"
)

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # File handler
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return StructuredLogger(logger)
