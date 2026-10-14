import json
import logging
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime
from dss.common.paths import log_dir

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class JSONFormatter(logging.Formatter):
    """Optional JSON formatter for structured log aggregators."""
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def setup_logger(name: str = "dss", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a cleanly formatted logger.
    Outputs human-readable, aligned logs to both stdout and logs/app.log.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

        # Standard console output
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(level)
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)

        # Human-readable rotating log file (2 MB cap per file, 2 backups)
        log_file = log_dir() / "app.log"
        f_handler = RotatingFileHandler(
            log_file,
            maxBytes=2 * 1024 * 1024,
            backupCount=2,
            encoding="utf-8"
        )
        f_handler.setLevel(level)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    return logger

