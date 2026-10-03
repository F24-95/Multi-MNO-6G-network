import json
import logging
import sys
from datetime import datetime
from dss.common.paths import log_dir

class JSONFormatter(logging.Formatter):
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

def setup_logger(name: str = "dss") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s")
    c_handler.setFormatter(c_formatter)
    logger.addHandler(c_handler)

    # File Handler (JSON lines)
    f_path = log_dir() / "app.log"
    f_handler = logging.FileHandler(f_path, encoding="utf-8")
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(JSONFormatter())
    logger.addHandler(f_handler)

    return logger
