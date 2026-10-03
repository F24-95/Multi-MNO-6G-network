import json
from dss.common.logging_setup import setup_logger
from dss.common.paths import log_dir

def test_logger_creates_file_and_logs():
    logger = setup_logger("test_dss")
    test_msg = "test log message 12345"
    logger.info(test_msg)

    log_file = log_dir() / "app.log"
    assert log_file.exists()

    found = False
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if data.get("message") == test_msg:
                        found = True
                        break
                except json.JSONDecodeError:
                    pass
    assert found, "Log message was not found in JSON format in app.log"
