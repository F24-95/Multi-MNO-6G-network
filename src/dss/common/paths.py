from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def data_dir(kind: str = "raw") -> Path:
    d = PROJECT_ROOT / "data" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d

def model_dir(model_type: str, version: str) -> Path:
    d = PROJECT_ROOT / "models" / model_type / version
    d.mkdir(parents=True, exist_ok=True)
    return d

def experiment_dir(exp_id: str) -> Path:
    d = PROJECT_ROOT / "experiments" / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def log_dir() -> Path:
    d = PROJECT_ROOT / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d
