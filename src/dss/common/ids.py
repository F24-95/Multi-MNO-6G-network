import datetime
import hashlib
import json
from typing import Any, Dict, List

def compute_config_hash(cfg_dict: Dict[str, Any], length: int = 8) -> str:
    s = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]

def dataset_key(scenario: str, seed: int, config_hash: str) -> str:
    return f"ds_{scenario}_s{seed}_{config_hash[:8]}"

def experiment_id(scenario: str, allocator: str, seed: int) -> str:
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"exp_{scenario}_{allocator}_s{seed}_{now_str}"

def next_model_version(existing_versions: List[str]) -> str:
    nums = []
    for v in existing_versions:
        if v.startswith("v") and v[1:].isdigit():
            nums.append(int(v[1:]))
    next_num = max(nums, default=0) + 1
    return f"v{next_num:03d}"
