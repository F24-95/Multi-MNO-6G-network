import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from dss.common.paths import PROJECT_ROOT
from dss.database.connection import get_db_url  # single source for DATABASE_URL


@dataclass(frozen=True)
class SimConfig:
    n_mno: int = 4
    n_channels: int = 16
    channel_bw_mhz: float = 20.0
    interval_min: int = 5
    days: int = 14
    start_ts: str = "2026-01-05T00:00:00"
    d_cap_mhz: float = 160.0
    se_ref: float = 4.0
    p_link_ref_dbm: float = -80.0
    u_peak: int = 250
    r_user_mbps: float = 0.8
    seed_train: int = 42
    seed_unseen: int = 2026


@dataclass(frozen=True)
class SplitConfig:
    train_days: int = 10
    val_days: int = 2
    test_days: int = 2


@dataclass(frozen=True)
class WindowsConfig:
    L: int = 12
    H: int = 3


@dataclass(frozen=True)
class AllocationConfig:
    tau_avail: float = 0.3
    i_max_dbm: float = -95.0
    alpha: float = 1.0
    beta: float = 2.0
    release_margin_mhz: float = 5.0
    theta_c: float = 0.8
    q_min: float = 0.8
    min_channels: int = 1
    weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])


@dataclass(frozen=True)
class Config:
    sim: SimConfig
    split: SplitConfig
    windows: WindowsConfig
    allocation: AllocationConfig
    scenarios: Dict[str, Any]
    models: Dict[str, Any]
    env: str = "dev"
    database_url: str = ""


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CONFIG_CACHE: Config | None = None


def get_config(force_reload: bool = False) -> Config:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    cfg_dir = PROJECT_ROOT / "configs"
    default_cfg = load_yaml(cfg_dir / "default.yaml")
    alloc_cfg   = load_yaml(cfg_dir / "allocation.yaml")
    scen_cfg    = load_yaml(cfg_dir / "scenarios.yaml")
    models_cfg  = load_yaml(cfg_dir / "models.yaml")

    sim        = SimConfig(**default_cfg.get("sim", {}))
    split      = SplitConfig(**default_cfg.get("split", {}))
    windows    = WindowsConfig(**default_cfg.get("windows", {}))
    allocation = AllocationConfig(**alloc_cfg)

    env    = os.getenv("DSS_ENV", "dev")
    db_url = get_db_url()  # reads DATABASE_URL — load_dotenv() was already called in connection.py

    _CONFIG_CACHE = Config(
        sim=sim,
        split=split,
        windows=windows,
        allocation=allocation,
        scenarios=scen_cfg.get("scenarios", {}),
        models=models_cfg,
        env=env,
        database_url=db_url,
    )
    return _CONFIG_CACHE
