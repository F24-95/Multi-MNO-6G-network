from dataclasses import dataclass
from typing import Dict, Any
from dss.common.config import get_config

@dataclass(frozen=True)
class ScenarioParams:
    name: str
    scen_mult: float
    spike_lambda: float
    pu_duty_min: float
    pu_duty_max: float
    sigma_u: float
    p_burst: float
    description: str

def get_scenario(name: str) -> ScenarioParams:
    cfg = get_config()
    scens = cfg.scenarios
    if name not in scens:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(scens.keys())}")
    data = scens[name]
    return ScenarioParams(
        name=name,
        scen_mult=float(data["scen_mult"]),
        spike_lambda=float(data["spike_lambda"]),
        pu_duty_min=float(data["pu_duty_min"]),
        pu_duty_max=float(data["pu_duty_max"]),
        sigma_u=float(data["sigma_u"]),
        p_burst=float(data["p_burst"]),
        description=str(data.get("description", "")),
    )

def list_scenarios() -> list[str]:
    cfg = get_config()
    return list(cfg.scenarios.keys())
