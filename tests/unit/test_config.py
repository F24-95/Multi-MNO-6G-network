import os
import pytest
from dss.common.config import get_config, load_yaml
from dss.common.paths import PROJECT_ROOT

def test_config_loads_defaults():
    cfg = get_config(force_reload=True)
    assert cfg.sim.n_mno == 4
    assert cfg.sim.n_channels == 16
    assert cfg.sim.channel_bw_mhz == 20.0
    assert cfg.sim.d_cap_mhz == 160.0
    assert cfg.split.train_days == 10
    assert cfg.windows.L == 12
    assert cfg.windows.H == 3
    assert len(cfg.scenarios) == 9
    assert "vae_gmm" in cfg.models
    assert "autoencoder" in cfg.models
    assert "gan" in cfg.models
    assert "forecaster" in cfg.models

def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_yaml(PROJECT_ROOT / "configs" / "non_existent.yaml")

def test_env_override():
    os.environ["DSS_ENV"] = "test_env"
    cfg = get_config(force_reload=True)
    assert cfg.env == "test_env"
    os.environ["DSS_ENV"] = "dev"
