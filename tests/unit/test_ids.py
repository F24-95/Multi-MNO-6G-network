import re
from dss.common.ids import dataset_key, experiment_id, next_model_version, compute_config_hash

def test_dataset_key_format():
    key = dataset_key("normal", 42, "abcd1234ef")
    assert re.match(r"^ds_normal_s42_[a-f0-9]{8}$", key)

def test_experiment_id_format():
    exp = experiment_id("high_traffic", "dynamic", 42)
    assert re.match(r"^exp_high_traffic_dynamic_s42_\d{8}_\d{6}$", exp)

def test_next_model_version():
    assert next_model_version([]) == "v001"
    assert next_model_version(["v001", "v002"]) == "v003"
    assert next_model_version(["v005", "v001"]) == "v006"

def test_compute_config_hash():
    h1 = compute_config_hash({"a": 1, "b": 2})
    h2 = compute_config_hash({"b": 2, "a": 1})
    assert h1 == h2
    assert len(h1) == 8
