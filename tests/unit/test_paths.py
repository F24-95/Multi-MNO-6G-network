from dss.common.paths import PROJECT_ROOT, data_dir, model_dir, experiment_dir, log_dir

def test_paths_structure():
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "src").exists()
    assert data_dir("raw").exists()
    assert data_dir("processed").exists()
    assert model_dir("vae_gmm", "v001").exists()
    assert experiment_dir("exp_test").exists()
    assert log_dir().exists()
