"""Backend model registry: path resolution, one-time load, honest failures."""
from __future__ import annotations

import pytest

import model_registry as mr
from config import load_config


def test_find_ct_image_model_resolves_backend_path():
    cfg = load_config()
    path = mr.find_ct_image_model(cfg)
    assert path is not None and path.is_file()
    assert path.suffix == ".pt"


def test_load_ct_image_model_once_and_eval():
    cfg = load_config()
    mr.clear_ct_cache()
    model, meta, msg, ok = mr.load_ct_image_model(cfg)
    assert ok, msg
    assert meta["class_names"] == ("Low", "Mild", "Moderate", "Severe")
    import torch
    assert not model.training  # eval mode - never retrained at startup
    # second call uses the same cached object (loaded once per process)
    model2, _, _, ok2 = mr.load_ct_image_model(cfg)
    assert ok2 and model2 is model


def test_missing_weights_reported_not_faked(tmp_path):
    cfg = load_config()
    cfg.ct_image_model_path = str(tmp_path / "no_such_model.pt")
    # also hide fallbacks by pointing BASE search away: find still falls back,
    # so only assert the API contract: (model None, ok False) when nothing found
    # Simulate by clearing cache and monkeypatching find to None.
    orig = mr.find_ct_image_model
    mr.find_ct_image_model = lambda c: None
    try:
        model, meta, msg, ok = mr.load_ct_image_model(cfg)
    finally:
        mr.find_ct_image_model = orig
    assert model is None and meta is None and not ok
    assert "not found" in msg.lower()
    assert "no prediction" in msg.lower()


def test_fusion_without_ecg_is_honest():
    """Missing ECG must fail with an error, never a fabricated fused score."""
    import numpy as np
    import assessment_core as core
    from config import load_config
    cfg = load_config()
    model, msg, ok = core.prepare_model(cfg)
    assert ok, msg
    result = core.run_recording(model, np.empty(0), 100, cfg, ct_level=2)
    assert result["status"] != "completed"
    assert result["fused_score"] is None
    assert result["error"]
