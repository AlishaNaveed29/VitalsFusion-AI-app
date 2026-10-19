"""Run with:  python -m pytest -q   (from this folder)

The Keras comparison and the real-model checks run only when the ECG zip is
found next to this file (or via app_config.json); the Keras comparison also
needs `pip install keras` plus a Keras 3 backend (jax or torch), e.g.
KERAS_BACKEND=torch.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from ecg_model import (EcgMiModel, analyze_signal, assess_quality, resample_signal,
                       segment_windows, synthetic_ecg, zscore)
from fusion import FusionConfig, adjusted_thresholds, fuse


def _find_zip():
    try:
        from config import load_config
        from assessment_core import find_model_zip
        p = find_model_zip(load_config())
        if p is not None:
            return Path(p)
    except Exception:
        pass
    for name in ("ecg_lead2_mi_model_v2.keras.zip", "ecg_lead2_mi_model_v2_keras.zip"):
        p = Path(__file__).with_name(name)
        if p.exists():
            return p
    return Path(__file__).with_name("ecg_lead2_mi_model_v2_keras.zip")


ZIP = _find_zip()


# ------------------------------------------------------------------ fusion rules
def test_ct_level_lowers_thresholds():
    cfg = FusionConfig()
    assert adjusted_thresholds(cfg, None) == (0.50, 0.80)
    assert adjusted_thresholds(cfg, 0) == (0.50, 0.80)
    w, a = adjusted_thresholds(cfg, 3)
    assert (round(w, 2), round(a, 2)) == (0.35, 0.65)


def test_threshold_floor_and_order():
    cfg = FusionConfig(thr_watch=0.2, thr_alert=0.25, step_per_ct_level=0.2, min_threshold=0.1)
    w, a = adjusted_thresholds(cfg, 3)
    assert w == 0.1 and a >= w


def test_levels():
    cfg = FusionConfig()
    assert fuse([0.1, 0.2, 0.1], None, cfg)["level"] == 0
    assert fuse([0.6, 0.7, 0.1], None, cfg)["level"] == 1
    assert fuse([0.9, 0.95, 0.1], None, cfg)["level"] == 2


def test_persistence_blocks_single_spike():
    assert fuse([0.1, 0.99, 0.1], None, FusionConfig())["level"] == 0


def test_ct_can_change_outcome():
    scores = [0.40, 0.42, 0.41]
    assert fuse(scores, None)["level"] == 0
    assert fuse(scores, 3)["level"] == 1          # watch threshold drops to 0.35


def test_invalid_windows_are_ignored_and_no_decision():
    assert fuse([None, None, None], 2)["level"] == -1
    r = fuse([None, 0.9, None], None)             # one valid window -> needs only 1
    assert r["level"] == 2 and r["windows_valid"] == 1


def test_only_last_n_windows_count():
    assert fuse([0.99, 0.99, 0.1, 0.1, 0.1], None, FusionConfig())["level"] == 0


def test_bad_config_and_ct_level():
    with pytest.raises(ValueError):
        fuse([0.5], None, FusionConfig(thr_watch=0.9, thr_alert=0.5))
    with pytest.raises(ValueError):
        fuse([0.5], None, FusionConfig(k_required=4, n_windows=3))
    with pytest.raises(ValueError):
        fuse([0.5], 7)


# ------------------------------------------------------------------ signal helpers
def test_windows_and_zscore():
    x = np.arange(2500, dtype=float)
    assert segment_windows(x, 1000).shape == (2, 1000)
    assert segment_windows(x[:900], 1000).shape == (0, 1000)
    z = zscore(segment_windows(synthetic_ecg(20, 100), 1000))
    assert np.allclose(z.mean(axis=1), 0, atol=1e-9) and np.allclose(z.std(axis=1), 1, atol=1e-6)


def test_resample_length():
    assert len(resample_signal(np.zeros(2500), 250, 100)) == 1000
    assert len(resample_signal(np.zeros(1000), 100, 100)) == 1000


def test_quality_gate():
    fs = 100
    clean = synthetic_ecg(10, fs, 72, 0.05, 0)[:1000]
    assert assess_quality(clean, fs)["ok"]
    assert 70 <= assess_quality(clean, fs)["hr_bpm"] <= 74
    assert not assess_quality(np.zeros(1000), fs)["ok"]
    assert not assess_quality(np.r_[clean[:999], np.nan], fs)["ok"]
    assert not assess_quality(np.clip(clean * 3, -1, 1), fs)["ok"]
    rejected = sum(not assess_quality(np.random.default_rng(s).normal(size=1000), fs)["ok"] for s in range(50))
    assert rejected >= 45                          # noise is almost always rejected


# ------------------------------------------------------------------ real model
needs_zip = pytest.mark.skipif(not ZIP.exists(), reason="ECG zip not next to the tests")


@needs_zip
def test_real_model_loads_and_predicts():
    m = EcgMiModel.from_path(ZIP)
    assert m.input_len == 1000
    p = m.predict(zscore(segment_windows(synthetic_ecg(30, 100), 1000)))
    assert p.shape == (3,) and np.all((p >= 0) & (p <= 1))
    with pytest.raises(ValueError):
        m.predict(np.zeros((1, 999)))


@needs_zip
def test_pipeline_rejects_flat_and_scores_clean():
    m = EcgMiModel.from_path(ZIP)
    rows, _ = analyze_signal(m, np.zeros(3000), 100)
    assert all(not r["quality_ok"] and r["score"] is None for r in rows)
    rows, _ = analyze_signal(m, synthetic_ecg(30, 250, 72, 0.05, 0), 250)   # needs resampling
    assert len(rows) == 3 and all(r["quality_ok"] and r["score"] is not None for r in rows)


@needs_zip
@pytest.mark.skipif(os.environ.get("KERAS_BACKEND") not in ("numpy", "torch"),
                     reason="set KERAS_BACKEND=numpy (or torch) to compare with Keras")
def test_matches_keras():
    keras = pytest.importorskip("keras")
    km = keras.saving.load_model(str(ZIP), compile=False) if ZIP.suffix == ".keras" else None
    if km is None:
        import shutil, tempfile
        tmp = Path(tempfile.mkdtemp()) / "m.keras"
        shutil.copy(ZIP, tmp)
        km = keras.saving.load_model(str(tmp), compile=False)
    m = EcgMiModel.from_path(ZIP)
    X = np.vstack([zscore(segment_windows(synthetic_ecg(60, 100, 75, 0.05, 1), 1000)),
                   np.random.default_rng(0).normal(size=(4, 1000)) * 0.3])
    pk = np.asarray(km.predict(X[:, :, None].astype("float32"), verbose=0)).ravel()
    assert np.abs(pk - m.predict(X)).max() < 1e-4
