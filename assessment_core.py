"""Shared AI assessment pipeline for the patient monitor and the doctor lab.

Uses the existing ECG CNN-LSTM Keras model (``ecg_model.py``), the radiology
report CT features (``ct_features.py``) and the late-fusion layer
(``fusion.py``) exactly as before - this module only wires them into the
two-role application and returns honest statuses when something is missing.

Fusion contract (Case A): ``run_recording`` combines per-window ECG scores
with a *scalar* CT context level 0..3 via ``fusion.fuse`` /
``fusion.fused_risk_score``. No learned fusion checkpoint exists in this
repository; inventing embedding-level fusion would require retraining data
and weights that are not available here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from config import AppConfig, BASE_DIR
from ct_features import encode_ct_report
from ecg_model import DEFAULT_MODEL_FS, EcgMiModel, analyze_signal
from fusion import FusionConfig, fused_risk_score, fuse


def find_model_zip(cfg: AppConfig) -> Optional[Path]:
    name = cfg.ecg_model_filename
    for p in (BASE_DIR / name, Path.cwd() / name):
        if p.exists():
            return p
    # fallback: any ECG model zip in the app folder (tolerates small
    # differences in the configured file name, e.g. extra/missing underscores)
    for folder in (BASE_DIR, Path.cwd()):
        for pat in ("ecg_lead2_mi_model*.zip", "*ecg*model*.zip"):
            matches = sorted(p for p in folder.glob(pat) if p.is_file())
            if matches:
                return matches[0]
    return None


def fusion_config(cfg: AppConfig) -> FusionConfig:
    return FusionConfig(**cfg.fusion_kwargs())


def ct_level_from_record(record: Optional[dict]) -> Optional[int]:
    if not record:
        return None
    level = record.get("context_level")
    return None if level is None else int(level)


def ct_level_from_report(cac_score=None, cac_category=None, stenosis_grade=None,
                         vessels_ge50=None) -> Optional[int]:
    feats = encode_ct_report(cac_score=cac_score, cac_category=cac_category,
                             stenosis_grade=stenosis_grade, vessels_ge50=vessels_ge50)
    return feats.context_level


# ---------------------------------------------------------------------------
# model loading (Streamlit-cached; called only inside a running app)
# ---------------------------------------------------------------------------
def load_model_cached(data: bytes) -> EcgMiModel:
    import streamlit as st

    @st.cache_resource(show_spinner="Loading ECG model...")
    def _load(payload: bytes) -> EcgMiModel:
        return EcgMiModel.from_zip_bytes(payload)

    return _load(data)


def prepare_model(cfg: AppConfig, uploaded_bytes: Optional[bytes] = None):
    """Return ``(model, message, ok)``. Never raises for a missing model."""
    if uploaded_bytes:
        try:
            return load_model_cached(uploaded_bytes), "ECG model loaded from the uploaded file.", True
        except Exception as exc:
            return None, f"Could not load the uploaded ECG model: {exc}", False
    path = find_model_zip(cfg)
    if path is None:
        return None, (f"ECG model file {cfg.ecg_model_filename} was not found next to "
                      "app.py and none was uploaded. AI assessment is UNAVAILABLE - "
                      "no score will be invented."), False
    try:
        return load_model_cached(path.read_bytes()), f"ECG model loaded from {path.name}.", True
    except Exception as exc:
        return None, f"Could not load the ECG model: {exc}", False


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
def run_recording(model: Optional[EcgMiModel], signal, fs_in: Optional[int],
                  cfg: AppConfig, ct_level: Optional[int],
                  use_bandpass: bool = False, use_zscore: bool = True,
                  model_fs: int = DEFAULT_MODEL_FS) -> dict:
    """raw signal -> per-window scores -> fused level + fused risk score.

    Returns a dict with ``status`` of ``completed`` or ``failed`` and an
    ``error`` message for every non-completed case. ``fused_score`` is ``None``
    whenever the model produced no usable output - never a fabricated value.
    """
    out = {
        "status": "failed", "model_name": None, "rows": [], "scores": [],
        "fused": None, "fused_score": None, "fused_level": None,
        "ecg_score_mean": None, "heart_rate_bpm": None, "ct_level": ct_level,
        "error": None, "config": {},
    }
    fc = fusion_config(cfg)
    out["config"] = {
        "high_risk_threshold": cfg.high_risk_threshold,
        "model_fs": int(model_fs),
        "use_bandpass": bool(use_bandpass),
        "use_zscore": bool(use_zscore),
        "ct_level": ct_level,
        "fusion": cfg.fusion_kwargs(),
        "model_filename": cfg.ecg_model_filename,
    }

    if model is None:
        out["error"] = ("ECG model not loaded - assessment UNAVAILABLE. "
                        "No risk score was computed.")
        out["status"] = "unavailable"
        return out
    out["model_name"] = model.name

    x = np.asarray(signal, dtype=float) if signal is not None else np.empty(0)
    if x.size == 0:
        out["error"] = "No ECG signal available (empty recording)."
        return out
    if fs_in is None or int(fs_in) <= 0:
        out["error"] = "Sampling rate of the signal is unknown or invalid."
        return out

    needed_s = model.input_len / float(model_fs)
    effective_len = len(x) * float(model_fs) / float(fs_in)
    if effective_len < model.input_len:
        out["error"] = (f"Recording too short: at least {needed_s:.0f} seconds are "
                        f"needed for one window at {model_fs} Hz.")
        return out

    try:
        rows, _processed = analyze_signal(model, x, int(fs_in), int(model_fs),
                                          use_bandpass, use_zscore)
    except Exception as exc:
        out["error"] = f"Signal processing failed: {exc}"
        return out

    out["rows"] = rows
    scores = [r["score"] for r in rows]
    out["scores"] = scores
    hrs = [r["hr_bpm"] for r in rows if r.get("hr_bpm") is not None]
    out["heart_rate_bpm"] = round(float(np.mean(hrs)), 1) if hrs else None

    valid = [s for s in scores if s is not None]
    if not valid:
        reasons = sorted({r["problems"] for r in rows if r.get("problems")})
        out["error"] = ("Every window failed the signal-quality gate - no score was "
                        "computed. " + ("; ".join(reasons) if reasons else ""))
        return out

    fused = fuse(scores, ct_level, fc)
    out["fused"] = fused
    out["fused_level"] = fused["level"]
    out["fused_score"] = fused_risk_score(scores, ct_level, fc)
    out["ecg_score_mean"] = round(float(np.mean(valid)), 4)
    out["status"] = "completed"
    if out["fused_score"] is None:
        out["status"] = "failed"
        out["error"] = "Fused risk score could not be computed from the model output."
    return out


def assessment_kwargs(result: dict, *, ecg_session_id=None, ct_record_id=None) -> dict:
    """Map a pipeline result onto :func:`emergency.process_assessment` kwargs."""
    return dict(
        status=result["status"],
        ecg_session_id=ecg_session_id,
        ct_record_id=ct_record_id,
        model_name=result.get("model_name"),
        fused_score=result.get("fused_score"),
        fused_level=result.get("fused_level"),
        ecg_score_mean=result.get("ecg_score_mean"),
        ct_level=result.get("ct_level"),
        heart_rate_bpm=result.get("heart_rate_bpm"),
        config=result.get("config"),
        windows=result.get("rows") or None,
        error_message=result.get("error"),
    )
