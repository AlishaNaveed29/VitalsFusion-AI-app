"""Backend model registry for VitalsFusion AI.

Loads the trained model weights ONCE per process from local backend paths.
Never trains, never accepts weights through the Streamlit UI, and never
invents predictions when a file is missing - it returns an honest status
describing what is needed instead.

Models
    * ECG  : ``ecg_lead2_mi_model_v2.keras.zip``  (see assessment_core)
    * CT   : ``models/ResNet18_2.5D_best.pt``     (see ct_image_model)
    * Fusion: rule-based late fusion in ``fusion.py`` / ``assessment_core``.
      No learned fusion checkpoint exists in this repository; if one is
      trained later it can be registered here the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from config import AppConfig, BASE_DIR

# process-level cache: path -> (model, meta)
_CT_CACHE: dict[str, tuple] = {}


def find_ct_image_model(cfg: AppConfig) -> Optional[Path]:
    """Resolve the CT image-model checkpoint on the backend disk."""
    raw = Path(cfg.ct_image_model_path or "models/ResNet18_2.5D_best.pt")
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(BASE_DIR / raw)
        candidates.append(Path.cwd() / raw)
    candidates += [
        BASE_DIR / "models" / "ResNet18_2.5D_best.pt",
        BASE_DIR / "ResNet18_2.5D_best.pt",
    ]
    for p in candidates:
        if p.is_file():
            return p
    models_dir = BASE_DIR / "models"
    if models_dir.is_dir():
        matches = sorted(models_dir.glob("ResNet18_2.5D*.pt"))
        if matches:
            return matches[0]
    return None


def load_ct_image_model(cfg: AppConfig) -> Tuple[object, Optional[dict], str, bool]:
    """Return ``(model, meta, message, ok)``. Loads weights once per process.

    The network is built from the checkpoint's own metadata and put in
    ``eval()`` mode by ``ct_image_model.load_model`` - never retrained here.
    """
    path = find_ct_image_model(cfg)
    if path is None:
        return None, None, (
            f"CT image-model weights not found on the backend. Expected "
            f"'{cfg.ct_image_model_path}' (for example models/ResNet18_2.5D_best.pt) "
            "next to app.py. Image inference is UNAVAILABLE - no prediction "
            "will be invented. Report-values and manual CT levels still work."
        ), False
    key = str(path.resolve())
    if key not in _CT_CACHE:
        try:
            import ct_image_model as cim
        except ImportError as exc:
            return None, None, (
                f"CT image-model packages are not installed ({exc}). "
                "Install torch, torchvision and pydicom to enable image "
                "inference, or use manual/report CT levels."
            ), False
        try:
            _CT_CACHE[key] = cim.load_model(path.read_bytes())
        except Exception as exc:
            return None, None, f"Could not load CT image model from {path.name}: {exc}", False
    model, meta = _CT_CACHE[key]
    return model, meta, f"CT image model loaded from backend file {path.name}.", True


def clear_ct_cache() -> None:
    """Testing helper: drop the cached CT model."""
    _CT_CACHE.clear()
