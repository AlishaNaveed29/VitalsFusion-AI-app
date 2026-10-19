"""Application configuration for VitalsFusion AI (offline-first prototype).

Defaults are read first, then overridden by an optional ``app_config.json``
next to this file, then by environment variables:

    VITALSFUSION_DB        path of the SQLite database file
    VITALSFUSION_CONFIG    path of a JSON config file
    VITALSFUSION_OFFLINE   "1" / "0" force offline / online mode

No secrets and no default passwords are stored here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DB_PATH = BASE_DIR / "vitalsfusion.db"
DEFAULT_CONFIG_PATH = BASE_DIR / "app_config.json"


@dataclass
class AppConfig:
    # --- emergency workflow -------------------------------------------------
    # High-risk threshold for a fused assessment (subject to model validation).
    high_risk_threshold: float = 0.85
    # Configurable telephone link for the Call Ambulance button.
    emergency_service_number: str = "1122"
    emergency_service_label: str = "Local emergency number (Lahore/Pakistan prototype: 1122)"

    # --- sessions / accounts ------------------------------------------------
    session_absolute_ttl_minutes: int = 240
    session_idle_ttl_minutes: int = 60
    max_failed_logins: int = 5
    lockout_minutes: int = 15
    min_password_length: int = 8

    # --- fusion design parameters (see fusion.py) ---------------------------
    thr_watch: float = 0.50
    thr_alert: float = 0.80
    step_per_ct_level: float = 0.05
    min_threshold: float = 0.10
    k_required: int = 2
    n_windows: int = 3

    # --- notifications ------------------------------------------------------
    # local_dashboard always works offline. Remote channels are never faked.
    local_notification_enabled: bool = True
    remote_channel: Optional[str] = None      # None / "none" / "sms"
    sms_gateway_module: str = "sms_gateway"   # real integration hook (SIM800L etc.)
    offline_mode: str = "auto"                # "auto" | "on" | "off"

    # --- storage ------------------------------------------------------------
    db_path: str = str(DEFAULT_DB_PATH)

    ecg_model_filename: str = "ecg_lead2_mi_model_v2_keras.zip"
    # Backend CT image-model weights (never uploaded through the UI).
    ct_image_model_path: str = "models/ResNet18_2.5D_best.pt"

    # ------------------------------------------------------------------
    def fusion_kwargs(self) -> dict:
        return dict(
            thr_watch=self.thr_watch,
            thr_alert=self.thr_alert,
            step_per_ct_level=self.step_per_ct_level,
            min_threshold=self.min_threshold,
            k_required=self.k_required,
            n_windows=self.n_windows,
        )

    def validate(self) -> None:
        if not (0.0 < self.high_risk_threshold <= 1.0):
            raise ValueError("high_risk_threshold must be in (0, 1]")
        if self.offline_mode not in ("auto", "on", "off"):
            raise ValueError("offline_mode must be auto, on or off")
        if self.remote_channel not in (None, "none", "sms"):
            raise ValueError("remote_channel must be none or sms")
        if not str(self.emergency_service_number).strip():
            raise ValueError("emergency_service_number must not be empty")


def _coerce(data: dict) -> dict:
    known = {f: data[f] for f in AppConfig.__dataclass_fields__ if f in data}
    return known


def load_config(path: Optional[Path] = None) -> AppConfig:
    path = Path(path or os.environ.get("VITALSFUSION_CONFIG") or DEFAULT_CONFIG_PATH)
    data = {}
    if path.exists():
        try:
            data = _coerce(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read config file {path}: {exc}") from exc

    env_db = os.environ.get("VITALSFUSION_DB")
    if env_db:
        data["db_path"] = env_db

    cfg = AppConfig(**data)
    # resolve a relative database path against the app folder, not the CWD
    dbp = Path(cfg.db_path)
    cfg.db_path = str(dbp if dbp.is_absolute() else (BASE_DIR / dbp))
    cfg.validate()
    return cfg


def offline(cfg: AppConfig) -> bool:
    """True when the app is operating without a usable remote connection."""
    if cfg.offline_mode == "on":
        return True
    if cfg.offline_mode == "off":
        return False
    return not _network_reachable()


def _network_reachable(timeout: float = 0.8) -> bool:
    """Best-effort connectivity probe; never raises. Result is cached 60 s."""
    import socket
    import time

    cached = getattr(_network_reachable, "_cache", None)
    now = time.time()
    if cached and now - cached[0] < 60:
        return cached[1]
    ok = False
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=timeout):
            ok = True
    except OSError:
        ok = False
    _network_reachable._cache = (now, ok)
    return ok


def save_config(cfg: AppConfig, path: Optional[Path] = None) -> Path:
    path = Path(path or DEFAULT_CONFIG_PATH)
    path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return path
