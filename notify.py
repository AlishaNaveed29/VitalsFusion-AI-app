"""Doctor notification channels for high-risk events.

Design rules (see project requirements):

* The app must distinguish: ``queued``, ``sent``, ``delivery_confirmed``,
  ``failed``, ``unavailable_offline`` and ``unavailable`` (not configured).
* A local dashboard notification works fully offline.
* A remote SMS to the doctor's phone only works with a real, configured
  communication channel (e.g. a SIM800L 2G modem **with** cellular service).
  This module never simulates a successful remote delivery: without a working
  channel it records ``failed`` or ``unavailable_offline`` with the reason.
"""

from __future__ import annotations

import importlib
from typing import Optional

import db as dbm
from config import AppConfig, offline

LOCAL = "local_dashboard"


def _queue_local(conn, cfg: AppConfig, event: dict) -> dict:
    nid = dbm.add_notification(
        conn, event["id"], LOCAL, "queued",
        "event queued into the local dashboard inbox",
    )
    dbm.update_notification_status(
        conn, nid, "sent",
        "stored in the local doctor dashboard (available offline; "
        "the assigned doctor sees it at next sign-in or refresh)",
    )
    return {"id": nid, "channel": LOCAL, "status": "sent"}


def _remote(conn, cfg: AppConfig, event: dict, is_offline: bool) -> dict:
    channel = cfg.remote_channel or "none"

    if channel in (None, "none"):
        nid = dbm.add_notification(
            conn, event["id"], "none", "unavailable",
            "No remote notification channel is configured "
            "(set remote_channel and a working gateway in app_config.json).",
        )
        return {"id": nid, "channel": "none", "status": "unavailable"}

    if is_offline:
        nid = dbm.add_notification(
            conn, event["id"], channel, "unavailable_offline",
            "Device is offline: no remote notification was attempted. "
            "A remote doctor phone cannot be reached without a network or "
            "cellular service (SIM800L needs 2G coverage).",
        )
        return {"id": nid, "channel": channel, "status": "unavailable_offline"}

    try:
        gateway = importlib.import_module(cfg.sms_gateway_module)
        send = getattr(gateway, "send_doctor_alert", None)
        if send is None:
            raise AttributeError("module has no send_doctor_alert(event) function")
    except Exception as exc:  # import/attribute problems -> honest failure
        nid = dbm.add_notification(
            conn, event["id"], channel, "failed",
            f"Remote channel '{channel}' is not usable ({exc.__class__.__name__}: {exc}). "
            "No message was sent - delivery is NOT simulated.",
        )
        return {"id": nid, "channel": channel, "status": "failed"}

    try:
        result = send(event)  # type: ignore[misc]
    except Exception as exc:
        nid = dbm.add_notification(
            conn, event["id"], channel, "failed",
            f"Gateway raised {exc.__class__.__name__}: {exc}. No delivery confirmed.",
        )
        return {"id": nid, "channel": channel, "status": "failed"}

    if isinstance(result, dict):
        confirmed = bool(result.get("confirmed"))
        ok = bool(result.get("ok"))
        detail = str(result.get("detail") or "")
    else:
        confirmed, ok, detail = False, bool(result), ""

    if ok and confirmed:
        status, msg = "delivery_confirmed", detail or "gateway confirmed delivery"
    elif ok:
        status, msg = "sent", detail or "gateway accepted the message (delivery not confirmed)"
    else:
        status, msg = "failed", detail or "gateway reported failure; no delivery"
    nid = dbm.add_notification(conn, event["id"], channel, status, msg,
                               confirmed=confirmed)
    return {"id": nid, "channel": channel, "status": status}


def notify_doctor_alert(conn, cfg: AppConfig, event: dict,
                        is_offline: Optional[bool] = None) -> list[dict]:
    """Record notification attempts for a newly created high-risk event.

    Returns one row per channel with its recorded delivery status.
    Never raises: a notification problem must not lose the clinical event.
    """
    outcomes: list[dict] = []
    if cfg.local_notification_enabled:
        try:
            outcomes.append(_queue_local(conn, cfg, event))
        except Exception as exc:
            outcomes.append({"channel": LOCAL, "status": "failed",
                             "detail": f"{exc.__class__.__name__}: {exc}"})
    try:
        off = offline(cfg) if is_offline is None else bool(is_offline)
        outcomes.append(_remote(conn, cfg, event, off))
    except Exception as exc:
        outcomes.append({"channel": cfg.remote_channel or "none", "status": "failed",
                         "detail": f"{exc.__class__.__name__}: {exc}"})
    return outcomes


def mark_alert_opened(conn, viewer: dbm.Viewer, event_id: int) -> int:
    """The assigned doctor opened the alert -> local delivery confirmed."""
    if not viewer.is_doctor:
        return 0
    return dbm.confirm_local_delivery(conn, event_id)
