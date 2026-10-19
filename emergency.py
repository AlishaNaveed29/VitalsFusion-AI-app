"""High-risk emergency workflow (project requirement section 4/5/7).

Flow, strictly in order:

1. The real model result + timestamp is saved as an ``ai_assessments`` row.
2. If the fused score >= configured threshold (default 0.85) a **unique**
   high-risk event is created (one active event per patient; reruns reuse it).
3. The patient dashboard shows a prominent red warning with the measured heart
   rate and the actual model output.
4. A clearly labelled ``Call Ambulance`` action (``tel:`` link) is available;
   pressing it is recorded - the app never claims a call was connected.
5. The event appears on the assigned doctor's dashboard and a notification
   attempt is recorded per configured channel (local works offline; remote is
   never simulated).
6. The doctor can acknowledge / resolve the event and add notes.
"""

from __future__ import annotations

from typing import Optional

import db as dbm
from config import AppConfig
import notify


# ---------------------------------------------------------------------------
# assessments
# ---------------------------------------------------------------------------
def record_assessment(conn, viewer: dbm.Viewer, patient_id: int, *,
                      status: str, ecg_session_id=None, ct_record_id=None,
                      model_name=None, fused_score=None, fused_level=None,
                      ecg_score_mean=None, ct_level=None, heart_rate_bpm=None,
                      config=None, windows=None, error_message=None) -> dict:
    """Step 1: persist the actual model result (or the honest failure status)."""
    assessment_id = dbm.create_assessment(
        conn, viewer, patient_id,
        status=status, ecg_session_id=ecg_session_id, ct_record_id=ct_record_id,
        model_name=model_name, fused_score=fused_score, fused_level=fused_level,
        ecg_score_mean=ecg_score_mean, ct_level=ct_level, heart_rate_bpm=heart_rate_bpm,
        config=config, windows=windows, error_message=error_message,
    )
    return dbm.get_assessment(conn, viewer, assessment_id)


def evaluate_assessment(conn, cfg: AppConfig, assessment: dict,
                        is_offline: Optional[bool] = None) -> Optional[dict]:
    """Steps 2-5: raise (or refresh) the high-risk event for a completed
    assessment at/above the threshold. Returns the event or None.

    ``None`` is returned for failed/unavailable/low-score assessments - no
    event and no invented result.
    """
    if not assessment or assessment.get("status") != "completed":
        return None
    score = assessment.get("fused_score")
    if score is None:
        return None
    threshold = float(cfg.high_risk_threshold)
    if float(score) < threshold:
        return None

    event, created = dbm.create_or_refresh_event(
        conn,
        assessment["patient_id"],
        score=float(score),
        threshold=threshold,
        assessment_id=assessment["id"],
        heart_rate_bpm=assessment.get("heart_rate_bpm"),
    )
    if created:
        notify.notify_doctor_alert(conn, cfg, event, is_offline=is_offline)
        row = conn.execute("SELECT * FROM emergency_events WHERE id=?",
                           (event["id"],)).fetchone()
        event = dict(row) if row else event
    return event


def process_assessment(conn, viewer: dbm.Viewer, cfg: AppConfig, patient_id: int, **fields):
    """Convenience: save the assessment, then evaluate the emergency threshold."""
    is_offline = fields.pop("_is_offline", None)
    assessment = record_assessment(conn, viewer, patient_id, **fields)
    event = evaluate_assessment(conn, cfg, assessment, is_offline=is_offline)
    return assessment, event


# ---------------------------------------------------------------------------
# test-data simulation (clearly labelled; for testing the workflow only)
# ---------------------------------------------------------------------------
TEST_MODEL_TAG = "TEST-SIMULATION (not a real model output)"

_TEST_PRESETS = {
    "high": (0.93, 0.91, 116.0, "score forced above the threshold on purpose"),
    "low": (0.42, 0.40, 72.0, "score forced below the threshold on purpose"),
}


def simulate_test_assessment(conn, viewer: dbm.Viewer, cfg: AppConfig,
                             patient_id: int, kind: str = "high"):
    """Save a clearly labelled synthetic assessment through the REAL workflow.

    For testing the system without the trained model zip: ``kind="high"``
    saves a completed assessment above the threshold (raises a real high-risk
    event + notifications + patient banner); ``kind="low"`` saves one below
    the threshold (no event). Records are tagged TEST_MODEL_TAG and shown
    with a TEST warning everywhere; real model outputs are never overwritten
    (every assessment is a new row) and one active event per patient still
    applies.

    Returns ``(assessment, event)``.
    """
    if kind not in _TEST_PRESETS:
        raise ValueError(f"unknown test kind: {kind!r}")
    fused, ecg_mean, hr, note = _TEST_PRESETS[kind]
    ct = dbm.latest_ct_record(conn, viewer, patient_id)
    sessions = dbm.list_ecg_sessions(conn, viewer, patient_id, limit=1)
    ct_level = int(ct["context_level"]) if ct and ct.get("context_level") is not None else 0
    return process_assessment(
        conn, viewer, cfg, patient_id,
        status="completed",
        ecg_session_id=int(sessions[0]["id"]) if sessions else None,
        ct_record_id=int(ct["id"]) if ct else None,
        model_name=TEST_MODEL_TAG,
        fused_score=fused, fused_level=2 if kind == "high" else 0,
        ecg_score_mean=ecg_mean, ct_level=ct_level, heart_rate_bpm=hr,
        config={"test_simulation": True, "kind": kind, "note": note,
                "high_risk_threshold": cfg.high_risk_threshold},
        windows=[{"window": i, "start_s": (i - 1) * 10, "quality_ok": True,
                  "problems": "", "hr_bpm": hr, "score": fused,
                  "test_simulation": True} for i in (1, 2, 3)],
    )


# ---------------------------------------------------------------------------
# patient actions
# ---------------------------------------------------------------------------
def active_event(conn, viewer: dbm.Viewer, patient_id: int) -> Optional[dict]:
    dbm.require_patient_access(conn, viewer, patient_id)
    return dbm.active_event_for_patient(conn, patient_id)


def record_call_action(conn, viewer: dbm.Viewer, event_id: int) -> dict:
    """The patient pressed 'Call Ambulance'.

    Only the patient themselves may do this.  This records the *action*; the
    ``tel:`` link merely opens whatever calling app the device provides, so the
    system must never report that an ambulance was contacted.
    """
    if not viewer.is_patient:
        raise dbm.DataAccessDenied("Only the patient may start an emergency call.")
    event = dbm.get_event(conn, viewer, event_id)  # access check (self only)
    if viewer.id != event["patient_id"]:
        raise dbm.DataAccessDenied("You may only use the call button for your own event.")
    return dbm.set_patient_call(conn, viewer.id, event_id)


def dismiss_warning(conn, viewer: dbm.Viewer, event_id: int) -> None:
    """Patient hides the big banner (event stays open for the doctor)."""
    if not viewer.is_patient:
        raise dbm.DataAccessDenied("Only the patient can dismiss their own warning.")
    event = dbm.get_event(conn, viewer, event_id)
    if viewer.id != event["patient_id"]:
        raise dbm.DataAccessDenied("You may only dismiss your own warning.")
    dbm.set_patient_dismissed(conn, viewer.id, event_id)


# ---------------------------------------------------------------------------
# doctor actions
# ---------------------------------------------------------------------------
def _doctor_event(conn, viewer: dbm.Viewer, event_id: int) -> dict:
    if not viewer.is_doctor:
        raise dbm.DataAccessDenied("Doctor account required.")
    return dbm.get_event(conn, viewer, event_id)  # assignment/broader access enforced


def acknowledge_event(conn, viewer: dbm.Viewer, event_id: int, note: str = "") -> dict:
    _doctor_event(conn, viewer, event_id)
    event = dbm.set_event_status(conn, viewer.id, event_id, "acknowledged", note)
    notify.mark_alert_opened(conn, viewer, event_id)
    return event


def resolve_event(conn, viewer: dbm.Viewer, event_id: int, note: str = "") -> dict:
    _doctor_event(conn, viewer, event_id)
    event = dbm.set_event_status(conn, viewer.id, event_id, "resolved", note)
    notify.mark_alert_opened(conn, viewer, event_id)
    return event


def add_note(conn, viewer: dbm.Viewer, event_id: int, note: str) -> None:
    _doctor_event(conn, viewer, event_id)
    if not (note or "").strip():
        raise ValueError("Note must not be empty.")
    dbm.add_event_note(conn, viewer.id, event_id, note.strip())


def mark_opened(conn, viewer: dbm.Viewer, event_id: int) -> int:
    """Doctor opened the alert -> local notification becomes delivery_confirmed."""
    _doctor_event(conn, viewer, event_id)
    return notify.mark_alert_opened(conn, viewer, event_id)


def open_events_for_patient(conn, viewer: dbm.Viewer, patient_id: int) -> list[dict]:
    """Undismissed, unresolved events that must be shown as a red warning."""
    dbm.require_patient_access(conn, viewer, patient_id)
    rows = conn.execute(
        "SELECT * FROM emergency_events WHERE patient_id=? AND status IN ('open','acknowledged')"
        " AND patient_dismissed_at IS NULL ORDER BY id DESC",
        (int(patient_id),),
    ).fetchall()
    return [dict(r) for r in rows]


def all_active_events(conn, viewer: dbm.Viewer, patient_id: int) -> list[dict]:
    """Active events regardless of dismissal (doctor view / compact strip)."""
    dbm.require_patient_access(conn, viewer, patient_id)
    rows = conn.execute(
        "SELECT * FROM emergency_events WHERE patient_id=? AND status IN ('open','acknowledged')"
        " ORDER BY id DESC",
        (int(patient_id),),
    ).fetchall()
    return [dict(r) for r in rows]
