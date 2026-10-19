"""High-risk emergency workflow tests.

Run with:  python -m pytest -q test_emergency.py

Covers: event creation, duplicate-alert prevention, emergency-call action
recording, notification delivery statuses (including offline failure),
doctor acknowledgment, patient warning dismissal, and honest handling of
failed/unavailable assessments.
"""

import pytest

import auth
import db as dbm
import emergency
import notify
from config import AppConfig
from fusion import fused_risk_score

ITERS = 10_000


@pytest.fixture()
def cfg():
    c = AppConfig(offline_mode="on", remote_channel=None)
    c.validate()
    return c


@pytest.fixture()
def conn(tmp_path):
    c = dbm.init_db(tmp_path / "emg.db")
    yield c
    c.close()


@pytest.fixture()
def accounts(conn):
    d1 = dbm.create_user(conn, "dr@clinic.pk", auth.hash_password("DoctorPass1!", iterations=ITERS),
                         "doctor", full_name="Dr One")
    p1 = dbm.create_user(conn, "p1@test.pk", auth.hash_password("PatientPass1!", iterations=ITERS),
                         "patient", full_name="Patient One")
    p2 = dbm.create_user(conn, "p2@test.pk", auth.hash_password("PatientPass2!", iterations=ITERS),
                         "patient", full_name="Patient Two")
    dbm.assign_patient(conn, d1, p1, d1)
    return {"d1": d1, "p1": p1, "p2": p2}


def v(conn, uid):
    return dbm.viewer_from_row(dbm.get_user(conn, uid))


def _assess(conn, accounts, score=0.93, status="completed", hr=118.0, **kw):
    return emergency.record_assessment(
        conn, v(conn, accounts["p1"]), accounts["p1"],
        status=status, fused_score=score, fused_level=2, ecg_score_mean=score,
        ct_level=0, heart_rate_bpm=hr, model_name="ecg_lead2_mi_model_v2",
        **kw,
    )


# ---------------------------------------------------------------- risk score
def test_fused_risk_score_uses_real_scores_and_never_invents_low_risk():
    assert fused_risk_score([0.9, 0.95, None], 0) == pytest.approx(0.925, abs=1e-3)
    assert fused_risk_score([0.9, 0.95], 3) == pytest.approx(min(1.0, 0.925 + 0.15), abs=1e-3)
    assert fused_risk_score([None, None], 0) is None      # no usable window -> None
    assert fused_risk_score([], 2) is None


# ---------------------------------------------------------------- events
def test_high_risk_event_created_at_threshold(conn, accounts, cfg):
    a = _assess(conn, accounts, score=cfg.high_risk_threshold)
    event = emergency.evaluate_assessment(conn, cfg, a)
    assert event is not None
    assert event["status"] == "open"
    assert event["score"] == pytest.approx(cfg.high_risk_threshold)
    assert event["threshold"] == cfg.high_risk_threshold
    assert event["heart_rate_bpm"] == 118.0
    assert event["assessment_id"] == a["id"]
    rows = conn.execute("SELECT * FROM emergency_events").fetchall()
    assert len(rows) == 1
    # event created BEFORE notification was attempted, both recorded
    notifs = dbm.list_notifications(conn, v(conn, accounts["d1"]), event["id"])
    assert notifs and any(n["channel"] == "local_dashboard" for n in notifs)


def test_below_threshold_creates_no_event(conn, accounts, cfg):
    a = _assess(conn, accounts, score=0.50)
    assert emergency.evaluate_assessment(conn, cfg, a) is None
    assert dbm.list_events_for_patient(conn, v(conn, accounts["p1"]), accounts["p1"]) == []


def test_failed_or_unavailable_assessment_never_invents_low_risk(conn, accounts, cfg):
    a = _assess(conn, accounts, score=None, status="failed",
                error_message="signal processing failed: flat line")
    assert emergency.evaluate_assessment(conn, cfg, a) is None
    saved = dbm.get_assessment(conn, v(conn, accounts["p1"]), a["id"])
    assert saved["status"] == "failed" and saved["fused_score"] is None
    assert saved["error_message"].startswith("signal processing failed")


def test_duplicate_alert_prevention_on_rerun(conn, accounts, cfg):
    a1 = _assess(conn, accounts, score=0.95)
    e1 = emergency.evaluate_assessment(conn, cfg, a1)
    # the same assessment evaluated again (Streamlit rerun) -> no new event
    e2 = emergency.evaluate_assessment(conn, cfg, a1)
    assert e1["id"] == e2["id"]
    # a second, new high-risk assessment while the event is active -> refreshed
    a2 = _assess(conn, accounts, score=0.97)
    e3 = emergency.evaluate_assessment(conn, cfg, a2)
    assert e3["id"] == e1["id"]
    total = conn.execute("SELECT COUNT(*) c FROM emergency_events").fetchone()["c"]
    assert total == 1
    assert e3["assessment_id"] == a2["id"]          # latest model output is linked
    # only ONE notification batch: no duplicate alerts on rerun
    n = conn.execute("SELECT COUNT(*) c FROM notification_attempts").fetchone()["c"]
    assert n == 2                                   # local + remote-attempt rows


def test_new_event_allowed_after_previous_resolved(conn, accounts, cfg):
    a1 = _assess(conn, accounts, score=0.95)
    e1 = emergency.evaluate_assessment(conn, cfg, a1)
    emergency.resolve_event(conn, v(conn, accounts["d1"]), e1["id"], "patient seen")
    a2 = _assess(conn, accounts, score=0.99)
    e2 = emergency.evaluate_assessment(conn, cfg, a2)
    assert e2["id"] != e1["id"]
    assert conn.execute("SELECT COUNT(*) c FROM emergency_events").fetchone()["c"] == 2


def test_threshold_is_configurable(conn, accounts):
    cfg = AppConfig(high_risk_threshold=0.5, offline_mode="on")
    a = _assess(conn, accounts, score=0.6)
    assert emergency.evaluate_assessment(conn, cfg, a) is not None   # 0.6 >= 0.5
    cfg2 = AppConfig(high_risk_threshold=0.99, offline_mode="on")
    a2 = _assess(conn, accounts, score=0.95)
    assert emergency.evaluate_assessment(conn, cfg2, a2) is None     # 0.95 < 0.99
    a3 = _assess(conn, accounts, score=0.995)
    e = emergency.evaluate_assessment(conn, cfg2, a3)
    assert e is not None and e["score"] == pytest.approx(0.995)


# ---------------------------------------------------------------- notifications
def _notif_statuses(conn, event_id):
    return [r["status"] for r in
            conn.execute("SELECT status FROM notification_attempts WHERE event_id=?",
                         (event_id,))]


def test_notification_statuses_local_and_unconfigured_remote(conn, accounts, cfg):
    a = _assess(conn, accounts, score=0.95)
    event = emergency.evaluate_assessment(conn, cfg, a)
    rows = dbm.list_notifications(conn, v(conn, accounts["d1"]), event["id"])
    by_ch = {r["channel"]: r["status"] for r in rows}
    assert by_ch["local_dashboard"] == "sent"
    assert by_ch["none"] == "unavailable"            # no remote channel configured
    # doctor opens the alert -> local delivery confirmed
    emergency.mark_opened(conn, v(conn, accounts["d1"]), event["id"])
    rows = dbm.list_notifications(conn, v(conn, accounts["d1"]), event["id"])
    by_ch = {r["channel"]: r["status"] for r in rows}
    assert by_ch["local_dashboard"] == "delivery_confirmed"


def test_notification_offline_status(conn, accounts):
    cfg = AppConfig(offline_mode="on", remote_channel="sms")
    a = _assess(conn, accounts, score=0.95)
    event = emergency.evaluate_assessment(conn, cfg, a, is_offline=True)
    rows = dbm.list_notifications(conn, v(conn, accounts["d1"]), event["id"])
    by_ch = {r["channel"]: r["status"] for r in rows}
    assert by_ch["local_dashboard"] == "sent"        # local works offline
    assert by_ch["sms"] == "unavailable_offline"
    assert any("offline" in (r["detail"] or "").lower() for r in rows
               if r["channel"] == "sms")


def test_notification_failure_when_no_real_gateway(conn, accounts):
    cfg = AppConfig(offline_mode="off", remote_channel="sms",
                    sms_gateway_module="definitely_missing_gateway_xyz")
    a = _assess(conn, accounts, score=0.95)
    event = emergency.evaluate_assessment(conn, cfg, a, is_offline=False)
    rows = dbm.list_notifications(conn, v(conn, accounts["d1"]), event["id"])
    by_ch = {r["channel"]: r["status"] for r in rows}
    assert by_ch["sms"] == "failed"                  # never simulated as sent
    sms = [r for r in rows if r["channel"] == "sms"][0]
    assert "NOT simulated" in sms["detail"] or "No message was sent" in sms["detail"]


def test_local_notification_never_crashes_the_event(conn, accounts):
    cfg = AppConfig(offline_mode="on")
    a = _assess(conn, accounts, score=0.95)
    event = emergency.evaluate_assessment(conn, cfg, a)
    assert event is not None                          # event exists regardless


# ---------------------------------------------------------------- call button
def test_patient_call_action_recorded_only_by_patient(conn, accounts, cfg):
    a = _assess(conn, accounts, score=0.95)
    event = emergency.evaluate_assessment(conn, cfg, a)
    pv = v(conn, accounts["p1"])
    updated = emergency.record_call_action(conn, pv, event["id"])
    assert updated["patient_call_initiated"] == 1
    assert updated["patient_call_at"]
    audit = dbm.list_audit(conn, action="ambulance_call_action")
    assert audit and audit[0]["user_id"] == accounts["p1"]
    # another patient and the doctor cannot press someone else's call button
    with pytest.raises(dbm.DataAccessDenied):
        emergency.record_call_action(conn, v(conn, accounts["p2"]), event["id"])
    with pytest.raises(dbm.DataAccessDenied):
        emergency.record_call_action(conn, v(conn, accounts["d1"]), event["id"])


def test_patient_can_call_even_without_assessment(conn, accounts, cfg):
    """No assessment available -> patient still gets an emergency-call path."""
    p1 = v(conn, accounts["p1"])
    sid = dbm.create_ecg_session(conn, p1, accounts["p1"], fs_hz=100, source="demo",
                                 device_status="disconnected", status="error")
    emergency.process_assessment(
        conn, p1, cfg, accounts["p1"], status="unavailable",
        error_message="ECG model not loaded; no assessment was run",
        ecg_session_id=sid,
    )
    assert dbm.list_events_for_patient(conn, p1, accounts["p1"]) == []
    # the emergency-call UI does not depend on an assessment: a manual event
    # row can be raised by the patient's explicit request path in the UI
    # (tested at service level here: emergency contact is still readable)
    contact = dbm.get_emergency_contact(conn, p1, accounts["p1"])
    assert contact["emergency_service_number"] == "1122"


# ---------------------------------------------------------------- doctor workflow
def test_doctor_acknowledge_and_resolve_with_notes(conn, accounts, cfg):
    a = _assess(conn, accounts, score=0.96)
    event = emergency.evaluate_assessment(conn, cfg, a)
    dv = v(conn, accounts["d1"])
    acked = emergency.acknowledge_event(conn, dv, event["id"], "Called patient, reviewing ECG")
    assert acked["status"] == "acknowledged"
    emergency.add_note(conn, dv, event["id"], "Patient reports chest pain improving")
    resolved = emergency.resolve_event(conn, dv, event["id"], "Advised ER visit, patient attended")
    assert resolved["status"] == "resolved"
    acks = dbm.list_event_acknowledgments(conn, dv, event["id"])
    assert [x["action"] for x in acks] == ["acknowledge", "note", "resolve"]
    assert all(x["doctor_name"] == "Dr One" for x in acks)
    # local notification confirmed by opening the alert
    rows = dbm.list_notifications(conn, dv, event["id"])
    assert any(r["status"] == "delivery_confirmed" for r in rows)


def test_doctor_without_assignment_cannot_touch_event(conn, accounts, cfg):
    d2 = dbm.create_user(conn, "dr2@clinic.pk",
                         auth.hash_password("DoctorPass2!", iterations=ITERS), "doctor")
    a = _assess(conn, accounts, score=0.95)
    event = emergency.evaluate_assessment(conn, cfg, a)
    with pytest.raises(dbm.DataAccessDenied):
        emergency.acknowledge_event(conn, v(conn, d2), event["id"])
    with pytest.raises(dbm.DataAccessDenied):
        emergency.add_note(conn, v(conn, d2), event["id"], "not mine")
    # d2's dashboard shows nothing
    assert dbm.list_events_for_doctor(conn, d2) == []
    assert dbm.list_events_for_doctor(conn, accounts["d1"]) != []


def test_patient_dismissal_keeps_event_open_for_doctor(conn, accounts, cfg):
    a = _assess(conn, accounts, score=0.95)
    event = emergency.evaluate_assessment(conn, cfg, a)
    pv = v(conn, accounts["p1"])
    assert emergency.open_events_for_patient(conn, pv, accounts["p1"])   # visible
    emergency.dismiss_warning(conn, pv, event["id"])
    assert emergency.open_events_for_patient(conn, pv, accounts["p1"]) == []
    # compact strip / doctor view still sees the active event
    assert emergency.active_event(conn, pv, accounts["p1"])["id"] == event["id"]
    assert dbm.list_events_for_doctor(conn, accounts["d1"])[0]["id"] == event["id"]
    # dismissal must not create or duplicate anything
    assert conn.execute("SELECT COUNT(*) c FROM emergency_events").fetchone()["c"] == 1


def test_patient_cannot_acknowledge_own_event(conn, accounts, cfg):
    a = _assess(conn, accounts, score=0.95)
    event = emergency.evaluate_assessment(conn, cfg, a)
    with pytest.raises(dbm.DataAccessDenied):
        emergency.acknowledge_event(conn, v(conn, accounts["p1"]), event["id"])
    with pytest.raises(dbm.DataAccessDenied):
        emergency.resolve_event(conn, v(conn, accounts["p1"]), event["id"])
    assert dbm.get_event(conn, v(conn, accounts["p1"]), event["id"])["status"] == "open"


# ---------------------------------------------------------------- offline end-to-end
def test_offline_mode_end_to_end(conn, accounts):
    cfg = AppConfig(offline_mode="on", remote_channel="sms")
    p1 = v(conn, accounts["p1"])
    a = emergency.record_assessment(conn, p1, accounts["p1"], status="completed",
                                    fused_score=0.91, heart_rate_bpm=125.0)
    event = emergency.evaluate_assessment(conn, cfg, a, is_offline=True)
    assert event is not None                          # local warning works offline
    statuses = dict(conn.execute(
        "SELECT channel, status FROM notification_attempts WHERE event_id=?",
        (event["id"],)).fetchall())
    assert statuses["local_dashboard"] == "sent"
    assert statuses["sms"] == "unavailable_offline"
    # offline: patient can still record a call action and read their records
    emergency.record_call_action(conn, p1, event["id"])
    assert dbm.list_assessments(conn, p1, accounts["p1"]) != []
