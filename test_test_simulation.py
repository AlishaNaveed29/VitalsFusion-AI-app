"""In-app TEST-simulation tests (fake data through the REAL workflow).

Run with:  python -m pytest -q test_test_simulation.py

Covers: labelled TEST high/low assessments via
emergency.simulate_test_assessment, event creation + dedup, no event below
threshold, doctor/patient access to the resulting records, and the
synthetic-record badge (ui_common.demo_note).
"""

import pytest

import auth
import db as dbm
import emergency
from config import AppConfig
from ui_common import demo_note

ITERS = 10_000


@pytest.fixture()
def cfg():
    c = AppConfig(offline_mode="on", remote_channel=None)
    c.validate()
    return c


@pytest.fixture()
def conn(tmp_path):
    c = dbm.init_db(tmp_path / "sim.db")
    yield c
    c.close()


@pytest.fixture()
def accounts(conn):
    d1 = dbm.create_user(conn, "dr@clinic.pk",
                         auth.hash_password("DoctorPass1!", iterations=ITERS),
                         "doctor", full_name="Dr One")
    p1 = dbm.create_user(conn, "p1@test.pk",
                         auth.hash_password("PatientPass1!", iterations=ITERS),
                         "patient", full_name="Patient One")
    p2 = dbm.create_user(conn, "p2@test.pk",
                         auth.hash_password("PatientPass2!", iterations=ITERS),
                         "patient", full_name="Patient Two")
    dbm.assign_patient(conn, d1, p1, d1)
    return {"d1": d1, "p1": p1, "p2": p2}


def v(conn, uid):
    return dbm.viewer_from_row(dbm.get_user(conn, uid))


def test_high_simulation_raises_labelled_event(conn, cfg, accounts):
    dview = v(conn, accounts["d1"])
    assessment, event = emergency.simulate_test_assessment(
        conn, dview, cfg, accounts["p1"], kind="high")
    assert assessment["status"] == "completed"
    assert assessment["fused_score"] == pytest.approx(0.93)
    assert assessment["model_name"] == emergency.TEST_MODEL_TAG
    assert event is not None
    assert event["score"] == pytest.approx(0.93)
    assert event["threshold"] == pytest.approx(cfg.high_risk_threshold)
    # local dashboard notification row exists (real workflow, offline)
    notifs = dbm.list_notifications(conn, dview, event["id"])
    assert any(n["channel"] == "local_dashboard" for n in notifs)
    # the patient sees the active event on their own dashboard path
    pview = v(conn, accounts["p1"])
    events = emergency.all_active_events(conn, pview, accounts["p1"])
    assert len(events) == 1


def test_high_simulation_deduplicates_active_event(conn, cfg, accounts):
    dview = v(conn, accounts["d1"])
    emergency.simulate_test_assessment(conn, dview, cfg, accounts["p1"], kind="high")
    a2, e2 = emergency.simulate_test_assessment(conn, dview, cfg, accounts["p1"],
                                                 kind="high")
    assert e2 is not None and e2["status"] in ("open", "acknowledged")
    n_events = conn.execute("SELECT COUNT(*) c FROM emergency_events").fetchone()["c"]
    n_assess = conn.execute("SELECT COUNT(*) c FROM ai_assessments").fetchone()["c"]
    assert n_events == 1        # one active event per patient still applies
    assert n_assess == 2        # each run is a new assessment row


def test_low_simulation_creates_no_event(conn, cfg, accounts):
    dview = v(conn, accounts["d1"])
    assessment, event = emergency.simulate_test_assessment(
        conn, dview, cfg, accounts["p1"], kind="low")
    assert assessment["status"] == "completed"
    assert assessment["fused_score"] == pytest.approx(0.42)
    assert assessment["model_name"] == emergency.TEST_MODEL_TAG
    assert event is None
    assert conn.execute("SELECT COUNT(*) c FROM emergency_events").fetchone()["c"] == 0


def test_simulation_is_patient_isolated(conn, cfg, accounts):
    dview = v(conn, accounts["d1"])
    emergency.simulate_test_assessment(conn, dview, cfg, accounts["p1"], kind="high")
    # p2 is a real (unassigned) patient - unaffected
    p2view = v(conn, accounts["p2"])
    assert emergency.all_active_events(conn, p2view, accounts["p2"]) == []
    assert dbm.list_assessments(conn, p2view, accounts["p2"]) == []


def test_unassigned_doctor_cannot_simulate_for_other_patients(conn, cfg, accounts):
    d2 = dbm.create_user(conn, "dr2@clinic.pk",
                         auth.hash_password("DoctorPass2!", iterations=ITERS),
                         "doctor", full_name="Dr Two")
    d2view = v(conn, d2)
    with pytest.raises(dbm.DataAccessDenied):
        emergency.simulate_test_assessment(conn, d2view, cfg, accounts["p1"],
                                           kind="high")


def test_unknown_kind_rejected(conn, cfg, accounts):
    dview = v(conn, accounts["d1"])
    with pytest.raises(ValueError):
        emergency.simulate_test_assessment(conn, dview, cfg, accounts["p1"],
                                           kind="invented")


def test_synthetic_record_badge():
    assert demo_note(emergency.TEST_MODEL_TAG)
    assert demo_note("DEMO-SYNTHETIC (not a real model output)")
    assert demo_note("ecg_lead2_mi_model_v2") == ""
    assert demo_note(None) == ""
    assert demo_note("") == ""
