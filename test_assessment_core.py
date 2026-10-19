"""Tests for the shared assessment pipeline (``assessment_core.py``).

The real Keras zip is not required: a tiny stub with the same interface as
``EcgMiModel`` (``input_len`` + ``predict``) stands in for the model so the
windowing, quality gate, fusion, risk score and emergency wiring can be tested
end to end. The stub is a test double, not a fabricated clinical result.
"""

import numpy as np
import pytest

import auth
import assessment_core as core
import db as dbm
import emergency
from config import AppConfig
from ecg_model import synthetic_ecg

ITERS = 10_000


class StubModel:
    """Interface-compatible stand-in for EcgMiModel (tests only)."""

    name = "stub-model-for-tests"
    input_len = 1000

    def __init__(self, score: float = 0.99):
        self.score = float(score)

    def predict(self, windows):
        w = np.asarray(windows)
        return np.full(len(w), self.score)


@pytest.fixture()
def cfg():
    return AppConfig(offline_mode="on")


@pytest.fixture()
def conn(tmp_path):
    c = dbm.init_db(tmp_path / "pipe.db")
    yield c
    c.close()


def _accounts(conn):
    d = dbm.create_user(conn, "dr@t.pk", auth.hash_password("DoctorPass1!", iterations=ITERS),
                        "doctor")
    p = dbm.create_user(conn, "pt@t.pk", auth.hash_password("PatientPass1!", iterations=ITERS),
                        "patient")
    dbm.assign_patient(conn, d, p, d)
    return d, p


def test_no_model_is_unavailable_not_low_risk(cfg):
    r = core.run_recording(None, synthetic_ecg(30, 100), 100, cfg, ct_level=None)
    assert r["status"] == "unavailable"
    assert r["fused_score"] is None
    assert "UNAVAILABLE" in r["error"]


def test_too_short_and_empty_signal_fail_honestly(cfg):
    m = StubModel()
    r = core.run_recording(m, np.zeros(500), 100, cfg, None)
    assert r["status"] == "failed" and r["fused_score"] is None
    assert "too short" in r["error"].lower()
    r2 = core.run_recording(m, np.empty(0), 100, cfg, None)
    assert r2["status"] == "failed" and "no ecg signal" in r2["error"].lower()
    r3 = core.run_recording(m, synthetic_ecg(30, 100), None, cfg, None)
    assert r3["status"] == "failed"


def test_flat_line_is_rejected_by_quality_gate(cfg):
    r = core.run_recording(StubModel(0.5), np.zeros(3000), 100, cfg, None)
    assert r["status"] == "failed"
    assert "quality" in r["error"].lower()
    assert all(not row["quality_ok"] and row["score"] is None for row in r["rows"])


def test_completed_pipeline_and_event_creation(conn, cfg):
    d, p = _accounts(conn)
    viewer = dbm.viewer_from_row(dbm.get_user(conn, d))
    sig = synthetic_ecg(30, 100, hr=72, noise=0.05, seed=0)
    result = core.run_recording(StubModel(0.99), sig, 100, cfg, ct_level=3)
    assert result["status"] == "completed"
    assert result["fused_score"] is not None and result["fused_score"] >= cfg.high_risk_threshold
    assert result["heart_rate_bpm"] is not None
    assessment, event = emergency.process_assessment(
        conn, viewer, cfg, p, **core.assessment_kwargs(result))
    assert assessment["status"] == "completed"
    assert event is not None
    assert event["score"] == pytest.approx(result["fused_score"])
    assert event["heart_rate_bpm"] == result["heart_rate_bpm"]


def test_low_score_completes_without_event(conn, cfg):
    d, p = _accounts(conn)
    viewer = dbm.viewer_from_row(dbm.get_user(conn, d))
    sig = synthetic_ecg(30, 100, hr=72, noise=0.05, seed=0)
    result = core.run_recording(StubModel(0.10), sig, 100, cfg, ct_level=None)
    assert result["status"] == "completed"
    assert result["fused_score"] < cfg.high_risk_threshold
    assessment, event = emergency.process_assessment(
        conn, viewer, cfg, p, **core.assessment_kwargs(result))
    assert assessment["fused_score"] == pytest.approx(result["fused_score"])
    assert event is None


def test_ct_level_from_saved_record(conn, cfg):
    d, p = _accounts(conn)
    viewer = dbm.viewer_from_row(dbm.get_user(conn, d))
    level = core.ct_level_from_report(cac_score=450, stenosis_grade=4, vessels_ge50=3)
    assert level == 3                       # severe CAC + severe stenosis + 3 vessels
    rid = dbm.create_ct_record(conn, viewer, p, cac_score=450, stenosis_grade=4,
                               vessels_ge50=3, context_level=level, source="report")
    rec = dbm.latest_ct_record(conn, viewer, p)
    assert rec["id"] == rid and rec["context_level"] == 3
    assert core.ct_level_from_record(rec) == 3
    assert core.ct_level_from_record(None) is None
