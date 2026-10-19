"""CT upload UI helpers: filtering, readable formatting, access control."""
from __future__ import annotations

from datetime import date

import pytest

import auth
import ct_upload as cu
import db as dbm

ITERS = 10_000


@pytest.fixture()
def conn(tmp_path):
    c = dbm.init_db(tmp_path / "ctui.db")
    yield c
    c.close()


@pytest.fixture()
def accounts(conn):
    pat = dbm.create_user(conn, "p1@test.pk",
                          auth.hash_password("PatientPass1!", iterations=ITERS),
                          "patient", full_name="Patient One")
    other = dbm.create_user(conn, "p2@test.pk",
                            auth.hash_password("PatientPass2!", iterations=ITERS),
                            "patient", full_name="Patient Two")
    return {"pat": pat, "other": other}


def _rec(rid, created="2026-01-15T10:00:00+00:00", source="report",
         level=2, text="mild calcification"):
    return {"id": rid, "created_at": created, "source": source,
            "context_level": level, "report_text": text, "cac_score": None,
            "cac_category": 1, "stenosis_grade": None, "vessels_ge50": None,
            "recorded_by": 3}


def test_filter_by_source_and_query():
    recs = [_rec(1, source="report"), _rec(2, source="manual"),
            _rec(3, source="image_model", text="severe stenosis")]
    out = cu.filter_ct_records(recs, sources=["manual"])
    assert [r["id"] for r in out] == [2]
    out = cu.filter_ct_records(recs, query="stenosis")
    assert [r["id"] for r in out] == [3]


def test_filter_by_date_range():
    recs = [_rec(1, created="2026-01-10T00:00:00+00:00"),
            _rec(2, created="2026-02-01T00:00:00+00:00")]
    out = cu.filter_ct_records(recs, date_from=date(2026, 1, 15),
                               date_to=date(2026, 2, 28))
    assert [r["id"] for r in out] == [2]


def test_format_row_readable_missing_values():
    row = cu.format_ct_row(_rec(9, level=None, text=""))
    assert row["CAC score"] == "—"
    assert row["stenosis grade"] == "—"
    assert row["CT level"] == "—"
    assert row["source"] == "Report values"
    row2 = cu.format_ct_row(_rec(10, level=3, source="manual"))
    assert row2["CT level"] == "3"
    assert row2["source"] == "Manual image review"


def test_patient_cannot_list_or_write_other_patients_ct(conn, accounts):
    p1 = accounts["pat"]
    other = accounts["other"]
    view = dbm.Viewer(id=other, role="patient", email="o@x", full_name="O")
    with pytest.raises(dbm.DataAccessDenied):
        dbm.list_ct_records(conn, view, p1)
    with pytest.raises(dbm.DataAccessDenied):
        dbm.create_ct_record(conn, view, p1, context_level=1, source="manual")


def test_fingerprint_includes_patient_id():
    """Same upload bytes for different patients must produce different keys."""
    names = ["a.dcm", "b.dcm"]
    blobs = [b"\x00\x01", b"\x02\x03"]
    fp_a = cu._upload_fingerprint(names, blobs, pid=1)
    fp_b = cu._upload_fingerprint(names, blobs, pid=2)
    assert fp_a != fp_b
    # same patient + same files is stable
    assert cu._upload_fingerprint(names, blobs, pid=1) == fp_a


def test_save_ai_prediction_rejects_wrong_patient(monkeypatch):
    """A stored prediction for patient A must not save onto patient B."""
    st_pred = {
        "patient_id": 1,
        "probs": [0.7, 0.1, 0.1, 0.1],
        "pred": 0,
        "class_names": ["Low", "Mild", "Moderate", "Severe"],
        "n_slices": 32,
        "model_name": "ResNet18_2.5D",
        "saved_record_id": None,
    }
    with pytest.raises(ValueError, match="different patient"):
        cu._save_ai_prediction(None, None, pid=2, st_pred=st_pred)


def test_save_ai_prediction_rejects_double_save():
    st_pred = {
        "patient_id": 1,
        "probs": [0.7, 0.1, 0.1, 0.1],
        "pred": 0,
        "class_names": ["Low", "Mild", "Moderate", "Severe"],
        "saved_record_id": 99,
    }
    with pytest.raises(ValueError, match="already saved"):
        cu._save_ai_prediction(None, None, pid=1, st_pred=st_pred)
