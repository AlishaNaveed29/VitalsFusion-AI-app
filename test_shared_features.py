"""Tests for the shared doctor+patient features:

- CT upload available to BOTH roles (patient saves own record, doctor saves
  on an assigned patient, others denied)
- ECG monitoring sessions created by the doctor for an assigned patient are
  visible to that patient; unassigned doctors denied
- registration allows both roles (db-level path used by the web form)
"""

import pytest

import auth
import db as dbm

ITERS = 10_000


@pytest.fixture()
def conn(tmp_path):
    c = dbm.init_db(tmp_path / "shared.db")
    yield c
    c.close()


@pytest.fixture()
def accounts(conn):
    doc = dbm.create_user(conn, "dr@clinic.pk",
                          auth.hash_password("DoctorPass1!", iterations=ITERS),
                          "doctor", full_name="Dr One")
    pat = dbm.create_user(conn, "p1@test.pk",
                          auth.hash_password("PatientPass1!", iterations=ITERS),
                          "patient", full_name="Patient One")
    other = dbm.create_user(conn, "p2@test.pk",
                            auth.hash_password("PatientPass2!", iterations=ITERS),
                            "patient", full_name="Patient Two")
    doc2 = dbm.create_user(conn, "dr2@clinic.pk",
                           auth.hash_password("DoctorPass2!", iterations=ITERS),
                           "doctor", full_name="Dr Two")
    dbm.assign_patient(conn, doc, pat, doc)
    return {"doc": doc, "pat": pat, "other": other, "doc2": doc2}


def v(conn, uid):
    return dbm.viewer_from_row(dbm.get_user(conn, uid))


# ---------------------------------------------------------------------------
# CT upload for both roles
# ---------------------------------------------------------------------------
def test_patient_can_upload_ct_for_self(conn, accounts):
    pview = v(conn, accounts["pat"])
    sid = dbm.create_ct_record(conn, pview, accounts["pat"],
                               cac_score=120.0, stenosis_grade=2, vessels_ge50=1,
                               context_level=2, report_text="self-reported report",
                               source="report")
    recs = dbm.list_ct_records(conn, pview, accounts["pat"])
    assert [r["id"] for r in recs] == [sid]
    assert recs[0]["context_level"] == 2


def test_patient_cannot_upload_ct_for_other_patient(conn, accounts):
    pview = v(conn, accounts["pat"])
    with pytest.raises(dbm.DataAccessDenied):
        dbm.create_ct_record(conn, pview, accounts["other"], context_level=1,
                             source="manual")


def test_doctor_can_upload_ct_for_assigned_patient(conn, accounts):
    dview = v(conn, accounts["doc"])
    dbm.create_ct_record(conn, dview, accounts["pat"], cac_score=45.0,
                         stenosis_grade=1, vessels_ge50=0, context_level=1,
                         report_text="mild", source="report")
    recs = dbm.list_ct_records(conn, dview, accounts["pat"])
    assert len(recs) == 1 and recs[0]["source"] == "report"


def test_unassigned_doctor_cannot_upload_ct(conn, accounts):
    d2view = v(conn, accounts["doc2"])
    with pytest.raises(dbm.DataAccessDenied):
        dbm.create_ct_record(conn, d2view, accounts["pat"], context_level=0,
                             source="manual")


# ---------------------------------------------------------------------------
# patient-recorded ECG is visible to the assigned doctor (view-only on the
# doctor side - doctors do not connect devices or run the demo generator)
# ---------------------------------------------------------------------------
def test_patient_recorded_session_seen_by_assigned_doctor(conn, accounts):
    pview = v(conn, accounts["pat"])
    sid = dbm.create_ecg_session(conn, pview, accounts["pat"], fs_hz=100,
                                 device_label="VitalsFusion demo generator",
                                 device_status="connected", source="demo",
                                 status="running")
    dbm.update_ecg_session(conn, pview, sid, signal=[0.1] * 1500, ended=True,
                           mean_hr_bpm=72.0, status="stopped")
    # assigned doctor sees the patient's recording (read/review path)
    dview = v(conn, accounts["doc"])
    sessions = dbm.list_ecg_sessions(conn, dview, accounts["pat"])
    assert [s["id"] for s in sessions] == [sid]
    assert sessions[0]["n_samples"] == 1500
    # another patient cannot see it
    oview = v(conn, accounts["other"])
    assert dbm.list_ecg_sessions(conn, oview, accounts["other"]) == []


def test_unassigned_doctor_cannot_read_other_patients_ecg(conn, accounts):
    pview = v(conn, accounts["pat"])
    sid = dbm.create_ecg_session(conn, pview, accounts["pat"], fs_hz=100,
                                 device_label="demo", device_status="connected",
                                 source="demo", status="stopped")
    dbm.update_ecg_session(conn, pview, sid, signal=[0.0] * 100, ended=True)
    d2view = v(conn, accounts["doc2"])
    with pytest.raises(dbm.DataAccessDenied):
        dbm.list_ecg_sessions(conn, d2view, accounts["pat"])


# ---------------------------------------------------------------------------
# registration roles (same db path as the web registration form)
# ---------------------------------------------------------------------------
def test_registration_creates_both_roles(conn):
    d = dbm.create_user(conn, "newdoc@clinic.pk",
                        auth.hash_password("DoctorPass1!", iterations=ITERS),
                        "doctor", full_name="New Doc")
    p = dbm.create_user(conn, "newpat@test.pk",
                        auth.hash_password("PatientPass1!", iterations=ITERS),
                        "patient", full_name="New Pat")
    assert dbm.get_user(conn, d)["role"] == "doctor"
    assert dbm.get_user(conn, p)["role"] == "patient"
    # new doctors start with assigned-patients-only access
    row = conn.execute("SELECT broader_access FROM users WHERE id=?", (d,)).fetchone()
    assert row["broader_access"] == 0
    # and they can authenticate like any other account
    res = auth.authenticate(conn, "newdoc@clinic.pk", "DoctorPass1!",
                            __import__("config").AppConfig(offline_mode="on"))
    assert res.ok and res.user["role"] == "doctor"


def test_only_two_roles_allowed(conn):
    with pytest.raises(Exception):
        dbm.create_user(conn, "x@y.z", "hash", "admin")
