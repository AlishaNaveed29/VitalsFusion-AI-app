"""Authentication, sessions and access-control tests.

Run with:  python -m pytest -q test_auth_access.py

All patients used here are synthetic test accounts created in a throwaway
database.
"""

import pytest

import auth
import db as dbm
from config import AppConfig

ITERS = 10_000  # fast hashes for tests only


@pytest.fixture()
def cfg():
    c = AppConfig(offline_mode="on")
    c.validate()
    return c


@pytest.fixture()
def conn(tmp_path):
    c = dbm.init_db(tmp_path / "test.db")
    yield c
    c.close()


@pytest.fixture()
def accounts(conn):
    """doctor d1, patients p1/p2, p1 assigned to d1; doctor d2 unassigned."""
    d1 = dbm.create_user(conn, "d1@clinic.pk", auth.hash_password("DoctorPass1!", iterations=ITERS),
                         "doctor", full_name="Dr One")
    d2 = dbm.create_user(conn, "d2@clinic.pk", auth.hash_password("DoctorPass2!", iterations=ITERS),
                         "doctor", full_name="Dr Two")
    p1 = dbm.create_user(conn, "p1@test.pk", auth.hash_password("PatientPass1!", iterations=ITERS),
                         "patient", full_name="Patient One")
    p2 = dbm.create_user(conn, "p2@test.pk", auth.hash_password("PatientPass2!", iterations=ITERS),
                         "patient", full_name="Patient Two")
    dbm.assign_patient(conn, d1, p1, d1)
    return {"d1": d1, "d2": d2, "p1": p1, "p2": p2}


def viewer(conn, uid):
    return dbm.viewer_from_row(dbm.get_user(conn, uid))


# ---------------------------------------------------------------- passwords
def test_password_hash_is_salted_and_verifies():
    h1 = auth.hash_password("SecretPass99", iterations=ITERS)
    h2 = auth.hash_password("SecretPass99", iterations=ITERS)
    assert h1 != h2                       # random salt
    assert "SecretPass99" not in h1       # no plaintext
    assert h1.startswith("pbkdf2_sha256$")
    assert auth.verify_password("SecretPass99", h1)
    assert not auth.verify_password("wrong", h1)
    assert not auth.verify_password("x", "garbage-format")


def test_password_policy_and_no_default_password():
    assert auth.check_password_policy("short") is not None
    assert auth.check_password_policy("aaaaaaaa") is not None
    assert auth.check_password_policy("LongEnough1") is None
    assert auth.check_password_policy(None) is not None
    assert len(auth.generate_password(16)) >= 16


# ---------------------------------------------------------------- login
def test_doctor_login_success(conn, accounts, cfg):
    res = auth.authenticate(conn, "d1@clinic.pk", "DoctorPass1!", cfg)
    assert res.ok and res.user["role"] == "doctor"
    row = dbm.get_user(conn, accounts["d1"])
    assert row["last_login_at"] is not None and row["failed_attempts"] == 0


def test_patient_login_with_username_or_email(conn, accounts, cfg):
    dbm.create_user(conn, "ali", auth.hash_password("PatientPass9!", iterations=ITERS),
                    "patient", full_name="Ali")
    assert auth.authenticate(conn, "ali", "PatientPass9!", cfg).ok
    assert auth.authenticate(conn, "ALI", "PatientPass9!", cfg).ok  # case-insensitive


def test_invalid_credentials_give_identical_message(conn, accounts, cfg):
    bad_pw = auth.authenticate(conn, "d1@clinic.pk", "nope", cfg)
    bad_id = auth.authenticate(conn, "ghost@clinic.pk", "nope", cfg)
    assert not bad_pw.ok and not bad_id.ok
    assert bad_pw.reason == bad_id.reason == "Invalid email/username or password."


def test_account_lockout_after_max_failures(conn, accounts, cfg):
    for _ in range(cfg.max_failed_logins):
        assert not auth.authenticate(conn, "p1@test.pk", "wrong", cfg).ok
    locked = auth.authenticate(conn, "p1@test.pk", "PatientPass1!", cfg)
    assert not locked.ok and "locked" in locked.reason.lower()


def test_deactivated_account_cannot_login(conn, accounts, cfg):
    dbm.set_user_active(conn, accounts["d1"], accounts["p2"], False, "test")
    res = auth.authenticate(conn, "p2@test.pk", "PatientPass2!", cfg)
    assert not res.ok and "deactivated" in res.reason.lower()
    dbm.set_user_active(conn, accounts["d1"], accounts["p2"], True, "test")
    assert auth.authenticate(conn, "p2@test.pk", "PatientPass2!", cfg).ok


def test_session_expiry_reason(cfg):
    import datetime as dt
    s = auth.new_session({"id": 1, "role": "doctor", "email": "x", "full_name": ""}, cfg)
    assert auth.session_expired_reason(s) is None
    s.absolute_expiry = (dt.datetime.now(dt.timezone.utc) -
                         dt.timedelta(minutes=1)).isoformat()
    assert "expired" in auth.session_expired_reason(s).lower()
    s2 = auth.new_session({"id": 1, "role": "patient", "email": "x", "full_name": ""}, cfg)
    s2.idle_expiry = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)).isoformat()
    assert "inactivity" in auth.session_expired_reason(s2).lower()


def test_viewer_reflects_database_not_session_claims(conn, accounts, cfg):
    res = auth.authenticate(conn, "p1@test.pk", "PatientPass1!", cfg)
    session = auth.new_session(res.user, cfg)
    v = auth.make_viewer(conn, session)
    assert v.is_patient and v.id == accounts["p1"]
    # forged session claiming doctor role is rejected by make_viewer
    session.role = "doctor"
    with pytest.raises(dbm.DataAccessDenied):
        auth.make_viewer(conn, session)


# ---------------------------------------------------------------- roles
def test_only_two_roles_allowed(conn):
    with pytest.raises(ValueError):
        dbm.create_user(conn, "a@b.c", "hash", "administrator")
    with pytest.raises(ValueError):
        dbm.create_user(conn, "a@b.c", "hash", "researcher")
    roles = {u["role"] for u in dbm.list_users(conn)}
    assert roles <= {"doctor", "patient"}


def test_patient_cannot_change_role_or_privilege_fields(conn, accounts):
    v = viewer(conn, accounts["p1"])
    before = dbm.get_user(conn, accounts["p1"])
    # whitelist: unknown keys such as role/broader_access/is_active are ignored
    dbm.update_profile(conn, v, accounts["p1"],
                       {"phone": "0300", "role": "doctor", "broader_access": 1,
                        "is_active": 1})
    after = dbm.get_user(conn, accounts["p1"])
    assert after["role"] == "patient"
    assert after["broader_access"] == 0 and after["is_active"] == 1
    assert before["role"] == after["role"]


# ---------------------------------------------------------------- access control
def test_patient_reads_only_own_records(conn, accounts):
    v1 = viewer(conn, accounts["p1"])
    v2 = viewer(conn, accounts["p2"])
    dbm.create_ecg_session(conn, v2, accounts["p2"], fs_hz=100, source="demo")
    own = dbm.list_ecg_sessions(conn, v1, accounts["p1"])
    assert own == []                       # p1 has no sessions yet
    with pytest.raises(dbm.DataAccessDenied):
        dbm.list_ecg_sessions(conn, v1, accounts["p2"])
    with pytest.raises(dbm.DataAccessDenied):
        dbm.list_assessments(conn, v1, accounts["p2"])
    with pytest.raises(dbm.DataAccessDenied):
        dbm.get_emergency_contact(conn, v1, accounts["p2"])
    # audit trail records the denied attempt
    assert any(r["action"] == "access_denied" for r in dbm.list_audit(conn, action="access_denied"))


def test_patient_cannot_use_doctor_entry_points(conn, accounts):
    v1 = viewer(conn, accounts["p1"])
    with pytest.raises(dbm.DataAccessDenied):
        dbm.list_events_for_doctor(conn, accounts["p1"])
    with pytest.raises(dbm.DataAccessDenied):
        dbm.add_clinical_note(conn, accounts["p1"], accounts["p1"], "self note")
    with pytest.raises(dbm.DataAccessDenied):
        dbm.set_event_status(conn, accounts["p1"], 1, "acknowledged")
    with pytest.raises(dbm.DataAccessDenied):
        dbm.assign_patient(conn, accounts["p1"], accounts["p2"], accounts["p1"])
    # privilege fields cannot be granted by a patient
    with pytest.raises(dbm.DataAccessDenied):
        dbm.set_broader_access(conn, accounts["p1"], accounts["p1"], True)
    assert dbm.get_user(conn, accounts["p1"])["broader_access"] == 0
    assert dbm.get_user(conn, accounts["p1"])["role"] == "patient"
    _ = v1


def test_doctor_sees_only_assigned_patients(conn, accounts):
    d1v, d2v = viewer(conn, accounts["d1"]), viewer(conn, accounts["d2"])
    assert [p["id"] for p in dbm.list_assigned_patients(conn, accounts["d1"])] == [accounts["p1"]]
    assert dbm.list_assigned_patients(conn, accounts["d2"]) == []
    # d2 (no assignment) cannot read p1
    with pytest.raises(dbm.DataAccessDenied):
        dbm.list_ecg_sessions(conn, d2v, accounts["p1"])
    with pytest.raises(dbm.DataAccessDenied):
        dbm.create_ct_record(conn, d2v, accounts["p1"], cac_score=10)
    # after assignment d1 can
    assert dbm.can_access_patient(conn, d1v, accounts["p1"])
    assert not dbm.can_access_patient(conn, d1v, accounts["p2"])
    # revoke again
    dbm.unassign_patient(conn, accounts["d1"], accounts["d1"], accounts["p1"])
    assert not dbm.can_access_patient(conn, d1v, accounts["p1"])


def test_broader_access_must_be_explicit(conn, accounts):
    d2v = viewer(conn, accounts["d2"])
    assert not dbm.can_access_patient(conn, d2v, accounts["p1"])
    dbm.set_broader_access(conn, accounts["d1"], accounts["d2"], True)
    d2v = viewer(conn, accounts["d2"])
    assert dbm.can_access_patient(conn, d2v, accounts["p1"])
    dbm.set_broader_access(conn, accounts["d1"], accounts["d2"], False)
    d2v = viewer(conn, accounts["d2"])
    assert not dbm.can_access_patient(conn, d2v, accounts["p1"])


def test_unauthorized_role_access_to_alert_list(conn, accounts):
    with pytest.raises(dbm.DataAccessDenied):
        dbm.list_events_for_doctor(conn, accounts["p1"])


# ---------------------------------------------------------------- persistence
def test_records_survive_restart(tmp_path):
    path = tmp_path / "persist.db"
    c1 = dbm.init_db(path)
    uid = dbm.create_user(c1, "persist@test.pk",
                          auth.hash_password("PersistPass1!", iterations=ITERS),
                          "patient", full_name="Persist")
    v = dbm.viewer_from_row(dbm.get_user(c1, uid))
    dbm.update_emergency_contact(conn=c1, viewer=v, patient_id=uid,
                                 values={"contact_name": "Mother",
                                         "contact_number": "0300-1234567"})
    c1.close()

    c2 = dbm.init_db(path)                    # application restart
    row = dbm.get_user_by_identifier(c2, "persist@test.pk")
    assert row is not None and row["role"] == "patient"
    v2 = dbm.viewer_from_row(row)
    contact = dbm.get_emergency_contact(c2, v2, uid)
    assert contact["contact_name"] == "Mother"
    assert contact["contact_number"] == "0300-1234567"
    c2.close()
