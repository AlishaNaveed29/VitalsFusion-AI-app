"""SQLite storage layer for VitalsFusion AI.

Two application roles only: ``doctor`` and ``patient``.

Access control is enforced **inside this module**: every patient-scoped read
and write goes through ``require_patient_access``, so a caller that forgets the
check cannot leak another patient's records.  The Streamlit navigation layer is
never the only gate.

Migrations are applied automatically by :func:`init_db` (idempotent), or
explicitly with ``python migrate.py``.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

SCHEMA_VERSION = 2

ROLES = ("doctor", "patient")

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "vitalsfusion.db"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_path(db_path) -> Path:
    return Path(db_path or os.environ.get("VITALSFUSION_DB") or DEFAULT_DB_PATH)


def encode_signal(values: Sequence[float]) -> bytes:
    import numpy as np

    return np.asarray(values, dtype="<f4").tobytes()


def decode_signal(blob: Optional[bytes]):
    import numpy as np

    if not blob:
        return np.empty(0, dtype=np.float32)
    return np.frombuffer(blob, dtype="<f4")


class DataAccessDenied(PermissionError):
    """Raised when a viewer tries to touch a patient they are not allowed to see."""


@dataclass(frozen=True)
class Viewer:
    """The authenticated principal performing an operation."""

    id: int
    role: str
    email: str = ""
    full_name: str = ""
    broader_access: bool = False  # doctors only: explicitly granted wider access

    @property
    def is_doctor(self) -> bool:
        return self.role == "doctor"

    @property
    def is_patient(self) -> bool:
        return self.role == "patient"


def viewer_from_row(row) -> Viewer:
    return Viewer(
        id=int(row["id"]),
        role=row["role"],
        email=row["email"],
        full_name=row["full_name"] or "",
        broader_access=bool(row["broader_access"]),
    )


# ---------------------------------------------------------------------------
# connection / migration
# ---------------------------------------------------------------------------
def connect(db_path=None) -> sqlite3.Connection:
    path = _as_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Exactly two application roles are permitted.
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    email           TEXT NOT NULL COLLATE NOCASE UNIQUE,
    username        TEXT COLLATE NOCASE UNIQUE,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('doctor','patient')),
    full_name       TEXT NOT NULL DEFAULT '',
    is_active       INTEGER NOT NULL DEFAULT 1,
    broader_access  INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_login_at   TEXT,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until    TEXT
);

CREATE TABLE IF NOT EXISTS doctor_patient_assignments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id   INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    patient_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    assigned_by INTEGER REFERENCES users(id),
    assigned_at TEXT NOT NULL,
    is_active   INTEGER NOT NULL DEFAULT 1,
    UNIQUE (doctor_id, patient_id)
);

CREATE TABLE IF NOT EXISTS patient_profiles (
    patient_id          INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    date_of_birth       TEXT,
    sex                 TEXT,
    phone               TEXT,
    location_text       TEXT,      -- voluntarily provided by the patient
    accessibility_notes TEXT,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS emergency_contacts (
    patient_id              INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    contact_name            TEXT NOT NULL DEFAULT '',
    relationship            TEXT NOT NULL DEFAULT '',
    contact_number          TEXT NOT NULL DEFAULT '',
    emergency_service_number TEXT NOT NULL DEFAULT '1122',
    location_text           TEXT NOT NULL DEFAULT '',
    accessibility_notes     TEXT NOT NULL DEFAULT '',
    updated_at              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ecg_sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_by    INTEGER NOT NULL REFERENCES users(id),
    started_at    TEXT,
    ended_at      TEXT,
    status        TEXT NOT NULL DEFAULT 'running'
                  CHECK (status IN ('running','paused','stopped','error')),
    device_label  TEXT NOT NULL DEFAULT '',
    device_status TEXT NOT NULL DEFAULT 'disconnected'
                  CHECK (device_status IN ('connected','disconnected','unknown')),
    source        TEXT NOT NULL DEFAULT 'demo'
                  CHECK (source IN ('device','upload','demo')),
    fs_hz         INTEGER,
    n_samples     INTEGER NOT NULL DEFAULT 0,
    signal        BLOB,
    mean_hr_bpm   REAL,
    analysis_json TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ecg_sessions_patient ON ecg_sessions(patient_id, created_at);

CREATE TABLE IF NOT EXISTS ct_records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    recorded_by   INTEGER NOT NULL REFERENCES users(id),
    source        TEXT NOT NULL DEFAULT 'report'
                  CHECK (source IN ('report','image_model','manual')),
    cac_score     REAL,
    cac_category  INTEGER,
    stenosis_grade INTEGER,
    vessels_ge50  INTEGER,
    context_level INTEGER,
    report_text   TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ct_records_patient ON ct_records(patient_id, created_at);

CREATE TABLE IF NOT EXISTS ai_assessments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ecg_session_id  INTEGER REFERENCES ecg_sessions(id) ON DELETE SET NULL,
    ct_record_id    INTEGER REFERENCES ct_records(id) ON DELETE SET NULL,
    status          TEXT NOT NULL CHECK (status IN ('completed','failed','unavailable')),
    model_name      TEXT,
    fused_score     REAL,
    fused_level     INTEGER,
    ecg_score_mean  REAL,
    ct_level        INTEGER,
    heart_rate_bpm  REAL,
    config_json     TEXT,
    windows_json    TEXT,
    error_message   TEXT,
    created_by      INTEGER REFERENCES users(id),
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_assessments_patient ON ai_assessments(patient_id, created_at);

CREATE TABLE IF NOT EXISTS emergency_events (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uid            TEXT NOT NULL UNIQUE,
    patient_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    assessment_id        INTEGER REFERENCES ai_assessments(id) ON DELETE SET NULL,
    score                REAL NOT NULL,
    threshold            REAL NOT NULL,
    heart_rate_bpm       REAL,
    status               TEXT NOT NULL DEFAULT 'open'
                         CHECK (status IN ('open','acknowledged','resolved')),
    patient_call_initiated INTEGER NOT NULL DEFAULT 0,
    patient_call_at      TEXT,
    patient_dismissed_at TEXT,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);
-- One active high-risk event per patient: blocks duplicate alerts at the DB level.
CREATE UNIQUE INDEX IF NOT EXISTS ux_active_event_per_patient
    ON emergency_events(patient_id) WHERE status IN ('open','acknowledged');
CREATE INDEX IF NOT EXISTS ix_events_patient ON emergency_events(patient_id, created_at);

CREATE TABLE IF NOT EXISTS notification_attempts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    INTEGER NOT NULL REFERENCES emergency_events(id) ON DELETE CASCADE,
    channel     TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN
                ('queued','sent','delivery_confirmed','failed',
                 'unavailable_offline','unavailable')),
    detail      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    confirmed_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_notifications_event ON notification_attempts(event_id);

CREATE TABLE IF NOT EXISTS doctor_acknowledgments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   INTEGER NOT NULL REFERENCES emergency_events(id) ON DELETE CASCADE,
    doctor_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    action     TEXT NOT NULL CHECK (action IN ('acknowledge','resolve','note')),
    note       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clinical_notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    doctor_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    note       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER,
    action        TEXT NOT NULL,
    resource_type TEXT,
    resource_id   TEXT,
    detail        TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_audit_user ON audit_log(user_id, created_at);
"""

# Schema v1 -> v2: nothing structural yet; kept to exercise the migration path.
_MIGRATIONS: dict[int, str] = {}


def init_db(db_path=None, target_version: int = SCHEMA_VERSION) -> sqlite3.Connection:
    """Open the database, create/upgrade the schema, return a connection."""
    conn = connect(db_path)
    conn.executescript(_SCHEMA)
    row = conn.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone()
    version = int(row["value"]) if row else 0
    for next_version in range(max(version, 0) + 1, target_version + 1):
        script = _MIGRATIONS.get(next_version)
        if script:
            conn.executescript(script)
        conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES('version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(next_version),),
        )
        conn.commit()
    if version == 0:
        audit(conn, None, "schema_init", detail=f"schema version {target_version}")
    return conn


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def audit(conn, user_id, action: str, resource_type: str = None,
          resource_id=None, detail: str = "") -> None:
    conn.execute(
        "INSERT INTO audit_log(user_id, action, resource_type, resource_id, detail, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (user_id, action, resource_type, None if resource_id is None else str(resource_id), detail, utcnow()),
    )
    conn.commit()


def list_audit(conn, limit: int = 200, action: Optional[str] = None) -> list[dict]:
    q = "SELECT * FROM audit_log"
    args: tuple = ()
    if action:
        q += " WHERE action = ?"
        args = (action,)
    q += " ORDER BY id DESC LIMIT ?"
    return [dict(r) for r in conn.execute(q, args + (limit,))]


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------
def create_user(conn, email: str, password_hash: str, role: str,
                full_name: str = "", username: Optional[str] = None,
                is_active: bool = True, broader_access: bool = False) -> int:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    now = utcnow()
    try:
        cur = conn.execute(
            "INSERT INTO users(email, username, password_hash, role, full_name, is_active,"
            " broader_access, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (email.strip(), (username or "").strip() or None, password_hash, role,
             full_name.strip(), int(is_active), int(broader_access), now, now),
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError("An account with that email/username already exists.") from exc
    conn.commit()
    audit(conn, cur.lastrowid, "user_created", "user", cur.lastrowid, f"role={role}")
    return int(cur.lastrowid)


def get_user(conn, user_id: int) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM users WHERE id=?", (int(user_id),)).fetchone()


def get_user_by_identifier(conn, identifier: str) -> Optional[sqlite3.Row]:
    ident = (identifier or "").strip()
    if not ident:
        return None
    return conn.execute(
        "SELECT * FROM users WHERE email=? OR username=?", (ident, ident)
    ).fetchone()


def set_password_hash(conn, user_id: int, password_hash: str) -> None:
    conn.execute(
        "UPDATE users SET password_hash=?, updated_at=?, failed_attempts=0, locked_until=NULL WHERE id=?",
        (password_hash, utcnow(), int(user_id)),
    )
    conn.commit()


def set_user_active(conn, actor_id, user_id: int, active: bool, reason: str = "") -> None:
    """Account activation / deactivation. Doctors only when initiated in-app."""
    if actor_id is not None:
        actor = get_user(conn, actor_id)
        if actor is None or actor["role"] != "doctor":
            audit(conn, actor_id, "access_denied", "user", user_id, "activation by non-doctor")
            raise DataAccessDenied("Only a doctor account may activate or deactivate users.")
    conn.execute(
        "UPDATE users SET is_active=?, updated_at=?, locked_until=NULL WHERE id=?",
        (int(active), utcnow(), int(user_id)),
    )
    conn.commit()
    audit(conn, actor_id, "account_activated" if active else "account_deactivated",
          "user", user_id, reason)


def set_broader_access(conn, actor_id, doctor_id: int, granted: bool) -> None:
    if actor_id is not None:
        actor = get_user(conn, actor_id)
        if actor is None or actor["role"] != "doctor":
            audit(conn, actor_id, "access_denied", "user", doctor_id, "grant by non-doctor")
            raise DataAccessDenied("Only a doctor account may grant broader access.")
    conn.execute("UPDATE users SET broader_access=?, updated_at=? WHERE id=?",
                 (int(granted), utcnow(), int(doctor_id)))
    conn.commit()
    audit(conn, actor_id, "broader_access_granted" if granted else "broader_access_revoked",
          "user", doctor_id, "explicit wider patient access grant")


def list_users(conn, role: Optional[str] = None) -> list[dict]:
    if role:
        rows = conn.execute(
            "SELECT id, email, username, full_name, role, is_active, broader_access,"
            " created_at, last_login_at FROM users WHERE role=? ORDER BY full_name, email",
            (role,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, email, username, full_name, role, is_active, broader_access,"
            " created_at, last_login_at FROM users ORDER BY role, full_name, email"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# access control
# ---------------------------------------------------------------------------
def can_access_patient(conn, viewer: Viewer, patient_id: int) -> bool:
    """True when *viewer* is allowed to read/write records of *patient_id*.

    * patients: only themselves;
    * doctors: assigned patients, plus any patient when ``broader_access``
      was explicitly granted to that doctor account.
    """
    patient_id = int(patient_id)
    if viewer is None:
        return False
    if viewer.is_patient:
        return viewer.id == patient_id
    if not viewer.is_doctor:
        return False
    if viewer.broader_access:
        return True
    row = conn.execute(
        "SELECT 1 FROM doctor_patient_assignments WHERE doctor_id=? AND patient_id=? "
        "AND is_active=1",
        (viewer.id, patient_id),
    ).fetchone()
    return row is not None


def require_patient_access(conn, viewer: Viewer, patient_id: int) -> int:
    """Access gate used by every patient-scoped function below."""
    patient_id = int(patient_id)
    if not can_access_patient(conn, viewer, patient_id):
        audit(conn, viewer.id if viewer else None, "access_denied", "patient", patient_id,
              f"role={viewer.role if viewer else 'anon'}")
        raise DataAccessDenied("You are not authorized to access this patient's records.")
    return patient_id


def is_patient_self(conn, viewer: Viewer, patient_id: int) -> bool:
    return viewer is not None and viewer.is_patient and viewer.id == int(patient_id)


def _require_doctor(conn, user_id) -> Viewer:
    """Resolve *user_id* to a Viewer, refusing anything but a doctor account."""
    row = get_user(conn, user_id) if user_id is not None else None
    if row is None or row["role"] != "doctor":
        audit(conn, user_id, "access_denied", None, user_id, "doctor account required")
        raise DataAccessDenied("Doctor account required for this operation.")
    return viewer_from_row(row)


# -- assignments -------------------------------------------------------------
def assign_patient(conn, doctor_id: int, patient_id: int, assigned_by) -> int:
    if int(doctor_id) == int(patient_id):
        raise ValueError("A doctor cannot be assigned as their own patient.")
    if assigned_by is not None:
        actor = get_user(conn, assigned_by)
        if actor is None or actor["role"] != "doctor":
            audit(conn, assigned_by, "access_denied", "assignment", patient_id,
                  "assignment attempted by non-doctor")
            raise DataAccessDenied("Only a doctor account may assign patients.")
    d = get_user(conn, doctor_id)
    p = get_user(conn, patient_id)
    if d is None or d["role"] != "doctor":
        raise ValueError("Target doctor account not found.")
    if p is None or p["role"] != "patient":
        raise ValueError("Target patient account not found.")
    cur = conn.execute(
        "INSERT INTO doctor_patient_assignments(doctor_id, patient_id, assigned_by, assigned_at)"
        " VALUES (?,?,?,?) ON CONFLICT(doctor_id, patient_id) DO UPDATE SET is_active=1",
        (int(doctor_id), int(patient_id), assigned_by, utcnow()),
    )
    conn.commit()
    audit(conn, assigned_by, "patient_assigned", "assignment", patient_id,
          f"doctor={doctor_id}")
    return int(cur.lastrowid or 0)


def unassign_patient(conn, actor_id, doctor_id: int, patient_id: int) -> None:
    conn.execute(
        "UPDATE doctor_patient_assignments SET is_active=0 WHERE doctor_id=? AND patient_id=?",
        (int(doctor_id), int(patient_id)),
    )
    conn.commit()
    audit(conn, actor_id, "patient_unassigned", "assignment", patient_id,
          f"doctor={doctor_id}")


def list_assigned_patients(conn, doctor_id: int, active_only: bool = True) -> list[dict]:
    q = (
        "SELECT u.id, u.email, u.full_name, u.is_active, a.assigned_at, a.is_active AS assigned "
        "FROM doctor_patient_assignments a JOIN users u ON u.id = a.patient_id "
        "WHERE a.doctor_id=? AND u.role='patient'"
    )
    if active_only:
        q += " AND a.is_active=1"
    q += " ORDER BY u.full_name, u.email"
    return [dict(r) for r in conn.execute(q, (int(doctor_id),))]


def list_doctors_for_patient(conn, patient_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT u.id, u.email, u.full_name, a.assigned_at FROM doctor_patient_assignments a "
        "JOIN users u ON u.id = a.doctor_id WHERE a.patient_id=? AND a.is_active=1 "
        "AND u.role='doctor' ORDER BY u.full_name",
        (int(patient_id),),
    ).fetchall()
    return [dict(r) for r in rows]


def accessible_patient_ids(conn, viewer: Viewer) -> list[int]:
    """All patient ids this viewer may read (patients: only themselves)."""
    if viewer.is_patient:
        return [viewer.id]
    if viewer.broader_access:
        rows = conn.execute("SELECT id FROM users WHERE role='patient' AND is_active=1")
    else:
        rows = conn.execute(
            "SELECT patient_id AS id FROM doctor_patient_assignments "
            "WHERE doctor_id=? AND is_active=1",
            (viewer.id,),
        )
    return [int(r["id"]) for r in rows]


# ---------------------------------------------------------------------------
# profiles & emergency contacts
# ---------------------------------------------------------------------------
PROFILE_FIELDS = ("date_of_birth", "sex", "phone", "location_text", "accessibility_notes")
CONTACT_FIELDS = ("contact_name", "relationship", "contact_number",
                  "emergency_service_number", "location_text", "accessibility_notes")


def get_profile(conn, viewer: Viewer, patient_id: int) -> dict:
    require_patient_access(conn, viewer, patient_id)
    row = conn.execute("SELECT * FROM patient_profiles WHERE patient_id=?",
                       (int(patient_id),)).fetchone()
    data = {f: None for f in PROFILE_FIELDS}
    if row:
        data.update({f: row[f] for f in PROFILE_FIELDS})
    return data


def update_profile(conn, viewer: Viewer, patient_id: int, values: dict) -> None:
    """Only the patient themselves (or a doctor with access) may edit."""
    require_patient_access(conn, viewer, patient_id)
    if viewer.is_patient and viewer.id != int(patient_id):
        raise DataAccessDenied("Patients may only edit their own profile.")
    clean = {k: (str(v).strip() if v is not None else None)
             for k, v in values.items() if k in PROFILE_FIELDS}
    conn.execute(
        "INSERT INTO patient_profiles(patient_id, date_of_birth, sex, phone, location_text,"
        " accessibility_notes, updated_at) VALUES (?,?,?,?,?,?,?) "
        "ON CONFLICT(patient_id) DO UPDATE SET date_of_birth=excluded.date_of_birth,"
        " sex=excluded.sex, phone=excluded.phone, location_text=excluded.location_text,"
        " accessibility_notes=excluded.accessibility_notes, updated_at=excluded.updated_at",
        (int(patient_id), clean.get("date_of_birth"), clean.get("sex"),
         clean.get("phone"), clean.get("location_text"),
         clean.get("accessibility_notes"), utcnow()),
    )
    conn.commit()
    audit(conn, viewer.id, "profile_updated", "patient", patient_id, "")


def get_emergency_contact(conn, viewer: Viewer, patient_id: int) -> dict:
    require_patient_access(conn, viewer, patient_id)
    row = conn.execute("SELECT * FROM emergency_contacts WHERE patient_id=?",
                       (int(patient_id),)).fetchone()
    data = {f: "" for f in CONTACT_FIELDS}
    if row:
        data.update({f: row[f] or "" for f in CONTACT_FIELDS})
    if not data["emergency_service_number"]:
        data["emergency_service_number"] = "1122"
    return data


def update_emergency_contact(conn, viewer: Viewer, patient_id: int, values: dict) -> None:
    """The patient may edit their own emergency contact details."""
    require_patient_access(conn, viewer, patient_id)
    if viewer.is_patient and viewer.id != int(patient_id):
        raise DataAccessDenied("Patients may only edit their own emergency contact.")
    clean = {k: str(v).strip() if v is not None else ""
             for k, v in values.items() if k in CONTACT_FIELDS}
    if "emergency_service_number" in values and not clean.get("emergency_service_number"):
        clean["emergency_service_number"] = "1122"
    conn.execute(
        "INSERT INTO emergency_contacts(patient_id, contact_name, relationship, contact_number,"
        " emergency_service_number, location_text, accessibility_notes, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(patient_id) DO UPDATE SET"
        " contact_name=excluded.contact_name, relationship=excluded.relationship,"
        " contact_number=excluded.contact_number,"
        " emergency_service_number=excluded.emergency_service_number,"
        " location_text=excluded.location_text,"
        " accessibility_notes=excluded.accessibility_notes, updated_at=excluded.updated_at",
        (int(patient_id), clean.get("contact_name", ""), clean.get("relationship", ""),
         clean.get("contact_number", ""), clean.get("emergency_service_number", "1122"),
         clean.get("location_text", ""), clean.get("accessibility_notes", ""), utcnow()),
    )
    conn.commit()
    audit(conn, viewer.id, "emergency_contact_updated", "patient", patient_id, "")


# ---------------------------------------------------------------------------
# ECG sessions
# ---------------------------------------------------------------------------
def create_ecg_session(conn, viewer: Viewer, patient_id: int, *, fs_hz=None,
                       device_label: str = "", device_status: str = "connected",
                       source: str = "demo", status: str = "running") -> int:
    require_patient_access(conn, viewer, patient_id)
    now = utcnow()
    cur = conn.execute(
        "INSERT INTO ecg_sessions(patient_id, created_by, started_at, status, device_label,"
        " device_status, source, fs_hz, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (int(patient_id), viewer.id, now, status, device_label, device_status, source,
         fs_hz, now, now),
    )
    conn.commit()
    audit(conn, viewer.id, "ecg_session_started", "ecg_session", cur.lastrowid,
          f"patient={patient_id}")
    return int(cur.lastrowid)


def update_ecg_session(conn, viewer: Viewer, session_id: int, *, status=None,
                       device_status=None, signal=None, n_samples=None,
                       mean_hr_bpm=None, analysis=None, ended: bool = False) -> None:
    row = get_ecg_session(conn, viewer, session_id)  # enforces access
    fields, vals = [], []
    if status is not None:
        fields.append("status=?"); vals.append(status)
    if device_status is not None:
        fields.append("device_status=?"); vals.append(device_status)
    if signal is not None:
        fields.append("signal=?"); vals.append(encode_signal(signal))
        fields.append("n_samples=?"); vals.append(len(signal))
    if n_samples is not None:
        fields.append("n_samples=?"); vals.append(int(n_samples))
    if mean_hr_bpm is not None:
        fields.append("mean_hr_bpm=?"); vals.append(mean_hr_bpm)
    if analysis is not None:
        import json
        fields.append("analysis_json=?"); vals.append(json.dumps(analysis, default=float))
    if ended:
        fields.append("ended_at=?"); vals.append(utcnow())
    if not fields:
        return
    fields.append("updated_at=?"); vals.append(utcnow())
    conn.execute(f"UPDATE ecg_sessions SET {', '.join(fields)} WHERE id=?", vals + [int(session_id)])
    conn.commit()
    _ = row


def get_ecg_session(conn, viewer: Viewer, session_id: int) -> dict:
    row = conn.execute("SELECT * FROM ecg_sessions WHERE id=?", (int(session_id),)).fetchone()
    if row is None:
        raise KeyError(f"ECG session {session_id} not found")
    require_patient_access(conn, viewer, row["patient_id"])
    data = dict(row)
    data["signal_array"] = decode_signal(data.get("signal"))
    return data


def list_ecg_sessions(conn, viewer: Viewer, patient_id: int, limit: int = 100) -> list[dict]:
    require_patient_access(conn, viewer, patient_id)
    rows = conn.execute(
        "SELECT id, patient_id, created_by, started_at, ended_at, status, device_label,"
        " device_status, source, fs_hz, n_samples, mean_hr_bpm, created_at, updated_at"
        " FROM ecg_sessions WHERE patient_id=? ORDER BY id DESC LIMIT ?",
        (int(patient_id), limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# CT / radiology records
# ---------------------------------------------------------------------------
def create_ct_record(conn, viewer: Viewer, patient_id: int, *, cac_score=None,
                     cac_category=None, stenosis_grade=None, vessels_ge50=None,
                     context_level=None, report_text: str = "", source: str = "report") -> int:
    require_patient_access(conn, viewer, patient_id)
    cur = conn.execute(
        "INSERT INTO ct_records(patient_id, recorded_by, source, cac_score, cac_category,"
        " stenosis_grade, vessels_ge50, context_level, report_text, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (int(patient_id), viewer.id, source, cac_score, cac_category, stenosis_grade,
         vessels_ge50, context_level, report_text, utcnow()),
    )
    conn.commit()
    audit(conn, viewer.id, "ct_record_created", "ct_record", cur.lastrowid,
          f"patient={patient_id}")
    return int(cur.lastrowid)


def list_ct_records(conn, viewer: Viewer, patient_id: int, limit: int = 100) -> list[dict]:
    require_patient_access(conn, viewer, patient_id)
    rows = conn.execute(
        "SELECT * FROM ct_records WHERE patient_id=? ORDER BY id DESC LIMIT ?",
        (int(patient_id), limit),
    ).fetchall()
    return [dict(r) for r in rows]


def latest_ct_record(conn, viewer: Viewer, patient_id: int) -> Optional[dict]:
    recs = list_ct_records(conn, viewer, patient_id, limit=1)
    return recs[0] if recs else None


# ---------------------------------------------------------------------------
# AI assessments
# ---------------------------------------------------------------------------
def create_assessment(conn, viewer: Viewer, patient_id: int, *, status: str,
                      ecg_session_id=None, ct_record_id=None, model_name=None,
                      fused_score=None, fused_level=None, ecg_score_mean=None,
                      ct_level=None, heart_rate_bpm=None, config=None,
                      windows=None, error_message=None) -> int:
    require_patient_access(conn, viewer, patient_id)
    if status not in ("completed", "failed", "unavailable"):
        raise ValueError("bad assessment status")
    import json
    cur = conn.execute(
        "INSERT INTO ai_assessments(patient_id, ecg_session_id, ct_record_id, status,"
        " model_name, fused_score, fused_level, ecg_score_mean, ct_level, heart_rate_bpm,"
        " config_json, windows_json, error_message, created_by, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (int(patient_id), ecg_session_id, ct_record_id, status, model_name, fused_score,
         fused_level, ecg_score_mean, ct_level, heart_rate_bpm,
         json.dumps(config, default=float) if config is not None else None,
         json.dumps(windows, default=float) if windows is not None else None,
         error_message, viewer.id, utcnow()),
    )
    conn.commit()
    audit(conn, viewer.id, "assessment_created", "assessment", cur.lastrowid,
          f"patient={patient_id} status={status}")
    return int(cur.lastrowid)


def get_assessment(conn, viewer: Viewer, assessment_id: int) -> dict:
    row = conn.execute("SELECT * FROM ai_assessments WHERE id=?",
                       (int(assessment_id),)).fetchone()
    if row is None:
        raise KeyError(f"Assessment {assessment_id} not found")
    require_patient_access(conn, viewer, row["patient_id"])
    return dict(row)


def list_assessments(conn, viewer: Viewer, patient_id: int, limit: int = 100) -> list[dict]:
    require_patient_access(conn, viewer, patient_id)
    rows = conn.execute(
        "SELECT * FROM ai_assessments WHERE patient_id=? ORDER BY id DESC LIMIT ?",
        (int(patient_id), limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# high-risk events
# ---------------------------------------------------------------------------
def active_event_for_patient(conn, patient_id: int) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM emergency_events WHERE patient_id=? AND status IN ('open','acknowledged')"
        " ORDER BY id DESC LIMIT 1",
        (int(patient_id),),
    ).fetchone()
    return dict(row) if row else None


def create_or_refresh_event(conn, patient_id: int, *, score: float, threshold: float,
                            assessment_id=None, heart_rate_bpm=None) -> tuple[dict, bool]:
    """Create a unique high-risk event, or refresh the patient's active event.

    Returns ``(event, created)``.  Duplicate prevention:
    * one active event per patient is enforced by a partial UNIQUE index;
    * the app-level check reuses the active event instead of inserting.
    """
    existing = active_event_for_patient(conn, patient_id)
    now = utcnow()
    if existing:
        conn.execute(
            "UPDATE emergency_events SET score=?, threshold=?, assessment_id=?,"
            " heart_rate_bpm=?, updated_at=? WHERE id=?",
            (float(score), float(threshold), assessment_id, heart_rate_bpm, now, existing["id"]),
        )
        conn.commit()
        audit(conn, None, "event_refreshed", "emergency_event", existing["id"],
              f"patient={patient_id} score={score:.4f}")
        row = conn.execute("SELECT * FROM emergency_events WHERE id=?", (existing["id"],)).fetchone()
        return dict(row), False

    event_uid = uuid.uuid4().hex
    try:
        cur = conn.execute(
            "INSERT INTO emergency_events(event_uid, patient_id, assessment_id, score, threshold,"
            " heart_rate_bpm, status, created_at, updated_at) VALUES (?,?,?,?,?,?, 'open', ?, ?)",
            (event_uid, int(patient_id), assessment_id, float(score), float(threshold),
             heart_rate_bpm, now, now),
        )
    except sqlite3.IntegrityError:
        # lost a race with another session: reuse whatever won
        winner = active_event_for_patient(conn, patient_id)
        conn.commit()
        return winner, False
    conn.commit()
    audit(conn, None, "high_risk_event_created", "emergency_event", cur.lastrowid,
          f"patient={patient_id} score={score:.4f} threshold={threshold}")
    row = conn.execute("SELECT * FROM emergency_events WHERE id=?", (cur.lastrowid,)).fetchone()
    return dict(row), True


def get_event(conn, viewer: Viewer, event_id: int) -> dict:
    row = conn.execute("SELECT * FROM emergency_events WHERE id=?", (int(event_id),)).fetchone()
    if row is None:
        raise KeyError(f"Event {event_id} not found")
    require_patient_access(conn, viewer, row["patient_id"])
    return dict(row)


def list_events_for_patient(conn, viewer: Viewer, patient_id: int,
                            include_resolved: bool = True) -> list[dict]:
    require_patient_access(conn, viewer, patient_id)
    q = "SELECT * FROM emergency_events WHERE patient_id=?"
    if not include_resolved:
        q += " AND status IN ('open','acknowledged')"
    q += " ORDER BY id DESC"
    return [dict(r) for r in conn.execute(q, (int(patient_id),))]


def list_events_for_doctor(conn, doctor_id: int, include_resolved: bool = True) -> list[dict]:
    """Events of assigned patients only (plus broader-access doctors)."""
    doc = get_user(conn, doctor_id)
    if doc is None or doc["role"] != "doctor":
        raise DataAccessDenied("Not a doctor account.")
    viewer = viewer_from_row(doc)
    ids = accessible_patient_ids(conn, viewer)
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    q = (f"SELECT * FROM emergency_events WHERE patient_id IN ({marks})")
    args: list = list(ids)
    if not include_resolved:
        q += " AND status IN ('open','acknowledged')"
    q += " ORDER BY CASE status WHEN 'open' THEN 0 WHEN 'acknowledged' THEN 1 ELSE 2 END, id DESC"
    return [dict(r) for r in conn.execute(q, args)]


def set_patient_call(conn, patient_id: int, event_id: int) -> dict:
    now = utcnow()
    conn.execute(
        "UPDATE emergency_events SET patient_call_initiated=1, patient_call_at=?, updated_at=?"
        " WHERE id=? AND patient_id=?",
        (now, now, int(event_id), int(patient_id)),
    )
    conn.commit()
    audit(conn, patient_id, "ambulance_call_action", "emergency_event", event_id,
          "patient pressed Call Ambulance (tel: link opened; connection not verified)")
    row = conn.execute("SELECT * FROM emergency_events WHERE id=?", (int(event_id),)).fetchone()
    return dict(row)


def set_patient_dismissed(conn, patient_id: int, event_id: int) -> None:
    now = utcnow()
    conn.execute(
        "UPDATE emergency_events SET patient_dismissed_at=?, updated_at=?"
        " WHERE id=? AND patient_id=? AND status IN ('open','acknowledged')",
        (now, now, int(event_id), int(patient_id)),
    )
    conn.commit()
    audit(conn, patient_id, "warning_dismissed", "emergency_event", event_id, "")


def set_event_status(conn, doctor_id: int, event_id: int, status: str,
                     note: str = "") -> dict:
    if status not in ("open", "acknowledged", "resolved"):
        raise ValueError("bad event status")
    viewer = _require_doctor(conn, doctor_id)
    row = conn.execute("SELECT * FROM emergency_events WHERE id=?", (int(event_id),)).fetchone()
    if row is None:
        raise KeyError(f"Event {event_id} not found")
    require_patient_access(conn, viewer, row["patient_id"])
    now = utcnow()
    conn.execute("UPDATE emergency_events SET status=?, updated_at=? WHERE id=?",
                 (status, now, int(event_id)))
    action = {"open": "note", "acknowledged": "acknowledge", "resolved": "resolve"}[status]
    conn.execute(
        "INSERT INTO doctor_acknowledgments(event_id, doctor_id, action, note, created_at)"
        " VALUES (?,?,?,?,?)",
        (int(event_id), int(doctor_id), action, note or "", now),
    )
    conn.commit()
    audit(conn, doctor_id, f"event_{action}d", "emergency_event", event_id,
          note[:200])
    row = conn.execute("SELECT * FROM emergency_events WHERE id=?", (int(event_id),)).fetchone()
    return dict(row)


def add_event_note(conn, doctor_id: int, event_id: int, note: str) -> None:
    set_event_status_doctor_note(conn, doctor_id, event_id, note)


def set_event_status_doctor_note(conn, doctor_id: int, event_id: int, note: str) -> None:
    viewer = _require_doctor(conn, doctor_id)
    row = conn.execute("SELECT * FROM emergency_events WHERE id=?", (int(event_id),)).fetchone()
    if row is None:
        raise KeyError(f"Event {event_id} not found")
    require_patient_access(conn, viewer, row["patient_id"])
    conn.execute(
        "INSERT INTO doctor_acknowledgments(event_id, doctor_id, action, note, created_at)"
        " VALUES (?,?,?,?,?)",
        (int(event_id), int(doctor_id), "note", note, utcnow()),
    )
    conn.commit()
    audit(conn, doctor_id, "event_note_added", "emergency_event", event_id, note[:200])


def list_event_acknowledgments(conn, viewer: Viewer, event_id: int) -> list[dict]:
    event = get_event(conn, viewer, event_id)  # enforces access
    _ = event
    rows = conn.execute(
        "SELECT ack.*, u.full_name AS doctor_name, u.email AS doctor_email"
        " FROM doctor_acknowledgments ack"
        " JOIN users u ON u.id = ack.doctor_id WHERE ack.event_id=? ORDER BY ack.id",
        (int(event_id),),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# notifications
# ---------------------------------------------------------------------------
def add_notification(conn, event_id: int, channel: str, status: str,
                     detail: str = "", confirmed: bool = False) -> int:
    now = utcnow()
    cur = conn.execute(
        "INSERT INTO notification_attempts(event_id, channel, status, detail, created_at,"
        " confirmed_at) VALUES (?,?,?,?,?,?)",
        (int(event_id), channel, status, detail, now, now if confirmed else None),
    )
    conn.commit()
    return int(cur.lastrowid)


def update_notification_status(conn, notification_id: int, status: str, detail: str = "",
                               confirmed: bool = False) -> None:
    conn.execute(
        "UPDATE notification_attempts SET status=?, detail=?, confirmed_at=? WHERE id=?",
        (status, detail, utcnow() if confirmed else None, int(notification_id)),
    )
    conn.commit()


def list_notifications(conn, viewer: Viewer, event_id: int) -> list[dict]:
    get_event(conn, viewer, event_id)  # enforces access
    rows = conn.execute(
        "SELECT * FROM notification_attempts WHERE event_id=? ORDER BY id",
        (int(event_id),),
    ).fetchall()
    return [dict(r) for r in rows]


def confirm_local_delivery(conn, event_id: int) -> int:
    """Called when the assigned doctor actually opens the alert (offline-proof)."""
    rows = conn.execute(
        "SELECT id FROM notification_attempts WHERE event_id=? AND channel='local_dashboard'"
        " AND status IN ('queued','sent')",
        (int(event_id),),
    ).fetchall()
    for r in rows:
        update_notification_status(conn, r["id"], "delivery_confirmed",
                                   "doctor opened the alert in the local dashboard",
                                   confirmed=True)
    return len(rows)


# ---------------------------------------------------------------------------
# clinical notes
# ---------------------------------------------------------------------------
def add_clinical_note(conn, doctor_id: int, patient_id: int, note: str) -> int:
    viewer = _require_doctor(conn, doctor_id)
    require_patient_access(conn, viewer, patient_id)
    if not (note or "").strip():
        raise ValueError("Note must not be empty")
    cur = conn.execute(
        "INSERT INTO clinical_notes(patient_id, doctor_id, note, created_at) VALUES (?,?,?,?)",
        (int(patient_id), int(doctor_id), note.strip(), utcnow()),
    )
    conn.commit()
    audit(conn, doctor_id, "clinical_note_added", "patient", patient_id, note[:200])
    return int(cur.lastrowid)


def list_clinical_notes(conn, viewer: Viewer, patient_id: int, limit: int = 100) -> list[dict]:
    require_patient_access(conn, viewer, patient_id)
    rows = conn.execute(
        "SELECT n.*, u.full_name AS doctor_name, u.email AS doctor_email"
        " FROM clinical_notes n JOIN users u ON u.id = n.doctor_id"
        " WHERE n.patient_id=? ORDER BY n.id DESC LIMIT ?",
        (int(patient_id), limit),
    ).fetchall()
    return [dict(r) for r in rows]
