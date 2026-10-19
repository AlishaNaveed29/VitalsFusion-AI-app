"""Authentication for VitalsFusion AI.

* Password hashing: PBKDF2-HMAC-SHA256, per-user random salt, constant-time
  comparison (stdlib only - no external services, works fully offline).
* Persistent accounts live in the SQLite database (see ``db.py``).
* Roles are exactly ``doctor`` and ``patient``.  There is no doctor
  self-registration path in the application; the first doctor account is
  created by ``setup_doctor.py`` from a terminal prompt.
* Session state (Streamlit) carries an expiry timestamp checked on every rerun.
"""

from __future__ import annotations

import hmac
import hashlib
import re
import secrets
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import db as dbm

ALGORITHM = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 240_000


# ---------------------------------------------------------------------------
# password hashing
# ---------------------------------------------------------------------------
def hash_password(password: str, *, iterations: int = DEFAULT_ITERATIONS,
                  salt: Optional[bytes] = None) -> str:
    if not isinstance(password, str) or not password:
        raise ValueError("Password must be a non-empty string")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{ALGORITHM}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt_hex, hash_hex = (stored or "").split("$")
        if algo != ALGORITHM:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", (password or "").encode("utf-8"),
            bytes.fromhex(salt_hex), int(iterations),
        )
        return hmac.compare_digest(digest.hex(), hash_hex)
    except (ValueError, TypeError, AttributeError):
        return False


def generate_password(length: int = 16) -> str:
    """Random password for the one-time initial doctor setup (never stored)."""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%&*"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def check_password_policy(password: str, min_length: int = 8) -> Optional[str]:
    """Return an error message when the password is unacceptable, else None."""
    if password is None or len(password) < min_length:
        return f"Password must be at least {min_length} characters long."
    if password.strip() != password or not password.strip():
        return "Password must not start or end with whitespace."
    if len(set(password)) < 4:
        return "Password is too repetitive."
    if re.fullmatch(r"(.)\1*", password or ""):
        return "Password is too weak."
    return None


# ---------------------------------------------------------------------------
# login
# ---------------------------------------------------------------------------
@dataclass
class LoginResult:
    ok: bool
    user: Optional[dict] = None
    reason: str = ""


def _register_failure(conn, row, cfg) -> str:
    attempts = int(row["failed_attempts"] or 0) + 1
    locked_until = None
    reason = "Invalid email/username or password."
    if attempts >= cfg.max_failed_logins:
        locked_until = (datetime.now(timezone.utc) +
                        timedelta(minutes=cfg.lockout_minutes)).isoformat(timespec="seconds")
        reason = "Account temporarily locked after too many failed attempts."
    conn.execute(
        "UPDATE users SET failed_attempts=?, locked_until=? WHERE id=?",
        (attempts, locked_until, row["id"]),
    )
    conn.commit()
    dbm.audit(conn, row["id"], "login_failed", "user", row["id"],
              f"attempts={attempts} locked={locked_until is not None}")
    return reason


def _lock_active(row) -> bool:
    lu = row["locked_until"]
    if not lu:
        return False
    try:
        return datetime.fromisoformat(lu) > datetime.now(timezone.utc)
    except ValueError:
        return False


def authenticate(conn, identifier: str, password: str, cfg) -> LoginResult:
    """Verify credentials.  Unknown user and wrong password give the same
    message so account existence cannot be enumerated."""
    identifier = (identifier or "").strip()
    if not identifier or not password:
        return LoginResult(False, None, "Enter your email/username and password.")

    row = dbm.get_user_by_identifier(conn, identifier)
    if row is None:
        # run a dummy verification so timing is not obviously different
        verify_password(password, hash_password("dummy-timing-equaliser", iterations=10_000))
        dbm.audit(conn, None, "login_failed", None, None, "unknown identifier")
        return LoginResult(False, None, "Invalid email/username or password.")

    if _lock_active(row):
        dbm.audit(conn, row["id"], "login_blocked_locked", "user", row["id"], "")
        return LoginResult(False, None, "Account temporarily locked. Try again later.")

    if not row["is_active"]:
        dbm.audit(conn, row["id"], "login_blocked_inactive", "user", row["id"], "")
        return LoginResult(False, None, "This account has been deactivated. "
                                        "Contact the clinic administrator (doctor account).")

    if not verify_password(password, row["password_hash"]):
        reason = _register_failure(conn, row, cfg)
        return LoginResult(False, None, reason)

    if check_password_policy(password, cfg.min_password_length) and row["role"] == "doctor":
        # Extremely old/weak hashes still work; policy only gates new passwords.
        pass

    now = dbm.utcnow()
    conn.execute(
        "UPDATE users SET failed_attempts=0, locked_until=NULL, last_login_at=? WHERE id=?",
        (now, row["id"]),
    )
    conn.commit()
    dbm.audit(conn, row["id"], "login_success", "user", row["id"], "")
    fresh = dbm.get_user(conn, row["id"])
    return LoginResult(True, dict(fresh), "")


# ---------------------------------------------------------------------------
# session (Streamlit)
# ---------------------------------------------------------------------------
SESSION_KEY = "vf_auth_session"


@dataclass
class AuthSession:
    user_id: int
    role: str
    email: str
    full_name: str
    session_uid: str
    login_at: str
    absolute_expiry: str       # ISO timestamp
    idle_expiry: str           # ISO timestamp, refreshed on activity

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AuthSession":
        return cls(**data)

    @property
    def expired(self) -> bool:
        return datetime.now(timezone.utc) >= _parse(self.absolute_expiry)

    @property
    def idle_expired(self) -> bool:
        return datetime.now(timezone.utc) >= _parse(self.idle_expiry)


def _parse(iso: str) -> datetime:
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def new_session(user: dict, cfg) -> AuthSession:
    now = datetime.now(timezone.utc)
    return AuthSession(
        user_id=int(user["id"]),
        role=user["role"],
        email=user["email"],
        full_name=user.get("full_name") or "",
        session_uid=secrets.token_urlsafe(12),
        login_at=now.isoformat(timespec="seconds"),
        absolute_expiry=(now + timedelta(minutes=cfg.session_absolute_ttl_minutes))
                        .isoformat(timespec="seconds"),
        idle_expiry=(now + timedelta(minutes=cfg.session_idle_ttl_minutes))
                    .isoformat(timespec="seconds"),
    )


def session_state(session: AuthSession, cfg) -> AuthSession:
    """Slide the idle window forward; returns the session unchanged if expired."""
    if session.expired or session.idle_expired:
        return session
    now = datetime.now(timezone.utc)
    session.idle_expiry = min(
        _parse(session.absolute_expiry),
        now + timedelta(minutes=cfg.session_idle_ttl_minutes),
    ).isoformat(timespec="seconds")
    return session


def session_expired_reason(session: Optional[AuthSession]) -> Optional[str]:
    if session is None:
        return None
    if session.expired:
        return "Your session has expired. Please sign in again."
    if session.idle_expired:
        return "You were signed out after a period of inactivity. Please sign in again."
    return None


def make_viewer(conn, session: AuthSession) -> dbm.Viewer:
    """Build a Viewer from the *database record* (never from client input)."""
    row = dbm.get_user(conn, session.user_id)
    if row is None:
        raise dbm.DataAccessDenied("Account no longer exists.")
    if not row["is_active"]:
        raise dbm.DataAccessDenied("Account has been deactivated.")
    if row["role"] != session.role:
        raise dbm.DataAccessDenied("Session role no longer matches the account.")
    return dbm.viewer_from_row(row)


def logout(conn, session: Optional[AuthSession]) -> None:
    if session is not None:
        dbm.audit(conn, session.user_id, "logout", "user", session.user_id, "")
