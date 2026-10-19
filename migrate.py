"""Create or upgrade the VitalsFusion AI database.

Run from this folder:

    python migrate.py

Idempotent - safe to run on every upgrade. Prints the schema version and the
database path when finished.
"""

from __future__ import annotations

import db as dbm
from config import load_config


def main() -> None:
    cfg = load_config()
    conn = dbm.init_db(cfg.db_path)
    row = conn.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone()
    n_users = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
    print(f"Database      : {dbm._as_path(cfg.db_path)}")
    print(f"Schema version: {row['value'] if row else '?'} (target {dbm.SCHEMA_VERSION})")
    print(f"Accounts      : {n_users}")
    if n_users == 0:
        print("\nNo accounts yet. Create the first DOCTOR account with:")
        print("    python setup_doctor.py")
        print("Patients register through the app's patient sign-up form, or are")
        print("added by an existing doctor.")
    conn.close()


if __name__ == "__main__":
    main()
