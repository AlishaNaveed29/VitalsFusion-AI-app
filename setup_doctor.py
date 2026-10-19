"""Secure initial doctor-account setup (terminal only, never in the web app).

The application has NO public doctor self-registration.  The first doctor
account - and any later one - is created from this local script, which:

* prompts for the password with echo disabled (or takes ``--password`` on a
  local trusted shell), or generates a strong one-time password;
* enforces the password policy;
* stores only a PBKDF2-SHA256 hash.

Examples
--------
    python setup_doctor.py                          # interactive prompt
    python setup_doctor.py --email dr@clinic.pk --name "Dr Ayesha"
    python setup_doctor.py --generate-password      # print a random password
    python setup_doctor.py --grant-broad-access     # explicit wider access
    python setup_doctor.py --reset-password dr@clinic.pk
    python setup_doctor.py --deactivate dr@clinic.pk
    python setup_doctor.py --activate dr@clinic.pk
    python setup_doctor.py --create-patient demo@example.org --name "Test Patient"
"""

from __future__ import annotations

import argparse
import getpass
import sys

import auth
import db as dbm
from config import load_config


def _new_password(args, cfg) -> str:
    if args.password:
        pw = args.password
    elif args.generate_password:
        pw = auth.generate_password()
        print(f"Generated password: {pw}")
    else:
        pw = getpass.getpass("Password for the doctor account: ")
        pw2 = getpass.getpass("Repeat password: ")
        if pw != pw2:
            sys.exit("Passwords do not match.")
    err = auth.check_password_policy(pw, cfg.min_password_length)
    if err:
        sys.exit(f"Password rejected: {err}")
    return pw


def main(argv=None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Manage VitalsFusion AI accounts (local terminal).")
    p.add_argument("--email", help="doctor email / login")
    p.add_argument("--username", help="optional alternative login name")
    p.add_argument("--name", default="", help="full name")
    p.add_argument("--password", help="password (prefer the interactive prompt)")
    p.add_argument("--generate-password", action="store_true",
                   help="generate a strong one-time password")
    p.add_argument("--grant-broad-access", action="store_true",
                   help="explicitly grant this doctor access beyond assigned patients")
    p.add_argument("--reset-password", metavar="IDENTIFIER",
                   help="set a new password for an existing account")
    p.add_argument("--deactivate", metavar="IDENTIFIER", help="deactivate an account")
    p.add_argument("--activate", metavar="IDENTIFIER", help="activate an account")
    p.add_argument("--create-patient", metavar="EMAIL",
                   help="create a patient account (optional; patients can also self-register)")
    p.add_argument("--assign", nargs=2, metavar=("DOCTOR", "PATIENT"),
                   help="assign a patient to a doctor (identifiers)")
    args = p.parse_args(argv)

    conn = dbm.init_db(cfg.db_path)

    # ---- password reset / activation -------------------------------------
    if args.reset_password:
        row = dbm.get_user_by_identifier(conn, args.reset_password)
        if row is None:
            sys.exit("No such account.")
        pw = auth.generate_password() if args.generate_password else _new_password(args, cfg)
        dbm.set_password_hash(conn, row["id"], auth.hash_password(pw))
        if args.generate_password:
            print(f"New password for {row['email']}: {pw}")
        else:
            print(f"Password updated for {row['email']}.")
        conn.close()
        return 0

    if args.deactivate or args.activate:
        ident = args.deactivate or args.activate
        row = dbm.get_user_by_identifier(conn, ident)
        if row is None:
            sys.exit("No such account.")
        dbm.set_user_active(conn, None, row["id"], args.activate is not None,
                            "terminal setup script")
        print(f"{'Activated' if args.activate else 'Deactivated'} {row['email']}.")
        conn.close()
        return 0

    # ---- optional patient creation by the operator ------------------------
    if args.create_patient:
        pw = _new_password(args, cfg)
        pid = dbm.create_user(conn, args.create_patient, auth.hash_password(pw),
                              "patient", full_name=args.name)
        print(f"Created patient account id={pid} ({args.create_patient}).")
        conn.close()
        return 0

    # ---- assignment --------------------------------------------------------
    if args.assign:
        drow = dbm.get_user_by_identifier(conn, args.assign[0])
        prow = dbm.get_user_by_identifier(conn, args.assign[1])
        if drow is None or drow["role"] != "doctor":
            sys.exit("Doctor identifier not found.")
        if prow is None or prow["role"] != "patient":
            sys.exit("Patient identifier not found.")
        dbm.assign_patient(conn, drow["id"], prow["id"], drow["id"])
        print(f"Assigned {prow['email']} to {drow['email']}.")
        conn.close()
        return 0

    # ---- create doctor -----------------------------------------------------
    if not args.email:
        p.print_help()
        print("\nNo action given. Create the first doctor with:  python setup_doctor.py --email you@clinic.pk")
        conn.close()
        return 1

    if dbm.get_user_by_identifier(conn, args.email):
        sys.exit("An account with that email/username already exists.")

    pw = _new_password(args, cfg)
    uid = dbm.create_user(conn, args.email, auth.hash_password(pw), "doctor",
                          full_name=args.name, username=args.username,
                          broader_access=args.grant_broad_access)
    print(f"Created DOCTOR account id={uid} ({args.email}).")
    if args.grant_broad_access:
        print("NOTE: this doctor has explicitly granted access beyond assigned patients.")
    print("Sign in at the app's login page.")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
