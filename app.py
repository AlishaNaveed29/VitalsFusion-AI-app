"""VitalsFusion AI - two-role (Doctor / Patient) Streamlit application
======================================================================
Offline-first ECG + CT fusion prototype with authentication, role dashboards
and a patient-controlled ambulance-call emergency workflow.

Run:
    pip install -r requirements.txt
    python migrate.py                      # create/upgrade the database
    streamlit run app.py --server.maxUploadSize 1000
    # register as Doctor or Patient on the login page, or create a doctor
    # from the terminal: python setup_doctor.py --email ...

Put ecg_lead2_mi_model_v2_keras.zip next to this file (or upload it) to enable
AI assessments; without it the app runs and reports assessments as UNAVAILABLE
instead of inventing scores.
"""

from __future__ import annotations

import streamlit as st

import auth
import db as dbm
import doctor_dashboard
import patient_dashboard
import ui_common as ui
from config import load_config

st.set_page_config(page_title="VitalsFusion AI", layout="wide", page_icon="\U0001F49A")


# ---------------------------------------------------------------------------
# login / registration
# ---------------------------------------------------------------------------
def _login_page(conn, cfg) -> None:
    left, mid, right = st.columns([1.0, 2.6, 1.0], gap="medium")

    with mid:
        st.title("\U0001F49A VitalsFusion AI", anchor=False)
        st.markdown(
            "<p class='vf-login-tag'>Secure ECG + CT cardiac risk fusion "
            "&middot; offline-first &middot; doctor &amp; patient portal</p>",
            unsafe_allow_html=True)
        st.markdown(
            f"<div class='vf-disclaimer'>"
            f"<b>Research prototype &mdash; not a medical device.</b> "
            f"In an emergency, call <b>{cfg.emergency_service_number}</b>. "
            f"A low AI score never rules out a heart attack."
            f"</div>",
            unsafe_allow_html=True)

        n_doctors = conn.execute(
            "SELECT COUNT(*) c FROM users WHERE role='doctor'").fetchone()["c"]
        if n_doctors == 0:
            st.info("No doctor account exists yet - you can register as a "
                    "Doctor below.")

        tab_in, tab_up = st.tabs(["Sign in", "Create account"])

        with tab_in:
            with st.form("vf_login_form", clear_on_submit=False):
                ident = st.text_input("Email or username",
                                      placeholder="you@clinic.pk")
                pw = st.text_input("Password", type="password",
                                   placeholder="Your password")
                submitted = st.form_submit_button("Sign in", type="primary")
                if submitted:
                    result = auth.authenticate(conn, ident, pw, cfg)
                    if result.ok:
                        session = auth.new_session(result.user, cfg)
                        st.session_state[auth.SESSION_KEY] = session.to_dict()
                        st.rerun()
                    else:
                        st.error(result.reason)

        with tab_up:
            st.caption("A doctor added this way only sees patients they "
                       "assign to themselves.")
            with st.form("vf_signup_form"):
                role = st.radio("I am registering as", ["Patient", "Doctor"],
                                horizontal=True, key="vf_signup_role")
                email = st.text_input("Email / login", placeholder="you@clinic.pk")
                name = st.text_input("Full name", placeholder="Your full name")
                pw1 = st.text_input("Password (min 8 characters)", type="password",
                                    placeholder="Choose a password")
                pw2 = st.text_input("Repeat password", type="password",
                                    placeholder="Repeat the password")
                ok = st.form_submit_button("Create account", type="primary")
                if ok:
                    err = auth.check_password_policy(pw1, cfg.min_password_length)
                    role_val = "doctor" if role == "Doctor" else "patient"
                    if err:
                        st.error(err)
                    elif pw1 != pw2:
                        st.error("Passwords do not match.")
                    elif not email.strip():
                        st.error("Email is required.")
                    elif dbm.get_user_by_identifier(conn, email):
                        st.error("An account with that email/username already exists.")
                    else:
                        dbm.create_user(conn, email.strip(), auth.hash_password(pw1),
                                        role_val, full_name=name.strip())
                        st.success(f"{role} account created. Sign in with your "
                                   "email/username and password.")
                        dbm.audit(conn, None, f"{role_val}_self_registered", "user",
                                  email, "web registration form")

    with st.sidebar:
        st.markdown("### \U0001F49A VitalsFusion AI")
        st.caption("ECG + CT cardiac risk fusion for doctors and patients.")
        st.divider()
        st.markdown(f"**Emergency number:** {cfg.emergency_service_number}")
        st.caption("Works offline for records, ECG processing and AI "
                   "inference. Remote notifications are never simulated.")
        with st.expander("How the system works (flow)"):
            st.markdown(
                "**1. Register / sign in** as Patient or Doctor "
                "(doctors only see assigned patients).\n\n"
                "**2. Patient records ECG** on the ECG monitoring page "
                "(demo generator - no hardware needed) and **uploads CT** "
                "on My records (report values or image/manual level).\n\n"
                "**3. Fused assessment** (patient or doctor): the trained ECG "
                "model scores each 10-s window; the mean is combined with the "
                "patient's CT context level; the fused score is compared with "
                "the high-risk threshold (default 0.85).\n\n"
                "**4. If high-risk**: the patient sees a red emergency banner "
                "on every page with measured HR and the Call Ambulance button; "
                "the assigned doctor gets the event in **Alerts** with "
                "notification attempts; the doctor acknowledges / resolves.\n\n"
                "**5. The doctor reviews** the patient's ECG recordings "
                "(read-only) and CT records on Patient review, and can re-run "
                "the fused assessment on any recording.")


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    conn = ui.get_conn(cfg.db_path)

    session = ui.load_session(cfg)
    if session is not None:
        reason = auth.session_expired_reason(session)
        if reason:
            ui.force_logout(conn, session)
            st.warning(reason)
            session = None

    if session is None:
        ui.inject_css()
        _login_page(conn, cfg)
        return

    try:
        viewer = ui.viewer_or_none(conn, session)
    except Exception:
        viewer = None
    if viewer is None:
        ui.force_logout(conn, session)
        st.error("Your session is no longer valid (account deactivated or removed). "
                 "Please sign in again.")
        _login_page(conn, cfg)
        return

    ui.inject_css()
    try:
        if viewer.is_patient:
            ui.render_patient_banner(conn, viewer, cfg)
            ui.render_sidebar_user(conn, session, cfg)
            patient_dashboard.render(conn, viewer, cfg)
        else:
            ui.render_doctor_banner(conn, viewer, cfg)
            with st.sidebar:
                st.header("Doctor menu")
                st.caption("Use the navigation row on the main page.")
            ui.render_sidebar_user(conn, session, cfg)
            doctor_dashboard.render(conn, viewer, cfg)
    except dbm.DataAccessDenied as exc:
        st.error(f"Not authorized: {exc}")
    except Exception as exc:  # keep the app alive; show honest errors
        st.exception(exc)


main()
