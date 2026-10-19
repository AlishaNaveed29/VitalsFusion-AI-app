"""Doctor dashboard: assigned patients, clinical review, alerts, notes, reports.

Doctors only see patients assigned to them (unless broader access was
explicitly granted). Every query is access-checked in ``db.py``.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import assessment_core as core
import assessment_lab
import auth
import ct_upload
import db as dbm
import emergency
from config import AppConfig
from ecg_model import DEFAULT_MODEL_FS
from ui_common import (
    combined_review_card,
    demo_note,
    fmt_time,
    fusion_panel,
    page_header,
)


def render(conn, viewer: dbm.Viewer, cfg: AppConfig) -> None:
    with st.sidebar:
        st.markdown("#### \U0001F9ED Navigate")
        page = st.radio(
            "Go to",
            ["👥 My patients", "🩺 Patient review", "🚨 Alerts",
             "🧪 Assessment lab", "👤 Account"],
            key="vf_doc_nav", label_visibility="collapsed",
        )
        st.caption("Doctor portal: you see only your assigned patients. "
                   "Patients record their own ECG and upload CT reports; "
                   "you review and assess here.")
    page_header(f"Doctor dashboard - {viewer.full_name or viewer.email}",
                "You see only your assigned patients. Patients record their "
                "own ECG and upload CT reports; you review and assess here.")
    if page == "👥 My patients":
        _patients(conn, viewer, cfg)
    elif page == "🩺 Patient review":
        _review(conn, viewer, cfg)
    elif page == "🚨 Alerts":
        _alerts(conn, viewer, cfg)
    elif page == "🧪 Assessment lab":
        assessment_lab.render(conn, viewer, cfg)
    else:
        _account(conn, viewer, cfg)


# ---------------------------------------------------------------------------
# overview helpers
# ---------------------------------------------------------------------------
def _patient_rows(conn, viewer: dbm.Viewer) -> list[dict]:
    rows = []
    for p in dbm.list_assigned_patients(conn, viewer.id):
        pid = p["id"]
        sessions = dbm.list_ecg_sessions(conn, viewer, pid, limit=1)
        assessments = dbm.list_assessments(conn, viewer, pid, limit=1)
        events = [e for e in dbm.list_events_for_patient(conn, viewer, pid)
                  if e["status"] in ("open", "acknowledged")]
        latest = assessments[0] if assessments else None
        sess = sessions[0] if sessions else None
        rows.append({
            "id": pid,
            "name": p["full_name"] or "-",
            "email": p["email"],
            "account": "active" if p["is_active"] else "DEACTIVATED",
            "monitoring": sess["status"] if sess else "never started",
            "device": (sess["device_status"] if sess else "-"),
            "last session": fmt_time(sess["started_at"]) if sess else "-",
            "last assessment": (f"{latest['status']}"
                                + (f" ({latest['fused_score']:.3f})"
                                   if latest and latest["fused_score"] is not None else "")
                                if latest else "none"),
            "assessment time": fmt_time(latest["created_at"]) if latest else "-",
            "open alerts": len(events),
        })
    return rows


def _patients(conn, viewer: dbm.Viewer, cfg: AppConfig) -> None:
    st.subheader("My patients")
    rows = _patient_rows(conn, viewer)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No patients assigned yet. Add or assign one below.")

    with st.expander("Add a new patient account and assign to me"):
        with st.form("vf_doc_new_patient"):
            email = st.text_input("Patient email / login")
            name = st.text_input("Full name")
            pw = st.text_input("Temporary password (min 8 chars)", type="password",
                               help="Share it securely; the patient can change it later "
                                    "via password reset by you.")
            if st.form_submit_button("Create and assign patient"):
                err = auth.check_password_policy(pw, cfg.min_password_length)
                if err:
                    st.error(err)
                elif not email.strip():
                    st.error("Email is required.")
                elif dbm.get_user_by_identifier(conn, email):
                    st.error("An account with that email/username already exists.")
                else:
                    pid = dbm.create_user(conn, email.strip(),
                                          auth.hash_password(pw), "patient",
                                          full_name=name.strip())
                    dbm.assign_patient(conn, viewer.id, pid, viewer.id)
                    st.success(f"Patient {email} created and assigned to you.")

    with st.expander("Assign an existing patient to me"):
        all_patients = [u for u in dbm.list_users(conn, "patient")
                        if u["id"] not in {r["id"] for r in rows}]
        if not all_patients:
            st.caption("No unassigned patient accounts available.")
        else:
            pick = st.selectbox(
                "Patient", all_patients,
                format_func=lambda u: f"{u['full_name'] or '-'} <{u['email']}> (id {u['id']})",
                key="vf_doc_assign_pick")
            if st.button("Assign to me", key="vf_doc_assign_btn"):
                dbm.assign_patient(conn, viewer.id, int(pick["id"]), viewer.id)
                st.success("Assigned.")
                st.rerun()

    if rows:
        st.markdown("### Account activation (assigned patients)")
        act = st.selectbox("Patient", rows, format_func=lambda r: f"{r['name']} <{r['email']}>",
                           key="vf_doc_act_pick")
        target = int(act["id"])
        current = dbm.get_user(conn, target)
        c1, c2 = st.columns(2)
        if c1.button("Deactivate account", disabled=not current["is_active"],
                     key="vf_doc_deact"):
            dbm.set_user_active(conn, viewer.id, target, False, "deactivated by doctor")
            st.success("Account deactivated - the patient can no longer sign in.")
            st.rerun()
        if c2.button("Activate account", disabled=bool(current["is_active"]),
                     key="vf_doc_act"):
            dbm.set_user_active(conn, viewer.id, target, True, "activated by doctor")
            st.success("Account activated.")
            st.rerun()


# ---------------------------------------------------------------------------
# alerts
# ---------------------------------------------------------------------------
def _alerts(conn, viewer: dbm.Viewer, cfg: AppConfig) -> None:
    st.subheader("High-risk alerts")
    events = dbm.list_events_for_doctor(conn, viewer.id)
    if not events:
        st.success("No high-risk events for your assigned patients.")
        return
    # confirm local delivery for alerts this doctor is looking at now
    for e in events:
        if e["status"] in ("open", "acknowledged"):
            try:
                emergency.mark_opened(conn, viewer, e["id"])
            except Exception:
                pass

    for e in events:
        patient = dbm.get_user(conn, e["patient_id"])
        color = {"open": "🔴", "acknowledged": "🟠", "resolved": "🟢"}[e["status"]]
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 2])
            c1.markdown(f"**{color} Event #{e['id']}** - "
                        f"{patient['full_name'] or patient['email']}")
            c2.metric("Score / threshold", f"{e['score']:.3f} / {e['threshold']:.2f}")
            c3.metric("Heart rate",
                      f"{e['heart_rate_bpm']:.0f} bpm" if e["heart_rate_bpm"] else "n/a")
            st.caption(f"Created {fmt_time(e['created_at'])} | status **{e['status']}** | "
                       f"assessment #{e['assessment_id'] or '-'} | "
                       f"patient call action: "
                       f"{'YES at ' + fmt_time(e['patient_call_at']) if e['patient_call_initiated'] else 'no'}")

            notifs = dbm.list_notifications(conn, viewer, e["id"])
            if notifs:
                st.markdown("Notification attempts: " + " | ".join(
                    f"`{n['channel']}` → **{n['status']}**" for n in notifs))
                with st.expander("Notification details"):
                    st.dataframe(pd.DataFrame([{
                        "channel": n["channel"], "status": n["status"],
                        "detail": n["detail"], "time": fmt_time(n["created_at"]),
                        "confirmed": fmt_time(n["confirmed_at"]),
                    } for n in notifs]), hide_index=True)

            if e.get("assessment_id"):
                try:
                    a_row = dbm.get_assessment(conn, viewer, e["assessment_id"])
                    if demo_note(a_row.get("model_name")):
                        st.warning(demo_note(a_row.get("model_name")))
                except Exception:
                    pass

            acks = dbm.list_event_acknowledgments(conn, viewer, e["id"])
            if acks:
                st.markdown("Your actions:")
                for a in acks:
                    st.caption(f"- {a['action']} {fmt_time(a['created_at'])}"
                               + (f": {a['note']}" if a["note"] else ""))

            if e["status"] == "open":
                note = st.text_input("Acknowledge note", key=f"vf_ack_note_{e['id']}")
                if st.button("Acknowledge", type="primary", key=f"vf_ack_{e['id']}"):
                    emergency.acknowledge_event(conn, viewer, e["id"], note)
                    st.rerun()
            if e["status"] in ("open", "acknowledged"):
                r1, r2 = st.columns([2, 1])
                rnote = r1.text_input("Resolution note", key=f"vf_res_note_{e['id']}")
                if r2.button("Resolve", key=f"vf_res_{e['id']}"):
                    emergency.resolve_event(conn, viewer, e["id"], rnote)
                    st.rerun()
            add = st.text_input("Add a note", key=f"vf_note_{e['id']}")
            if st.button("Save note", key=f"vf_note_btn_{e['id']}"):
                try:
                    emergency.add_note(conn, viewer, e["id"], add)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))


# ---------------------------------------------------------------------------
# patient review
# ---------------------------------------------------------------------------
def _review(conn, viewer: dbm.Viewer, cfg: AppConfig) -> None:
    st.subheader("Patient review")
    patients = dbm.list_assigned_patients(conn, viewer.id)
    if not patients:
        st.info("No assigned patients. Add or assign one from 'My patients'.")
        return
    selected = st.selectbox(
        "Patient", patients,
        format_func=lambda p: f"{p['full_name'] or '-'} <{p['email']}>", key="vf_rev_pick")
    pid = int(selected["id"])
    dbm.require_patient_access(conn, viewer, pid)   # defensive re-check

    tabs = st.tabs(["Overview", "ECG", "CT / radiology", "AI assessments",
                    "Alerts", "Clinical notes", "Emergency contact", "Reports"])
    with tabs[0]:
        _rev_overview(conn, viewer, cfg, pid)
    with tabs[1]:
        _rev_ecg(conn, viewer, cfg, pid)
    with tabs[2]:
        _rev_ct(conn, viewer, cfg, pid)
    with tabs[3]:
        _rev_assessments(conn, viewer, cfg, pid)
        _rev_test_data(conn, viewer, cfg, pid)
    with tabs[4]:
        _rev_alerts(conn, viewer, cfg, pid)
    with tabs[5]:
        _rev_notes(conn, viewer, cfg, pid)
    with tabs[6]:
        _rev_contact(conn, viewer, cfg, pid)
    with tabs[7]:
        _rev_reports(conn, viewer, cfg, pid)


def _rev_overview(conn, viewer, cfg, pid: int) -> None:
    user = dbm.get_user(conn, pid)
    prof = dbm.get_profile(conn, viewer, pid)
    st.write(f"**{user['full_name'] or '-'}** ({user['email']}) - account "
             f"{'active' if user['is_active'] else 'DEACTIVATED'}")
    st.write(f"DOB: {prof['date_of_birth'] or '-'} | sex: {prof['sex'] or '-'} | "
             f"phone: {prof['phone'] or '-'}")
    st.write(f"Location (patient-provided): {prof['location_text'] or 'not provided'}")
    st.write(f"Accessibility notes: {prof['accessibility_notes'] or '-'}")
    st.markdown("---")
    # Combined CT + ECG + fusion + recommended next steps in one place
    combined_review_card(conn, viewer, pid, cfg)


def _rev_ecg(conn, viewer, cfg, pid: int) -> None:
    page_header("Patient ECG recordings",
                "Recordings the patient made on their own ECG monitoring page. "
                "You review them here and run the fused ECG + CT assessment.")
    sessions = dbm.list_ecg_sessions(conn, viewer, pid, limit=50)
    if not sessions:
        st.info("No ECG recordings yet - ask the patient to record from their "
                "**ECG monitoring** page (demo generator or file upload).")
        return
    pick = st.selectbox(
        "Recording", sessions,
        format_func=lambda s: (f"#{s['id']} · {fmt_time(s['started_at'])} · "
                               f"{s['status']} · {s['n_samples']} samples @ "
                               f"{s['fs_hz'] or '?'} Hz · {s['source']}"),
        key="vf_rev_ecg_pick")
    sess = dbm.get_ecg_session(conn, viewer, int(pick["id"]))
    sig = sess["signal_array"]
    if sig.size:
        st.line_chart(pd.DataFrame({"ECG": sig[:8000]}), height=260)
        recorder = "patient"
        if sess.get("recorded_by") and sess["recorded_by"] != pid:
            row = conn.execute("SELECT email, full_name FROM users WHERE id=?",
                               (sess["recorded_by"],)).fetchone()
            if row:
                recorder = row["full_name"] or row["email"]
        st.caption(f"**{sig.size} samples** @ {sess['fs_hz']} Hz · mean HR "
                   f"{sess['mean_hr_bpm'] or 'n/a'} bpm · source {sess['source']} · "
                   f"device {sess['device_status']} · recorded by {recorder} · "
                   f"started {fmt_time(sess['started_at'])}.")
    else:
        st.caption("This recording has no samples stored.")
    if sess.get("analysis_json"):
        with st.expander("Signal analysis details"):
            try:
                st.dataframe(pd.DataFrame(json.loads(sess["analysis_json"])),
                             hide_index=True)
            except Exception:
                st.code(sess["analysis_json"][:2000])

    st.markdown("### Run fused AI assessment (ECG + CT)")
    st.caption("The trained ECG model scores this recording; the patient's "
               "latest CT context level is added by the fusion rule; the fused "
               "score is compared with the high-risk threshold.")
    ct = dbm.latest_ct_record(conn, viewer, pid)
    if ct:
        st.caption(f"Patient's latest CT: record #{ct['id']}, source "
                   f"`{ct['source']}`, context level **{ct['context_level']}** "
                   f"(recorded {fmt_time(ct['created_at'])}).")
    else:
        st.caption("Patient has no CT record yet - the assessment runs with "
                   "CT 'not reported' (context level 0 contribution).")
    if st.button("Run fused assessment on this recording", key="vf_rev_fused_run"):
        model, msg, _ok = core.prepare_model(cfg)
        st.info(msg)
        result = core.run_recording(
            model, sig, sess["fs_hz"] or DEFAULT_MODEL_FS, cfg,
            core.ct_level_from_record(ct))
        assessment, event = emergency.process_assessment(
            conn, viewer, cfg, pid,
            **core.assessment_kwargs(result, ecg_session_id=sess["id"],
                                     ct_record_id=ct["id"] if ct else None))
        fusion_panel(result, assessment, event, cfg)


def _rev_ct(conn, viewer, cfg, pid: int) -> None:
    ct_upload.ct_upload_ui(conn, viewer, pid, key_prefix="vf_rev_ct",
                           heading="### Add CT record - choose an input option")


def _rev_assessments(conn, viewer, cfg, pid: int) -> None:
    rows = dbm.list_assessments(conn, viewer, pid, limit=100)
    if rows:
        st.dataframe(pd.DataFrame([{
            "id": r["id"], "time": fmt_time(r["created_at"]), "status": r["status"],
            "fused score": r["fused_score"], "level": r["fused_level"],
            "ECG mean": r["ecg_score_mean"], "CT level": r["ct_level"],
            "HR": r["heart_rate_bpm"], "model": r["model_name"],
            "by": r["created_by"], "error": r["error_message"],
        } for r in rows]), hide_index=True)
        n_demo = sum(1 for r in rows if demo_note(r.get("model_name")))
        if n_demo:
            st.warning(f"{n_demo} of these assessments are synthetic TEST records "
                       "for testing only - not real model outputs.")
        for r in rows[:3]:
            if r.get("windows_json"):
                with st.expander(f"Assessment #{r['id']} windows"):
                    try:
                        st.dataframe(pd.DataFrame(json.loads(r["windows_json"])),
                                     hide_index=True)
                    except Exception:
                        st.code((r["windows_json"] or "")[:1500])
    else:
        st.info("No assessments yet - run one from the Assessment lab.")

    st.markdown("### Quick run (latest saved recording)")
    st.caption("Fused ECG + CT: the model scores the recording, the patient's "
               "latest CT context level is added, and the fused score is "
               "compared with the high-risk threshold.")
    sessions = dbm.list_ecg_sessions(conn, viewer, pid, limit=10)
    if not sessions:
        st.caption("No saved ECG recording to assess - the patient records "
                   "from their ECG monitoring page.")
        return
    sess_pick = st.selectbox(
        "Recording", sessions,
        format_func=lambda s: f"#{s['id']} {fmt_time(s['started_at'])} "
                              f"{s['n_samples']} samples",
        key="vf_rev_quick_sess")
    if st.button("Run fused AI assessment on this recording", key="vf_rev_quick_run"):
        sess = dbm.get_ecg_session(conn, viewer, int(sess_pick["id"]))
        model, msg, _ok = core.prepare_model(cfg)
        st.info(msg)
        ct = dbm.latest_ct_record(conn, viewer, pid)
        result = core.run_recording(model, sess["signal_array"].astype(float),
                                    sess["fs_hz"] or DEFAULT_MODEL_FS, cfg,
                                    core.ct_level_from_record(ct))
        assessment, event = emergency.process_assessment(
            conn, viewer, cfg, pid,
            **core.assessment_kwargs(result, ecg_session_id=sess["id"],
                                     ct_record_id=ct["id"] if ct else None))
        fusion_panel(result, assessment, event, cfg)


def _rev_test_data(conn, viewer, cfg, pid: int) -> None:
    with st.expander("Test data (simulation) - for testing the workflow only"):
        st.caption("Saves clearly labelled synthetic assessments (tagged "
                   "TEST-SIMULATION) through the real pipeline: 'high' raises a "
                   "real high-risk event with notifications, the patient banner "
                   "and Call Ambulance; 'low' saves a completed assessment "
                   "without an event. Nothing is overwritten (each run is a new "
                   "row) and one active event per patient still applies.")
        c1, c2 = st.columns(2)
        if c1.button("Simulate TEST high-risk assessment", key="vf_test_high"):
            assessment, event = emergency.simulate_test_assessment(
                conn, viewer, cfg, pid, kind="high")
            if event:
                st.success(f"TEST assessment #{assessment['id']} saved; "
                           f"high-risk event #{event['id']} raised.")
            else:
                st.success(f"TEST assessment #{assessment['id']} saved; "
                           "existing active event was refreshed.")
            st.rerun()
        if c2.button("Simulate TEST low-risk assessment", key="vf_test_low"):
            assessment, event = emergency.simulate_test_assessment(
                conn, viewer, cfg, pid, kind="low")
            st.success(f"TEST assessment #{assessment['id']} saved "
                       "(no event - below threshold).")
            st.rerun()


def _rev_alerts(conn, viewer, cfg, pid: int) -> None:
    events = dbm.list_events_for_patient(conn, viewer, pid)
    if not events:
        st.success("No high-risk events for this patient.")
        return
    for e in events:
        with st.container(border=True):
            st.markdown(f"**Event #{e['id']}** - {e['status']} - "
                        f"{fmt_time(e['created_at'])}")
            st.write(f"Score {e['score']:.3f} vs threshold {e['threshold']:.2f}; "
                     f"HR {e['heart_rate_bpm'] or 'n/a'} bpm; "
                     f"assessment #{e['assessment_id'] or '-'}; "
                     f"call action: "
                     f"{'YES ' + fmt_time(e['patient_call_at']) if e['patient_call_initiated'] else 'no'}")
            notifs = dbm.list_notifications(conn, viewer, e["id"])
            st.markdown("Notifications: " + ", ".join(
                f"`{n['channel']}` → **{n['status']}**" for n in notifs) or "none")
            acks = dbm.list_event_acknowledgments(conn, viewer, e["id"])
            for a in acks:
                st.caption(f"{a['action']} by {a['doctor_name'] or a['doctor_email']} "
                           f"at {fmt_time(a['created_at'])}"
                           + (f": {a['note']}" if a["note"] else ""))
            if e["status"] == "open":
                if st.button("Acknowledge", key=f"vf_rev_ack_{e['id']}"):
                    emergency.acknowledge_event(conn, viewer, e["id"], "")
                    st.rerun()
            if e["status"] in ("open", "acknowledged"):
                if st.button("Resolve", key=f"vf_rev_res_{e['id']}"):
                    emergency.resolve_event(conn, viewer, e["id"], "")
                    st.rerun()


def _rev_notes(conn, viewer, cfg, pid: int) -> None:
    with st.form("vf_rev_note_form"):
        note = st.text_area("Clinical note", height=120)
        if st.form_submit_button("Add note"):
            try:
                dbm.add_clinical_note(conn, viewer.id, pid, note)
                st.success("Note saved.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    notes = dbm.list_clinical_notes(conn, viewer, pid, limit=100)
    for n in notes:
        st.markdown(f"**{fmt_time(n['created_at'])}** - "
                    f"{n['doctor_name'] or n['doctor_email']}")
        st.write(n["note"])


def _rev_contact(conn, viewer, cfg, pid: int) -> None:
    st.caption("Read-only view. The patient edits their own emergency contact.")
    contact = dbm.get_emergency_contact(conn, viewer, pid)
    st.write(f"**Contact:** {contact['contact_name'] or '-'} "
             f"({contact['relationship'] or '-'}) - {contact['contact_number'] or '-'}")
    st.write(f"**Emergency services:** {contact['emergency_service_number'] or '1122'}")
    st.write(f"**Location (patient-provided):** {contact['location_text'] or 'not provided'}")
    st.write(f"**Accessibility notes:** {contact['accessibility_notes'] or '-'}")
    events = [e for e in dbm.list_events_for_patient(conn, viewer, pid)
              if e["patient_call_initiated"]]
    if events:
        st.markdown("**Ambulance-call actions recorded:**")
        for e in events:
            st.write(f"- Event #{e['id']}: patient pressed Call Ambulance at "
                     f"{fmt_time(e['patient_call_at'])} "
                     "(call connection is not verifiable by the app).")
    else:
        st.caption("No ambulance-call actions recorded.")


def _rev_reports(conn, viewer, cfg, pid: int) -> None:
    import reports as rep
    user = dbm.get_user(conn, pid)
    patient = {"full_name": user["full_name"], "email": user["email"]}
    assessments = dbm.list_assessments(conn, viewer, pid, limit=200)
    events = dbm.list_events_for_patient(conn, viewer, pid)
    notes = dbm.list_clinical_notes(conn, viewer, pid, limit=100)
    ct_recs = dbm.list_ct_records(conn, viewer, pid, limit=100)
    sessions = dbm.list_ecg_sessions(conn, viewer, pid, limit=100)
    csv_data = rep.multi_section_csv([
        ("Assessments", assessments),
        ("High-risk events", events),
        ("CT records", ct_recs),
        ("ECG sessions", sessions),
        ("Clinical notes", notes),
    ])
    b1, b2 = st.columns(2)
    b1.download_button("Download CSV", csv_data,
                       file_name=f"patient{pid}_report.csv", mime="text/csv",
                       key="vf_rev_csv")
    pdf = rep.assessment_report_pdf(patient, assessments, events, notes,
                                    ct_records=ct_recs, ecg_sessions=sessions)
    b2.download_button("Download PDF", pdf,
                       file_name=f"patient{pid}_report.pdf", mime="application/pdf",
                       key="vf_rev_pdf")


# ---------------------------------------------------------------------------
# account
# ---------------------------------------------------------------------------
def _account(conn, viewer: dbm.Viewer, cfg: AppConfig) -> None:
    st.subheader("My account")
    row = dbm.get_user(conn, viewer.id)
    st.write(f"**{row['full_name'] or '-'}** ({row['email']}) - role: **{row['role']}**")
    st.write(f"Account active: {'yes' if row['is_active'] else 'NO'} | "
             f"broader patient access: {'GRANTED' if row['broader_access'] else 'no (assigned patients only)'}")
    st.write(f"Signed in since / session expiry is shown in the sidebar.")
    n = len(dbm.list_assigned_patients(conn, viewer.id))
    st.write(f"Assigned patients: **{n}**")
    st.caption("Password changes and doctor deactivation are handled on this "
               "machine with: python setup_doctor.py --reset-password / --deactivate")

    st.markdown("### Recent security audit entries")
    entries = [a for a in dbm.list_audit(conn, limit=50)
               if a.get("user_id") == viewer.id]
    if entries:
        st.dataframe(pd.DataFrame([{
            "time": fmt_time(a["created_at"]), "action": a["action"],
            "resource": f"{a['resource_type'] or ''} {a['resource_id'] or ''}".strip(),
            "detail": a["detail"],
        } for a in entries]), hide_index=True)
    else:
        st.caption("No recent audit entries for your account.")
