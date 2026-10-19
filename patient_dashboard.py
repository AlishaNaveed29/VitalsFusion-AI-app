"""Patient dashboard: personal monitoring, records, emergency workflow, profile.

A patient only ever sees their own data: every read/write goes through the
access-controlled service functions in ``db.py`` using the authenticated
session identity - never a URL parameter, patient id box or other client input.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import assessment_core as core
import ct_upload
import db as dbm
import emergency
import monitoring
from ui_common import (
    SEEK_HELP_TEXT,
    combined_review_card,
    demo_note,
    fmt_time,
    fusion_panel,
    page_header,
    render_call_ambulance,
    render_emergency_contact,
)


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def render(conn, viewer: dbm.Viewer, cfg) -> None:
    with st.sidebar:
        st.markdown("#### \U0001F9ED Navigate")
        page = st.radio(
            "Go to",
            ["🏠 Home", "📈 ECG monitoring", "📁 My records", "🆘 Emergency",
             "📄 Reports", "👤 Profile"],
            key="vf_pat_nav", label_visibility="collapsed",
        )
        st.caption("Private portal: your records are visible only to you and "
                   "your assigned doctor.")
    page_header(f"Welcome, {viewer.full_name or viewer.email}",
                "Your personal ECG + CT fusion dashboard. "
                "Everything here is private to you unless shared with your doctor.")
    if page == "🏠 Home":
        _home(conn, viewer, cfg)
    elif page == "📈 ECG monitoring":
        _monitoring(conn, viewer, cfg)
    elif page == "📁 My records":
        _records(conn, viewer, cfg)
    elif page == "🆘 Emergency":
        _emergency_page(conn, viewer, cfg)
    elif page == "📄 Reports":
        _reports(conn, viewer, cfg)
    else:
        _profile(conn, viewer, cfg)


def _home(conn, viewer: dbm.Viewer, cfg) -> None:
    mon = monitoring.mon_state(viewer, viewer.id)
    sessions = dbm.list_ecg_sessions(conn, viewer, viewer.id, limit=5)
    assessments = dbm.list_assessments(conn, viewer, viewer.id, limit=10)
    events = emergency.all_active_events(conn, viewer, viewer.id)

    st.markdown("#### My vitals at a glance")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    status = mon["status"] if mon.get("session_id") else (
        sessions[0]["status"] if sessions else "never started")
    c1.metric("Monitoring status", status.capitalize() if status else "-",
              help="Live ECG session state, or your most recent saved session.")
    if mon.get("last_hr"):
        c2.metric("Heart rate (live)", f"{mon['last_hr']:.0f} bpm",
                  help="Updated while a live monitoring session is running.")
    elif sessions and sessions[0].get("mean_hr_bpm"):
        c2.metric("Heart rate (last session)",
                  f"{sessions[0]['mean_hr_bpm']:.0f} bpm",
                  help="Mean heart rate from your most recent saved ECG session.")
    else:
        c2.metric("Heart rate", "no data",
                  help="Record an ECG on the monitoring page to populate this.")
    latest = assessments[0] if assessments else None
    c3.metric("Latest AI assessment",
              (f"{latest['fused_score']:.3f}" if latest and latest["fused_score"] is not None
               else (latest["status"] if latest else "none")),
              help="Fused ECG + CT score (0-1). A higher score means higher "
                   "model-flagged risk - it is not a diagnosis.")
    c4.metric("Active high-risk events", len(events),
              help="Alerts from high-risk assessments that are still open or "
                   "acknowledged (not yet resolved by your doctor).")

    if latest and latest["status"] != "completed":
        st.warning(f"Your last assessment status is **{latest['status']}**"
                   + (f": {latest['error_message']}" if latest.get("error_message") else "")
                   + ". A missing or failed assessment never means 'low risk'.")
    elif latest and latest.get("fused_score") is not None:
        st.info(f"Last fused score {latest['fused_score']:.3f} "
                f"(threshold {cfg.high_risk_threshold}) on {fmt_time(latest['created_at'])}.")
    st.markdown(
        f"<div class='vf-guidance'><b>Safety guidance</b><br/>"
        f"{SEEK_HELP_TEXT.replace('**', '')}</div>",
        unsafe_allow_html=True)

    doctors = dbm.list_doctors_for_patient(conn, viewer.id)
    st.markdown("**My care team:** " + (", ".join(
        f"{d['full_name'] or d['email']}" for d in doctors) if doctors
        else "not assigned to a doctor yet"))

    if sessions:
        st.markdown("**Recent ECG recordings**")
        st.caption("Your last 5 recordings - private to you and your assigned doctor.")
        st.dataframe(pd.DataFrame([{
            "id": s["id"], "started": fmt_time(s["started_at"]),
            "status": s["status"], "device": s["device_label"],
            "device status": s["device_status"],
            "samples": s["n_samples"], "mean HR": s["mean_hr_bpm"],
        } for s in sessions]), hide_index=True)


def _monitoring(conn, viewer: dbm.Viewer, cfg) -> None:
    st.markdown("#### Record my ECG")
    st.caption("You record here with the demo generator (no hardware needed). "
               "Your assigned doctor sees the saved recordings on their side "
               "and can run the fused ECG + CT assessment on them.")
    monitoring.render_monitoring(conn, viewer, viewer.id, cfg,
                                 key_prefix="vf_pat_mon",
                                 assessment_ui=_run_assessment)


def _run_assessment(conn, viewer: dbm.Viewer, cfg, mon: dict, pid: int) -> None:
    st.caption("Fused ECG + CT: your ECG model scores + your latest CT context "
               "level. If the model or inputs are unavailable the status is "
               "reported honestly - no low-risk result is invented.")
    emergency_symptoms = st.checkbox(
        "I currently have chest pain, shortness of breath, fainting, or other "
        "possible heart-attack symptoms",
        key="vf_pat_emergency_symptoms",
        help="If checked, the recommendation will always be Emergency evaluation "
             "regardless of the AI score.")
    if not st.button("Run fused assessment on the current recording",
                     key="vf_pat_run_mon_assess",
                     disabled=mon["samples"].size == 0):
        return
    model, msg, ok = core.prepare_model(cfg)
    st.info(msg)
    ct = dbm.latest_ct_record(conn, viewer, viewer.id)
    ct_level = core.ct_level_from_record(ct)
    if ct_level is not None:
        st.caption(f"Using your saved CT context level {ct_level} "
                   f"(record #{ct['id']}).")
    else:
        st.caption("No CT record saved - assessment runs with CT 'not reported'.")

    result = core.run_recording(model, mon["samples"], mon["fs"], cfg, ct_level)
    assessment, event = emergency.process_assessment(
        conn, viewer, cfg, viewer.id,
        **core.assessment_kwargs(result, ecg_session_id=mon.get("session_id"),
                                 ct_record_id=ct["id"] if ct else None),
    )
    fusion_panel(result, assessment, event, cfg,
                 emergency_symptoms=emergency_symptoms)
    if event:
        st.rerun()  # show the global emergency banner


def _records(conn, viewer: dbm.Viewer, cfg) -> None:
    st.markdown("#### My records")
    st.caption("Your AI assessments, CT findings and high-risk events - "
               "private to you and your assigned doctor.")

    # Combined CT + ECG + fusion + next steps in one place
    with st.expander("Combined assessment summary (CT + ECG + next steps)",
                     expanded=True):
        emergency_symptoms = st.checkbox(
            "I currently have emergency symptoms (chest pain, breathlessness, "
            "fainting, etc.)",
            key="vf_pat_rec_symptoms",
            help="If checked, the recommendation is always Emergency evaluation.")
        combined_review_card(conn, viewer, viewer.id, cfg,
                             emergency_symptoms=emergency_symptoms)

    st.markdown("### AI assessment history")
    rows = dbm.list_assessments(conn, viewer, viewer.id, limit=100)
    if rows:
        df = pd.DataFrame([{
            "id": r["id"], "time": fmt_time(r["created_at"]), "status": r["status"],
            "fused score": r["fused_score"], "fused level": r["fused_level"],
            "ECG mean": r["ecg_score_mean"], "CT level": r["ct_level"],
            "HR (bpm)": r["heart_rate_bpm"], "model": r["model_name"],
            "error": r["error_message"],
        } for r in rows])
        st.dataframe(df, hide_index=True)
        n_demo = sum(1 for r in rows if demo_note(r.get("model_name")))
        if n_demo:
            st.warning(f"{n_demo} of these assessments are synthetic TEST records "
                       "created for testing only - not real model outputs.")
        scored = [r for r in rows if r["fused_score"] is not None]
        if scored:
            st.line_chart(pd.DataFrame(scored[::-1]).set_index("id")["fused_score"])
        for r in rows[:5]:
            if r.get("windows_json"):
                with st.expander(f"Assessment #{r['id']} window details"):
                    try:
                        st.dataframe(pd.DataFrame(json.loads(r["windows_json"])),
                                     hide_index=True)
                    except Exception:
                        st.code((r["windows_json"] or "")[:2000])
    else:
        st.info("No AI assessments yet. Run one from ECG monitoring.")

    with st.expander("Add CT record (report values or image / manual level)"):
        ct_upload.ct_upload_ui(conn, viewer, viewer.id, key_prefix="vf_pat_ct")

    st.markdown("### CT / radiology findings")
    st.caption("Context levels are computed from report values or set manually "
               "when you upload a CT image - used as the CT branch of the "
               "fused assessment.")
    recs = dbm.list_ct_records(conn, viewer, viewer.id, limit=50)
    if recs:
        st.dataframe(pd.DataFrame([{
            "id": r["id"], "recorded": fmt_time(r["created_at"]), "source": r["source"],
            "CAC score": r["cac_score"], "CAC category": r["cac_category"],
            "stenosis grade": r["stenosis_grade"], "vessels>=50%": r["vessels_ge50"],
            "CT context level": r["context_level"], "report": (r["report_text"] or "")[:80],
        } for r in recs]), hide_index=True)
    else:
        st.info("No CT records yet. Add one above, or your doctor can add "
                "radiology report values from Patient review.")

    st.markdown("### High-risk event history")
    events = dbm.list_events_for_patient(conn, viewer, viewer.id)
    if events:
        st.dataframe(pd.DataFrame([{
            "id": e["id"], "created": fmt_time(e["created_at"]), "status": e["status"],
            "score": e["score"], "threshold": e["threshold"],
            "HR": e["heart_rate_bpm"],
            "call initiated": bool(e["patient_call_initiated"]),
        } for e in events]), hide_index=True)
    else:
        st.caption("No high-risk events recorded.")


def _emergency_page(conn, viewer: dbm.Viewer, cfg) -> None:
    st.markdown(
        "<div class='vf-guidance'><b>If you have concerning symptoms, call your "
        "local emergency number or go to the nearest emergency department now - "
        "do not wait for this app.</b><br/>"
        "Chest pain, pressure, shortness of breath, sweating or fainting need "
        "immediate medical attention, regardless of any AI score shown here."
        "</div>",
        unsafe_allow_html=True)
    active = emergency.open_events_for_patient(conn, viewer, viewer.id)
    number = (cfg.emergency_service_number or "1122").strip()
    if not active:
        st.markdown(
            "<div class='vf-guidance vf-guidance-teal'><b>No AI assessment is "
            "required to call for help.</b><br/>"
            "Use the emergency button below at any time - it opens your "
            "phone's calling app.</div>",
            unsafe_allow_html=True)
        st.link_button(f"\U0001F4DE Call Ambulance ({number})", f"tel:{number}",
                       type="primary", key="vf_emg_page_call")
    else:
        st.caption("You have an open high-risk event - the emergency banner at "
                   "the top of every page keeps the Call Ambulance action and "
                   "your details visible.")

    st.markdown("### My emergency contact details")
    st.caption("Stored privately on this device. Location is NOT shared "
               "automatically with any ambulance service - you choose what to "
               "record here and can read it out when you call.")
    current = dbm.get_emergency_contact(conn, viewer, viewer.id)
    with st.form("vf_contact_form"):
        name = st.text_input("Emergency contact name", current["contact_name"])
        rel = st.text_input("Relationship", current["relationship"])
        num = st.text_input("Emergency contact number", current["contact_number"])
        svc = st.text_input("Emergency service number",
                            current["emergency_service_number"] or "1122",
                            help="Used for the Call Ambulance button (default 1122 "
                                 "for the Lahore/Pakistan prototype).")
        loc = st.text_area("My location / address (voluntary)",
                           current["location_text"])
        acc = st.text_area("Accessibility or assistance notes", current["accessibility_notes"])
        if st.form_submit_button("Save emergency contact"):
            dbm.update_emergency_contact(conn, viewer, viewer.id, {
                "contact_name": name, "relationship": rel, "contact_number": num,
                "emergency_service_number": svc, "location_text": loc,
                "accessibility_notes": acc,
            })
            st.success("Saved.")
    st.markdown("### My doctors")
    doctors = dbm.list_doctors_for_patient(conn, viewer.id)
    if doctors:
        for d in doctors:
            st.write(f"- {d['full_name'] or d['email']} ({d['email']})")
    else:
        st.caption("Not assigned to a doctor yet.")

    st.markdown("### Event history")
    events = dbm.list_events_for_patient(conn, viewer, viewer.id)
    if not events:
        st.caption("No high-risk events recorded.")
    for e in events:
        with st.expander(f"Event #{e['id']} - {e['status']} - {fmt_time(e['created_at'])}"):
            st.write(f"Score {e['score']:.3f} vs threshold {e['threshold']}; "
                     f"HR {e['heart_rate_bpm'] or 'n/a'} bpm.")
            st.write(f"Call action recorded: "
                     f"{'yes at ' + fmt_time(e['patient_call_at']) if e['patient_call_initiated'] else 'no'}")
            render_call_ambulance(conn, viewer, e, cfg, key=f"vf_emg_hist_{e['id']}")


def _reports(conn, viewer: dbm.Viewer, cfg) -> None:
    st.markdown("#### My reports")
    st.caption("Downloads are limited to your own records (CSV or PDF).")
    import reports as rep

    assessments = dbm.list_assessments(conn, viewer, viewer.id, limit=200)
    events = dbm.list_events_for_patient(conn, viewer, viewer.id)
    notes = dbm.list_clinical_notes(conn, viewer, viewer.id, limit=100)
    sessions = dbm.list_ecg_sessions(conn, viewer, viewer.id, limit=100)
    ct_recs = dbm.list_ct_records(conn, viewer, viewer.id, limit=100)
    user = dbm.get_user(conn, viewer.id)
    patient = {"full_name": viewer.full_name, "email": viewer.email}

    csv_data = rep.multi_section_csv([
        ("Assessments", [{k: a[k] for k in ("id", "created_at", "status", "fused_score",
                                            "fused_level", "ecg_score_mean", "ct_level",
                                            "heart_rate_bpm", "model_name", "error_message")}
                         for a in assessments]),
        ("High-risk events", [{k: e[k] for k in ("id", "event_uid", "created_at", "status",
                                                 "score", "threshold", "heart_rate_bpm",
                                                 "patient_call_initiated", "patient_call_at")}
                              for e in events]),
        ("CT records", [{k: c.get(k) for k in ("id", "created_at", "source",
                                               "cac_score", "cac_category",
                                               "stenosis_grade", "vessels_ge50",
                                               "context_level", "report_text")}
                        for c in ct_recs]),
        ("ECG sessions", [{k: s[k] for k in ("id", "started_at", "ended_at", "status",
                                             "device_label", "device_status", "source",
                                             "fs_hz", "n_samples", "mean_hr_bpm")}
                          for s in sessions]),
        ("Clinical notes", [{k: n[k] for k in ("id", "created_at", "doctor_name", "note")}
                            for n in notes]),
    ])
    b1, b2 = st.columns(2)
    b1.download_button("Download CSV", csv_data,
                       file_name="my_vitalsfusion_report.csv", mime="text/csv")
    pdf = rep.assessment_report_pdf(patient, assessments, events, notes,
                                    ct_records=ct_recs, ecg_sessions=sessions)
    b2.download_button("Download PDF", pdf,
                       file_name="my_vitalsfusion_report.pdf", mime="application/pdf")


def _profile(conn, viewer: dbm.Viewer, cfg) -> None:
    st.markdown("#### My profile")
    st.caption("Login email and account role cannot be changed here. "
               "Profile details stay on this device unless shared with your doctor.")
    prof = dbm.get_profile(conn, viewer, viewer.id)
    with st.form("vf_profile_form"):
        full_name = st.text_input("Full name", viewer.full_name)
        dob = st.text_input("Date of birth (YYYY-MM-DD)", prof["date_of_birth"] or "")
        sex = st.text_input("Sex", prof["sex"] or "")
        phone = st.text_input("Phone", prof["phone"] or "")
        loc = st.text_area("Location / address (voluntary)", prof["location_text"] or "")
        acc = st.text_area("Accessibility or assistance notes",
                           prof["accessibility_notes"] or "")
        if st.form_submit_button("Save profile"):
            dbm.update_profile(conn, viewer, viewer.id, {
                "date_of_birth": dob, "sex": sex, "phone": phone,
                "location_text": loc, "accessibility_notes": acc,
            })
            if full_name.strip() and full_name.strip() != viewer.full_name:
                conn.execute("UPDATE users SET full_name=?, updated_at=? WHERE id=?",
                             (full_name.strip(), dbm.utcnow(), viewer.id))
                conn.commit()
            st.success("Profile saved.")
    st.markdown("### Account")
    st.write(f"Login: **{viewer.email}**  |  Role: **{viewer.role}**")
    st.write("To change your password or deactivate your account, contact your doctor.")
