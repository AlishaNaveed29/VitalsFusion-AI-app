"""Shared Streamlit UI helpers: styles, session plumbing, and the emergency
warning that must stay reachable from every dashboard page."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import streamlit as st

import auth
import db as dbm
import emergency
import recommend as rec
from config import AppConfig

CSS = """
<style>
/* ---------- base ---------- */
.stApp { background: #f4f6f9; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b3d66 0%, #0e5a8a 100%);
}
[data-testid="stSidebar"] * { color: #eef4fa; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff; }
[data-testid="stSidebar"] .stButton button {
  background: rgba(255,255,255,.12); color: #ffffff;
  border: 1px solid rgba(255,255,255,.25); border-radius: 8px;
}
h1, h2, h3 { letter-spacing: .2px; }
h1 { color: #0b3d66; font-weight: 700; }
h2 { color: #12557f; font-weight: 650; }
h3 { color: #1a6a9c; font-weight: 600; }
section[data-testid="stVerticalBlock"] div[data-testid="stMetric"] {
  background: #ffffff; border: 1px solid #dde5ee; border-radius: 12px;
  padding: 14px 16px; box-shadow: 0 1px 3px rgba(16,42,67,.06);
  border-left: 4px solid #0e7490;
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  color:#5b7083; font-weight:600; font-size:.86rem; letter-spacing:.2px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color:#0b3d66; font-weight:700; font-size:1.55rem; line-height:1.2;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] > div { font-size:.85rem; }
div[data-testid="stForm"] {
  background: #ffffff; border: 1px solid #dde5ee; border-radius: 12px;
  padding: 8px 14px 4px 14px;
}
.stButton > button[kind="primary"], div.stButton > button[type="primary"] {
  background: #0e7490; border: none; font-weight: 600;
}
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
.vf-page { border-bottom: 2px solid #d7e3ee; padding-bottom: 6px; margin-bottom: 10px; }
.vf-fusion {
  background:#ffffff; border:1px solid #cfe0ee; border-left:5px solid #0e7490;
  border-radius:10px; padding:12px 16px; margin:8px 0 12px 0;
}
.vf-fusion h4 { margin:0 0 6px 0; color:#0b3d66; }
.vf-badge { display:inline-block; border-radius:999px; padding:3px 10px;
  font-size:.82rem; font-weight:600; margin:2px 4px 2px 0; }
.vf-badge-ok { background:#e6f4ea; color:#137333; border:1px solid #b7dfc0; }
.vf-badge-warn { background:#fef7e0; color:#b06000; border:1px solid #f0d48a; }
.vf-badge-bad { background:#fce8e6; color:#c5221f; border:1px solid #f2b8b5; }
.vf-badge-info { background:#e8f0fe; color:#174ea6; border:1px solid #b6d0f7; }
.vf-emg {
  background:linear-gradient(180deg, #b00020 0%, #8f001a 100%);
  color:#ffffff; border:3px solid #6d0012;
  border-radius:12px; padding:16px 20px; margin:8px 0 12px 0;
  box-shadow:0 4px 14px rgba(143,0,26,.40); line-height:1.5;
}
.vf-emg h3 { color:#ffffff; margin:0 0 6px 0; font-size:1.2rem;
  letter-spacing:.3px; }
.vf-emg .vf-emg-lead { font-size:1.02rem; margin:2px 0 8px 0; }
.vf-emg .vf-score { background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28);
  border-radius:8px; padding:6px 12px; display:inline-block;
  margin:3px 6px 3px 0; font-size:.92rem; }
.vf-emg .vf-emg-help { margin-top:8px; font-size:.95rem; opacity:.95; }
/* emergency call-to-action (st.link_button ambulance action) */
a[data-testid="stLinkButton"], [data-testid="stLinkButton"] {
  background:#b00020; color:#ffffff !important; border:2px solid #6d0012;
  border-radius:10px; padding:10px 22px; font-weight:700; font-size:1rem;
  letter-spacing:.2px; box-shadow:0 3px 10px rgba(176,0,32,.35);
  text-decoration:none; display:inline-block; transition:background .15s ease;
}
a[data-testid="stLinkButton"]:hover, [data-testid="stLinkButton"]:hover {
  background:#8f001a; color:#ffffff !important;
}
/* safety-guidance callout */
.vf-guidance {
  background:#fff8e6; border:1px solid #f0d48a; border-left:5px solid #b06000;
  border-radius:10px; padding:12px 16px; color:#5b4200;
  font-size:.95rem; line-height:1.55; margin:4px 0 12px 0;
}
.vf-guidance b { color:#7a5800; }
.vf-guidance-teal {
  background:#e7f5f8; border:1px solid #bfe0e8; border-left:5px solid #0e7490;
  color:#1f4d5c;
}
.vf-guidance-teal b { color:#0b3d66; }
.vf-strip { background:#c62828; color:#fff; font-weight:600; border-radius:6px;
  padding:8px 12px; margin:4px 0 10px 0; }
.vf-strip-amber { background:#e65100; color:#fff; font-weight:600; border-radius:6px;
  padding:8px 12px; margin:4px 0 10px 0; }
.vf-strip-info { background:#0d47a1; color:#fff; font-weight:600; border-radius:6px;
  padding:8px 12px; margin:4px 0 10px 0; }
/* sidebar navigation (patient/doctor radio nav) */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  background: rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.22);
  border-radius:8px; padding:4px 10px; color:#ffffff; font-weight:500;
  margin-bottom:4px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] input:checked + label,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background: rgba(14,116,144,.65); border-color:#5fd0c8; color:#ffffff;
}
[data-testid="stSidebar"] [role="radiogroup"] { gap:6px; }
footer { visibility: hidden; }
/* ---------- login page ---------- */
div[data-testid="stTitle"] { text-align:center; color:#0b3d66; padding-top:26px; }
.vf-login-tag { text-align:center; color:#5b7083; font-size:.98rem;
  margin:-6px 0 14px 0; }
.vf-disclaimer { background:#eef4fa; border:1px solid #cfe0ee;
  border-left:5px solid #0e7490; border-radius:10px; padding:10px 14px;
  color:#33506b; font-size:.9rem; margin:0 0 14px 0; line-height:1.45; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px;
  box-shadow:0 6px 18px rgba(16,42,67,.10); border-color:#d7e3ee; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:10px; }
[data-testid="stTabs"] [data-baseweb="tab"] {
  border-radius:10px 10px 2px 2px; padding:6px 14px;
  font-weight:600; color:#33506b;
}
[data-testid="stTabs"] [aria-selected="true"] { color:#0e7490; }
div[data-testid="stForm"] { border-radius:12px; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    """Consistent page heading with a thin divider and optional subtitle."""
    st.markdown(f"<div class='vf-page'><h1 style='margin:0'>{title}</h1>"
                + (f"<div style='color:#5b7083;font-size:.92rem'>{subtitle}</div>"
                   if subtitle else "")
                + "</div>", unsafe_allow_html=True)


def fusion_panel(result: dict, assessment: Optional[dict] = None,
                 event: Optional[dict] = None, cfg: Optional[AppConfig] = None,
                 emergency_symptoms: bool = False) -> None:
    """Show a fused ECG+CT assessment result the same way everywhere.

    Always includes the experimental action recommendation and the
    emergency-symptom warning so no page can show a fused score without them.
    """
    threshold = cfg.high_risk_threshold if cfg is not None else 0.85
    if result.get("status") == "completed":
        thr_txt = f" · threshold {threshold:.2f}"
        head = (f"Fused assessment #{assessment['id']} saved"
                if assessment else "Fused assessment ready")
        st.markdown(
            f"<div class='vf-strip-info'>✓ {head} — ECG model + CT context "
            f"fusion{thr_txt}</div>",
            unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("ECG model (mean score)",
                  f"{result['ecg_score_mean']:.3f}"
                  if result.get("ecg_score_mean") is not None else "n/a")
        ct_level = result.get("ct_level")
        c2.metric("CT context level",
                  f"{ct_level} / 3" if ct_level is not None else "not reported")
        c3.metric("Fused risk score",
                  f"{result['fused_score']:.3f}")
        fused = result.get("fused") or {}
        if fused.get("explanation"):
            st.caption(fused["explanation"])
        if event is not None:
            st.error(f"HIGH-RISK event #{event['id']} created - the patient's "
                     "emergency banner and Call Ambulance are active, and the "
                     "doctor alert list has been notified.")
    else:
        st.error(f"Assessment status **{result.get('status')}**: "
                 f"{result.get('error') or 'no detail'}"
                 + (f" (saved as #{assessment['id']})" if assessment else ""))
        st.caption("A missing or failed assessment never means 'low risk'.")

    # ---- always show action recommendation + symptom warning ---------------
    show_action_recommendation(
        status=result.get("status", "failed"),
        ecg_score_mean=result.get("ecg_score_mean"),
        ct_level=result.get("ct_level"),
        fused_score=result.get("fused_score"),
        fused_level=(result.get("fused") or {}).get("level"),
        heart_rate_bpm=result.get("heart_rate_bpm"),
        high_risk_event_open=event is not None and event.get("status") in ("open", "acknowledged"),
        emergency_symptoms=emergency_symptoms,
        cfg=cfg,
    )


def show_action_recommendation(
    *,
    status: str,
    ecg_score_mean: Optional[float] = None,
    ct_level: Optional[int] = None,
    fused_score: Optional[float] = None,
    fused_level: Optional[int] = None,
    heart_rate_bpm: Optional[float] = None,
    high_risk_event_open: bool = False,
    emergency_symptoms: bool = False,
    cfg: Optional[AppConfig] = None,
) -> dict:
    """Render the action-recommendation card and return the structured result."""
    from fusion import FusionConfig

    thr = float(cfg.high_risk_threshold) if cfg is not None else 0.85
    fcfg = None
    if cfg is not None:
        try:
            fcfg = FusionConfig(**cfg.fusion_kwargs())
        except Exception:
            fcfg = None

    out = rec.recommend_action(
        status=status,
        ecg_score_mean=ecg_score_mean,
        ct_level=ct_level,
        fused_score=fused_score,
        fused_level=fused_level,
        heart_rate_bpm=heart_rate_bpm,
        high_risk_event_open=high_risk_event_open,
        emergency_symptoms=emergency_symptoms,
        cfg_high_risk_threshold=thr,
        fusion_cfg=fcfg,
    )

    css = rec.CATEGORY_CSS.get(out["category"], "vf-strip-info")
    st.markdown(
        f"<div class='{css}'>Recommended next step: <b>{out['label']}</b></div>",
        unsafe_allow_html=True)
    st.markdown(out["explanation"])
    with st.expander("Why this recommendation? (factors)", expanded=False):
        for f in out["factors"]:
            st.markdown(f"- {f}")
        st.caption(out["disclaimer"])
    st.markdown(
        f"<div class='vf-guidance'><b>Emergency warning</b><br/>"
        f"{out['symptom_warning']}</div>",
        unsafe_allow_html=True)
    return out


# ---------------------------------------------------------------------------
# connection + session plumbing
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_conn(db_path: str):
    """One SQLite connection per process (WAL, foreign keys on)."""
    return dbm.init_db(db_path)


def load_session(cfg: AppConfig) -> Optional[auth.AuthSession]:
    raw = st.session_state.get(auth.SESSION_KEY)
    if not raw:
        return None
    try:
        session = auth.AuthSession.from_dict(raw)
    except TypeError:
        st.session_state.pop(auth.SESSION_KEY, None)
        return None
    if auth.session_expired_reason(session):
        return session  # caller must check reason and force logout
    session = auth.session_state(session, cfg)
    st.session_state[auth.SESSION_KEY] = session.to_dict()
    return session


def force_logout(conn, session: Optional[auth.AuthSession]) -> None:
    auth.logout(conn, session)
    st.session_state.pop(auth.SESSION_KEY, None)
    for k in list(st.session_state):
        if k.startswith("vf_mon") or k.startswith("vf_lab"):
            st.session_state.pop(k, None)


def viewer_or_none(conn, session) -> Optional[dbm.Viewer]:
    try:
        return auth.make_viewer(conn, session)
    except dbm.DataAccessDenied:
        return None


def fmt_time(iso: Optional[str]) -> str:
    if not iso:
        return "-"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return str(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def demo_note(model_name: Optional[str]) -> str:
    """Non-empty when a record is synthetic test data (DEMO/TEST labelled)."""
    name = str(model_name or "")
    if name.startswith(("DEMO", "TEST")):
        return ("SYNTHETIC TEST RECORD - created for testing only, "
                "NOT a real model output.")
    return ""


def short_uid(uid: str) -> str:
    return (uid or "")[:8]


# ---------------------------------------------------------------------------
# Call Ambulance action (never auto-dials, never claims a connected call)
# ---------------------------------------------------------------------------
def render_call_ambulance(conn, viewer: dbm.Viewer, event: dict, cfg: AppConfig,
                          key: str, show_recorded: bool = True) -> None:
    """Emergency-call button using a configurable ``tel:`` link.

    Pressing it records the patient's action and opens whatever calling app
    the device provides. The app cannot verify that a call was connected, and
    it says so.
    """
    number = (cfg.emergency_service_number or "1122").strip()
    tel_url = f"tel:{number}"

    def _record(event_id=event["id"]):
        try:
            emergency.record_call_action(conn, viewer, event_id)
        except Exception:
            pass  # never block the dialer because of a logging problem

    st.link_button(
        f"\U0001F4DE Call Ambulance ({number})",
        tel_url,
        key=key,
        on_click=_record,
        type="primary",
    )
    if event.get("patient_call_initiated") and show_recorded:
        st.caption(
            f"Your call action was recorded at {fmt_time(event.get('patient_call_at'))}. "
            "This app only opens your phone's dialer - it cannot verify that a call "
            "was connected or that an ambulance was dispatched."
        )
    else:
        st.caption(
            "Opens your phone's calling app. A compatible phone or calling "
            "application is required; this app cannot verify the call outcome."
        )


def render_emergency_contact(conn, viewer: dbm.Viewer, patient_id: int) -> dict:
    contact = dbm.get_emergency_contact(conn, viewer, patient_id)
    st.markdown(f"**Emergency contact:** {contact['contact_name'] or 'not set'} "
                f"({contact['relationship'] or 'relationship not set'}) - "
                f"**{contact['contact_number'] or 'number not set'}**")
    st.markdown(f"**Emergency services:** {contact['emergency_service_number'] or '1122'}")
    if contact.get("location_text"):
        st.caption(f"Location shared by patient (voluntary): {contact['location_text']}")
    else:
        st.caption("No location provided by the patient. Location is NOT shared "
                   "automatically with any ambulance service.")
    if contact.get("accessibility_notes"):
        st.caption(f"Accessibility / assistance notes: {contact['accessibility_notes']}")
    return contact


SEEK_HELP_TEXT = (
    "**Seek emergency medical help now** if you have chest pain, pressure, "
    "shortness of breath, sweating, fainting or other concerning symptoms - "
    "regardless of any AI score shown here. This system cannot rule out a "
    "heart attack, and a low score does not mean you are safe."
)


# ---------------------------------------------------------------------------
# global banners (rendered on every page of both dashboards)
# ---------------------------------------------------------------------------
def render_patient_banner(conn, viewer: dbm.Viewer, cfg: AppConfig) -> None:
    """Red high-risk warning, plus a persistent strip for dismissed events."""
    active = emergency.open_events_for_patient(conn, viewer, viewer.id)
    others = [e for e in emergency.all_active_events(conn, viewer, viewer.id)
              if e["id"] not in {x["id"] for x in active}]

    for ev in active:
        _full_warning(conn, viewer, cfg, ev)

    if others:
        ev = others[0]
        st.markdown(
            f"<div class='vf-strip'>\U0001F6A8 High-risk event #{ev['id']} is still open "
            f"(awaiting doctor review). Emergency assistance stays available below.</div>",
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns([1, 1])
        number = (cfg.emergency_service_number or "1122").strip()
        with col_a:
            st.link_button(f"\U0001F4DE Call Ambulance ({number})", f"tel:{number}",
                           key=f"vf_strip_call_{ev['id']}",
                           on_click=lambda eid=ev["id"]: _safe_call(conn, viewer, eid),
                           type="primary")
        with col_b:
            contact = dbm.get_emergency_contact(conn, viewer, viewer.id)
            st.caption(f"Emergency contact: {contact['contact_name'] or '-'} "
                       f"{contact['contact_number'] or '-'}")


def _safe_call(conn, viewer, event_id) -> None:
    try:
        emergency.record_call_action(conn, viewer, event_id)
    except Exception:
        pass


def _full_warning(conn, viewer: dbm.Viewer, cfg: AppConfig, ev: dict) -> None:
    score = ev.get("score")
    hr = ev.get("heart_rate_bpm")
    threshold = ev.get("threshold")
    badges = ""
    if score is not None:
        badges += (f"<span class='vf-score'>Model output (fused): "
                   f"<b>{score:.3f}</b></span>")
    if threshold is not None:
        badges += (f"<span class='vf-score'>High-risk threshold: "
                   f"<b>{threshold:.2f}</b></span>")
    if hr is not None:
        badges += (f"<span class='vf-score'>Measured heart rate: "
                   f"<b>{hr:.0f} bpm</b></span>")
    else:
        badges += "<span class='vf-score'>Heart rate: <b>not available</b></span>"

    st.markdown(
        f"<div class='vf-emg'><h3>\u26A0\ufe0f EMERGENCY WARNING - HIGH-RISK "
        f"RESULT</h3>"
        f"<div class='vf-emg-lead'>An AI assessment for you reached the "
        f"high-risk threshold (event #{ev['id']}, "
        f"{fmt_time(ev.get('created_at'))}).</div>"
        f"<div>{badges}</div>"
        f"<div class='vf-emg-help'>{SEEK_HELP_TEXT.replace('**', '')}</div></div>",
        unsafe_allow_html=True,
    )
    contact = render_emergency_contact(conn, viewer, viewer.id)
    c1, c2 = st.columns([1, 1])
    with c1:
        render_call_ambulance(conn, viewer, ev, cfg, key=f"vf_call_{ev['id']}")
    with c2:
        st.caption(f"Configured emergency number: "
                   f"**{contact.get('emergency_service_number') or cfg.emergency_service_number}** "
                   "- changeable in your Emergency page.")
    if ev.get("status") == "acknowledged":
        st.info("Your doctor has acknowledged this alert and is reviewing it. "
                "Keep this warning visible until the situation is handled.")
    if st.button("I have read this warning (hide the banner)", key=f"vf_dismiss_{ev['id']}"):
        emergency.dismiss_warning(conn, viewer, ev["id"])
        st.rerun()


def render_doctor_banner(conn, viewer: dbm.Viewer, cfg: AppConfig,
                         go_to_alerts=None) -> None:
    """Red strip on the doctor side: open events among assigned patients."""
    if not viewer.is_doctor:
        return
    events = [e for e in dbm.list_events_for_doctor(conn, viewer.id, include_resolved=False)]
    open_events = [e for e in events if e["status"] == "open"]
    acked = [e for e in events if e["status"] == "acknowledged"]
    if not events:
        return
    msg = (f"\U0001F6A8 {len(open_events)} open high-risk event(s), "
           f"{len(acked)} acknowledged - review required.")
    css = "vf-strip" if open_events else "vf-strip-amber"
    st.markdown(f"<div class='{css}'>{msg}</div>", unsafe_allow_html=True)
    if go_to_alerts is not None:
        if st.button("Go to Alerts", key="vf_doctor_goto_alerts"):
            go_to_alerts()
            st.rerun()


def offline_badge(cfg: AppConfig) -> str:
    try:
        from config import offline
        return "OFFLINE mode (local features only)" if offline(cfg) else "Online"
    except Exception:
        return "unknown"


def render_sidebar_user(conn, session: auth.AuthSession, cfg: AppConfig) -> None:
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Signed in as **{session.full_name or session.email}** "
                       f"({session.role})")
    st.sidebar.caption(f"Session expires: {fmt_time(session.absolute_expiry)}")
    st.sidebar.caption(f"Connectivity: {offline_badge(cfg)}")
    if st.sidebar.button("Sign out"):
        force_logout(conn, session)
        st.rerun()


# ---------------------------------------------------------------------------
# combined CT + ECG + fusion + next-steps review card
# ---------------------------------------------------------------------------
def combined_review_card(conn, viewer: dbm.Viewer, pid: int,
                         cfg: Optional[AppConfig] = None,
                         emergency_symptoms: bool = False) -> None:
    """Single combined review: CT card, ECG card, fusion, recommended next steps.

    Shows the latest saved CT record, the latest ECG session summary, the
    latest fused assessment (if any), and the action recommendation built
    from those real outputs. Never invents missing data.
    """
    st.markdown("#### Combined CT + ECG review")

    ct = dbm.latest_ct_record(conn, viewer, pid)
    sessions = dbm.list_ecg_sessions(conn, viewer, pid, limit=1)
    assessments = dbm.list_assessments(conn, viewer, pid, limit=1)
    latest = assessments[0] if assessments else None
    events = [e for e in dbm.list_events_for_patient(conn, viewer, pid)
              if e.get("status") in ("open", "acknowledged")]

    # --- CT card ---
    st.markdown("**CT result**")
    if ct and ct.get("context_level") is not None:
        from ct_features import CONTEXT_LEVELS
        c1, c2, c3 = st.columns(3)
        c1.metric("CT context level", f"{ct['context_level']} / 3")
        c2.metric("Source", str(ct.get("source") or "—"))
        c3.metric("Recorded", fmt_time(ct.get("created_at")))
        st.caption(CONTEXT_LEVELS.get(ct["context_level"], ""))
        if ct.get("report_text"):
            st.caption(f"Note: {ct['report_text'][:200]}")
    else:
        st.info("No CT record on file. Add report values or an image review "
                "in the CT upload section.")

    # --- ECG card ---
    st.markdown("**ECG result**")
    if sessions:
        s = sessions[0]
        e1, e2, e3 = st.columns(3)
        e1.metric("Session", f"#{s['id']}")
        e2.metric("Samples", f"{s.get('n_samples') or 0}")
        e3.metric("Mean HR", f"{s['mean_hr_bpm']:.0f} bpm"
                  if s.get("mean_hr_bpm") else "n/a")
        st.caption(f"Status: {s.get('status')} · started {fmt_time(s.get('started_at'))}")
    else:
        st.info("No ECG recording on file. Record one on the ECG monitoring page.")

    # --- Fusion + assessment card ---
    st.markdown("**Fused assessment**")
    if latest and latest.get("status") == "completed":
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Status", "completed")
        f2.metric("Fused score",
                  f"{latest['fused_score']:.3f}"
                  if latest.get("fused_score") is not None else "n/a")
        f3.metric("ECG mean",
                  f"{latest['ecg_score_mean']:.3f}"
                  if latest.get("ecg_score_mean") is not None else "n/a")
        f4.metric("CT level used",
                  f"{latest['ct_level']} / 3"
                  if latest.get("ct_level") is not None else "not reported")
        st.caption(f"Saved assessment #{latest['id']} · "
                   f"{fmt_time(latest.get('created_at'))} · "
                   f"model: {latest.get('model_name') or '—'}")
    elif latest:
        st.warning(f"Latest assessment status: **{latest.get('status')}**"
                   + (f" — {latest['error_message']}" if latest.get("error_message") else "")
                   + ". A failed assessment never means low risk.")
    else:
        st.info("No fused assessment yet. Run one from ECG monitoring or the "
                "assessment lab.")

    # --- Action recommendation ---
    show_action_recommendation(
        status=(latest.get("status") if latest else "missing"),
        ecg_score_mean=(latest.get("ecg_score_mean") if latest else None),
        ct_level=(latest.get("ct_level") if latest else None),
        fused_score=(latest.get("fused_score") if latest else None),
        fused_level=(latest.get("fused_level") if latest else None),
        heart_rate_bpm=(latest.get("heart_rate_bpm") if latest else None),
        high_risk_event_open=bool(events),
        emergency_symptoms=emergency_symptoms,
        cfg=cfg,
    )
