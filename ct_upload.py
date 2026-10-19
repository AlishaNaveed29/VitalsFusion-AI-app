"""Shared CT upload UI - available to BOTH doctors and patients.

Two input options, same as the assessment lab:
  1. Report values (radiology report) -> source='report'
  2. Image (DICOM series) - experimental model if ``ct_image_model`` is
      installed, otherwise a manual image-review level -> source='image_model'
      / 'manual'

Access control is enforced by ``db.create_ct_record`` / ``db.list_ct_records``
(``require_patient_access``): a patient can only save on their own record,
a doctor only on an assigned patient. Denied attempts are shown as clear
errors and audited by the database layer.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

import assessment_core as core
import db as dbm
import emergency
import model_registry
from ct_features import CAC_CATEGORIES, CONTEXT_LEVELS, STENOSIS_GRADES, stenosis_grade_from_percent
from ui_common import fmt_time, fusion_panel

CT_MODE_REPORT = "1. Report values (radiology report)"
CT_MODE_IMAGE = "2. Image (DICOM series)"

SOURCE_LABELS = {
    "report": "Report values",
    "image_model": "AI image model",
    "manual": "Manual image review",
}

_LEVEL_HELP = (
    "CT context level used by the fused ECG + CT assessment: "
    "0 = low context, 1 = mildly raised, 2 = raised, 3 = high. "
    "Choose the level that matches the images you reviewed - this is not a diagnosis."
)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def ct_upload_ui(conn, viewer: dbm.Viewer, pid: int, *, key_prefix: str,
                 heading: str | None = None, show_saved: bool = True,
                 cfg=None) -> None:
    """Render the two-option CT upload area for ``pid``.

    All widget / session-state keys are scoped by ``key_prefix`` **and** the
    patient id so switching patients cannot leak a previous patient's
    prediction or form values into the next one.
    """
    if heading:
        st.markdown(heading)
    if cfg is None:
        from config import load_config
        cfg = load_config()
    # early access check (also enforced inside every db call below)
    try:
        dbm.list_ct_records(conn, viewer, pid, limit=1)
    except dbm.DataAccessDenied:
        st.error("Not authorized to view or edit CT records for this patient.")
        return

    kp = f"{key_prefix}_p{int(pid)}"

    st.caption("Option 1: values from the radiology report. "
               "Option 2: CT images (experimental model, or manual level).")
    mode = st.radio("CT input option", [CT_MODE_REPORT, CT_MODE_IMAGE],
                    horizontal=True, key=f"{kp}_mode")
    try:
        if mode == CT_MODE_REPORT:
            _report_form(conn, viewer, pid, kp)
        else:
            _image_section(conn, viewer, pid, kp, cfg)
    except dbm.DataAccessDenied as exc:
        st.error(f"Not authorized: {exc}")
    if show_saved:
        _saved_records(conn, viewer, pid, kp)


# ---------------------------------------------------------------------------
# option 1 - report values
# ---------------------------------------------------------------------------
def _report_form(conn, viewer: dbm.Viewer, pid: int, kp: str) -> None:
    with st.form(f"{kp}_report_form"):
        c1, c2, c3 = st.columns(3)
        cac_mode = c1.radio("Calcium", ["Agatston score", "Category only", "Not reported"],
                            key=f"{kp}_cac_mode")
        cac_score = cac_category = None
        if cac_mode == "Agatston score":
            cac_score = c1.number_input("Agatston score", min_value=0.0, value=0.0,
                                        step=1.0, key=f"{kp}_cac_score")
        elif cac_mode == "Category only":
            cac_category = c1.selectbox("CAC category", list(CAC_CATEGORIES),
                                        format_func=lambda k: CAC_CATEGORIES[k],
                                        key=f"{kp}_cac_cat")
        sten_mode = c2.radio("Stenosis", ["Grade", "Worst stenosis %", "Not reported"],
                             key=f"{kp}_sten_mode")
        stenosis_grade = None
        if sten_mode == "Grade":
            stenosis_grade = c2.selectbox("Stenosis grade", list(STENOSIS_GRADES),
                                          format_func=lambda k: STENOSIS_GRADES[k],
                                          key=f"{kp}_sten_grade")
        elif sten_mode == "Worst stenosis %":
            pct = c2.number_input("Worst stenosis (%)", 0.0, 100.0, 0.0, 1.0,
                                  key=f"{kp}_sten_pct")
            stenosis_grade = stenosis_grade_from_percent(pct)
        v_choice = c3.selectbox("Vessels >=50% stenosis", ["Not reported", 0, 1, 2, 3],
                                key=f"{kp}_vessels")
        vessels = None if v_choice == "Not reported" else int(v_choice)
        report_text = st.text_area("Report text", "", key=f"{kp}_report_text",
                                   max_chars=4000)
        if st.form_submit_button("Save CT record (report values)", type="primary"):
            try:
                level = core.ct_level_from_report(cac_score=cac_score,
                                                  cac_category=cac_category,
                                                  stenosis_grade=stenosis_grade,
                                                  vessels_ge50=vessels)
            except ValueError as exc:
                st.error(str(exc)); level = None
            if level is not None:
                rid = dbm.create_ct_record(conn, viewer, pid, cac_score=cac_score,
                                           cac_category=cac_category,
                                           stenosis_grade=stenosis_grade,
                                           vessels_ge50=vessels, context_level=level,
                                           report_text=report_text, source="report")
                st.success(f"CT record #{rid} saved - context level {level} "
                           f"({CONTEXT_LEVELS[level]}), source: report values.")


# ---------------------------------------------------------------------------
# option 2 - images (experimental model + manual level)
# ---------------------------------------------------------------------------
def _image_section(conn, viewer: dbm.Viewer, pid: int, kp: str, cfg) -> None:
    st.warning("Experimental. The optional image model (ct_image_model.py + "
               "torch/pydicom) estimates a severity quartile from DICOM slices; "
               "without it you can still record a manual image-review level. "
               "Model weights are loaded from the backend - never uploaded here.")
    try:
        import ct_image_model as cim  # type: ignore
        image_ready = True
    except ImportError:
        image_ready = False
        st.info("Image model packages not installed - use the manual level below.")

    if image_ready:
        _model_panel(conn, viewer, pid, kp, cim, cfg)

    _manual_panel(conn, viewer, pid, kp)
    _fusion_section(conn, viewer, pid, kp, cfg)


def _model_panel(conn, viewer: dbm.Viewer, pid: int, kp: str, cim, cfg) -> None:
    st.markdown("#### AI image model")
    with st.expander("About this model", expanded=False):
        st.markdown(
            "- **Architecture:** ResNet18 2.5D (3 adjacent slices as channels), "
            "4 severity classes (Low / Mild / Moderate / Severe -> levels 0-3).\n"
            "- **Weights:** loaded automatically from the backend path "
            "(`models/ResNet18_2.5D_best.pt`) - not uploaded through the UI.\n"
            "- **Input:** cardiac CT series as DICOM slices or a .zip of them.\n"
            "- **Output:** class probabilities; the top class becomes the CT "
            "context level saved on this patient's record.\n"
            "- Research prototype only - **not a diagnosis** and not a "
            "prediction of a heart attack.")

    model_ct, meta, msg, ok = model_registry.load_ct_image_model(cfg)
    if ok:
        st.caption(msg)
    else:
        st.warning(msg)

    # Stage indicator for the image workflow
    st.markdown(
        "<div class='vf-guidance vf-guidance-teal'><b>Workflow stages</b><br/>"
        "1. Upload CT series &nbsp;→&nbsp; 2. Preview slices &nbsp;→&nbsp; "
        "3. Run image model &nbsp;→&nbsp; 4. Save CT record &nbsp;→&nbsp; "
        "5. Run ECG + CT fusion below</div>",
        unsafe_allow_html=True)

    dicoms = st.file_uploader("DICOM slices (select all) or a .zip",
                              accept_multiple_files=True, key=f"{kp}_dicom")

    names, blobs, file_warnings = _validate_uploaded_files(dicoms)
    for w in file_warnings:
        st.warning(w)

    fingerprint = _upload_fingerprint(names, blobs, pid) if blobs else None
    state_key = f"{kp}_img_pred"
    _invalidate_stale_prediction(state_key, fingerprint)

    volume = _preview_series(cim, names, blobs, kp) if blobs else None

    run = st.button("Run image model", key=f"{kp}_img_run",
                    type="primary",
                    disabled=not (ok and blobs))
    if run:
        if not ok or not blobs:
            st.error("Model or upload unavailable - inference cancelled.")
        else:
            _infer_and_store_prediction(cim, model_ct, meta, names, blobs,
                                        volume, state_key, fingerprint, pid)

    # Always render whatever prediction is stored (survives Streamlit reruns)
    _render_stored_prediction(conn, viewer, pid, kp, state_key)


def _upload_fingerprint(names: Sequence[str], blobs: Sequence[bytes],
                        pid: Optional[int] = None) -> str:
    """Stable id for the current upload set + patient (invalidates stale results).

    Includes ``pid`` so a prediction for patient A can never be re-saved onto
    patient B after a selectbox switch.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(f"pid:{int(pid) if pid is not None else -1}".encode())
    for n, b in zip(names, blobs):
        h.update((n or "").encode("utf-8", errors="replace"))
        h.update(b)
    return h.hexdigest()[:32]


def _invalidate_stale_prediction(state_key: str, fingerprint: Optional[str]) -> None:
    """Drop a stored prediction when the upload changed or was cleared."""
    pred = st.session_state.get(state_key)
    if pred is None:
        return
    if fingerprint is None or pred.get("fingerprint") != fingerprint:
        st.session_state.pop(state_key, None)


def _validate_uploaded_files(files) -> tuple[list[str], list[bytes], list[str]]:
    """Basic validation of the multi-file uploader; returns names/blobs/warnings."""
    names: list[str] = []
    blobs: list[bytes] = []
    warnings: list[str] = []
    if not files:
        return names, blobs, warnings
    max_total = 500 * 1024 * 1024  # 500 MB combined upload cap
    total = 0
    for f in files:
        data = f.getvalue()
        if not data:
            warnings.append(f"'{f.name}' is empty and was skipped.")
            continue
        total += len(data)
        if total > max_total:
            warnings.append("Combined upload exceeds 500 MB - remaining files skipped.")
            break
        if f.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    members = [m for m in zf.namelist()
                               if not m.endswith("/") and "__macosx" not in m.lower()]
                    if not members:
                        warnings.append(f"'{f.name}' contains no files.")
                        continue
            except zipfile.BadZipFile:
                warnings.append(f"'{f.name}' is not a valid zip file.")
                continue
        names.append(f.name)
        blobs.append(data)
    if blobs:
        mb = sum(len(b) for b in blobs) / (1024 * 1024)
        st.caption(f"{len(blobs)} file(s), {mb:.1f} MB total. "
                   "Non-DICOM or empty files are skipped with a warning above.")
    return names, blobs, warnings


@st.cache_data(show_spinner=False)
def _read_volume_cached(names: tuple, blobs: tuple) -> np.ndarray:
    import ct_image_model as cim
    return cim.read_dicom_volume(list(zip(names, blobs)))


def _preview_series(cim, names: list[str], blobs: list[bytes], kp: str):
    """Read the series once (cached) and show slice previews + basic checks."""
    try:
        with st.spinner("Reading DICOM series..."):
            volume = _read_volume_cached(tuple(names), tuple(blobs))
    except Exception as exc:
        st.error(f"Could not read DICOM series: {exc}")
        return None

    n = int(volume.shape[0])
    st.markdown(f"**Series loaded:** {n} slices, "
                f"{volume.shape[1]}×{volume.shape[2]} px")
    if n < 8:
        st.error("Series has fewer than 8 slices - too short for a meaningful "
                 "prediction. Check that all slices were selected.")
    elif n < 32:
        st.warning(f"Series has only {n} slices (model training used slabs of "
                   "32); the prediction may be less reliable.")

    c1, c2, c3 = st.columns(3)
    mid = n // 2
    for col, idx, label in ((c1, 0, "First slice"),
                            (c2, mid, "Middle slice"),
                            (c3, n - 1, "Last slice")):
        with col:
            st.image(_slice_to_uint8(volume[idx]), caption=f"{label} ({idx + 1}/{n})",
                     use_container_width=True)
    pick = st.slider("Preview slice", 0, n - 1, mid, key=f"{kp}_preview_slice")
    st.image(_slice_to_uint8(volume[pick]),
             caption=f"Slice {pick + 1} of {n}", use_container_width=True)
    return volume


def _slice_to_uint8(sl: np.ndarray) -> np.ndarray:
    """HU slice -> uint8 for st.image (windowed to soft-tissue range)."""
    lo, hi = -100.0, 400.0
    x = np.clip(sl.astype(np.float64), lo, hi)
    x = (x - lo) / (hi - lo) * 255.0
    return x.astype(np.uint8)


def _infer_and_store_prediction(cim, model_ct, meta, names, blobs, volume,
                                state_key: str, fingerprint: str,
                                pid: Optional[int] = None) -> None:
    """Run inference once and persist the result in session state."""
    if volume is None:
        try:
            with st.spinner("Reading DICOM series..."):
                volume = _read_volume_cached(tuple(names), tuple(blobs))
        except Exception as exc:
            st.error(f"Could not read DICOM series: {exc}")
            return
    if volume.shape[0] < 8:
        st.error("Series too short - inference cancelled. Fix the upload above.")
        return

    stage = st.empty()
    try:
        stage.info("Running the image model "
                   "(weights already loaded from the backend)...")
        probs = np.asarray(cim.predict_probs(model_ct, volume), dtype=float)
        if probs.ndim != 1 or not np.all(np.isfinite(probs)):
            raise ValueError("invalid probability vector from the model")
        pred = int(np.argmax(probs))
    except Exception as exc:
        stage.empty()
        st.error(f"Image model failed: {exc}")
        return
    stage.empty()

    class_names = tuple((meta or {}).get("class_names") or
                        ("Low", "Mild", "Moderate", "Severe"))
    st.session_state[state_key] = {
        "fingerprint": fingerprint,
        "patient_id": int(pid) if pid is not None else None,
        "probs": [float(p) for p in probs],
        "pred": pred,
        "class_names": list(class_names),
        "n_slices": int(volume.shape[0]),
        "model_name": (meta or {}).get("model_name", "ResNet18_2.5D"),
        "hu_clip": list((meta or {}).get("hu_clip") or [-100.0, 1000.0]),
        "center_crop_frac": float((meta or {}).get("center_crop_frac") or 0.75),
        "saved_record_id": None,
    }


def _render_stored_prediction(conn, viewer, pid, kp: str, state_key: str) -> None:
    """Display the stored prediction and handle its one-time save."""
    st_pred = st.session_state.get(state_key)
    if not st_pred:
        return
    probs = np.asarray(st_pred["probs"], dtype=float)
    pred = int(st_pred["pred"])
    class_names = tuple(st_pred.get("class_names") or CLASS_NAMES_FALLBACK)
    meta_view = {
        "model_name": st_pred.get("model_name"),
        "class_names": class_names,
        "hu_clip": tuple(st_pred.get("hu_clip") or (-100.0, 1000.0)),
        "center_crop_frac": st_pred.get("center_crop_frac"),
    }
    _render_ai_results(meta_view, probs, pred, st_pred.get("n_slices", 0))

    saved_id = st_pred.get("saved_record_id")
    if saved_id:
        st.success(f"This prediction is already saved as CT record #{saved_id} "
                   "(source: AI image model). Upload new images to run again.")
        return
    if st.button("Save AI result as CT record", key=f"{kp}_img_save_result",
                 type="primary"):
        try:
            rid = _save_ai_prediction(conn, viewer, pid, st_pred)
        except dbm.DataAccessDenied as exc:
            st.error(f"Not authorized: {exc}")
            return
        except ValueError as exc:
            st.error(str(exc))
            return
        st_pred["saved_record_id"] = rid
        st.session_state[state_key] = st_pred
        st.success(f"CT record #{rid} saved - context level {pred} "
                   f"({class_names[pred]}), source: AI image model "
                   f"(p={probs[pred]:.3f}).")
        st.rerun()


def _save_ai_prediction(conn, viewer, pid: int, st_pred: dict) -> int:
    """Persist a stored prediction exactly once. Returns the new record id."""
    if st_pred.get("saved_record_id"):
        raise ValueError("This prediction was already saved.")
    stored_pid = st_pred.get("patient_id")
    if stored_pid is not None and int(stored_pid) != int(pid):
        raise ValueError(
            "Stored prediction belongs to a different patient - not saving. "
            "Upload the CT series again for this patient.")
    probs = st_pred.get("probs") or []
    if len(probs) < 2 or not all(np.isfinite(probs)):
        raise ValueError("Stored prediction is invalid - not saving.")
    pred = int(st_pred.get("pred", -1))
    class_names = tuple(st_pred.get("class_names") or CLASS_NAMES_FALLBACK)
    if pred < 0 or pred >= len(class_names):
        raise ValueError("Stored class index is out of range - not saving.")
    return dbm.create_ct_record(
        conn, viewer, pid, context_level=pred, source="image_model",
        report_text=f"image model class {pred} ({class_names[pred]}), "
                    f"p={probs[pred]:.3f}, {st_pred.get('n_slices', '?')} slices, "
                    f"model={st_pred.get('model_name', '?')}")


CLASS_NAMES_FALLBACK = ("Low", "Mild", "Moderate", "Severe")


def _render_ai_results(meta: dict, probs, pred: int, n_slices: int) -> None:
    """Show the actual prediction as a result card with probability bars."""
    class_names = tuple(meta.get("class_names") or
                        ("Low", "Mild", "Moderate", "Severe"))
    probs = np.asarray(probs, dtype=float)
    st.markdown("#### AI result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted severity", class_names[pred],
              help="Top class of the 4-class severity head.")
    c2.metric("CT context level", f"{pred} / 3",
              help="Saved as the CT branch input of the fused assessment.")
    c3.metric("Confidence (top class)", f"{probs[pred]:.2f}",
              help="Model softmax probability for the predicted class.")

    # Coloured horizontal probability bars (no external chart library)
    st.markdown("**Class probabilities**")
    bar_colors = ["#2e7d32", "#f9a825", "#ef6c00", "#c62828"]
    for i, name in enumerate(class_names):
        p = float(probs[i]) if i < len(probs) else 0.0
        pct = max(0.0, min(1.0, p)) * 100.0
        color = bar_colors[i] if i < len(bar_colors) else "#0e7490"
        highlight = " ← predicted" if i == pred else ""
        st.markdown(
            f"<div style='margin:2px 0 6px 0'>"
            f"<span style='font-size:.9rem'>{name}{highlight}</span>"
            f"<div style='background:#e8eef4;border-radius:6px;height:18px;"
            f"overflow:hidden'>"
            f"<div style='width:{pct:.1f}%;background:{color};height:100%;"
            f"border-radius:6px'></div></div>"
            f"<span style='font-size:.8rem;color:#5b7083'>{pct:.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True)

    if probs[pred] < 0.5:
        st.info("Low confidence: the top class probability is below 50%. "
                "Interpret this prediction cautiously.")

    with st.expander("Model information", expanded=False):
        st.markdown(
            f"- **Model:** {meta.get('model_name', 'ResNet18_2.5D')}\n"
            f"- **Classes:** {', '.join(class_names)}\n"
            f"- **HU window:** {meta.get('hu_clip')}\n"
            f"- **Centre crop:** {meta.get('center_crop_frac')}\n"
            f"- **Series:** {n_slices} slices")

    st.markdown(
        "<div class='vf-guidance'><b>Warnings</b><br/>"
        "This is an experimental research model, not a diagnosis. "
        "A low predicted class never rules out coronary disease; "
        "always interpret it together with the radiology report and a clinician. "
        "Predictions on incomplete or non-cardiac series are unreliable.</div>",
        unsafe_allow_html=True)


def _manual_panel(conn, viewer: dbm.Viewer, pid: int, kp: str) -> None:
    st.markdown("#### Manual image-review level")
    st.caption("Open the images in your viewer, judge the overall severity, "
               "then save the matching context level. Nothing is uploaded - "
               "records stay on this device.")
    level = st.select_slider("CT context level read from the images",
                             options=[0, 1, 2, 3], value=0, key=f"{kp}_img_manual",
                             format_func=lambda k: CONTEXT_LEVELS[k],
                             help=_LEVEL_HELP)
    note = st.text_area("Image review note", "", key=f"{kp}_img_note",
                        max_chars=1000,
                        help="Optional: what you saw (calcification, stenosis "
                             "locations, limitations of the series).")
    if note:
        st.caption(f"{len(note)}/1000 characters")
    if st.button("Save manual image-review CT level", key=f"{kp}_img_save",
                 type="primary"):
        note_clean = (note or "").strip()
        if len(note_clean) > 1000:
            st.error("Review note is too long (max 1000 characters).")
            return
        try:
            rid = dbm.create_ct_record(conn, viewer, pid, context_level=int(level),
                                       source="manual",
                                       report_text=(note_clean or
                                                    "entered manually from image review"))
        except dbm.DataAccessDenied as exc:
            st.error(f"Not authorized: {exc}")
            return
        st.success(f"CT record #{rid} saved - {CONTEXT_LEVELS[int(level)]}, "
                   "source: manual image review"
                   + (f" (note: {len(note_clean)} characters)." if note_clean else "."))
        st.rerun()


# ---------------------------------------------------------------------------
# ECG + CT fusion (real pipeline: trained ECG model + CT level + fusion layer)
# ---------------------------------------------------------------------------
def _fusion_section(conn, viewer: dbm.Viewer, pid: int, kp: str, cfg) -> None:
    """Fuse the patient's latest saved ECG recording with the latest CT level.

    Fusion contract (Case A - the only trained path in this repository):
    the ECG CNN-LSTM produces per-window MI-pattern scores; a scalar CT
    context level (0..3) from report / image-model / manual records shifts
    the alert thresholds in ``fusion.py``. There is no learned fusion
    checkpoint; a multilateral embedding model would require new training
    data and weights that are not present here.
    """
    st.markdown("#### ECG + CT fusion")
    st.caption(
        "Uses your most recent saved ECG recording (from ECG monitoring) with "
        "the latest CT context level on file. The trained ECG model scores each "
        "10-s window; the mean is combined with the CT level by the existing "
        "fusion layer. Missing ECG data is reported honestly - no score is invented.")

    ct = dbm.latest_ct_record(conn, viewer, pid)
    if not ct or ct.get("context_level") is None:
        st.info("No CT context level saved yet - add one with the options above first.")
        return
    st.markdown(
        f"**CT context:** level {ct['context_level']} "
        f"({CONTEXT_LEVELS.get(ct['context_level'], '?')}) - "
        f"record #{ct['id']}, {SOURCE_LABELS.get(ct.get('source'), ct.get('source'))}")

    sessions = dbm.list_ecg_sessions(conn, viewer, pid, limit=1)
    if not sessions:
        st.warning("No ECG recording found. Record one on the ECG monitoring "
                   "page first, then run fusion here.")
        return
    sess_meta = sessions[0]
    st.markdown(f"**Latest ECG:** session #{sess_meta['id']} "
                f"({fmt_time(sess_meta['started_at'])}, "
                f"{sess_meta['n_samples'] or 0} samples @ {sess_meta['fs_hz'] or '?'} Hz, "
                f"status: {sess_meta['status']})")

    if not st.button("Run fused ECG + CT assessment", key=f"{kp}_fuse_run",
                     type="primary"):
        return

    sid = int(sess_meta["id"])
    try:
        sess = dbm.get_ecg_session(conn, viewer, sid)
    except dbm.DataAccessDenied as exc:
        st.error(f"Not authorized: {exc}")
        return
    sig = sess.get("signal_array")
    if sig is None or getattr(sig, "size", 0) == 0:
        st.error("The latest ECG recording has no samples stored - fusion not run.")
        return

    model, msg, ok = core.prepare_model(cfg)
    if not ok:
        st.error(msg)
        return
    st.info(msg)

    ct_level = int(ct["context_level"])
    result = core.run_recording(model, sig, sess.get("fs_hz"), cfg, ct_level)
    assessment, event = emergency.process_assessment(
        conn, viewer, cfg, pid,
        **core.assessment_kwargs(result, ecg_session_id=sid,
                                 ct_record_id=int(ct["id"])))
    fusion_panel(result, assessment, event, cfg)


# ---------------------------------------------------------------------------
# saved records: filters, search, readable table, details
# ---------------------------------------------------------------------------
def filter_ct_records(records: Sequence[dict],
                      sources: Optional[Sequence[str]] = None,
                      date_from: Optional[date] = None,
                      date_to: Optional[date] = None,
                      query: str = "") -> list[dict]:
    """Pure filter used by the UI (also unit-tested)."""
    q = (query or "").strip().lower()
    out = []
    for r in records:
        if sources and r.get("source") not in sources:
            continue
        created = (r.get("created_at") or "")[:10]
        if date_from and created and created < date_from.isoformat():
            continue
        if date_to and created and created > date_to.isoformat():
            continue
        if q:
            hay = " ".join(str(r.get(k, "")) for k in
                           ("id", "source", "context_level", "report_text",
                            "cac_category", "stenosis_grade", "vessels_ge50",
                            "cac_score")).lower()
            if q not in hay:
                continue
        out.append(r)
    return out


def format_ct_row(r: dict) -> dict:
    """Table row with readable placeholders for missing values."""
    def fmt(v, suffix=""):
        return "—" if v is None or v == "" else f"{v}{suffix}"
    return {
        "id": r["id"],
        "when": fmt_time(r.get("created_at")),
        "source": SOURCE_LABELS.get(r.get("source"), fmt(r.get("source"))),
        "CAC score": fmt(r.get("cac_score")),
        "CAC category": fmt(r.get("cac_category")),
        "stenosis grade": fmt(r.get("stenosis_grade")),
        "vessels >=50%": fmt(r.get("vessels_ge50")),
        "CT level": fmt(r.get("context_level")),
        "by (user id)": fmt(r.get("recorded_by")),
    }


def _saved_records(conn, viewer: dbm.Viewer, pid: int, kp: str) -> None:
    st.markdown("### Saved CT records")
    try:
        recs = dbm.list_ct_records(conn, viewer, pid, limit=100)
    except dbm.DataAccessDenied as exc:
        st.error(f"Not authorized: {exc}")
        return
    if not recs:
        st.caption("No CT records yet - add one with the options above.")
        return

    f1, f2, f3 = st.columns([2, 2, 2])
    sources_pick = f1.multiselect(
        "Source", list(SOURCE_LABELS), default=list(SOURCE_LABELS),
        format_func=lambda s: SOURCE_LABELS[s], key=f"{kp}_flt_src")
    dcol1, dcol2 = f2.columns(2)
    d_from = dcol1.date_input("From", value=None, key=f"{kp}_flt_from")
    d_to = dcol2.date_input("To", value=None, key=f"{kp}_flt_to")
    query = f3.text_input("Search", "", key=f"{kp}_flt_q",
                          help="Matches id, level, source or report text.")

    filtered = filter_ct_records(
        recs, sources=sources_pick,
        date_from=d_from if isinstance(d_from, date) and d_from else None,
        date_to=d_to if isinstance(d_to, date) and d_to else None,
        query=query)
    st.caption(f"{len(filtered)} of {len(recs)} record(s) shown. "
               "'—' means not reported for that field.")
    if filtered:
        st.dataframe(pd.DataFrame([format_ct_row(r) for r in filtered]),
                     hide_index=True, use_container_width=True)
        with st.expander("Record details"):
            pick_id = st.selectbox("Record", filtered,
                                   format_func=lambda r: f"#{r['id']} - "
                                   f"{fmt_time(r.get('created_at'))} - "
                                   f"{SOURCE_LABELS.get(r.get('source'), r.get('source'))}",
                                   key=f"{kp}_detail")
            if pick_id:
                for k, v in format_ct_row(pick_id).items():
                    st.markdown(f"**{k}:** {v}")
                st.markdown(f"**Report / note text:** {pick_id.get('report_text') or '—'}")
    else:
        st.info("No records match the current filters.")


def _saved_table(conn, viewer: dbm.Viewer, pid: int, kp: str) -> None:
    """Backward-compatible alias (older callers)."""
    _saved_records(conn, viewer, pid, kp)
