"""Doctor assessment lab: the original VitalsFusion AI prototype, preserved.

Tabs:
    1. Final fusion : ECG CNN-LSTM score per window + CT context level
    2. CT upload    : choose ONE of two options for the CT context level -
                      **1. Report values** (radiology report fields) or
                      **2. Image** (DICOM series with the experimental model)

New in the two-role app: everything can be saved to the selected patient's
record (CT records + AI assessments), and a fused score at/above the
configured threshold raises the high-risk emergency workflow.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import streamlit as st

import assessment_core as core
import db as dbm
import emergency
from ct_features import (
    CAC_CATEGORIES,
    CONTEXT_LEVELS,
    STENOSIS_GRADES,
    cac_category_from_score,
    encode_ct_report,
    stenosis_grade_from_percent,
    to_report_dict,
)
from config import AppConfig
from ecg_model import DEFAULT_MODEL_FS, analyze_signal, synthetic_ecg
from fusion import DISCLAIMER, fuse
from ui_common import fmt_time

CT_MODE_REPORT = "1. Report values (radiology report)"
CT_MODE_IMAGE = "2. Image (DICOM series)"


def read_signal_file(name: str, data: bytes) -> pd.DataFrame:
    """CSV / TXT / NPY -> DataFrame of numeric columns (one column = one lead)."""
    if name.lower().endswith(".npy"):
        arr = np.asarray(np.load(io.BytesIO(data), allow_pickle=False), dtype=float)
        if arr.ndim == 1:
            return pd.DataFrame({"signal": arr})
        if arr.ndim == 2:
            if arr.shape[0] <= 16 < arr.shape[1]:
                arr = arr.T
            return pd.DataFrame(arr, columns=[f"col{i}" for i in range(arr.shape[1])])
        raise ValueError("The .npy file must be 1-D or 2-D.")
    df = pd.read_csv(io.BytesIO(data), header=None, sep=None, engine="python")
    first = pd.to_numeric(df.iloc[0], errors="coerce")
    if first.isna().any():
        df = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError("No numeric columns found in the file.")
    df.columns = [str(c) for c in df.columns]
    return df


def _patient_label(p: dict) -> str:
    return f"{p.get('full_name') or '-'} <{p['email']}> (id {p['id']})"


# ---------------------------------------------------------------------------
# CT upload area - option 1: report values
# ---------------------------------------------------------------------------
def _ct_report_section(conn, viewer: dbm.Viewer, pid: int) -> None:
    st.header("Option 1 - Radiology report values")
    st.caption("The CT branch uses values taken from the radiology report. "
               "It does not analyse images here.")
    col_a, col_b, col_c = st.columns(3)
    cac_score = cac_category = stenosis_grade = vessels = None

    with col_a:
        st.subheader("Calcium (CAC)")
        cac_mode = st.radio("How is it reported?",
                            ["Agatston score", "Category only", "Not reported"],
                            key="lab_cac_mode")
        if cac_mode == "Agatston score":
            cac_score = st.number_input("Agatston score", min_value=0.0, value=0.0,
                                        step=1.0, key="lab_cac_score")
            st.caption(f"Category: {CAC_CATEGORIES[cac_category_from_score(cac_score)]}")
        elif cac_mode == "Category only":
            cac_category = st.selectbox(
                "CAC category", list(CAC_CATEGORIES.keys()),
                format_func=lambda k: CAC_CATEGORIES[k], key="lab_cac_cat")

    with col_b:
        st.subheader("Stenosis")
        sten_mode = st.radio("How is it reported?",
                             ["Grade", "Worst stenosis %", "Not reported"],
                             key="lab_sten_mode")
        if sten_mode == "Grade":
            stenosis_grade = st.selectbox(
                "Stenosis grade", list(STENOSIS_GRADES.keys()),
                format_func=lambda k: STENOSIS_GRADES[k], key="lab_sten_grade")
        elif sten_mode == "Worst stenosis %":
            pct = st.number_input("Worst stenosis (%)", 0.0, 100.0, 0.0, 1.0,
                                  key="lab_sten_pct")
            stenosis_grade = stenosis_grade_from_percent(pct)
            st.caption(f"Grade: {STENOSIS_GRADES[stenosis_grade]}")

    with col_c:
        st.subheader("Vessels")
        v_choice = st.selectbox("Vessels with 50% or more stenosis",
                                ["Not reported", 0, 1, 2, 3], key="lab_vessels")
        vessels = None if v_choice == "Not reported" else int(v_choice)

    report_text = st.text_area("Radiology report text (optional)", "",
                               key="lab_report_text")

    st.header("Encoded CT features")
    feats, input_error = None, None
    try:
        feats = encode_ct_report(cac_score=cac_score, cac_category=cac_category,
                                 stenosis_grade=stenosis_grade, vessels_ge50=vessels)
    except ValueError as e:
        input_error = str(e)

    st.session_state["vf_lab_ct_level"] = None
    if input_error:
        st.error(input_error)
    elif feats.context_level is None:
        st.info("Nothing reported yet. Enter at least one value to get a CT "
                "context level.")
    else:
        st.session_state["vf_lab_ct_level"] = int(feats.context_level)
        m1, m2, m3 = st.columns(3)
        m1.metric("CT context level", f"{feats.context_level} / 3")
        m2.metric("CAC category",
                  "-" if feats.cac_category is None else str(feats.cac_category))
        m3.metric("Stenosis grade",
                  "-" if feats.stenosis_grade is None else str(feats.stenosis_grade))
        st.write(f"**{CONTEXT_LEVELS[feats.context_level]}**. The highest reported "
                 "item sets the level. This level is passed to the Final fusion tab.")

        table = pd.DataFrame(
            [["cac_cat_norm", feats.cac_cat_norm, feats.missing_cac],
             ["cac_score_log_norm", feats.cac_score_log_norm, feats.missing_cac],
             ["stenosis_norm", feats.stenosis_norm, feats.missing_stenosis],
             ["vessels_norm", feats.vessels_norm, feats.missing_vessels]],
            columns=["feature (0 to 1)", "value", "missing flag"],
        )
        table["value"] = table["value"].apply(
            lambda vv: "not reported" if vv is None else f"{vv:.3f}")
        table["missing flag"] = table["missing flag"].astype(int)
        st.dataframe(table, hide_index=True)

        report = to_report_dict(feats)
        json_text = json.dumps(report, indent=2)
        b1, b2 = st.columns(2)
        b1.download_button("Download CT features (JSON)", json_text,
                           file_name=f"patient{pid}_ct_features.json",
                           mime="application/json")
        if b2.button("Save CT record to patient file"):
            dbm.create_ct_record(
                conn, viewer, pid,
                cac_score=cac_score, cac_category=cac_category,
                stenosis_grade=stenosis_grade, vessels_ge50=vessels,
                context_level=feats.context_level,
                report_text=report_text or "", source="report",
            )
            st.success("CT record saved to the patient's file.")
        with st.expander("Show JSON"):
            st.code(json_text, language="json")


# ---------------------------------------------------------------------------
# CT upload area - option 2: image (experimental model)
# ---------------------------------------------------------------------------
def _ct_image_section(conn, viewer: dbm.Viewer, pid: int) -> None:
    st.header("Option 2 - CT image (DICOM series)")
    st.warning(
        "Experimental, research only. The model's test Macro-F1 was only about "
        "0.5, and its classes are quartiles of the training set's calcium score, "
        "not Agatston categories. The main design uses report values."
    )
    try:
        import ct_image_model as cim  # type: ignore
        image_ready = True
    except ImportError:
        image_ready = False
        st.info("Image mode needs extra packages. Run: pip install torch torchvision "
                "pydicom, provide ct_image_model.py, and restart the app. "
                "Without them you can still enter a severity level manually below.")

    if image_ready:
        import model_registry as mr
        from config import load_config
        cfg_lab = load_config()
        model_ct, meta_lab, msg_lab, ok_lab = mr.load_ct_image_model(cfg_lab)
        if ok_lab:
            st.caption(msg_lab + " Weights are loaded from the backend - "
                       "never uploaded through the UI.")
        else:
            st.warning(msg_lab)

        dicoms = st.file_uploader("One patient's DICOM slices (select all) or a .zip",
                                  accept_multiple_files=True, key="lab_dicom")

        # Lightweight preview when files are present (reuse ct_upload helpers)
        volume_preview = None
        if dicoms:
            try:
                from ct_upload import _slice_to_uint8
                with st.spinner("Reading DICOM series for preview..."):
                    vol = cim.read_dicom_volume([(f.name, f.getvalue())
                                                 for f in dicoms])
                volume_preview = vol
                n = int(vol.shape[0])
                st.markdown(f"**Series loaded:** {n} slices, "
                            f"{vol.shape[1]}×{vol.shape[2]} px")
                if n < 8:
                    st.error("Series has fewer than 8 slices - too short for a "
                             "meaningful prediction.")
                mid = n // 2
                pc1, pc2, pc3 = st.columns(3)
                with pc1:
                    st.image(_slice_to_uint8(vol[0]), caption="First slice",
                             use_container_width=True)
                with pc2:
                    st.image(_slice_to_uint8(vol[mid]), caption=f"Middle ({mid+1}/{n})",
                             use_container_width=True)
                with pc3:
                    st.image(_slice_to_uint8(vol[n - 1]), caption="Last slice",
                             use_container_width=True)
            except Exception as e:
                st.warning(f"Preview unavailable: {e}")

        if ok_lab and dicoms and st.button("Run trained model", type="primary"):
            try:
                with st.spinner("Reading CT series..."):
                    volume = (volume_preview if volume_preview is not None
                              else cim.read_dicom_volume([(f.name, f.getvalue())
                                                          for f in dicoms]))
                with st.spinner("Predicting..."):
                    probs = cim.predict_probs(model_ct, volume)
                st.session_state["vf_lab_img_result"] = {
                    "probs": probs.tolist(), "n_slices": int(volume.shape[0]),
                    "class_names": list((meta_lab or {}).get("class_names")
                                        or cim.CLASS_NAMES)}
            except Exception as e:
                st.session_state.pop("vf_lab_img_result", None)
                st.session_state["vf_lab_ct_image_level"] = None
                st.error(str(e))

        res = st.session_state.get("vf_lab_img_result")
        if res:
            probs = res["probs"]
            class_names = tuple(res.get("class_names") or cim.CLASS_NAMES)
            pred = max(range(len(probs)), key=lambda i: probs[i])
            st.session_state["vf_lab_ct_image_level"] = pred
            st.write(f"Used {res['n_slices']} slices.")
            r1, r2 = st.columns(2)
            r1.metric("Model severity class", class_names[pred],
                      f"{probs[pred] * 100:.1f}% confidence")
            r2.metric("As CT context level", f"{pred} / 3")
            st.bar_chart(pd.DataFrame({"class": list(class_names),
                                       "probability": probs}).set_index("class"))
            if probs[pred] < 0.5:
                st.info("Low confidence: the top class is below 50%.")
            if st.button("Save image-model CT level to patient file"):
                dbm.create_ct_record(conn, viewer, pid, context_level=int(pred),
                                     source="image_model",
                                     report_text=f"image model class {pred}, "
                                                 f"p={probs[pred]:.3f}")
                st.success("Saved.")
            st.caption("The class index (0 to 3) is used directly as the context "
                       "level. Because the classes are training-set quartiles, "
                       "this is a rough severity ranking, not a clinical category.")

    st.markdown("#### No image model? Enter the level manually")
    st.caption("Pick the CT context level you read from the images; it is stored "
               "as a 'manual' CT record and passed to the Final fusion tab.")
    manual_level = st.select_slider("CT context level from image review",
                                    options=[0, 1, 2, 3], value=0,
                                    key="lab_img_manual_level")
    st.session_state["vf_lab_ct_image_level"] = int(manual_level)
    if st.button("Save manual image-review CT level", key="lab_img_manual_save"):
        dbm.create_ct_record(conn, viewer, pid, context_level=int(manual_level),
                             source="manual",
                             report_text="CT context level entered manually from "
                                         "image review (no model run)")
        st.success(f"CT record saved (context level {manual_level}, source=manual).")


def _ct_upload_section(conn, viewer: dbm.Viewer, pid: int) -> None:
    """The CT upload area with its two input options: report values or image."""
    st.header("CT upload - choose an input option")
    mode = st.radio("CT input option", [CT_MODE_REPORT, CT_MODE_IMAGE],
                    horizontal=True, key="lab_ct_input_mode",
                    help="Option 1: values from the radiology report. "
                         "Option 2: DICOM images (experimental model or manual level).")
    if mode == CT_MODE_REPORT:
        _ct_report_section(conn, viewer, pid)
    else:
        _ct_image_section(conn, viewer, pid)

    st.subheader("Saved CT records (both options)")
    recs = dbm.list_ct_records(conn, viewer, pid, limit=20)
    if recs:
        df = pd.DataFrame([{
            "id": r["id"], "saved": fmt_time(r["created_at"]), "source": r["source"],
            "CAC score": r["cac_score"], "CAC cat": r["cac_category"],
            "stenosis grade": r["stenosis_grade"], "vessels>=50%": r["vessels_ge50"],
            "CT level": r["context_level"],
        } for r in recs])
        st.dataframe(df, hide_index=True)
    else:
        st.caption("No CT records saved for this patient yet.")


# ---------------------------------------------------------------------------
# main lab
# ---------------------------------------------------------------------------
def render(conn, viewer: dbm.Viewer, cfg: AppConfig) -> None:
    st.subheader("Assessment lab (ECG + CT fusion)")
    st.caption("Research prototype - run and review AI assessments for your "
               "assigned patients. Results are saved to the patient record only "
               "when you press Save.")

    patients = dbm.list_assigned_patients(conn, viewer.id)
    if not patients:
        st.info("No patients are assigned to you yet. Assign one from 'My patients'.")
        return
    selected = st.selectbox(
        "Patient", patients,
        format_func=_patient_label, key="vf_lab_patient",
    )
    pid = int(selected["id"])
    dbm.require_patient_access(conn, viewer, pid)

    # Clear lab session state when the selected patient changes so one
    # patient's CT level / image result / uploaded files cannot leak into
    # another patient's assessment.
    last_pid = st.session_state.get("vf_lab_last_pid")
    if last_pid is not None and int(last_pid) != pid:
        for k in list(st.session_state):
            if k.startswith("vf_lab_") or k.startswith("lab_"):
                st.session_state.pop(k, None)
        st.session_state["vf_lab_last_pid"] = pid
        st.rerun()
    st.session_state["vf_lab_last_pid"] = pid

    st.warning(DISCLAIMER)

    with st.expander("Fusion rule parameters (design parameters, not clinical values)"):
        thr_watch = st.number_input("Watch threshold (ECG score)", 0.05, 0.99,
                                    cfg.thr_watch, 0.05, key="lab_thr_watch")
        thr_alert = st.number_input("Alert threshold (ECG score)", 0.05, 1.00,
                                    cfg.thr_alert, 0.05, key="lab_thr_alert")
        step_ct = st.number_input("Lower thresholds by (per CT level)", 0.0, 0.20,
                                  cfg.step_per_ct_level, 0.01, key="lab_step_ct")
        k_req = st.number_input("Windows needed (K)", 1, 10, cfg.k_required, 1, key="lab_k")
        n_win = st.number_input("... out of the last N windows", 1, 10, cfg.n_windows,
                                1, key="lab_n")
        st.caption(f"High-risk emergency threshold: **{cfg.high_risk_threshold}** "
                   "(fused score; configured in app_config.json).")

    lab_cfg = AppConfig(**{**cfg.__dict__,
                           "thr_watch": float(thr_watch), "thr_alert": float(thr_alert),
                           "step_per_ct_level": float(step_ct),
                           "k_required": int(k_req), "n_windows": int(n_win)})

    tab_fusion, tab_ct = st.tabs(
        ["Final fusion (ECG + CT)",
         "CT upload (choose: 1. Report values / 2. Image)"]
    )

    # =====================================================================
    # TAB 2 - CT upload area (runs first so tab 1 can read its result)
    # =====================================================================
    with tab_ct:
        _ct_upload_section(conn, viewer, pid)

    # =====================================================================
    # TAB 1 - Final fusion
    # =====================================================================
    with tab_fusion:
        st.header("1. ECG model")
        ecg_upload = st.file_uploader(
            f"ECG model zip ({cfg.ecg_model_filename})", type=["zip", "keras"],
            key="lab_ecg_zip")
        model, model_msg, model_ok = core.prepare_model(
            cfg, ecg_upload.getvalue() if ecg_upload else None)
        (st.success if model_ok else st.warning)(model_msg)

        with st.expander("Match your ECG training settings (important)"):
            st.write("The model file does not store how the signals were prepared in "
                     "training. These must match your ECG training notebook, or the "
                     "scores will be wrong.")
            model_fs = st.number_input("Sampling rate the model was trained at (Hz)",
                                       25, 1000, DEFAULT_MODEL_FS, 1, key="lab_model_fs")
            use_z = st.checkbox("Z-score normalise each window", value=True, key="lab_z")
            use_bp = st.checkbox("Band-pass filter 0.5-40 Hz before windowing",
                                 value=False, key="lab_bp")

        st.header("2. ECG signal")
        sessions = dbm.list_ecg_sessions(conn, viewer, pid, limit=20)
        src_opts = ["Upload file (CSV / TXT / NPY)", "Synthetic demo (UI test only)",
                    "Saved ECG recording (patient file)"]
        source = st.radio("Signal source", src_opts, key="lab_sig_src")
        signal, fs_in = None, None

        if source.startswith("Upload"):
            sig_file = st.file_uploader("Lead II signal, one sample per row",
                                        type=["csv", "txt", "npy"], key="lab_sig_file")
            fs_in = st.number_input("Sampling rate of this file (Hz)", 25, 2000, 100,
                                    1, key="lab_fs_in")
            if sig_file:
                try:
                    df = read_signal_file(sig_file.name, sig_file.getvalue())
                    col = st.selectbox("Column to use", list(df.columns), key="lab_sig_col")
                    signal = df[col].to_numpy(dtype=float)
                except Exception as e:
                    st.error(str(e))
        elif source.startswith("Saved"):
            if not sessions:
                st.info("This patient has no saved ECG recordings yet.")
            else:
                label = st.selectbox(
                    "Recording", sessions,
                    format_func=lambda s: (f"#{s['id']} {fmt_time(s['started_at'])} "
                                           f"[{s['status']}] {s['n_samples']} samples "
                                           f"@ {s['fs_hz'] or '?'} Hz"),
                    key="lab_saved_session")
                if label:
                    sess = dbm.get_ecg_session(conn, viewer, int(label["id"]))
                    signal = sess["signal_array"].astype(float)
                    fs_in = sess["fs_hz"] or DEFAULT_MODEL_FS
                    if signal.size:
                        st.line_chart(pd.DataFrame({"signal": signal[:5000]}))
        else:
            st.caption("A made-up ECG-like trace, only to test the screens and the "
                       "quality gate. Its scores mean nothing medically.")
            kind = st.radio("Demo signal",
                            ["Clean synthetic ECG", "Flat line", "Random noise"],
                            horizontal=True, key="lab_demo_kind")
            secs = st.slider("Length (seconds)", 10, 60, 30, 10, key="lab_demo_secs")
            fs_in = 100
            if kind == "Clean synthetic ECG":
                signal = synthetic_ecg(seconds=secs, fs=fs_in)
            elif kind == "Flat line":
                signal = np.zeros(secs * fs_in)
            else:
                signal = np.random.default_rng(0).normal(size=secs * fs_in)

        st.header("3. CT context level")
        saved = dbm.latest_ct_record(conn, viewer, pid)
        ct_rep = st.session_state.get("vf_lab_ct_level")
        ct_img = st.session_state.get("vf_lab_ct_image_level")
        ct_saved = core.ct_level_from_record(saved)
        ct_choice = st.radio(
            "Where does the CT context level come from?",
            ["Latest saved CT record", "CT upload - report values",
             "CT upload - image model / manual", "Set manually", "Not reported"],
            key="lab_ct_src",
        )
        ct_level = None
        if ct_choice == "Latest saved CT record":
            ct_level = ct_saved
            st.caption("Current value: " + (
                f"level {ct_saved} (record #{saved['id']})" if ct_saved is not None
                else "no saved CT record for this patient"))
        elif ct_choice == "CT upload - report values":
            ct_level = ct_rep
            st.caption("Current value: " + ("not set (choose option 1 and enter "
                                            "report values in the CT upload tab)"
                                            if ct_rep is None else f"level {ct_rep}"))
        elif ct_choice == "CT upload - image model / manual":
            ct_level = ct_img
            st.caption("Current value: " + ("not set (choose option 2 in the CT "
                                            "upload tab)" if ct_img is None
                                            else f"level {ct_img}"))
        elif ct_choice == "Set manually":
            ct_level = st.select_slider("CT context level", options=[0, 1, 2, 3],
                                        value=0, key="lab_ct_manual")

        st.header("4. Result")
        cfg_error = None
        try:
            lab_cfg.validate()
        except ValueError as e:
            cfg_error = str(e)
            st.error(f"Fusion parameters: {cfg_error}")

        if model is None or signal is None or cfg_error:
            st.info("Load the ECG model and provide a signal to see the result.")
        else:
            result = core.run_recording(model, signal, fs_in, lab_cfg, ct_level,
                                        use_bandpass=use_bp, use_zscore=use_z,
                                        model_fs=int(model_fs))
            if result["status"] != "completed":
                st.error(result["error"])
            else:
                fused = result["fused"]
                show = {-1: st.info, 0: st.success, 1: st.warning, 2: st.error}[fused["level"]]
                show(f"**{fused['level_name']}**  -  {fused['explanation']}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Fused result", fused["level_name"])
                c2.metric("Fused risk score", f"{result['fused_score']:.3f}")
                c3.metric("ECG alone (no CT)",
                          fuse(result["scores"], None, core.fusion_config(lab_cfg))["level_name"])
                c4.metric("Thresholds used (watch / alert)",
                          f"{fused['watch_threshold']} / {fused['alert_threshold']}")

                if result["fused_score"] >= cfg.high_risk_threshold:
                    st.error(f"Fused score {result['fused_score']:.3f} is at/above the "
                             f"high-risk threshold ({cfg.high_risk_threshold}). Saving "
                             "this assessment will raise a high-risk event for the "
                             "patient and notify you.")

                # Action recommendation (experimental rule layer)
                from ui_common import show_action_recommendation
                show_action_recommendation(
                    status=result["status"],
                    ecg_score_mean=result.get("ecg_score_mean"),
                    ct_level=ct_level,
                    fused_score=result.get("fused_score"),
                    fused_level=fused.get("level"),
                    heart_rate_bpm=result.get("heart_rate_bpm"),
                    high_risk_event_open=False,
                    emergency_symptoms=False,
                    cfg=lab_cfg,
                )

                valid_scores = [s for s in result["scores"] if s is not None]
                if valid_scores and np.mean(np.array(valid_scores) > 0.98) >= 0.75:
                    st.info("Almost every valid window scores near 1.0. On real normal "
                            "ECGs that would be suspicious: check the sampling rate, "
                            "normalisation and filter settings above.")
                rows = result["rows"]
                n_bad = sum(not r["quality_ok"] for r in rows)
                if n_bad:
                    st.caption(f"{n_bad} of {len(rows)} window(s) were rejected by the "
                               "signal-quality check and not scored.")
                if result["heart_rate_bpm"] is not None:
                    st.metric("Mean measured heart rate", f"{result['heart_rate_bpm']:.1f} bpm")

                st.subheader("Per-window scores")
                st.dataframe(pd.DataFrame(rows), hide_index=True)

                chart = pd.DataFrame({
                    "ECG score": [np.nan if s is None else s for s in result["scores"]],
                    "watch threshold": fused["watch_threshold"],
                    "alert threshold": fused["alert_threshold"],
                }, index=[r["window"] for r in rows])
                st.line_chart(chart)

                win_no = st.slider("Inspect window", 1, len(rows), 1,
                                   key="lab_win_inspect") if len(rows) > 1 else 1
                L = model.input_len
                try:
                    _rows, processed = analyze_signal(
                        model, signal, int(fs_in), int(model_fs), use_bp, use_z)
                    st.line_chart(pd.DataFrame(
                        {"signal": processed[(win_no - 1) * L: win_no * L]}))
                except Exception:
                    pass

                out = {"fusion": fused,
                       "ecg_only": {"level": fuse(result["scores"], None,
                                                  core.fusion_config(lab_cfg))["level"]},
                       "windows": rows}
                st.download_button("Download result (JSON)",
                                   json.dumps(out, indent=2, default=float),
                                   file_name=f"patient{pid}_fusion_result.json",
                                   mime="application/json")

                if st.button("Save assessment to patient file",
                             type="primary", key="lab_save_assessment"):
                    ct_rec_id = saved["id"] if (ct_choice == "Latest saved CT record"
                                                and saved) else None
                    assessment, event = emergency.process_assessment(
                        conn, viewer, lab_cfg, pid,
                        **core.assessment_kwargs(result, ecg_session_id=None,
                                                 ct_record_id=ct_rec_id),
                    )
                    st.session_state["vf_lab_last_assessment"] = assessment["id"]
                    if event:
                        st.success(f"Assessment #{assessment['id']} saved. HIGH-RISK "
                                   f"event #{event['id']} created and the patient's "
                                   "dashboard now shows the emergency warning.")
                    else:
                        st.success(f"Assessment #{assessment['id']} saved to the "
                                   "patient's file.")

    last = st.session_state.get("vf_lab_last_assessment")
    if last:
        try:
            a = dbm.get_assessment(conn, viewer, int(last))
            if a["patient_id"] == pid:
                st.caption(f"Last saved assessment: #{a['id']} "
                           f"({a['status']}, score={a['fused_score']}, "
                           f"saved {fmt_time(a['created_at'])}).")
        except Exception:
            pass
