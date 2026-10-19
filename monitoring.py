"""Shared ECG monitoring - used by the patient dashboard AND the doctor's
patient-review tab (doctors can monitor an assigned patient's ECG too).

Sources: optional hardware driver (``ecg_device``, never faked) or the built-in
demo generator. The recording is saved as an ECG session
on the monitored patient's record via the access-controlled ``db`` functions.
"""

from __future__ import annotations

import importlib
import time

import numpy as np
import pandas as pd
import streamlit as st

import db as dbm
from ecg_model import assess_quality, synthetic_ecg
from ui_common import fmt_time

DEMO_FS = 100
DEMO_BUFFER_S = 600
_MON_PREFIX = "vf_mon_"


# ---------------------------------------------------------------------------
# state (keyed per viewer + patient so doctor/patient sessions never mix)
# ---------------------------------------------------------------------------
def mon_key(viewer: dbm.Viewer, pid: int) -> str:
    return f"{_MON_PREFIX}{viewer.id}_{pid}"


def mon_state(viewer: dbm.Viewer, pid: int) -> dict:
    key = mon_key(viewer, pid)
    if key not in st.session_state:
        st.session_state[key] = {
            "status": "idle", "session_id": None, "source": None,
            "device_label": "", "device_status": "disconnected", "fs": DEMO_FS,
            "buffer": None, "pos": 0, "samples": np.empty(0, dtype=float),
            "last_tick": None, "device_handle": None, "pid": int(pid),
        }
    return st.session_state[key]


def _stop_others(current_key: str) -> None:
    """Stop a monitoring session left running for a different patient/viewer."""
    for key in list(st.session_state.keys()):
        if not key.startswith(_MON_PREFIX) or key == current_key:
            continue
        mon = st.session_state[key]
        if mon.get("status") == "running" and mon.get("session_id"):
            _stop_session(mon, status="stopped")


# ---------------------------------------------------------------------------
# device / state machine
# ---------------------------------------------------------------------------
def open_device():
    """Optional hardware driver hook (e.g. BLE/serial). Never faked."""
    try:
        mod = importlib.import_module("ecg_device")
        return mod.connect()
    except Exception:
        return None


def _start(conn, viewer, pid: int, mon: dict) -> None:
    if mon["status"] == "running":
        return
    handle = open_device()
    if handle is None:
        st.error("No ECG device driver is installed (optional module "
                 "'ecg_device' not found). The device stays DISCONNECTED. "
                 "Use the demo generator instead.")
        mon["device_status"] = "disconnected"
        return
    buffer = None
    device_label, device_status, src = "hardware device", "connected", "device"
    sid = dbm.create_ecg_session(conn, viewer, pid, fs_hz=DEMO_FS,
                                 device_label=device_label,
                                 device_status=device_status, source=src,
                                 status="running")
    mon.update(status="running", session_id=sid, source=src, fs=DEMO_FS,
               device_label=device_label, device_status=device_status,
               buffer=buffer, pos=0, samples=np.empty(0, dtype=float),
               last_tick=time.time(), device_handle=handle)


def _start_demo(conn, viewer, pid: int, mon: dict) -> None:
    if mon["status"] == "running":
        return
    buffer = synthetic_ecg(seconds=DEMO_BUFFER_S, fs=DEMO_FS, hr=72,
                           noise=0.02, seed=1)
    sid = dbm.create_ecg_session(conn, viewer, pid, fs_hz=DEMO_FS,
                                 device_label="VitalsFusion demo generator",
                                 device_status="connected", source="demo",
                                 status="running")
    mon.update(status="running", session_id=sid, source="demo", fs=DEMO_FS,
               device_label="VitalsFusion demo generator", device_status="connected",
               buffer=buffer, pos=0, samples=np.empty(0, dtype=float),
               last_tick=time.time(), device_handle=None)


def _tick(conn, viewer, mon: dict) -> None:
    """Pull new samples since the last rerun (live monitoring loop)."""
    now = time.time()
    dt = min(max(now - (mon["last_tick"] or now), 0.0), 5.0)
    n = max(1, int(dt * mon["fs"]))
    if mon["source"] == "demo" and mon["buffer"] is not None:
        buf = mon["buffer"]
        start = mon["pos"] % len(buf)
        end = start + n
        chunk = (buf[start:end] if end <= len(buf)
                 else np.concatenate([buf[start:], buf[: end - len(buf)]]))
    elif mon["source"] == "device" and mon["device_handle"] is not None:
        try:
            mod = importlib.import_module("ecg_device")
            chunk = np.asarray(mod.read(mon["device_handle"], n), dtype=float)
        except Exception as exc:
            st.error(f"Device read failed: {exc}. Monitoring stopped.")
            _stop(conn, viewer, mon, status="error")
            return
    else:
        chunk = np.empty(0, dtype=float)

    if chunk.size:
        mon["samples"] = np.concatenate([mon["samples"], chunk])
        mon["pos"] += n
    mon["last_tick"] = now

    # measured heart rate from the last 10 s window (real measurement)
    wlen = int(10 * mon["fs"])
    if mon["samples"].size >= wlen:
        q = assess_quality(mon["samples"][-wlen:], mon["fs"])
        mon["last_hr"] = q.get("hr_bpm")
        mon["last_quality"] = q
    else:
        mon["last_hr"] = None
        mon["last_quality"] = None

    dbm.update_ecg_session(conn, viewer, mon["session_id"],
                           signal=mon["samples"],
                           mean_hr_bpm=mon["last_hr"],
                           device_status=mon["device_status"])


def _pause(conn, viewer, mon: dict) -> None:
    mon["status"] = "paused"
    dbm.update_ecg_session(conn, viewer, mon["session_id"], status="paused")


def _resume(mon: dict) -> None:
    mon["status"] = "running"
    mon["last_tick"] = time.time()


def _stop_session(mon: dict, status: str = "stopped") -> None:
    mon["status"] = status if status == "error" else "stopped"
    if mon.get("device_handle") is not None:
        try:
            importlib.import_module("ecg_device").disconnect(mon["device_handle"])
        except Exception:
            pass
        mon["device_handle"] = None
    mon["device_status"] = "disconnected"


def _stop(conn, viewer, mon: dict, status: str = "stopped") -> None:
    _stop_session(mon, status=status)
    if mon.get("session_id"):
        dbm.update_ecg_session(conn, viewer, mon["session_id"],
                               status=mon["status"], signal=mon["samples"],
                               device_status="disconnected", ended=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_monitoring(conn, viewer: dbm.Viewer, pid: int, cfg, *,
                      key_prefix: str = "mon",
                      assessment_ui=None,
                      show_saved: bool = True) -> None:
    """Start/pause/resume/stop + live waveform + saved recordings.

    ``assessment_ui(conn, viewer, cfg, mon, pid)`` - optional callable that
    renders a 'Run AI assessment' control for the live session.
    """
    mon = mon_state(viewer, pid)
    _stop_others(mon_key(viewer, pid))

    with st.expander("ECG device connection"):
        st.write(f"**Device status:** `{mon['device_status'].upper()}`")
        st.caption("Hardware support needs the optional 'ecg_device' module. "
                   "Without it the demo generator can be used; the device is "
                   "honestly reported as disconnected.")
        if st.button("Scan for device", key=f"{key_prefix}_scan"):
            handle = open_device()
            if handle:
                mon["device_handle"] = handle
                mon["device_status"] = "connected"
                st.success("Device driver found and connected.")
            else:
                mon["device_status"] = "disconnected"
                st.warning("No device driver available.")

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Start monitoring", type="primary", key=f"{key_prefix}_start",
                 disabled=mon["status"] == "running"):
        _start_demo(conn, viewer, pid, mon)
    if c2.button("Pause", key=f"{key_prefix}_pause",
                 disabled=mon["status"] != "running"):
        _pause(conn, viewer, mon); st.rerun()
    if c3.button("Resume", key=f"{key_prefix}_resume",
                 disabled=mon["status"] != "paused"):
        _resume(mon); st.rerun()
    if c4.button("Stop", key=f"{key_prefix}_stop",
                 disabled=mon["status"] not in ("running", "paused")):
        _stop(conn, viewer, mon); st.rerun()

    if mon["status"] == "running":
        _live_fragment(conn, viewer, mon)
    elif mon["samples"].size:
        wlen = int(10 * mon["fs"])
        tail = mon["samples"][-wlen:] if mon["samples"].size >= wlen else mon["samples"]
        q = assess_quality(tail, mon["fs"]) if tail.size >= mon["fs"] else None
        st.line_chart(pd.DataFrame({"ECG": tail}))
        if mon.get("last_hr") or (q and q.get("hr_bpm")):
            hr = mon.get("last_hr") or q.get("hr_bpm")
            st.metric("Heart rate", f"{hr:.0f} bpm")
        st.caption(f"Monitoring {mon['status']}. {mon['samples'].size} samples "
                   f"at {mon['fs']} Hz "
                   f"({mon['samples'].size / mon['fs']:.1f} s) saved in session "
                   f"#{mon['session_id']}.")
    else:
        st.info("Press **Start monitoring** to begin (demo generator). "
                "The waveform and heart rate update live.")

    if assessment_ui is not None:
        st.markdown("### Run AI assessment")
        assessment_ui(conn, viewer, cfg, mon, pid)

    if not show_saved:
        return
    st.markdown("### Saved recordings")
    sessions = dbm.list_ecg_sessions(conn, viewer, pid, limit=20)
    if not sessions:
        st.caption("No recordings yet.")
        return
    pick = st.selectbox(
        "Recording", sessions,
        format_func=lambda s: f"#{s['id']} {fmt_time(s['started_at'])} [{s['status']}] "
                              f"{s['n_samples']} samples ({s['source']}, "
                              f"device: {s['device_status']})",
        key=f"{key_prefix}_pick")
    if pick:
        sess = dbm.get_ecg_session(conn, viewer, int(pick["id"]))
        sig = sess["signal_array"]
        if sig.size:
            st.line_chart(pd.DataFrame({"ECG": sig[:6000]}))
            st.caption(f"{sig.size} samples @ {sess['fs_hz']} Hz; "
                       f"mean HR {sess['mean_hr_bpm'] or 'n/a'}; "
                       f"recorded {fmt_time(sess['started_at'])}.")
        else:
            st.caption("This recording has no samples stored.")


@st.fragment(run_every="1s")
def _live_fragment(conn, viewer, mon: dict) -> None:
    """Live panel: advances the recording, updates waveform and heart rate."""
    _tick(conn, viewer, mon)
    st.markdown(f"**Live monitoring** - device `{mon['device_status']}` "
                f"(session #{mon['session_id']})")
    wlen = int(10 * mon["fs"])
    tail = (mon["samples"][-wlen:] if mon["samples"].size >= wlen
            else mon["samples"])
    if tail.size:
        st.line_chart(pd.DataFrame({"ECG": tail}), height=200)
    if mon.get("last_hr"):
        st.metric("Heart rate", f"{mon['last_hr']:.0f} bpm")
    else:
        st.caption("Waiting for at least 10 s of signal to measure heart rate...")
    q = mon.get("last_quality")
    if q is not None and not q.get("ok"):
        st.warning("Signal quality: " + "; ".join(q.get("reasons") or ["unknown"]))
