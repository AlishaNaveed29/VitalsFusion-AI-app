"""
ECG branch for VitalsFusion AI
==============================
* Runs your Keras lead-II MI classifier (ecg_lead2_mi_model_v2_keras.zip) with
  plain NumPy, so the app needs only `numpy`, `scipy` and `h5py` (no
  TensorFlow). The forward pass was checked against Keras itself.
* Preprocessing helpers (resample, optional band-pass, z-score, windowing).
* A simple signal-quality gate, so a flat or noisy window never reaches the
  model (the model outputs ~1.0 on random noise).

The model outputs ONE number per window: the score of an MI-like pattern in
lead II. It is a research score, not a diagnosis.
"""

import io
import json
import math
import zipfile

import numpy as np

# ---------------------------------------------------------------------------
# ASSUMPTIONS you must match to your ECG training notebook
# (the model file does not store them):
#   - the sampling rate the model was trained at (1000 samples = 10 s at 100 Hz)
#   - whether each window was z-score normalised
#   - whether a band-pass filter was applied
# ---------------------------------------------------------------------------
DEFAULT_MODEL_FS = 100


# ===========================================================================
# Model
# ===========================================================================
def _activation(name, x):
    if name in (None, "linear"):
        return x
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
    if name == "softmax":
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
    raise ValueError(f"Unsupported activation: {name}")


class EcgMiModel:
    """NumPy re-implementation of the saved Keras Functional model."""

    SUPPORTED = {"InputLayer", "Conv1D", "BatchNormalization", "MaxPooling1D",
                 "GlobalAveragePooling1D", "Dense", "Dropout"}

    def __init__(self, ops, input_len, name):
        self.ops = ops
        self.input_len = input_len
        self.name = name

    # -- loading ------------------------------------------------------------
    @classmethod
    def from_zip_bytes(cls, data: bytes):
        import h5py

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            if "config.json" not in names or "model.weights.h5" not in names:
                raise ValueError("Not a Keras model archive: needs config.json and model.weights.h5")
            cfg = json.loads(zf.read("config.json"))
            weights_bytes = zf.read("model.weights.h5")

        layers = cfg["config"]["layers"]
        input_len = int(layers[0]["config"]["batch_shape"][1])

        ops = []
        with h5py.File(io.BytesIO(weights_bytes), "r") as f:
            for layer in layers:
                kind, c = layer["class_name"], layer["config"]
                if kind not in cls.SUPPORTED:
                    raise ValueError(f"Unsupported layer type for the NumPy runner: {kind}")
                if kind in ("InputLayer", "Dropout"):
                    continue

                def var(i, _name=c["name"]):
                    return np.asarray(f["layers"][_name]["vars"][str(i)][()], dtype=np.float64)

                if kind == "Conv1D":
                    if c["strides"] != [1] or c["dilation_rate"] != [1] or c["groups"] != 1:
                        raise ValueError("Only stride 1, dilation 1, groups 1 are supported")
                    ops.append(("conv", var(0), var(1), c["padding"], c["activation"]))
                elif kind == "BatchNormalization":
                    if not (c["center"] and c["scale"]):
                        raise ValueError("BatchNormalization without center/scale not supported")
                    # Keras 3 order: gamma, beta, moving_mean, moving_variance
                    ops.append(("bn", var(0), var(1), var(2), var(3), float(c["epsilon"])))
                elif kind == "MaxPooling1D":
                    ops.append(("pool", int(c["pool_size"][0]), int(c["strides"][0]), c["padding"]))
                elif kind == "GlobalAveragePooling1D":
                    ops.append(("gap",))
                elif kind == "Dense":
                    ops.append(("dense", var(0), var(1), c["activation"]))

        return cls(ops, input_len, cfg["config"]["name"])

    @classmethod
    def from_path(cls, path):
        with open(path, "rb") as fh:
            return cls.from_zip_bytes(fh.read())

    # -- inference ----------------------------------------------------------
    def _run(self, x, stop_after=None):
        for i, op in enumerate(self.ops):
            kind = op[0]
            if kind == "conv":
                _, W, b, padding, act = op
                K = W.shape[0]
                if padding == "same":
                    left = (K - 1) // 2
                    x = np.pad(x, ((0, 0), (left, K - 1 - left), (0, 0)))
                win = np.lib.stride_tricks.sliding_window_view(x, K, axis=1)  # B,L,Cin,K
                x = np.einsum("blck,kco->blo", win, W) + b
                x = _activation(act, x)
            elif kind == "bn":
                _, gamma, beta, mean, var_, eps = op
                x = (x - mean) / np.sqrt(var_ + eps) * gamma + beta
            elif kind == "pool":
                _, size, stride, padding = op
                if padding != "valid":
                    raise ValueError("Only 'valid' max pooling is supported")
                win = np.lib.stride_tricks.sliding_window_view(x, size, axis=1)[:, ::stride]
                x = win.max(axis=-1)
            elif kind == "gap":
                x = x.mean(axis=1)
            elif kind == "dense":
                _, W, b, act = op
                x = _activation(act, x @ W + b)
            if stop_after is not None and i == stop_after:
                break
        return x

    def predict(self, windows):
        """windows: array (N, input_len) -> probabilities (N,)."""
        w = np.asarray(windows, dtype=np.float64)
        if w.ndim == 1:
            w = w[None, :]
        if w.shape[1] != self.input_len:
            raise ValueError(f"Each window needs {self.input_len} samples, got {w.shape[1]}")
        out = self._run(w[:, :, None])
        return out.reshape(len(w), -1)[:, 0]

    def embedding(self, windows):
        """Output of the global-average-pooling layer (64 numbers per window)."""
        w = np.asarray(windows, dtype=np.float64)
        if w.ndim == 1:
            w = w[None, :]
        gap_idx = next(i for i, op in enumerate(self.ops) if op[0] == "gap")
        return self._run(w[:, :, None], stop_after=gap_idx)


# ===========================================================================
# Signal preprocessing
# ===========================================================================
def resample_signal(x, fs_in, fs_model=DEFAULT_MODEL_FS):
    x = np.asarray(x, dtype=np.float64)
    fs_in, fs_model = int(round(fs_in)), int(round(fs_model))
    if fs_in == fs_model:
        return x
    from scipy.signal import resample_poly

    g = math.gcd(fs_in, fs_model)
    return resample_poly(x, fs_model // g, fs_in // g)


def bandpass(x, fs, low=0.5, high=40.0):
    from scipy.signal import butter, filtfilt

    high = min(high, fs / 2 - 1)
    b, a = butter(3, [low, high], btype="band", fs=fs)
    return filtfilt(b, a, x)


def segment_windows(x, window_len):
    """Non-overlapping windows; the incomplete tail is dropped."""
    n = len(x) // window_len
    if n == 0:
        return np.empty((0, window_len))
    return np.asarray(x[: n * window_len]).reshape(n, window_len)


def zscore(w):
    w = np.asarray(w, dtype=np.float64)
    return (w - w.mean(axis=-1, keepdims=True)) / (w.std(axis=-1, keepdims=True) + 1e-8)


# ===========================================================================
# Signal quality gate
# ===========================================================================
def assess_quality(w, fs):
    """Return dict(ok, reasons, hr_bpm) for one window (before z-scoring).

    Checks: finite values, not flat, not clipped, at least 3 clear beats, a
    plausible heart rate (30-220 bpm) and that the beats look alike (template
    match). The last check is what separates real ECG from noise.
    """
    reasons, hr = [], None
    w = np.asarray(w, dtype=np.float64)

    if not np.all(np.isfinite(w)):
        return {"ok": False, "reasons": ["contains NaN/inf"], "hr_bpm": None}

    if float(np.ptp(w)) == 0.0 or float(w.std()) < 1e-9:
        return {"ok": False, "reasons": ["flat line (no signal)"], "hr_bpm": None}

    if max(np.mean(w == w.max()), np.mean(w == w.min())) > 0.02:
        reasons.append("saturated / clipped signal")

    try:
        from scipy.signal import butter, filtfilt, find_peaks

        b, a = butter(2, [5.0, min(15.0, fs / 2 - 1)], btype="band", fs=fs)
        env = filtfilt(b, a, w - np.median(w)) ** 2
        peaks, _ = find_peaks(env, height=0.4 * env.max(), distance=max(1, int(0.25 * fs)))

        if len(peaks) < 3:
            reasons.append("no clear heart beats found")
        else:
            hr = 60.0 / (np.median(np.diff(peaks)) / fs)
            if hr < 30 or hr > 220:
                reasons.append(f"implausible heart rate ({hr:.0f} bpm), probably noise")

            # Template match: every beat should resemble the median beat.
            half = int(0.25 * fs)
            usable = [p for p in peaks if p - half >= 0 and p + half < len(w)]
            if len(usable) >= 3:
                bh, ah = butter(2, 0.5, btype="high", fs=fs)
                xh = filtfilt(bh, ah, w)
                segs = np.array([xh[p - half:p + half] for p in usable])
                segs = segs - segs.mean(axis=1, keepdims=True)
                template = np.median(segs, axis=0)
                corr = float(np.mean([np.corrcoef(sg, template)[0, 1] for sg in segs]))
                if not np.isfinite(corr) or corr < 0.6:
                    reasons.append(f"signal too noisy (beats do not match, score {corr:.2f})")
    except Exception:
        pass  # the beat checks are a bonus; never crash on them

    return {"ok": not reasons, "reasons": reasons, "hr_bpm": None if hr is None else float(hr)}


# ===========================================================================
# Helpers for tests and for the UI demo
# ===========================================================================
def synthetic_ecg(seconds=30, fs=100, hr=72, noise=0.02, seed=0):
    """Toy ECG-like trace (P, QRS, T as Gaussians). For UI/testing ONLY."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    beat = 60.0 / hr
    sig = np.zeros(n)
    for k in range(int(seconds / beat) + 2):
        c = k * beat + 0.3
        for amp, off, wid in ((0.12, -0.18, 0.025), (-0.10, -0.035, 0.010),
                              (1.00, 0.0, 0.012), (-0.18, 0.035, 0.012), (0.30, 0.22, 0.045)):
            sig += amp * np.exp(-0.5 * ((t - c - off) / wid) ** 2)
    return sig + rng.normal(0, noise, n)


# ===========================================================================
# Full pipeline: raw samples -> per-window scores
# ===========================================================================
def analyze_signal(model, x, fs_in, model_fs=DEFAULT_MODEL_FS,
                   use_bandpass=False, use_zscore=True):
    """Run the whole ECG branch on one recording.

    Returns (rows, processed_signal). `rows` has one dict per 10-second window:
    window, start_s, quality_ok, problems, hr_bpm, score (None if the window
    was rejected by the quality gate, so it never reaches the model).
    """
    x = resample_signal(x, fs_in, model_fs)
    if use_bandpass:
        x = bandpass(x, model_fs)

    windows = segment_windows(x, model.input_len)
    rows, ok_idx = [], []
    for i, w in enumerate(windows):
        q = assess_quality(w, model_fs)
        rows.append({
            "window": i + 1,
            "start_s": round(i * model.input_len / model_fs, 2),
            "quality_ok": bool(q["ok"]),
            "problems": "; ".join(q["reasons"]),
            "hr_bpm": None if q["hr_bpm"] is None else round(q["hr_bpm"], 1),
            "score": None,
        })
        if q["ok"]:
            ok_idx.append(i)

    if ok_idx:
        batch = windows[ok_idx]
        batch = zscore(batch) if use_zscore else batch
        scores = model.predict(batch)
        for j, i in enumerate(ok_idx):
            rows[i]["score"] = float(scores[j])
    return rows, x
