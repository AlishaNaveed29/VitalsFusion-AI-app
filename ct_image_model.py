"""Experimental CT image model: severity quartile from DICOM slices.

Optional module. When importable (torch + torchvision + pydicom installed)
the CT upload UI offers a "Run image model" path; without it, a manual
image-review level can still be recorded.

The bundled checkpoint ``ResNet18_2.5D_best*.pt`` stores:
    * state dict of a torchvision-style ResNet18 (``features`` trunk,
      ``classifier`` = Linear(512 -> 4)); first conv expects 3 channels
      ("2.5D": three slices stacked as channels)
    * class names: Low / Mild / Moderate / Severe  ->  context levels 0..3
    * preprocessing metadata: HU clip window, centre-crop fraction,
      spatial/depth input size

API used by the Streamlit upload UI (``ct_upload.py``):
    load_model(ckpt_bytes)      -> (model, meta)
    read_dicom_volume(files)    -> float32 ndarray (N slices, H, W) in HU
    predict_probs(model, volume)-> sequence of 4 class probabilities

This is a research prototype - outputs are NOT a diagnosis.
"""

from __future__ import annotations

import io
import zipfile
from typing import List, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision.io import read_image  # noqa: F401  (kept optional)
    from torchvision.models import resnet18
    import torchvision.transforms.functional as TF
    import pydicom
except ImportError as exc:  # pragma: no cover - exercised on bare installs
    raise ImportError(
        "ct_image_model requires torch, torchvision and pydicom"
    ) from exc

N_CLASSES = 4
CLASS_NAMES = ("Low", "Mild", "Moderate", "Severe")
_DEFAULT_HU_CLIP = (-100.0, 1000.0)
_DEFAULT_CROP = 0.75
_DEFAULT_INPUT = (32, 128, 128)

# Security limits for untrusted uploads (never extract zips to the filesystem).
MAX_ZIP_MEMBERS = 4000
MAX_MEMBER_BYTES = 128 * 1024 * 1024        # 128 MB per member
MAX_TOTAL_UNCOMPRESSED = 2 * 1024 ** 3      # 2 GB expanded
MAX_INPUT_FILES = 4000
MAX_SLICES = 4096
_MAX_DIM = 4096                              # rows/cols sanity cap


class ResNet18_25D(nn.Module):
    """ResNet18 with a 3-channel 2.5D input and a 4-class severity head."""

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        base = resnet18(weights=None)
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.avgpool = base.avgpool
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def load_model(ckpt_bytes: bytes) -> Tuple[nn.Module, dict]:
    """Build the network and load weights from checkpoint *bytes*."""
    buf = io.BytesIO(ckpt_bytes)
    try:
        ckpt = torch.load(buf, map_location="cpu", weights_only=False)
    except Exception:
        buf.seek(0)
        ckpt = torch.load(buf, map_location="cpu")  # older torch
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint is not a dictionary.")
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    if not isinstance(sd, dict) or not sd:
        raise ValueError("Checkpoint has no state dict.")
    if all(hasattr(v, "shape") for v in sd.values()):
        pass  # the dict itself is the state dict
    else:
        raise ValueError("Checkpoint state dict looks invalid.")

    class_names = tuple(ckpt.get("class_names") or
                        ("Low", "Mild", "Moderate", "Severe"))
    n_classes = len(class_names)
    model = ResNet18_25D(n_classes=n_classes)
    cleaned = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    critical = [k for k in missing if not k.endswith(("running_mean",
                                                      "running_var",
                                                      "num_batches_tracked"))]
    if critical or unexpected:
        raise ValueError(
            f"Checkpoint does not match the ResNet18_2.5D architecture "
            f"(missing={critical[:5]}, unexpected={list(unexpected)[:5]}).")
    model.eval()

    input_size = tuple(ckpt.get("input_size") or _DEFAULT_INPUT)
    hu_clip = tuple(ckpt.get("hu_clip") or _DEFAULT_HU_CLIP)
    crop = float(ckpt.get("center_crop_frac") or _DEFAULT_CROP)
    meta = {
        "model_name": ckpt.get("model_name", "ResNet18_2.5D"),
        "class_names": class_names,
        "input_size": input_size,
        "hu_clip": (float(hu_clip[0]), float(hu_clip[1])),
        "center_crop_frac": crop,
    }
    # keep preprocessing settings with the model (callers may ignore meta)
    model._vf_meta = meta  # type: ignore[attr-defined]
    return model, meta


def _decode_dicom(data: bytes) -> "pydicom.dataset.Dataset":
    if not data:
        raise ValueError("Empty file.")
    if len(data) > MAX_MEMBER_BYTES:
        raise ValueError("File exceeds the per-file size limit.")
    ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=False, force=False)
    if not hasattr(ds, "pixel_array"):
        raise ValueError("File has no pixel data.")
    return ds


def _hu_slice(ds) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D slice, got shape {arr.shape}.")
    if arr.shape[0] > _MAX_DIM or arr.shape[1] > _MAX_DIM:
        raise ValueError(f"Slice dimensions {arr.shape} exceed the {_MAX_DIM} px cap.")
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr * slope + intercept


def _orientation_normal(ds) -> np.ndarray | None:
    """Unit slice normal from ImageOrientationPatient (row x col directions)."""
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None or len(iop) < 6:
        return None
    try:
        vals = [float(v) for v in iop[:6]]
    except (TypeError, ValueError):
        return None
    row = np.array(vals[:3], dtype=np.float64)
    col = np.array(vals[3:6], dtype=np.float64)
    nrm = np.cross(row, col)
    norm = float(np.linalg.norm(nrm))
    if norm < 1e-6:
        return None
    return nrm / norm


def _slice_position(ds, normal: np.ndarray | None) -> float | None:
    """Project ImagePositionPatient onto the series normal (any orientation)."""
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is None or len(ipp) < 3:
        return None
    try:
        p = np.array([float(ipp[0]), float(ipp[1]), float(ipp[2])], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if normal is None:
        return float(p[2])  # axial fallback when orientation is missing
    return float(np.dot(p, normal))


def _expand_inputs(files: Sequence[Tuple[str, bytes]]) -> List[Tuple[str, bytes]]:
    """Expand uploads (incl. in-memory zips) with decompression-bomb guards."""
    if not files:
        raise ValueError("No files provided.")
    if len(files) > MAX_INPUT_FILES:
        raise ValueError(f"Too many files (limit {MAX_INPUT_FILES}).")
    raw: List[Tuple[str, bytes]] = []
    total_uncompressed = 0
    for name, data in files:
        if not data:
            continue
        if len(data) > MAX_MEMBER_BYTES:
            raise ValueError(f"'{name}' exceeds the per-file size limit.")
        lname = (name or "").lower()
        if not lname.endswith(".zip"):
            raw.append((name, data))
            continue
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise ValueError(f"'{name}' is not a valid zip file.") from exc
        with zf:
            infos = [i for i in zf.infolist()
                     if not i.is_dir() and not i.filename.lower().startswith("__macosx")]
            if len(infos) > MAX_ZIP_MEMBERS:
                raise ValueError(f"'{name}' has too many members "
                                 f"({len(infos)} > {MAX_ZIP_MEMBERS}).")
            total_uncompressed += sum(i.file_size for i in infos)
            if total_uncompressed > MAX_TOTAL_UNCOMPRESSED:
                raise ValueError("Expanded upload is too large "
                                 "(decompression-bomb protection).")
            for info in infos:
                if info.file_size > MAX_MEMBER_BYTES:
                    raise ValueError(f"'{info.filename}' inside '{name}' is too large.")
                # read into memory only - never extract to the filesystem
                raw.append((info.filename, zf.read(info)))
    return raw


def _series_uid(ds) -> str:
    uid = getattr(ds, "SeriesInstanceUID", None)
    return str(uid) if uid else ""


def read_dicom_volume(files: Sequence[Tuple[str, bytes]]) -> np.ndarray:
    """Read DICOM slices (or a .zip of them) into a float32 HU volume.

    ``files`` is a list of ``(filename, bytes)`` pairs from the Streamlit
    multi-file uploader. Returns shape ``(N, H, W)`` sorted by the projection
    of each slice position onto the series normal (orientation-aware).

    Validation performed:
      * invalid / non-DICOM files are skipped and counted (see
        :func:`read_dicom_volume_ex` for the skip list)
      * one dominant SeriesInstanceUID is selected when several are mixed
      * Rows/Columns, bit depth and photometric interpretation must match
      * ImageOrientationPatient must be consistent across slices
      * duplicate slice positions are dropped
      * slice count is capped at MAX_SLICES

    Limitation: the training-time DICOM reader cannot be verified from this
    repository; ordering and HU handling follow standard DICOM practice and
    the checkpoint's stored HU window / crop metadata.
    """
    volume, _info = read_dicom_volume_ex(files)
    return volume


def read_dicom_volume_ex(files: Sequence[Tuple[str, bytes]]
                         ) -> Tuple[np.ndarray, dict]:
    """Like :func:`read_dicom_volume` but also returns validation info."""
    raw = _expand_inputs(files)
    if not raw:
        raise ValueError("No DICOM files found in the selection.")

    dss: List = []
    kept_names: List[str] = []
    skipped: List[Tuple[str, str]] = []
    for name, data in raw:
        try:
            dss.append(_decode_dicom(data))
            kept_names.append(name)
        except Exception as exc:
            skipped.append((name, str(exc)))
    if not dss:
        raise ValueError("None of the provided files could be read as DICOM "
                         f"({len(skipped)} file(s) skipped).")

    # keep the dominant series when multiple SeriesInstanceUIDs are mixed
    by_series: dict[str, List[int]] = {}
    for i, ds in enumerate(dss):
        by_series.setdefault(_series_uid(ds), []).append(i)
    chosen_uid = max(by_series, key=lambda k: len(by_series[k]))
    dropped_mixed = sum(len(v) for k, v in by_series.items() if k != chosen_uid)
    idxs = by_series[chosen_uid]

    # shared geometry checks on the chosen series
    ref = dss[idxs[0]]
    ref_shape = None
    try:
        ref_shape = tuple(ref.pixel_array.shape)
    except Exception:
        pass
    ref_iop = getattr(ref, "ImageOrientationPatient", None)
    ref_iop_vals = None
    if ref_iop is not None and len(ref_iop) >= 6:
        try:
            ref_iop_vals = np.array([float(v) for v in ref_iop[:6]])
        except (TypeError, ValueError):
            ref_iop_vals = None
    ref_bits = (int(getattr(ref, "BitsAllocated", 0) or 0),
                int(getattr(ref, "SamplesPerPixel", 1) or 1),
                str(getattr(ref, "PhotometricInterpretation", "")))

    normal = _orientation_normal(ref)
    filtered: List[int] = []
    for i in idxs:
        ds = dss[i]
        try:
            shape = tuple(ds.pixel_array.shape)
        except Exception as exc:
            skipped.append((kept_names[i], f"unreadable pixels: {exc}"))
            continue
        if ref_shape is not None and shape != ref_shape:
            skipped.append((kept_names[i], f"inconsistent dimensions {shape} "
                                          f"!= {ref_shape}"))
            continue
        bits = (int(getattr(ds, "BitsAllocated", 0) or 0),
                int(getattr(ds, "SamplesPerPixel", 1) or 1),
                str(getattr(ds, "PhotometricInterpretation", "")))
        if bits != ref_bits:
            skipped.append((kept_names[i], f"incompatible pixel format {bits}"))
            continue
        iop = getattr(ds, "ImageOrientationPatient", None)
        if ref_iop_vals is not None:
            if iop is None or len(iop) < 6:
                skipped.append((kept_names[i], "missing ImageOrientationPatient"))
                continue
            try:
                vals = np.array([float(v) for v in iop[:6]])
            except (TypeError, ValueError):
                skipped.append((kept_names[i], "invalid ImageOrientationPatient"))
                continue
            if not np.allclose(vals, ref_iop_vals, atol=1e-3):
                skipped.append((kept_names[i], "inconsistent image orientation"))
                continue
        filtered.append(i)
        if len(filtered) >= MAX_SLICES:
            break

    if len(filtered) < 1:
        raise ValueError("No usable slices left after validation "
                         f"({len(skipped)} skipped).")

    # orientation-aware sort: position projected on the slice normal
    order = sorted(filtered,
                   key=lambda i: (_slice_position(dss[i], normal) if
                                  _slice_position(dss[i], normal) is not None
                                  else float("inf"), kept_names[i]))

    # drop duplicate positions (same location within 1e-3 mm along the normal)
    dedup: List[int] = []
    last_pos = None
    n_dup = 0
    for i in order:
        pos = _slice_position(dss[i], normal)
        if pos is not None and last_pos is not None and abs(pos - last_pos) < 1e-3:
            n_dup += 1
            continue
        dedup.append(i)
        if pos is not None:
            last_pos = pos

    try:
        volume = np.stack([_hu_slice(dss[i]) for i in dedup]).astype(np.float32)
    except Exception as exc:
        raise ValueError(f"Could not build the HU volume: {exc}") from exc

    info = {
        "n_slices": int(volume.shape[0]),
        "shape": tuple(int(s) for s in volume.shape),
        "skipped": skipped,
        "dropped_mixed_series": int(dropped_mixed),
        "dropped_duplicates": int(n_dup),
        "series_uid": chosen_uid,
        "orientation": "orientation-normal" if normal is not None else "axial-fallback",
    }
    return volume, info


def _prep_sample(volume: np.ndarray, meta: dict) -> torch.Tensor:
    """Volume (N,H,W) HU -> model input tensor (1, 3, h, w) in [0, 1]."""
    n_slab, hw, _ = meta["input_size"] if len(meta["input_size"]) == 3 else \
        (meta["input_size"][0], meta["input_size"][-2], meta["input_size"][-1])
    h, w = int(meta["input_size"][-2]), int(meta["input_size"][-1])
    n = volume.shape[0]
    # central slab of n_slab slices (pad by repeating edges if too short)
    if n >= n_slab:
        start = (n - n_slab) // 2
        slab = volume[start:start + n_slab]
    else:
        pad = n_slab - n
        front = np.repeat(volume[:1], pad // 2, axis=0)
        back = np.repeat(volume[-1:], pad - pad // 2, axis=0)
        slab = np.concatenate([front, volume, back], axis=0)

    # 2.5D: three slices evenly spaced through the slab, stacked as channels
    idxs = [int(round(f * (n_slab - 1))) for f in (0.25, 0.5, 0.75)]
    chans = []
    for i in idxs:
        img = slab[i].astype(np.float32)
        # spatial centre-crop
        frac = float(meta.get("center_crop_frac", _DEFAULT_CROP))
        if 0.0 < frac < 1.0:
            hh, ww = img.shape
            ch, cw = int(hh * frac), int(ww * frac)
            r0, c0 = (hh - ch) // 2, (ww - cw) // 2
            img = img[r0:r0 + ch, c0:c0 + cw]
        t = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).unsqueeze(0)
        t = TF.resize(t, [h, w], antialias=True)
        chans.append(t.squeeze(0))
    x = torch.cat(chans, dim=0)  # (3, h, w)

    lo, hi = meta.get("hu_clip", _DEFAULT_HU_CLIP)
    x = torch.clamp(x, lo, hi)
    x = (x - lo) / max(hi - lo, 1e-6)  # scale to [0, 1]
    return x.unsqueeze(0)


def predict_probs(model: nn.Module, volume: np.ndarray, meta: dict | None = None) -> np.ndarray:
    """Run the model on a HU volume; returns (K,) class probabilities.

    Softmax is applied to logits under ``torch.no_grad()`` with the model in
    ``eval()`` mode (set by :func:`load_model`). The result is validated to be
    finite, non-negative and summing to ~1 before it is returned.
    """
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3 or volume.shape[0] < 1:
        raise ValueError("Volume must have shape (N, H, W) with N >= 1.")
    if meta is None:
        meta = getattr(model, "_vf_meta", None) or {
            "input_size": _DEFAULT_INPUT, "hu_clip": _DEFAULT_HU_CLIP,
            "center_crop_frac": _DEFAULT_CROP}
    x = _prep_sample(volume, meta)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    finally:
        if was_training:
            model.train()
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if probs.size < 2:
        raise ValueError("Model returned an invalid probability vector.")
    if not np.all(np.isfinite(probs)):
        raise ValueError("Model produced non-finite probabilities.")
    if np.any(probs < -1e-6) or abs(float(probs.sum()) - 1.0) > 1e-3:
        raise ValueError("Model probabilities are not a valid distribution.")
    probs = np.clip(probs, 0.0, None)
    s = float(probs.sum())
    if s > 0:
        probs = probs / s
    return probs


def predict_class(model: nn.Module, volume: np.ndarray,
                  meta: dict | None = None) -> Tuple[int, np.ndarray]:
    """Convenience: (argmax class 0..3, probs)."""
    probs = predict_probs(model, volume, meta)
    return int(np.argmax(probs)), probs
