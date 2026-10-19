"""ct_image_model: load real checkpoint, synthetic DICOM volume, predict."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pydicom")
pytest.importorskip("torchvision")

import ct_image_model as cim  # noqa: E402
import pydicom  # noqa: E402
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402

CKPT = Path(__file__).parent / "models" / "ResNet18_2.5D_best.pt"


def _dcm_bytes(z: int, arr_u16: np.ndarray) -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.InstanceNumber = z + 1
    ds.ImagePositionPatient = [0.0, 0.0, float(z)]
    ds.Rows, ds.Columns = arr_u16.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.PixelData = arr_u16.tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def test_load_real_checkpoint_and_predict():
    ckpt_bytes = CKPT.read_bytes()
    model, meta = cim.load_model(ckpt_bytes)
    assert meta["class_names"] == ("Low", "Mild", "Moderate", "Severe")
    # synthetic volume: bright "calcium" blobs on soft tissue
    rng = np.random.default_rng(0)
    n, h, w = 24, 64, 64
    vol = rng.normal(-60.0, 30.0, size=(n, h, w)).astype(np.float32)
    vol[:, 20:30, 25:35] = 300.0  # dense streak
    files = [(f"slice_{i:03d}.dcm", _dcm_bytes(i, (vol[i] + 1024).astype(np.uint16)))
             for i in range(n)]
    volume = cim.read_dicom_volume(files)
    assert volume.shape == (n, h, w)
    probs = cim.predict_probs(model, volume)
    assert len(probs) == 4
    assert np.isclose(sum(probs), 1.0, atol=1e-4)
    pred, probs2 = cim.predict_class(model, volume)
    assert pred in (0, 1, 2, 3)


def test_zip_input_supported():
    import zipfile
    ckpt_bytes = CKPT.read_bytes()
    model, _ = cim.load_model(ckpt_bytes)
    arr = (np.full((32, 32), 200, dtype=np.float32) + 1024).astype(np.uint16)
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        for i in range(4):
            zf.writestr(f"s{i}.dcm", _dcm_bytes(i, arr))
    volume = cim.read_dicom_volume([("series.zip", zbuf.getvalue())])
    assert volume.shape[0] == 4
    probs = cim.predict_probs(model, volume)
    assert np.isclose(sum(probs), 1.0, atol=1e-4)


def test_import_shows_image_model_path_in_upload_ui():
    # ct_upload must now find the module (no "packages not installed" fallback)
    import importlib
    assert importlib.util.find_spec("ct_image_model") is not None
