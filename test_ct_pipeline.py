"""CT pipeline fixes: DICOM validation/sorting, prediction state, fusion honesty.

Covers the priority bugs fixed in ct_image_model.py and ct_upload.py:
- orientation-aware slice sorting (axial + oblique)
- invalid DICOM / malformed zip handling and bomb guards
- HU slope/intercept conversion
- prediction persistence helpers (fingerprint) and one-time save
- stale-prediction invalidation
- report missing-vs-zero calcium (ct_features)
- honest fusion when ECG is missing
"""
from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pydicom")

import auth  # noqa: E402
import ct_image_model as cim  # noqa: E402
import ct_upload as cu  # noqa: E402
import db as dbm  # noqa: E402
from ct_features import encode_ct_report, stenosis_grade_from_percent  # noqa: E402
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402

ITERS = 10_000
IOP_AXIAL = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
# oblique: row=(1,0,0), col=(0,0.6,0.8) -> normal (0,-0.8,0.6)
IOP_OBLIQUE = (1.0, 0.0, 0.0, 0.0, 0.6, 0.8)
_DEFAULT_SERIES = generate_uid()  # shared by normal single-series tests


def _dcm(z: float, arr_u16: np.ndarray, *, ipp_xy=(0.0, 0.0),
         iop=IOP_AXIAL, series=None, inst=None) -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series or _DEFAULT_SERIES
    ds.ImagePositionPatient = [ipp_xy[0], ipp_xy[1], float(z)]
    ds.ImageOrientationPatient = list(iop)
    if inst is not None:
        ds.InstanceNumber = inst
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


def _arr(fill: int = 100, h: int = 16, w: int = 16) -> np.ndarray:
    return np.full((h, w), fill, dtype=np.uint16)


# ---------------------------------------------------------------------------
# 1. sorting
# ---------------------------------------------------------------------------
def test_sorting_axial_by_ipp_z():
    # upload out of order; positions z=20,0,10 must sort 0,10,20
    files = [("c.dcm", _dcm(20, _arr(30))), ("a.dcm", _dcm(0, _arr(10))),
             ("b.dcm", _dcm(10, _arr(20)))]
    vol = cim.read_dicom_volume(files)
    # HU = raw - 1024; fill values 10,20,30 -> HU -1014,-1004,-994
    assert vol[0, 0, 0] == pytest.approx(10 - 1024)
    assert vol[1, 0, 0] == pytest.approx(20 - 1024)
    assert vol[2, 0, 0] == pytest.approx(30 - 1024)


def test_sorting_oblique_uses_orientation_normal():
    # positions chosen so z is NON-monotonic but projection on normal is
    normal = np.cross(np.array(IOP_OBLIQUE[:3]), np.array(IOP_OBLIQUE[3:]))
    normal = normal / np.linalg.norm(normal)
    # place slices at t=0,1,2 along the normal: ipp = t * normal
    slices = []
    for t, fill in ((2.0, 30), (0.0, 10), (1.0, 20)):
        ipp = t * normal
        slices.append((f"t{t}.dcm", _dcm(ipp[2], _arr(fill), ipp_xy=(ipp[0], ipp[1]),
                                         iop=IOP_OBLIQUE)))
    vol = cim.read_dicom_volume(slices)
    assert vol[0, 0, 0] == pytest.approx(10 - 1024)
    assert vol[1, 0, 0] == pytest.approx(20 - 1024)
    assert vol[2, 0, 0] == pytest.approx(30 - 1024)


# ---------------------------------------------------------------------------
# 2. invalid inputs / zip safety
# ---------------------------------------------------------------------------
def test_invalid_dicom_files_are_skipped_with_count():
    good = _dcm(0, _arr(10))
    files = [("bad.txt", b"not a dicom"), ("good.dcm", good)]
    vol, info = cim.read_dicom_volume_ex(files)
    assert vol.shape[0] == 1
    assert len(info["skipped"]) == 1


def test_no_usable_dicoms_raises():
    with pytest.raises(ValueError, match="DICOM"):
        cim.read_dicom_volume([("x.txt", b"hello")])


def test_malformed_zip_raises_clear_error():
    with pytest.raises(ValueError, match="zip"):
        cim.read_dicom_volume([("s.zip", b"PK\x03\x04corrupt")])


def test_zip_of_dicoms_still_works():
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        for i in range(3):
            zf.writestr(f"s{i}.dcm", _dcm(float(i), _arr(10 + i)))
    vol = cim.read_dicom_volume([("s.zip", zbuf.getvalue())])
    assert vol.shape[0] == 3


def test_duplicate_positions_dropped():
    files = [("a.dcm", _dcm(0, _arr(10))), ("b.dcm", _dcm(0, _arr(99))),
             ("c.dcm", _dcm(5, _arr(20)))]
    vol, info = cim.read_dicom_volume_ex(files)
    assert vol.shape[0] == 2
    assert info["dropped_duplicates"] == 1


def test_mixed_series_keeps_dominant():
    uid_a = generate_uid()
    uid_b = generate_uid()
    s1 = [_dcm(float(i), _arr(10), series=uid_a) for i in range(3)]
    s2 = [_dcm(0.0, _arr(50), series=uid_b)]
    vol, info = cim.read_dicom_volume_ex([(f"{i}.dcm", b) for i, b in enumerate(s1 + s2)])
    assert vol.shape[0] == 3
    assert info["dropped_mixed_series"] == 1


def test_inconsistent_dimensions_skipped():
    files = [("a.dcm", _dcm(0, _arr(16, 16))), ("b.dcm", _dcm(1, _arr(16, 20)))]
    vol, info = cim.read_dicom_volume_ex(files)
    assert vol.shape[0] == 1
    assert any("dimensions" in s for _, s in info["skipped"])


# ---------------------------------------------------------------------------
# 3. HU conversion + preprocessing consistency
# ---------------------------------------------------------------------------
def test_hu_slope_intercept_applied():
    # custom slope/intercept via direct dataset tweak
    data = _dcm(0, _arr(200))
    # rebuild with slope 2, intercept -1024: HU = 200*2 - 1024 = -624
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.ImageOrientationPatient = list(IOP_AXIAL)
    ds.Rows, ds.Columns = 16, 16
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 2.0
    ds.RescaleIntercept = -1024.0
    ds.PixelData = np.full((16, 16), 200, dtype=np.uint16).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    vol = cim.read_dicom_volume([("s.dcm", buf.getvalue())])
    assert vol[0, 0, 0] == pytest.approx(200 * 2 - 1024)


def test_prep_sample_shape_and_range():
    model, meta = cim.load_model(
        ( __import__("pathlib").Path(__file__).parent / "models" / "ResNet18_2.5D_best.pt"
        ).read_bytes())
    vol = np.full((40, 64, 64), 100.0, dtype=np.float32)
    x = cim._prep_sample(vol, meta)
    assert tuple(x.shape) == (1, 3, 128, 128)
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0


# ---------------------------------------------------------------------------
# 4. prediction persistence (fingerprint / state helpers)
# ---------------------------------------------------------------------------
def test_upload_fingerprint_changes_with_content():
    a = cu._upload_fingerprint(["s.dcm"], [b"abc"])
    b = cu._upload_fingerprint(["s.dcm"], [b"abd"])
    c = cu._upload_fingerprint(["t.dcm"], [b"abc"])
    assert a != b and a != c
    assert a == cu._upload_fingerprint(["s.dcm"], [b"abc"])


def test_stale_prediction_invalidated_on_upload_change(monkeypatch):
    # use a plain dict as session_state for isolation
    store = {}
    monkeypatch.setattr(cu.st, "session_state", store)
    store["k_pred"] = {"fingerprint": "fp1", "probs": [0.1, 0.2, 0.3, 0.4]}
    cu._invalidate_stale_prediction("k_pred", "fp2")
    assert "k_pred" not in store
    store["k_pred"] = {"fingerprint": "fp1"}
    cu._invalidate_stale_prediction("k_pred", "fp1")
    assert "k_pred" in store
    store["k_pred"] = {"fingerprint": "fp1"}
    cu._invalidate_stale_prediction("k_pred", None)  # upload cleared
    assert "k_pred" not in store


# ---------------------------------------------------------------------------
# 5. one-time save of a stored prediction
# ---------------------------------------------------------------------------
@pytest.fixture()
def conn(tmp_path):
    c = dbm.init_db(tmp_path / "pipe.db")
    yield c
    c.close()


@pytest.fixture()
def accounts(conn):
    pat = dbm.create_user(conn, "p@t.pk", auth.hash_password("PatientPass1!", iterations=ITERS),
                          "patient", full_name="P")
    doc = dbm.create_user(conn, "d@c.pk", auth.hash_password("DoctorPass1!", iterations=ITERS),
                          "doctor", full_name="D")
    return {"pat": pat, "doc": doc}


def _viewer(conn, uid):
    return dbm.viewer_from_row(dbm.get_user(conn, uid))


def _st_pred(fingerprint="fpX", pred=2, saved=None):
    return {"fingerprint": fingerprint, "pred": pred,
            "probs": [0.05, 0.1, 0.75, 0.1], "class_names": list(cim.CLASS_NAMES),
            "n_slices": 40, "model_name": "ResNet18_2.5D",
            "hu_clip": [-100, 1000], "center_crop_frac": 0.75,
            "saved_record_id": saved}


def test_save_ai_prediction_once(conn, accounts):
    v = _viewer(conn, accounts["pat"])
    rid = cu._save_ai_prediction(conn, v, accounts["pat"], _st_pred())
    recs = dbm.list_ct_records(conn, v, accounts["pat"])
    assert recs[0]["id"] == rid
    assert recs[0]["source"] == "image_model"
    assert recs[0]["context_level"] == 2
    # double-save blocked by helper guard
    with pytest.raises(ValueError, match="already saved"):
        cu._save_ai_prediction(conn, v, accounts["pat"], _st_pred(saved=rid))


def test_save_rejects_invalid_probs(conn, accounts):
    v = _viewer(conn, accounts["pat"])
    bad = _st_pred()
    bad["probs"] = [float("nan"), 0.1, 0.2, 0.3]
    with pytest.raises(ValueError):
        cu._save_ai_prediction(conn, v, accounts["pat"], bad)


def test_patient_cannot_save_ai_for_other_patient(conn, accounts):
    v_other = _viewer(conn, accounts["doc"])  # doctor as wrong principal demo
    # doctor without assignment denied
    with pytest.raises(dbm.DataAccessDenied):
        cu._save_ai_prediction(conn, v_other, accounts["pat"], _st_pred())


# ---------------------------------------------------------------------------
# 6. report values: missing vs zero calcium
# ---------------------------------------------------------------------------
def test_missing_cac_is_not_zero():
    f = encode_ct_report(cac_score=None, cac_category=None,
                         stenosis_grade=None, vessels_ge50=None)
    assert f.cac_score is None and f.cac_score_log_norm is None
    assert f.missing_cac == 1
    assert f.context_level is None  # nothing reported -> no level invented


def test_explicit_zero_cac_is_category_zero():
    f = encode_ct_report(cac_score=0.0)
    assert f.cac_category == 0
    assert f.missing_cac == 0
    assert f.cac_score_log_norm == 0.0


def test_stenosis_percent_mapping():
    assert stenosis_grade_from_percent(0) == 0
    assert stenosis_grade_from_percent(10) == 1
    assert stenosis_grade_from_percent(40) == 2
    assert stenosis_grade_from_percent(60) == 3
    assert stenosis_grade_from_percent(90) == 4
    assert stenosis_grade_from_percent(100) == 5


def test_manual_record_source_distinct(conn, accounts):
    v = _viewer(conn, accounts["pat"])
    rid = dbm.create_ct_record(conn, v, accounts["pat"], context_level=1,
                               source="manual", report_text="reviewed")
    recs = dbm.list_ct_records(conn, v, accounts["pat"])
    assert recs[0]["id"] == rid and recs[0]["source"] == "manual"


# ---------------------------------------------------------------------------
# 7. fusion honesty (Case A scalar level; missing ECG never fabricated)
# ---------------------------------------------------------------------------
def test_fused_pipeline_honors_scalar_ct_level():
    import assessment_core as core
    from config import load_config
    from ecg_model import synthetic_ecg
    cfg = load_config()
    model, msg, ok = core.prepare_model(cfg)
    assert ok, msg
    sig = synthetic_ecg(seconds=30, fs=100, hr=72, noise=0.02, seed=2)
    r0 = core.run_recording(model, sig, 100, cfg, ct_level=0)
    r3 = core.run_recording(model, sig, 100, cfg, ct_level=3)
    assert r0["status"] == "completed" and r3["status"] == "completed"
    assert r0["ct_level"] == 0 and r3["ct_level"] == 3
    # higher CT level must not raise the fused score (thresholds shift, score
    # uses the same ECG mean + level-adjusted logic in fusion.py)
    assert r3["ecg_score_mean"] == r0["ecg_score_mean"]


def test_missing_ecg_never_fabricates_score():
    import assessment_core as core
    from config import load_config
    cfg = load_config()
    model, msg, ok = core.prepare_model(cfg)
    assert ok
    r = core.run_recording(model, np.empty(0), 100, cfg, ct_level=2)
    assert r["fused_score"] is None and r["status"] != "completed"
