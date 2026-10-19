"""Report generation tests (CSV + dependency-free PDF)."""

import reports as rep


def test_rows_to_csv_basic_and_empty():
    csv = rep.rows_to_csv([{"a": 1, "b": "x"}, {"a": 2, "b": None}])
    assert csv.splitlines()[0] == "a,b"
    assert "2," in csv
    assert rep.rows_to_csv([], columns=["a", "b"]) == "a,b\n"


def test_multi_section_csv_labels_sections():
    out = rep.multi_section_csv([
        ("Assessments", [{"id": 1, "status": "completed"}]),
        ("Events", []),
    ])
    assert "# Assessments" in out and "# Events" in out and "(no records)" in out


def test_build_pdf_is_structurally_valid_single_page():
    pdf = rep.build_pdf("Title here", ["line one", "line two"], footer="foot")
    assert pdf.startswith(b"%PDF-1.4")
    assert pdf.rstrip().endswith(b"%%EOF")
    assert b"Title here" in pdf
    # xref offset points at the xref table
    startxref = int(pdf.split(b"startxref")[1].split()[0])
    assert pdf[startxref:startxref + 4] == b"xref"


def test_build_pdf_paginates_long_content():
    lines = [f"item number {i} with some words" for i in range(120)]
    pdf = rep.build_pdf("Long report", lines)
    assert pdf.count(b"/Type /Page ") >= 3          # pages (not /Pages)
    assert b"item number 119" in pdf
    assert b"continued" in pdf


def test_assessment_report_pdf_contains_key_sections():
    patient = {"full_name": "Synthetic Patient", "email": "syn@test.pk"}
    assessments = [{"id": 1, "created_at": "2026-01-01T00:00:00+00:00",
                    "status": "completed", "fused_score": 0.93, "fused_level": 2,
                    "heart_rate_bpm": 118.0, "error_message": None}]
    events = [{"id": 7, "event_uid": "abcdef0123456789", "created_at": "2026-01-01T00:05:00+00:00",
               "status": "open", "score": 0.93, "threshold": 0.85,
               "heart_rate_bpm": 118.0, "patient_call_initiated": 1,
               "patient_call_at": "2026-01-01T00:06:00+00:00"}]
    notes = [{"created_at": "2026-01-01T00:10:00+00:00", "doctor_name": "Dr Test",
              "doctor_email": "dr@t.pk", "note": "Reviewed ECG windows."}]
    pdf = rep.assessment_report_pdf(patient, assessments, events, notes)
    assert pdf.startswith(b"%PDF-1.4")
    for token in (b"Synthetic Patient", b"AI ASSESSMENTS", b"HIGH-RISK EVENTS",
                  b"CLINICAL NOTES", b"DISCLAIMER", b"heart attack",
                  b"Reviewed ECG windows", b"RECOMMENDED NEXT STEPS",
                  b"Emergency evaluation", b"EMERGENCY WARNING"):
        assert token in pdf, token


def test_assessment_report_pdf_includes_ct_and_ecg_sections():
    patient = {"full_name": "P", "email": "p@x"}
    assessments = [{"id": 1, "created_at": "t", "status": "completed",
                    "fused_score": 0.2, "fused_level": 0,
                    "ecg_score_mean": 0.15, "ct_level": 0,
                    "heart_rate_bpm": 70.0, "error_message": None}]
    ct_records = [{"id": 3, "created_at": "t", "source": "report",
                   "context_level": 1, "cac_score": 50.0, "cac_category": 1,
                   "stenosis_grade": None, "vessels_ge50": None,
                   "report_text": "mild calcium"}]
    ecg_sessions = [{"id": 5, "started_at": "t", "status": "stopped",
                     "n_samples": 6000, "fs_hz": 100, "mean_hr_bpm": 72.0,
                     "source": "demo"}]
    pdf = rep.assessment_report_pdf(patient, assessments, [], (),
                                    ct_records=ct_records, ecg_sessions=ecg_sessions)
    for token in (b"CT / RADIOLOGY RECORDS", b"ECG RECORDINGS",
                  b"mild calcium", b"source=report"):
        assert token in pdf, token


def test_pdf_escapes_parentheses_and_backslashes():
    pdf = rep.build_pdf("T", ["weird (value) and \\ path"])
    assert b"weird \\(value\\) and \\\\ path" in pdf
