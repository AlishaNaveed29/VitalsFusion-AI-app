"""Report generation (CSV + PDF) with no external dependencies.

The PDF writer below emits a minimal but valid PDF 1.4 document using the
built-in Helvetica font, so reports work fully offline without reportlab/fpdf.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Optional, Sequence


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def rows_to_csv(rows: Sequence[dict], columns: Optional[Sequence[str]] = None) -> str:
    buf = io.StringIO()
    if not rows:
        return (",".join(columns) + "\n") if columns else ""
    cols = list(columns or rows[0].keys())
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    return buf.getvalue()


def multi_section_csv(sections: Sequence[tuple[str, Sequence[dict]]]) -> str:
    """Several labelled tables in one CSV file (patient/doctor export)."""
    buf = io.StringIO()
    for title, rows in sections:
        buf.write(f"# {title}\n")
        if rows:
            cols = list(rows[0].keys())
            writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
        else:
            buf.write("(no records)\n")
        buf.write("\n")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
_MAX_LINE = 95
_LINES_PER_PAGE = 48


def _esc(text: str) -> str:
    out = str(text).replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    return "".join(ch if ord(ch) < 128 else "?" for ch in out)


def _wrap(text: str, width: int = _MAX_LINE) -> list[str]:
    text = str(text)
    if not text:
        return [""]
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out or [""]


def build_pdf(title: str, body_lines: Sequence[str],
              footer: str = "") -> bytes:
    """Render *title* + *body_lines* into a multi-page PDF document."""
    wrapped: list[str] = []
    for ln in body_lines:
        wrapped.extend(_wrap(ln))
    pages = [wrapped[i:i + _LINES_PER_PAGE] for i in range(0, len(wrapped), _LINES_PER_PAGE)] or [[]]

    objects: list[bytes] = []          # 1-based object bodies (without "N 0 obj")
    page_ids = []
    font_id = 3
    next_id = 4

    # obj 1 catalog, obj 2 pages (filled later), obj 3 font
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")                # placeholder for pages
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for page_no, lines in enumerate(pages, start=1):
        content_lines = [
            "BT",
            "/F1 14 Tf",
            f"50 795 Td ({_esc(title)}) Tj",
            "ET",
            "BT",
            "/F1 9 Tf",
            "11 TL",
            "50 775 Td",
        ]
        if page_no > 1:
            content_lines.append(f"({_esc(title + ' (continued)')}) Tj")
            content_lines.append("T*")
        for ln in lines:
            content_lines.append(f"({_esc(ln)}) Tj")
            content_lines.append("T*")
        if footer and page_no == len(pages):
            content_lines.append("T*")
            content_lines.append(f"({_esc(footer)}) Tj")
        content_lines.append("ET")
        stream = "\n".join(content_lines).encode("latin-1", errors="replace")

        content_id = next_id
        page_id = next_id + 1
        next_id += 2
        objects.append(b"<< /Length " + str(len(stream)).encode()
                       + b" >>\nstream\n" + stream + b"\nendstream")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode()
        )
        page_ids.append(page_id)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode())
        out.write(body)
        out.write(b"\nendobj\n")
    xref_pos = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
              f"startxref\n{xref_pos}\n%%EOF\n".encode())
    return out.getvalue()


def assessment_report_pdf(patient: dict, assessments: Sequence[dict],
                          events: Sequence[dict], notes: Sequence[dict] = (),
                          extra: Sequence[str] = (),
                          ct_records: Sequence[dict] = (),
                          ecg_sessions: Sequence[dict] = ()) -> bytes:
    lines: list[str] = []
    lines.append(f"Patient: {patient.get('full_name') or '-'}  <{patient.get('email')}>")
    lines.append(f"Generated: {_now()}")
    lines.append("")

    # --- action recommendation from the latest completed assessment ---------
    lines.append("RECOMMENDED NEXT STEPS")
    lines.append("-" * _MAX_LINE)
    latest_completed = next((a for a in assessments
                             if a.get("status") == "completed"), None)
    if latest_completed:
        import recommend as rec
        r = rec.recommend_action(
            status=latest_completed.get("status", "completed"),
            ecg_score_mean=latest_completed.get("ecg_score_mean"),
            ct_level=latest_completed.get("ct_level"),
            fused_score=latest_completed.get("fused_score"),
            fused_level=latest_completed.get("fused_level"),
            heart_rate_bpm=latest_completed.get("heart_rate_bpm"),
            high_risk_event_open=any(e.get("status") in ("open", "acknowledged")
                                     for e in events),
        )
        lines.append(f"CATEGORY: {r['label']}")
        lines.append(r["explanation"])
        lines.append("Factors:")
        for f in r["factors"]:
            lines.append(f"  - {f}")
        lines.append("")
        lines.append("EMERGENCY WARNING")
        lines.append(r["symptom_warning"])
    else:
        lines.append("No completed assessment available to base a recommendation on.")
    lines.append("")

    # --- CT records --------------------------------------------------------
    lines.append("CT / RADIOLOGY RECORDS")
    lines.append("-" * _MAX_LINE)
    if not ct_records:
        lines.append("No CT records recorded.")
    for c in ct_records:
        lines.append(
            f"#{c['id']}  {c.get('created_at')}  source={c.get('source')}  "
            f"CT_level={c.get('context_level')}  "
            f"CAC_score={c.get('cac_score')}  CAC_cat={c.get('cac_category')}  "
            f"stenosis={c.get('stenosis_grade')}  vessels>=50%={c.get('vessels_ge50')}")
        if c.get("report_text"):
            lines.append(f"    note: {str(c['report_text'])[:200]}")
    lines.append("")

    # --- ECG sessions ------------------------------------------------------
    lines.append("ECG RECORDINGS")
    lines.append("-" * _MAX_LINE)
    if not ecg_sessions:
        lines.append("No ECG recordings.")
    for s in ecg_sessions:
        lines.append(
            f"#{s['id']}  {s.get('started_at')}  status={s.get('status')}  "
            f"samples={s.get('n_samples')}  fs={s.get('fs_hz')} Hz  "
            f"mean_HR={s.get('mean_hr_bpm')}  source={s.get('source')}")
    lines.append("")

    lines.append("AI ASSESSMENTS")
    lines.append("-" * _MAX_LINE)
    if not assessments:
        lines.append("No assessments recorded.")
    for a in assessments:
        score = "n/a" if a.get("fused_score") is None else f"{a['fused_score']:.4f}"
        hr = "n/a" if a.get("heart_rate_bpm") is None else f"{a['heart_rate_bpm']:.1f} bpm"
        ecg_m = "n/a" if a.get("ecg_score_mean") is None else f"{a['ecg_score_mean']:.4f}"
        ct_l = "n/a" if a.get("ct_level") is None else str(a["ct_level"])
        lines.append(f"#{a['id']}  {a['created_at']}  status={a['status']}  "
                     f"fused_score={score}  ecg_mean={ecg_m}  ct_level={ct_l}  "
                     f"HR={hr}  level={a.get('fused_level')}")
        if a.get("error_message"):
            lines.append(f"    note: {a['error_message']}")
    lines.append("")
    lines.append("HIGH-RISK EVENTS")
    lines.append("-" * _MAX_LINE)
    if not events:
        lines.append("No high-risk events recorded.")
    for e in events:
        lines.append(f"#{e['id']} ({e['event_uid'][:8]})  {e['created_at']}  "
                     f"status={e['status']}  score={e['score']:.4f} "
                     f"threshold={e['threshold']}  HR={e.get('heart_rate_bpm')}")
        lines.append(f"    call_initiated={bool(e.get('patient_call_initiated'))}"
                     f"  call_at={e.get('patient_call_at') or '-'}")
    lines.append("")
    lines.append("CLINICAL NOTES")
    lines.append("-" * _MAX_LINE)
    if not notes:
        lines.append("No clinical notes.")
    for n in notes:
        lines.append(f"{n['created_at']}  {n.get('doctor_name') or n.get('doctor_email')}: "
                     f"{n['note']}")
    lines.append("")
    lines.append("DISCLAIMER")
    lines.append("-" * _MAX_LINE)
    lines.append("Non-diagnostic research prototype. This report is not a diagnosis "
                 "and not a prediction of a heart attack. If you have chest pain or "
                 "other concerning symptoms, seek emergency medical care regardless "
                 "of any score shown here. Action categories are experimental, "
                 "rule-based design parameters - not clinically validated "
                 "recommendations.")
    lines.extend(extra)
    return build_pdf("VitalsFusion AI - Assessment report", lines)
