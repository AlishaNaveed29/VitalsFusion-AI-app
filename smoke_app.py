"""Headless UI smoke test using streamlit.testing.v1.AppTest.

Not part of the pytest suite (kept as scripts/smoke_app.py style); run:
    python smoke_app.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="vf_smoke_"))
os.environ["VITALSFUSION_DB"] = str(TMP / "smoke.db")
os.environ["VITALSFUSION_CONFIG"] = str(TMP / "no_config.json")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import auth  # noqa: E402
import db as dbm  # noqa: E402
from config import load_config  # noqa: E402

cfg = load_config()
conn = dbm.init_db(cfg.db_path)

DOC_EMAIL, DOC_PW = "smoke-doc@clinic.pk", "SmokeDocPass1!"
PAT_EMAIL, PAT_PW = "smoke-pat@test.pk", "SmokePatPass1!"
doc_id = dbm.create_user(conn, DOC_EMAIL, auth.hash_password(DOC_PW, iterations=10_000),
                         "doctor", full_name="Smoke Doctor")
pat_id = dbm.create_user(conn, PAT_EMAIL, auth.hash_password(PAT_PW, iterations=10_000),
                         "patient", full_name="Smoke Patient")
dbm.assign_patient(conn, doc_id, pat_id, doc_id)
conn.close()

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent / "app.py"


def run(at, timeout=60):
    at.run(timeout=timeout)
    if at.exception:
        for exc in at.exception:
            print("EXCEPTION:", exc.value)
        raise SystemExit(1)
    return at


print("== 1. login page renders ==")
at = run(AppTest.from_file(str(APP), default_timeout=60))
titles = [t.value for t in at.title]
print("titles:", titles)
assert any("VitalsFusion" in t for t in titles), titles

print("== 2. doctor login -> doctor dashboard ==")
at.text_input[0].set_value(DOC_EMAIL)          # email/username
at.text_input[1].set_value(DOC_PW)             # password
# find the login form submit button
submit = next(b for b in at.button if b.label == "Sign in")
submit.click()
run(at)
texts = [t.value for t in at.title] + [m.value for m in at.markdown]
print("headings:", texts[:3])
assert any("Doctor dashboard" in t for t in texts), texts

print("== 3. doctor sees assigned patient + lab ==")
# review tab navigation via radio
nav = at.radio[0]
nav.set_value("🩺 Patient review")
run(at)
print("patient review OK; selects:", len(at.selectbox))
assert len(at.selectbox) >= 1, "patient review page has no patient selector"

print("== 4. logout ==")
logout = next(b for b in at.button if b.label == "Sign out")
logout.click()
run(at)
assert any("VitalsFusion" in t.value for t in at.title)

print("== 5. patient login -> patient dashboard ==")
at.text_input[0].set_value(PAT_EMAIL)
at.text_input[1].set_value(PAT_PW)
next(b for b in at.button if b.label == "Sign in").click()
run(at)
texts = [t.value for t in at.title] + [m.value for m in at.markdown]
print("headings:", texts[:3])
assert any("Welcome" in t for t in texts), texts

print("== 6. patient visits Emergency page ==")
at.radio[0].set_value("🆘 Emergency")
run(at)
print("emergency page OK")

print("== 7. patient monitoring page (start/stop fragment) ==")
at.radio[0].set_value("📈 ECG monitoring")
run(at)
print("monitoring page OK")

print("== 8. invalid credentials on fresh session ==")
at2 = run(AppTest.from_file(str(APP), default_timeout=60))
at2.text_input[0].set_value("ghost@x.y")
at2.text_input[1].set_value("whatever123")
next(b for b in at2.button if b.label == "Sign in").click()
run(at2)
errs = [e.value for e in at2.error]
print("errors:", errs)
assert any("Invalid" in e for e in errs), errs

print("== 9. patient starts monitoring then runs an assessment ==")
at.text_input  # patient session from step 7
at.radio[0].set_value("📈 ECG monitoring")
run(at)
start = next(b for b in at.button if b.label == "Start monitoring")
start.click()
run(at)
# let the monitoring loop accumulate >= 10 s of samples (fragment caps 5 s/tick)
for _ in range(3):
    time.sleep(6)
    run(at)
run_btn = next((b for b in at.button
                if b.label.startswith("Run fused assessment on the current")), None)
assert run_btn is not None, "Run assessment button missing"
run_btn.click()
run(at)

conn2 = dbm.init_db(cfg.db_path)
rows = conn2.execute(
    "SELECT status, fused_score, error_message FROM ai_assessments "
    "WHERE patient_id=? ORDER BY id DESC", (pat_id,)).fetchall()
print("assessments:", [dict(r) for r in rows])
assert rows, "no assessment was recorded by the UI run"
latest = dict(rows[0])

import assessment_core as _core  # noqa: E402
model_path = _core.find_model_zip(cfg)
print("ECG model:", model_path)
if model_path is not None:
    # with the trained model present the real pipeline must COMPLETE with a score
    assert latest["status"] == "completed", latest
    assert latest["fused_score"] is not None, latest
    assert latest["error_message"] is None, latest
else:
    # without the model the app must report UNAVAILABLE - never a fake score
    assert latest["status"] in ("unavailable", "failed"), latest
    assert latest["fused_score"] is None, latest
    assert latest["error_message"], latest
conn2.close()

print("== 10. register a new doctor via the login-page form ==")
at3 = run(AppTest.from_file(str(APP), default_timeout=60))
role_radio = next((r for r in at3.radio if r.label == "I am registering as"), None)
assert role_radio is not None, "registration role choice missing"
role_radio.set_value("Doctor")
run(at3)
# widget order: [0]=login email, [1]=login password, [2..5]=signup form
at3.text_input[2].set_value("regdoc@clinic.pk")
at3.text_input[3].set_value("Reg Doctor")
at3.text_input[4].set_value("RegDocPass1!")
at3.text_input[5].set_value("RegDocPass1!")
next(b for b in at3.button if b.label == "Create account").click()
run(at3)
msgs = [s.value.lower() for s in at3.success]
print("signup messages:", msgs)
assert any("account created" in m for m in msgs), msgs
# sign in as the freshly registered doctor
at3.text_input[0].set_value("regdoc@clinic.pk")
at3.text_input[1].set_value("RegDocPass1!")
next(b for b in at3.button if b.label == "Sign in").click()
run(at3)
texts = [t.value for t in at3.title] + [m.value for m in at3.markdown]
print("headings:", texts[:3])
assert any("Doctor dashboard" in t for t in texts), texts
print("doctor self-registration + login OK")

print("\nALL SMOKE CHECKS PASSED")
