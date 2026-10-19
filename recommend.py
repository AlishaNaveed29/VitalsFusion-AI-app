"""Action recommendation layer for VitalsFusion AI (non-diagnostic prototype).

Produces one of four categories from real model outputs and workflow state:

  * Emergency evaluation   - fused score at/above the emergency threshold, OR
                              an open high-risk event, OR user-reported
                              emergency symptoms.
  * Prompt medical review  - concerning ECG/fusion evidence that is not yet
                              at the emergency threshold, OR high CT context
                              with any elevated ECG signal.
  * Routine follow-up      - completed assessment with non-concerning ECG
                              evidence and CT context that does not override it.
  * Insufficient data      - missing/failed inputs; never a reassurance.

Safety rules (hard):
  1. A reassuring CT result NEVER overrides concerning ECG findings.
  2. Emergency symptoms always force "Emergency evaluation".
  3. Missing/failed assessment NEVER maps to "Routine follow-up".
  4. Output is labelled experimental / non-validated.

This is a transparent, modular rule layer. It can be replaced by a properly
trained fusion model once paired CT-ECG patient data becomes available.
"""

from __future__ import annotations

from typing import List, Optional

from fusion import FusionConfig, adjusted_thresholds

# Category identifiers (stable keys for UI / persistence)
CAT_EMERGENCY = "emergency_evaluation"
CAT_PROMPT = "prompt_medical_review"
CAT_ROUTINE = "routine_follow_up"
CAT_INSUFFICIENT = "insufficient_data"

CATEGORY_LABELS = {
    CAT_EMERGENCY: "Emergency evaluation",
    CAT_PROMPT: "Prompt medical review",
    CAT_ROUTINE: "Routine follow-up",
    CAT_INSUFFICIENT: "Insufficient data - cannot assess",
}

CATEGORY_CSS = {
    CAT_EMERGENCY: "vf-strip",
    CAT_PROMPT: "vf-strip-amber",
    CAT_ROUTINE: "vf-strip-info",
    CAT_INSUFFICIENT: "vf-strip-info",
}

# Design thresholds (not clinical constants) ---------------------------------
# ECG mean score above this is "concerning" on its own (watch-level evidence)
ECG_CONCERNING = 0.50
# ECG mean score at/above this is high concern regardless of CT
ECG_HIGH_CONCERN = 0.80
# CT context level at/above this escalates to prompt review even with mild ECG
CT_HIGH_LEVEL = 2

DISCLAIMER = (
    "Experimental, non-validated assessment layer. These action categories are "
    "rule-based design parameters derived from model outputs - they are NOT "
    "clinically validated recommendations and do not replace a doctor's judgement. "
    "Coronary calcium severity alone cannot determine whether someone is having "
    "an acute heart attack."
)

SYMPTOM_WARNING = (
    "If you have chest pain, pressure, shortness of breath, sweating, fainting, "
    "or other possible heart-attack symptoms, seek emergency medical care "
    "immediately - do not wait for this application or for an AI score."
)


def recommend_action(
    *,
    status: str,
    ecg_score_mean: Optional[float] = None,
    ct_level: Optional[int] = None,
    fused_score: Optional[float] = None,
    fused_level: Optional[int] = None,
    heart_rate_bpm: Optional[float] = None,
    high_risk_event_open: bool = False,
    emergency_symptoms: bool = False,
    cfg_high_risk_threshold: float = 0.85,
    fusion_cfg: Optional[FusionConfig] = None,
) -> dict:
    """Return a structured action recommendation.

    Parameters mirror the fields already available after ``run_recording``
    plus workflow state (open event, symptom checkbox).

    Returns dict with:
        category      - one of CAT_*
        label         - human-readable category
        factors       - list of strings explaining every contributing factor
        explanation   - one-paragraph summary
        disclaimer    - fixed experimental-layer disclaimer
        symptom_warning - always present emergency-symptom advice
    """
    fusion_cfg = fusion_cfg or FusionConfig()
    factors: List[str] = []

    # ---- hard overrides ----------------------------------------------------
    if emergency_symptoms:
        factors.append(
            "Emergency symptoms reported by the user (chest pain / breathlessness "
            "/ fainting / similar) - immediate emergency evaluation required "
            "regardless of any AI score."
        )
        return _result(CAT_EMERGENCY, factors,
                       "Emergency symptoms take priority over every model output.")

    if high_risk_event_open:
        factors.append(
            f"An open high-risk event exists (fused score at/above the emergency "
            f"threshold {cfg_high_risk_threshold:.2f})."
        )
        return _result(CAT_EMERGENCY, factors,
                       "An open high-risk event requires emergency evaluation.")

    # ---- insufficient data -------------------------------------------------
    if status != "completed":
        factors.append(f"Assessment status is '{status}' - no reliable fused "
                       "result is available.")
        if ecg_score_mean is None:
            factors.append("No valid ECG model score (signal quality gate or "
                           "missing recording).")
        if ct_level is None:
            factors.append("No CT context level on file.")
        return _result(CAT_INSUFFICIENT, factors,
                       "Inputs are missing or the assessment failed. A failed "
                       "assessment never means low risk - re-record and re-run, "
                       "and seek care if symptoms are present.")

    if fused_score is None and ecg_score_mean is None:
        factors.append("Both fused score and ECG mean score are missing.")
        return _result(CAT_INSUFFICIENT, factors,
                       "No model scores available to base a recommendation on.")

    # ---- emergency via fused score ----------------------------------------
    if fused_score is not None and fused_score >= cfg_high_risk_threshold:
        factors.append(
            f"Fused risk score {fused_score:.3f} is at/above the emergency "
            f"threshold {cfg_high_risk_threshold:.2f}."
        )
        if ct_level is not None:
            factors.append(f"CT context level {ct_level}/3 contributed to the "
                           "threshold adjustment.")
        return _result(CAT_EMERGENCY, factors,
                       "The fused score crossed the emergency threshold - "
                       "immediate medical evaluation is advised.")

    # ---- ECG-alone concern (CT must not override) -------------------------
    watch_thr, alert_thr = adjusted_thresholds(fusion_cfg, ct_level)
    ecg_concerning = ecg_score_mean is not None and ecg_score_mean >= ECG_CONCERNING
    ecg_high = ecg_score_mean is not None and ecg_score_mean >= ECG_HIGH_CONCERN
    ct_elevated = ct_level is not None and ct_level >= CT_HIGH_LEVEL

    if ecg_score_mean is not None:
        factors.append(
            f"ECG model mean score {ecg_score_mean:.3f} "
            f"(watch threshold {watch_thr:.2f}, alert threshold {alert_thr:.2f} "
            f"after CT adjustment)."
        )
    if ct_level is not None:
        factors.append(f"CT context level {ct_level}/3 "
                       f"({'elevated' if ct_elevated else 'not elevated'}).")
    else:
        factors.append("CT not reported - recommendation relies on ECG evidence only.")
    if fused_level is not None:
        from fusion import LEVEL_NAMES
        factors.append(f"Fusion level: {LEVEL_NAMES.get(fused_level, fused_level)}.")

    # Rule: high ECG concern -> prompt review at minimum, never downgraded by CT
    if ecg_high or (ecg_concerning and ct_elevated):
        detail = ("ECG model score is in the high-concern range"
                  if ecg_high else
                  "ECG model score is concerning and CT context is elevated")
        factors.append(detail + " - a reassuring CT cannot cancel this.")
        return _result(CAT_PROMPT, factors,
                       "Concerning ECG evidence requires clinician review even if "
                       "the CT looks reassuring. Coronary calcium alone cannot "
                       "rule out an acute event.")

    if ecg_concerning:
        factors.append("ECG model score is in the watch range - clinician review "
                       "is recommended.")
        return _result(CAT_PROMPT, factors,
                       "The ECG branch shows watch-level evidence that warrants "
                       "prompt medical review.")

    if ct_elevated and fused_score is not None and fused_score >= 0.40:
        factors.append(
            f"Elevated CT context (level {ct_level}) with a moderate fused score "
            f"({fused_score:.3f}) - routine follow-up with the ordering clinician "
            "is advised."
        )
        return _result(CAT_ROUTINE, factors,
                       "CT findings plus a moderate fused score support a routine "
                       "follow-up conversation, not an emergency.")

    if ct_elevated:
        factors.append(
            f"CT context level {ct_level}/3 is elevated but the ECG model score "
            "is not concerning. Calcium severity alone cannot diagnose an acute "
            "event; discuss the CT result with your clinician at a routine visit."
        )
        return _result(CAT_ROUTINE, factors,
                       "Elevated CT context with non-concerning ECG supports "
                       "routine follow-up for the chronic finding.")

    # ---- low concern path --------------------------------------------------
    if ecg_score_mean is not None and fused_score is not None:
        factors.append(
            f"ECG mean {ecg_score_mean:.3f} and fused score {fused_score:.3f} "
            "are below the watch / emergency design thresholds."
        )
        if ct_level is not None and ct_level == 0:
            factors.append("CT context level 0 (low).")
        factors.append("This does NOT rule out disease - the models have known "
                       "limitations and the prototype is not clinically validated.")
        return _result(CAT_ROUTINE, factors,
                       "Current model outputs do not flag elevated risk. Routine "
                       "follow-up only; seek care immediately if symptoms appear.")

    factors.append("Not enough scored evidence to categorise beyond routine "
                   "awareness.")
    return _result(CAT_ROUTINE, factors,
                   "No concerning model evidence on this run. Stay aware of "
                   "symptoms and follow up with your clinician as usual.")


def _result(category: str, factors: List[str], explanation: str) -> dict:
    return {
        "category": category,
        "label": CATEGORY_LABELS[category],
        "factors": list(factors),
        "explanation": explanation,
        "disclaimer": DISCLAIMER,
        "symptom_warning": SYMPTOM_WARNING,
    }
