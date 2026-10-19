"""
Report-derived CT features for VitalsFusion AI (non-diagnostic prototype)
=========================================================================
The CT branch does NOT analyse images. It takes values a radiologist already
reported (calcium score, stenosis grade, number of diseased vessels), encodes
them as a small numeric feature vector, and turns them into a transparent
"CT context level" that the rule-based fusion layer can use.

No machine learning is involved here, so nothing needs paired CT+ECG data.
"""

import math
from dataclasses import dataclass, asdict
from typing import Optional

SCHEMA_VERSION = "ct-report-features/1.0"

# --- Agatston coronary artery calcium (CAC) categories -----------------------
CAC_CATEGORIES = {
    0: "No calcium (score 0)",
    1: "Mild (1-99)",
    2: "Moderate (100-399)",
    3: "Severe (400 or more)",
}

# --- Stenosis grades (CAD-RADS style, simplified) ----------------------------
STENOSIS_GRADES = {
    0: "None (0%)",
    1: "Minimal (1-24%)",
    2: "Mild (25-49%)",
    3: "Moderate (50-69%)",
    4: "Severe (70-99%)",
    5: "Total occlusion (100%)",
}

CONTEXT_LEVELS = {
    0: "Level 0 - low CT context",
    1: "Level 1 - mildly raised",
    2: "Level 2 - raised",
    3: "Level 3 - high",
}

# Stenosis grade -> context points (0..3)
STENOSIS_TO_POINTS = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 3}

DISCLAIMER = (
    "Non-diagnostic research prototype. Values come from a radiology report "
    "entered by the user. The output is a personalised-alert setting, not a "
    "diagnosis and not a prediction of a heart attack."
)


def cac_category_from_score(score: float) -> int:
    """Standard Agatston bands: 0 | 1-99 | 100-399 | >=400."""
    if score is None or score < 0 or math.isnan(score):
        raise ValueError("Agatston score must be a number >= 0")
    if score == 0:
        return 0
    if score < 100:
        return 1
    if score < 400:
        return 2
    return 3


def stenosis_grade_from_percent(percent: float) -> int:
    """Map the worst reported stenosis (%) to a grade 0..5."""
    if percent is None or percent < 0 or percent > 100 or math.isnan(percent):
        raise ValueError("Stenosis percent must be between 0 and 100")
    if percent == 0:
        return 0
    if percent < 25:
        return 1
    if percent < 50:
        return 2
    if percent < 70:
        return 3
    if percent < 100:
        return 4
    return 5


@dataclass
class CTFeatures:
    cac_score: Optional[float]          # raw Agatston score if reported
    cac_category: Optional[int]         # 0..3 or None if not reported
    stenosis_grade: Optional[int]       # 0..5 or None if not reported
    vessels_ge50: Optional[int]         # number of vessels with >=50% stenosis, 0..3

    # encoded, model/fusion-ready values in 0..1 (None = not reported)
    cac_cat_norm: Optional[float]
    cac_score_log_norm: Optional[float]
    stenosis_norm: Optional[float]
    vessels_norm: Optional[float]

    # which inputs are missing (1 = missing)
    missing_cac: int
    missing_stenosis: int
    missing_vessels: int

    context_level: Optional[int]        # 0..3 or None if nothing reported


def encode_ct_report(
    cac_score: Optional[float] = None,
    cac_category: Optional[int] = None,
    stenosis_grade: Optional[int] = None,
    vessels_ge50: Optional[int] = None,
) -> CTFeatures:
    """Encode report values. Provide cac_score OR cac_category (score wins).

    Any value left as None is treated as 'not reported' and flagged as missing,
    never silently replaced with zero.
    """
    if cac_score is not None:
        cac_category = cac_category_from_score(cac_score)
    if cac_category is not None and cac_category not in CAC_CATEGORIES:
        raise ValueError("cac_category must be 0..3")
    if stenosis_grade is not None and stenosis_grade not in STENOSIS_GRADES:
        raise ValueError("stenosis_grade must be 0..5")
    if vessels_ge50 is not None and vessels_ge50 not in (0, 1, 2, 3):
        raise ValueError("vessels_ge50 must be 0..3")

    cac_log = None
    if cac_score is not None:
        # log scale so 10 vs 100 matters more than 2000 vs 2090; 4000 = cap
        cac_log = min(1.0, math.log1p(cac_score) / math.log1p(4000.0))

    points = []
    if cac_category is not None:
        points.append(cac_category)
    if stenosis_grade is not None:
        points.append(STENOSIS_TO_POINTS[stenosis_grade])
    if vessels_ge50 is not None:
        points.append(min(3, vessels_ge50))
    # Conservative and easy to explain: the highest reported item sets the level.
    context_level = max(points) if points else None

    return CTFeatures(
        cac_score=cac_score,
        cac_category=cac_category,
        stenosis_grade=stenosis_grade,
        vessels_ge50=vessels_ge50,
        cac_cat_norm=None if cac_category is None else cac_category / 3.0,
        cac_score_log_norm=cac_log,
        stenosis_norm=None if stenosis_grade is None else stenosis_grade / 5.0,
        vessels_norm=None if vessels_ge50 is None else vessels_ge50 / 3.0,
        missing_cac=int(cac_category is None),
        missing_stenosis=int(stenosis_grade is None),
        missing_vessels=int(vessels_ge50 is None),
        context_level=context_level,
    )


def adjusted_threshold(base_threshold: float, context_level: Optional[int],
                       step_per_level: float) -> float:
    """Illustrative fusion rule: higher CT context level -> lower alert threshold.

    The step size is a tunable design parameter, not a clinical value.
    """
    level = 0 if context_level is None else context_level
    return base_threshold - step_per_level * level


def to_report_dict(features: CTFeatures) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "features": asdict(features),
        "context_level_name": None if features.context_level is None
        else CONTEXT_LEVELS[features.context_level],
        "disclaimer": DISCLAIMER,
    }
