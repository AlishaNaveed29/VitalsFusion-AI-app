"""Tests for the action-recommendation layer (recommend.py)."""
from __future__ import annotations

import pytest

from fusion import FusionConfig
from recommend import (
    CAT_EMERGENCY,
    CAT_INSUFFICIENT,
    CAT_PROMPT,
    CAT_ROUTINE,
    CATEGORY_LABELS,
    DISCLAIMER,
    SYMPTOM_WARNING,
    recommend_action,
)


def test_emergency_symptoms_force_emergency():
    r = recommend_action(status="completed", ecg_score_mean=0.10,
                         ct_level=0, fused_score=0.10, emergency_symptoms=True)
    assert r["category"] == CAT_EMERGENCY
    assert r["label"] == CATEGORY_LABELS[CAT_EMERGENCY]
    assert any("symptom" in f.lower() for f in r["factors"])


def test_open_event_forces_emergency():
    r = recommend_action(status="completed", fused_score=0.90,
                         high_risk_event_open=True)
    assert r["category"] == CAT_EMERGENCY


def test_fused_score_at_threshold_is_emergency():
    r = recommend_action(status="completed", fused_score=0.85,
                         cfg_high_risk_threshold=0.85)
    assert r["category"] == CAT_EMERGENCY


def test_high_ecg_not_downgraded_by_low_ct():
    """A reassuring CT must never cancel concerning ECG evidence."""
    r = recommend_action(status="completed", ecg_score_mean=0.90,
                         ct_level=0, fused_score=0.90,
                         cfg_high_risk_threshold=0.99)
    assert r["category"] == CAT_PROMPT
    assert any("reassuring CT" in f or "cannot cancel" in f.lower()
               or "high-concern" in f for f in r["factors"])


def test_ecg_concerning_plus_high_ct_is_prompt():
    r = recommend_action(status="completed", ecg_score_mean=0.60,
                         ct_level=3, fused_score=0.75,
                         cfg_high_risk_threshold=0.99)
    assert r["category"] == CAT_PROMPT


def test_ecg_watch_alone_is_prompt():
    r = recommend_action(status="completed", ecg_score_mean=0.55,
                         ct_level=0, fused_score=0.55,
                         cfg_high_risk_threshold=0.99)
    assert r["category"] == CAT_PROMPT


def test_failed_status_is_insufficient_not_routine():
    r = recommend_action(status="failed")
    assert r["category"] == CAT_INSUFFICIENT
    # never routine
    assert r["category"] != CAT_ROUTINE


def test_missing_scores_is_insufficient():
    r = recommend_action(status="completed", ecg_score_mean=None,
                         fused_score=None)
    assert r["category"] == CAT_INSUFFICIENT


def test_low_scores_with_low_ct_is_routine():
    r = recommend_action(status="completed", ecg_score_mean=0.20,
                         ct_level=0, fused_score=0.25,
                         cfg_high_risk_threshold=0.85)
    assert r["category"] == CAT_ROUTINE


def test_high_ct_alone_is_routine_not_emergency():
    """CAC severity alone cannot determine an acute event."""
    r = recommend_action(status="completed", ecg_score_mean=0.15,
                         ct_level=3, fused_score=0.30,
                         cfg_high_risk_threshold=0.85)
    assert r["category"] == CAT_ROUTINE
    assert any("calcium" in f.lower() or "acute" in f.lower()
               for f in r["factors"])


def test_high_ct_with_moderate_fused_is_routine():
    r = recommend_action(status="completed", ecg_score_mean=0.30,
                         ct_level=3, fused_score=0.45,
                         cfg_high_risk_threshold=0.85)
    assert r["category"] == CAT_ROUTINE


def test_all_results_have_disclaimer_and_symptom_warning():
    for kwargs in (
        dict(status="completed", ecg_score_mean=0.1, fused_score=0.1),
        dict(status="failed"),
        dict(status="completed", emergency_symptoms=True),
    ):
        r = recommend_action(**kwargs)
        assert r["disclaimer"] == DISCLAIMER
        assert r["symptom_warning"] == SYMPTOM_WARNING
        assert r["factors"], "every recommendation must list contributing factors"
        assert r["explanation"]


def test_missing_ct_still_allows_ecg_based_prompt():
    r = recommend_action(status="completed", ecg_score_mean=0.85,
                         ct_level=None, fused_score=0.85,
                         cfg_high_risk_threshold=0.99)
    assert r["category"] == CAT_PROMPT


def test_fusion_cfg_affects_thresholds_in_factors():
    fcfg = FusionConfig(thr_watch=0.5, thr_alert=0.8, step_per_ct_level=0.1)
    r = recommend_action(status="completed", ecg_score_mean=0.3,
                         ct_level=2, fused_score=0.35, fusion_cfg=fcfg)
    assert r["category"] == CAT_ROUTINE
    # factors should mention the adjusted thresholds
    assert any("threshold" in f.lower() for f in r["factors"])
