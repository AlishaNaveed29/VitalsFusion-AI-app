"""
Rule-based fusion for VitalsFusion AI (non-diagnostic prototype)
================================================================
Inputs
    * ECG branch : one MI-pattern score (0..1) per 10-second window
    * CT branch  : CT context level 0..3 from the radiology-report values
                   (see ct_features.py), or None if not reported

Output
    A personalised alert level:  No alert / Watch / Alert  (or "no decision"
    when the signal quality is too low).

Why rules and not a learned fusion model? No public dataset pairs CT and ECG
for the same patients, so a learned fusion model would have nothing real to
train on. Rules are transparent, tunable and explainable.

How the rules work
    1. A higher CT context level LOWERS the ECG score thresholds, so the same
       ECG pattern triggers earlier for a person whose CT report shows more
       coronary disease.
    2. Persistence: the alert needs K of the last N windows above the threshold,
       so one noisy window cannot raise an alarm (fewer false alarms).
    3. Windows that fail the signal-quality gate are ignored; if none are left
       the system makes no decision instead of guessing.

All numbers below are DESIGN PARAMETERS, not clinical values. Tune them on
your ECG validation data (for example, pick the alert threshold that gives the
false-alarm rate you are willing to accept).
"""

from dataclasses import dataclass, asdict
from typing import List, Optional

LEVEL_NAMES = {
    -1: "No decision (signal quality too low)",
    0: "No alert",
    1: "Watch",
    2: "Alert",
}

DISCLAIMER = (
    "Non-diagnostic research prototype. A single-lead wearable ECG is not suitable "
    "for clinical-grade monitoring or ST-segment analysis. This output is a "
    "personalised alert setting, not a diagnosis and not a prediction of a heart "
    "attack. If you have chest pain or other symptoms, seek medical care."
)


@dataclass
class FusionConfig:
    thr_watch: float = 0.50          # ECG score that raises a "Watch"
    thr_alert: float = 0.80          # ECG score that raises an "Alert"
    step_per_ct_level: float = 0.05  # thresholds are lowered by this much per CT level
    min_threshold: float = 0.10      # thresholds never go below this
    k_required: int = 2              # windows that must exceed the threshold ...
    n_windows: int = 3               # ... out of the last N windows

    def validate(self):
        if not (0.0 < self.thr_watch < self.thr_alert <= 1.0):
            raise ValueError("Need 0 < watch threshold < alert threshold <= 1")
        if self.step_per_ct_level < 0:
            raise ValueError("CT step must be >= 0")
        if not (0.0 < self.min_threshold < self.thr_watch):
            raise ValueError("Minimum threshold must be above 0 and below the watch threshold")
        if not (1 <= self.k_required <= self.n_windows):
            raise ValueError("Need 1 <= K <= N")


def adjusted_thresholds(cfg: FusionConfig, ct_level: Optional[int]):
    """(watch, alert) thresholds after applying the CT context level."""
    level = 0 if ct_level is None else int(ct_level)
    if level not in (0, 1, 2, 3):
        raise ValueError("ct_level must be 0..3 or None")
    watch = max(cfg.min_threshold, cfg.thr_watch - cfg.step_per_ct_level * level)
    alert = max(watch, cfg.thr_alert - cfg.step_per_ct_level * level)
    return watch, alert


def fuse(window_scores: List[Optional[float]], ct_level: Optional[int],
         cfg: Optional[FusionConfig] = None) -> dict:
    """Combine ECG window scores with the CT context level.

    window_scores: ECG score per window in time order; None = window rejected
    by the signal-quality gate.
    """
    cfg = cfg or FusionConfig()
    cfg.validate()
    watch_thr, alert_thr = adjusted_thresholds(cfg, ct_level)

    recent = list(window_scores)[-cfg.n_windows:]
    valid = [float(s) for s in recent if s is not None]

    result = {
        "ct_level_used": 0 if ct_level is None else int(ct_level),
        "ct_reported": ct_level is not None,
        "watch_threshold": round(watch_thr, 4),
        "alert_threshold": round(alert_thr, 4),
        "windows_considered": len(recent),
        "windows_valid": len(valid),
        "scores": [None if s is None else round(float(s), 4) for s in recent],
        "config": asdict(cfg),
        "disclaimer": DISCLAIMER,
    }

    if not valid:
        result.update(level=-1, level_name=LEVEL_NAMES[-1], n_over_watch=0, n_over_alert=0,
                      k_needed=0, explanation="No usable window: every recent window failed the signal-quality check.",
                      fused_risk_score=None)
        return result

    k_needed = min(cfg.k_required, len(valid))
    n_watch = sum(s >= watch_thr for s in valid)
    n_alert = sum(s >= alert_thr for s in valid)

    if n_alert >= k_needed:
        level = 2
    elif n_watch >= k_needed:
        level = 1
    else:
        level = 0

    ct_text = ("CT not reported, so the base thresholds were used" if ct_level is None
               else f"CT context level {int(ct_level)} lowered the thresholds by "
                    f"{cfg.step_per_ct_level * int(ct_level):.2f}")
    result.update(
        level=level,
        level_name=LEVEL_NAMES[level],
        n_over_watch=n_watch,
        n_over_alert=n_alert,
        k_needed=k_needed,
        fused_risk_score=fused_risk_score(window_scores, ct_level, cfg),
        explanation=(f"{ct_text}. Watch threshold {watch_thr:.2f}, alert threshold {alert_thr:.2f}. "
                     f"{n_watch} of {len(valid)} valid window(s) reached the watch threshold and "
                     f"{n_alert} reached the alert threshold; {k_needed} needed."),
    )
    return result


def fused_risk_score(window_scores: List[Optional[float]], ct_level: Optional[int],
                     cfg: Optional[FusionConfig] = None) -> Optional[float]:
    """Continuous 0..1 fused risk estimate for the high-risk emergency workflow.

    * Base value = mean of the recent valid ECG model scores (the actual model
      output; windows rejected by the quality gate are excluded, never replaced).
    * The CT context level adds a small transparent bonus of
      ``step_per_ct_level`` per level, capped at 1.0.
    * Returns ``None`` when no valid window exists - a missing or failed input
      must never look like a low-risk result.

    The emergency threshold (default 0.85, see config.py) is applied to this
    score. It is a DESIGN parameter pending model validation.
    """
    cfg = cfg or FusionConfig()
    valid = [float(s) for s in list(window_scores)[-cfg.n_windows:] if s is not None]
    if not valid:
        return None
    level = 0 if ct_level is None else int(ct_level)
    if level not in (0, 1, 2, 3):
        raise ValueError("ct_level must be 0..3 or None")
    base = sum(valid) / len(valid)
    return round(min(1.0, base + cfg.step_per_ct_level * level), 4)
