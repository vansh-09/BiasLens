"""
BiasLens — services/detector.py
Orchestrates the audit pipeline. All outputs derived from real data.
"""
import pandas as pd
import uuid
from typing import List
from models.schemas import (
    BiasIssue, DatasetInfo, AuditSummary, SeverityLevel, MetricResult
)
from utils.helpers import score_to_grade, round2, Timer
from config import settings


def build_dataset_info(df, sensitive_attrs, label_col, original_df) -> DatasetInfo:
    missing = {
        col: int(original_df[col].isna().sum())
        for col in original_df.columns
        if original_df[col].isna().sum() > 0
    }
    label_vals = df[label_col].value_counts()
    class_dist = {str(k): int(v) for k, v in label_vals.items()}

    demo_breakdown = {}
    for attr in sensitive_attrs:
        if attr in df.columns:
            vc = df[attr].value_counts()
            demo_breakdown[attr] = {str(k): int(v) for k, v in vc.items()}

    return DatasetInfo(
        total_records=len(df),
        total_features=len(df.columns) - 1,
        sensitive_attributes=sensitive_attrs,
        label_column=label_col,
        missing_values=missing,
        class_distribution=class_dist,
        demographic_breakdown=demo_breakdown
    )


def build_issues_from_metrics(metrics: List[MetricResult], proxy_vars: list) -> List[BiasIssue]:
    issues = []

    for m in metrics:
        if m.severity not in (SeverityLevel.CRITICAL, SeverityLevel.WARNING):
            continue

        legal_risk = None
        if m.severity == SeverityLevel.CRITICAL:
            name_lower = m.name.lower()
            if "disparate impact" in name_lower or "demographic parity" in name_lower:
                legal_risk = (
                    "Potential violation of EEOC 4/5ths rule. "
                    "May constitute illegal employment discrimination under Title VII (US) or Equality Act (UK)."
                )
            elif "statistical parity" in name_lower:
                legal_risk = "Potential violation of equal treatment standards under anti-discrimination law."

        issues.append(BiasIssue(
            id=f"issue_{uuid.uuid4().hex[:8]}",
            severity=m.severity,
            title=f"{m.name} — {m.attribute or 'Overall'}",
            description=m.description,
            metric_value=f"{m.value:.4f}",
            affected_attribute=m.attribute or "overall",
            affected_group=m.group,
            legal_risk=legal_risk,
            recommendation=_get_recommendation(m.name)
        ))

    for pv in proxy_vars:
        if pv.severity not in (SeverityLevel.CRITICAL, SeverityLevel.WARNING):
            continue
        issues.append(BiasIssue(
            id=f"issue_{uuid.uuid4().hex[:8]}",
            severity=pv.severity,
            title=f"Proxy Variable — '{pv.column}'",
            description=pv.description,
            metric_value=f"r={pv.correlation:.3f}",
            affected_attribute=pv.sensitive_attr,
            legal_risk=(
                "Using proxy variables may constitute indirect discrimination, "
                "which is illegal in many jurisdictions."
            ),
            recommendation=(
                f"Remove or transform '{pv.column}' before model training. "
                "Apply dimensionality reduction or fairness-aware feature selection."
            )
        ))

    return sorted(issues, key=lambda x: (
        0 if x.severity == SeverityLevel.CRITICAL else
        1 if x.severity == SeverityLevel.WARNING else 2
    ))


def _get_recommendation(metric_name: str) -> str:
    n = metric_name.lower()
    if "disparate impact" in n:
        return "Apply reweighing (AIF360) or adversarial debiasing. Remove proxy variables and rebalance training data."
    elif "statistical parity" in n:
        return "Apply prejudice remover regularization or calibrated equalized odds post-processing."
    elif "equal opportunity" in n:
        return "Use equalized odds post-processing or cost-sensitive learning to equalize TPR across groups."
    elif "average odds" in n:
        return "Apply reject option classification or per-group threshold optimization."
    elif "predictive parity" in n:
        return "Rebalance training data or apply group-specific calibration to equalize precision."
    elif "individual fairness" in n:
        return "Use fairness-aware similarity metrics or Lipschitz-constrained classification models."
    elif "calibration" in n:
        return "Apply Platt scaling or isotonic regression to recalibrate predictions per group."
    elif "theil" in n:
        return "Reduce inequality by resampling minority groups or applying group-aware loss functions."
    elif "demographic parity" in n:
        return "Apply constraint-based optimization or post-processing threshold adjustment per group."
    return "Review training data for imbalances and apply appropriate bias mitigation techniques from AIF360 or Fairlearn."


def compute_overall_score(metrics: List[MetricResult]) -> int:
    if not metrics:
        return 0

    # Metric weights — higher weight = more important for fairness score
    weights = {
        "disparate impact": 2.5,
        "statistical parity": 2.0,
        "equal opportunity": 2.0,
        "demographic parity ratio": 1.8,
        "average odds": 1.5,
        "predictive parity": 1.2,
        "calibration": 1.2,
        "individual fairness": 1.0,
        "theil index": 0.8,
    }

    total_weight = 0.0
    weighted_score = 0.0

    for m in metrics:
        w = 1.0
        m_lower = m.name.lower()
        for key, wt in weights.items():
            if key in m_lower:
                w = wt
                break

        pts = 100.0 if m.severity == SeverityLevel.PASS else \
              55.0  if m.severity == SeverityLevel.WARNING else 15.0

        weighted_score += pts * w
        total_weight += w

    return int(round(weighted_score / total_weight)) if total_weight > 0 else 0


def build_audit_summary(metrics: List[MetricResult], issues: List[BiasIssue], timer: Timer) -> AuditSummary:
    score = compute_overall_score(metrics)
    critical = sum(1 for i in issues if i.severity == SeverityLevel.CRITICAL)
    warning  = sum(1 for i in issues if i.severity == SeverityLevel.WARNING)
    passed   = sum(1 for m in metrics if m.pass_fail)

    return AuditSummary(
        overall_score=score,
        total_issues=len(issues),
        critical_count=critical,
        warning_count=warning,
        passed_count=passed,
        fairness_grade=score_to_grade(score),
        analysis_time_seconds=timer.elapsed()
    )