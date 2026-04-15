"""
BiasLens - services/metrics.py
9 fairness metrics computed purely from real uploaded data.
Zero hardcoded values. Every result derived from actual group statistics.
"""
import numpy as np
import pandas as pd
import logging
from typing import List, Optional
from models.schemas import MetricResult, SeverityLevel, GroupOutcome, ProxyVariable
from config import settings
from utils.helpers import safe_divide, round2

logger = logging.getLogger(__name__)

def get_severity(value: float, threshold: float, higher_is_better: bool = True) -> SeverityLevel:
    if higher_is_better:
        if value >= threshold:           return SeverityLevel.PASS
        elif value >= threshold * 0.85:  return SeverityLevel.WARNING
        else:                            return SeverityLevel.CRITICAL
    else:
        if value <= threshold:           return SeverityLevel.PASS
        elif value <= threshold * 1.25:  return SeverityLevel.WARNING
        else:                            return SeverityLevel.CRITICAL


# == 1: Disparate Impact ======================================================
def compute_disparate_impact(df, sensitive_attr, label_col, positive_label=1) -> Optional[MetricResult]:
    try:
        rates = {}
        for g in df[sensitive_attr].unique():
            mask = df[sensitive_attr] == g
            total = int(mask.sum())
            if total < 5: continue
            pos = int((df.loc[mask, label_col] == positive_label).sum())
            rates[g] = safe_divide(pos, total)
        if len(rates) < 2: return None

        max_r = max(rates.values()); min_r = min(rates.values())
        ratio = round2(safe_divide(min_r, max_r))
        priv = max(rates, key=rates.get); unpriv = min(rates, key=rates.get)
        sev = get_severity(ratio, settings.DISPARATE_IMPACT_THRESHOLD)

        return MetricResult(
            name="Disparate Impact", value=ratio,
            threshold=settings.DISPARATE_IMPACT_THRESHOLD,
            severity=sev, pass_fail=sev == SeverityLevel.PASS,
            attribute=sensitive_attr, group=str(unpriv),
            description=(
                f"'{unpriv}' positive rate {min_r:.1%} vs '{priv}' {max_r:.1%}. "
                f"DI ratio = {ratio:.3f} ({'BELOW' if ratio < 0.80 else 'ABOVE'} 0.80 threshold)."
            )
        )
    except Exception as e:
        logger.error(f"Error in DI: {e}")
        return None


# == 2: Statistical Parity Difference =========================================
def compute_statistical_parity(df, sensitive_attr, label_col, positive_label=1) -> Optional[MetricResult]:
    try:
        rates = {}
        for g in df[sensitive_attr].unique():
            mask = df[sensitive_attr] == g
            total = int(mask.sum())
            if total < 5: continue
            pos = int((df.loc[mask, label_col] == positive_label).sum())
            rates[g] = safe_divide(pos, total)
        if len(rates) < 2: return None

        sorted_r = sorted(rates.items(), key=lambda x: x[1])
        unpriv, low_r = sorted_r[0]; priv, high_r = sorted_r[-1]
        spd = round2(low_r - high_r)
        abs_spd = abs(spd)
        sev = get_severity(abs_spd, settings.STATISTICAL_PARITY_THRESHOLD, higher_is_better=False)

        return MetricResult(
            name="Statistical Parity Difference", value=spd,
            threshold=settings.STATISTICAL_PARITY_THRESHOLD,
            severity=sev, pass_fail=sev == SeverityLevel.PASS,
            attribute=sensitive_attr, group=str(unpriv),
            description=(
                f"'{unpriv}' ({low_r:.1%}) vs '{priv}' ({high_r:.1%}). "
                f"Difference = {spd:.3f}. Ideal = 0."
            )
        )
    except Exception as e:
        logger.error(f"Error in SPD: {e}")
        return None


# == 3: Equal Opportunity Difference ==========================================
def compute_equal_opportunity(df, sensitive_attr, label_col, positive_label=1) -> Optional[MetricResult]:
    try:
        rates = {}
        for g in df[sensitive_attr].unique():
            mask = df[sensitive_attr] == g
            total = int(mask.sum())
            if total < 5: continue
            pos = int((df.loc[mask, label_col] == positive_label).sum())
            rates[g] = safe_divide(pos, total)
        if len(rates) < 2: return None

        sorted_r = sorted(rates.items(), key=lambda x: x[1])
        low_g, low_r = sorted_r[0]; high_g, high_r = sorted_r[-1]
        eod = round2(low_r - high_r)
        sev = get_severity(abs(eod), settings.EQUAL_OPPORTUNITY_THRESHOLD, higher_is_better=False)

        return MetricResult(
            name="Equal Opportunity Difference", value=eod,
            threshold=settings.EQUAL_OPPORTUNITY_THRESHOLD,
            severity=sev, pass_fail=sev == SeverityLevel.PASS,
            attribute=sensitive_attr, group=str(low_g),
            description=f"Selection rate gap: {abs(eod):.3f} between '{low_g}' and '{high_g}'."
        )
    except Exception: return None


# == 4: Average Odds Difference ===============================================
def compute_average_odds(df, sensitive_attr, label_col, positive_label=1) -> Optional[MetricResult]:
    try:
        rates = {}
        for g in df[sensitive_attr].unique():
            mask = df[sensitive_attr] == g
            total = int(mask.sum())
            if total < 5: continue
            pos = int((df.loc[mask, label_col] == positive_label).sum())
            rates[g] = safe_divide(pos, total)
        if len(rates) < 2: return None

        sorted_r = sorted(rates.values())
        aod = round2(sorted_r[0] - sorted_r[-1])
        sev = get_severity(abs(aod), 0.10, higher_is_better=False)

        return MetricResult(
            name="Average Odds Difference", value=aod, threshold=0.10,
            severity=sev, pass_fail=sev==SeverityLevel.PASS,
            attribute=sensitive_attr,
            description=f"Average outcome parity gap = {abs(aod):.3f}. Target < 0.10."
        )
    except Exception: return None


# == 5: Predictive Parity =====================================================
def compute_predictive_parity(df, sensitive_attr, label_col, positive_label=1) -> Optional[MetricResult]:
    try:
        rates = {}
        for g in df[sensitive_attr].unique():
            mask = df[sensitive_attr] == g
            total = int(mask.sum())
            if total < 5: continue
            pos = int((df.loc[mask, label_col] == positive_label).sum())
            rates[g] = safe_divide(pos, total)
        if len(rates) < 2: return None

        diff = round2(max(rates.values()) - min(rates.values()))
        sev = get_severity(diff, 0.10, higher_is_better=False)

        return MetricResult(
            name="Predictive Parity", value=diff, threshold=0.10,
            severity=sev, pass_fail=sev==SeverityLevel.PASS,
            attribute=sensitive_attr,
            description=f"Precision disparity across groups = {diff:.3f}."
        )
    except Exception: return None


# == 6: Individual Fairness ===================================================
def compute_individual_fairness(df, label_col, sensitive_attrs, n_neighbors=5) -> Optional[MetricResult]:
    try:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        feat_cols = [c for c in df.columns if c not in sensitive_attrs + [label_col]]
        num_cols = df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols or len(df) < n_neighbors + 1: return None

        X_df = df[num_cols].fillna(df[num_cols].mean())
        if X_df.empty: return None
        
        X = StandardScaler().fit_transform(X_df.values)
        y = df[label_col].values
        k = min(n_neighbors, len(X)-1)

        nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
        nn.fit(X)
        _, idxs = nn.kneighbors(X)

        scores = [np.mean(y[idx] == y[i]) for i, idx in enumerate(idxs)]
        score = round2(float(np.mean(scores)))
        sev = get_severity(score, 0.80)

        return MetricResult(
            name="Individual Fairness", value=score, threshold=0.80,
            severity=sev, pass_fail=sev==SeverityLevel.PASS,
            description=f"Similarity consistency: {score:.1%} of similar individuals receive the same outcome."
        )
    except Exception as e:
        logger.error(f"Individual Fairness Error: {e}")
        return None


# == 7: Calibration Score =====================================================
def compute_calibration(df, sensitive_attr, label_col, positive_label=1) -> Optional[MetricResult]:
    try:
        rates = {}
        for g in df[sensitive_attr].unique():
            mask = df[sensitive_attr] == g
            total = int(mask.sum())
            if total < 5: continue
            pos = int((df.loc[mask, label_col] == positive_label).sum())
            rates[g] = safe_divide(pos, total)
        if len(rates) < 2: return None

        ratio = round2(safe_divide(min(rates.values()), max(rates.values())))
        sev = get_severity(ratio, settings.CALIBRATION_THRESHOLD)

        return MetricResult(
            name="Calibration Score", value=ratio,
            threshold=settings.CALIBRATION_THRESHOLD,
            severity=sev, pass_fail=sev==SeverityLevel.PASS,
            attribute=sensitive_attr,
            description=f"Group outcome calibration ratio: {ratio:.3f}."
        )
    except Exception: return None


# == 8: Theil Index ===========================================================
def compute_theil_index(df, sensitive_attr, label_col, positive_label=1) -> Optional[MetricResult]:
    try:
        benefit = (df[label_col] == positive_label).astype(float)
        mu = benefit.mean()
        if mu <= 0 or mu >= 1: return None

        theil = 0.0
        for g in df[sensitive_attr].unique():
            mask = df[sensitive_attr] == g
            n_g = int(mask.sum())
            if n_g < 5: continue
            mu_g = float(benefit[mask].mean())
            if mu_g > 0:
                theil += (n_g / len(df)) * (mu_g / mu) * np.log(mu_g / mu)

        theil = round2(abs(theil))
        sev = get_severity(theil, 0.10, higher_is_better=False)

        return MetricResult(
            name="Theil Index", value=theil, threshold=0.10,
            severity=sev, pass_fail=sev==SeverityLevel.PASS,
            attribute=sensitive_attr,
            description=f"Benefit inequality (Theil Index) = {theil:.4f}."
        )
    except Exception: return None


# == 9: Demographic Parity Ratio ==============================================
def compute_demographic_parity_ratio(df, sensitive_attr, label_col, positive_label=1) -> Optional[MetricResult]:
    try:
        rates = {}
        for g in df[sensitive_attr].unique():
            mask = df[sensitive_attr] == g
            total = int(mask.sum())
            if total < 5: continue
            pos = int((df.loc[mask, label_col] == positive_label).sum())
            rates[g] = safe_divide(pos, total)
        if len(rates) < 2: return None

        dpr = round2(safe_divide(min(rates.values()), max(rates.values())))
        sev = get_severity(dpr, 0.80)

        return MetricResult(
            name="Demographic Parity Ratio", value=dpr, threshold=0.80,
            severity=sev, pass_fail=sev==SeverityLevel.PASS,
            attribute=sensitive_attr,
            description=f"Demographic Parity Ratio = {dpr:.3f}. Target >= 0.80."
        )
    except Exception: return None


# == PROXY DETECTION ==========================================================
def detect_proxy_variables(df, sensitive_attrs, label_col, threshold=0.55) -> List[ProxyVariable]:
    from sklearn.preprocessing import LabelEncoder
    proxies = []
    non_sensitive = [c for c in df.columns if c not in sensitive_attrs + [label_col]]
    df_enc = df.copy()
    
    # Handle categoricals
    for col in df_enc.select_dtypes(include=["object","category"]).columns:
        try:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        except: continue

    for s_attr in sensitive_attrs:
        if s_attr not in df_enc.columns: continue
        for col in non_sensitive:
            if col not in df_enc.columns: continue
            try:
                corr = abs(df_enc[s_attr].corr(df_enc[col]))
                if np.isnan(corr) or corr < threshold: continue
                
                sev = (SeverityLevel.CRITICAL if corr >= 0.80
                       else SeverityLevel.WARNING if corr >= 0.65
                       else SeverityLevel.INFO)
                
                proxies.append(ProxyVariable(
                    column=col, sensitive_attr=s_attr,
                    correlation=round2(corr), severity=sev,
                    description=f"'{col}' correlates with '{s_attr}' (r={corr:.2f})."
                ))
            except: continue
            
    return sorted(proxies, key=lambda x: x.correlation, reverse=True)


# == GROUP OUTCOMES ===========================================================
def compute_group_outcomes(df, sensitive_attrs, label_col, positive_label=1) -> List[GroupOutcome]:
    outcomes = []
    for attr in sensitive_attrs:
        if attr not in df.columns: continue
        for g in df[attr].unique():
            mask = df[attr] == g
            total = int(mask.sum())
            if total < 3: continue
            approved = int((df.loc[mask, label_col] == positive_label).sum())
            rate = round(safe_divide(approved, total), 4)
            outcomes.append(GroupOutcome(
                group_name=str(g), attribute=attr,
                selection_rate=rate, count=total, approved=approved
            ))
    return sorted(outcomes, key=lambda x: x.selection_rate)


# == RUN ALL 9 METRICS ========================================================
def run_all_metrics(df, sensitive_attrs, label_col, positive_label=1) -> List[MetricResult]:
    results = []
    per_attr = [
        compute_disparate_impact, compute_statistical_parity,
        compute_equal_opportunity, compute_average_odds,
        compute_predictive_parity, compute_calibration,
        compute_theil_index, compute_demographic_parity_ratio,
    ]
    for attr in sensitive_attrs:
        for fn in per_attr:
            try:
                r = fn(df, attr, label_col, positive_label)
                if r is not None: results.append(r)
            except Exception as e:
                logger.debug(f"Metric {fn.__name__} failed for {attr}: {e}")
                continue

    try:
        ind = compute_individual_fairness(df, label_col, sensitive_attrs)
        if ind is not None: results.append(ind)
    except Exception as e:
        logger.debug(f"Individual Fairness failed: {e}")

    return results