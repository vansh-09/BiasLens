"""
BiasLens — services/analyzer.py (Debugged)
Advanced analysis for visualizations.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import LabelEncoder

def compute_correlation_heatmap(df: pd.DataFrame, sensitive_attrs: List[str], label_col: str) -> Dict[str, Any]:
    """
    Computes a correlation matrix between features and sensitive attributes.
    Optimized for JSON compatibility and front-end visualization.
    """
    df_enc = df.copy()
    
    # 1. Handle Categorical Encoding
    # Logic: Only encode what's necessary to avoid memory bloat
    for col in df_enc.columns:
        if df_enc[col].dtype == 'object' or df_enc[col].dtype.name == 'category':
            try:
                le = LabelEncoder()
                df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            except Exception:
                # Fallback: if encoding fails, convert to numeric codes manually
                df_enc[col] = pd.factorize(df_enc[col])[0]
            
    # 2. Define Features
    # We want to see how sensitive attributes correlate with OTHER features
    features = [c for c in df_enc.columns if c != label_col and c not in sensitive_attrs]
    
    # Tier 1 Optimization: If there are too many features, take the top 15 
    # to prevent the heatmap from becoming a mess in the UI.
    if len(features) > 15:
        features = features[:15]
    
    heatmap_data = []
    for s_attr in sensitive_attrs:
        row = []
        for feat in features:
            # Check for zero variance to avoid NaN errors
            if df_enc[s_attr].nunique() <= 1 or df_enc[feat].nunique() <= 1:
                corr = 0.0
            else:
                corr = df_enc[s_attr].corr(df_enc[feat])
            
            # Ensure the value is a standard float and not np.float64 (JSON requirement)
            val = float(abs(corr)) if not np.isnan(corr) else 0.0
            row.append(round(val, 3))
        heatmap_data.append(row)
        
    return {
        "x_labels": features,
        "y_labels": sensitive_attrs,
        "matrix": heatmap_data
    }