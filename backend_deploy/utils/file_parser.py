import pandas as pd
import numpy as np
import io
import json
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException
from config import settings


SUPPORTED_EXTENSIONS = {".csv", ".json", ".xlsx", ".xls", ".parquet"}


async def parse_uploaded_file(file: UploadFile) -> pd.DataFrame:
    """Parse any supported file format into a pandas DataFrame."""
    filename = file.filename.lower()

    # Check file extension
    ext = None
    for supported in SUPPORTED_EXTENSIONS:
        if filename.endswith(supported):
            ext = supported
            break

    if not ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Read file bytes
    content = await file.read()

    # Check file size
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. Maximum allowed: {settings.MAX_FILE_SIZE_MB}MB"
        )

    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(content))

        elif ext == ".json":
            data = json.loads(content)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame.from_dict(data)
            else:
                raise ValueError("JSON must be a list of records or a dict")

        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl" if ext == ".xlsx" else "xlrd")

        elif ext == ".parquet":
            df = pd.read_parquet(io.BytesIO(content))

        else:
            raise ValueError(f"Unhandled extension: {ext}")

    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse file: {str(e)}"
        )

    if df.empty:
        raise HTTPException(status_code=422, detail="Uploaded file contains no data.")

    if len(df) < 50:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset too small: {len(df)} records. Need at least 50 records for meaningful bias analysis."
        )

    # Clean column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    return df


def auto_detect_label_column(df: pd.DataFrame) -> Optional[str]:
    """Try to automatically detect the target/label column."""
    cols = df.columns.tolist()

    # Exact match first
    for kw in settings.LABEL_KEYWORDS:
        if kw in cols:
            return kw

    # Partial match
    for col in cols:
        for kw in settings.LABEL_KEYWORDS:
            if kw in col:
                return col

    # Heuristic: binary column at the end
    for col in reversed(cols):
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2:
            return col

    return None


def auto_detect_sensitive_attributes(df: pd.DataFrame, label_col: str) -> list:
    """Auto-detect sensitive attribute columns."""
    detected = []
    cols = [c for c in df.columns if c != label_col]

    for col in cols:
        col_lower = col.lower()
        for kw in settings.SENSITIVE_ATTR_KEYWORDS:
            if kw in col_lower:
                detected.append(col)
                break

    return detected


def preprocess_dataframe(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Clean and preprocess the dataframe."""
    df = df.copy()

    # Drop rows where label is missing
    df = df.dropna(subset=[label_col])

    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col != label_col:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if not df[col].empty and df[col].mode().shape[0] > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_dataframe(df: pd.DataFrame, label_col: str, sensitive_attrs: list) -> Tuple[pd.DataFrame, dict]:
    """Label encode categorical columns, return encoders map."""
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    encoders = {}

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = {
            "classes": le.classes_.tolist(),
            "encoder": le
        }

    return df, encoders