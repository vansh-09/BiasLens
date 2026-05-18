import json
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from fastapi import HTTPException


class ValidationError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 422):
        self.code = code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        error = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            error["details"] = self.details
        return {"success": False, "error": error}


def build_error_response(code: str, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 422) -> HTTPException:
    return HTTPException(status_code=status_code, detail=ValidationError(code, message, details, status_code).to_dict())


def normalize_column_name(name: Any) -> str:
    if name is None:
        return ""
    return str(name).strip().lower().replace(" ", "_")


def validate_empty_dataset(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValidationError(
            "EMPTY_DATASET",
            "Uploaded dataset contains no rows.",
            {"columns": df.columns.tolist()},
            status_code=422,
        )


def validate_duplicate_columns(df: pd.DataFrame) -> None:
    duplicate_columns = [col for col in df.columns if list(df.columns).count(col) > 1]
    if duplicate_columns:
        raise ValidationError(
            "DUPLICATE_COLUMNS",
            "Duplicate column names were found in the dataset.",
            {"duplicate_columns": sorted(set(duplicate_columns))},
            status_code=422,
        )


def validate_minimum_rows(df: pd.DataFrame, minimum_rows: int = 50) -> None:
    if len(df) < minimum_rows:
        raise ValidationError(
            "INSUFFICIENT_ROWS",
            f"Dataset too small: {len(df)} records. At least {minimum_rows} records are required for meaningful bias analysis.",
            {"row_count": len(df), "minimum_rows": minimum_rows},
            status_code=422,
        )


def validate_dataset_structure(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(
            "INVALID_DATASET",
            "Parsed dataset is not a valid table.",
            status_code=422,
        )

    if df.columns.empty:
        raise ValidationError(
            "INVALID_DATASET",
            "Dataset must contain at least one column.",
            status_code=422,
        )

    validate_duplicate_columns(df)
    validate_empty_dataset(df)

    invalid_columns = [c for c in df.columns if normalize_column_name(c) == ""]
    if invalid_columns:
        raise ValidationError(
            "INVALID_COLUMN_NAMES",
            "Dataset contains invalid column names.",
            {"invalid_columns": invalid_columns},
            status_code=422,
        )


def _is_categorical_usable(series: pd.Series) -> bool:
    values = series.dropna()
    unique_count = values.nunique(dropna=True)

    if unique_count < 2:
        return False

    if (
        pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return True

    if pd.api.types.is_integer_dtype(series):
        return unique_count <= 50

    if pd.api.types.is_float_dtype(series):
        return unique_count <= 25

    return False


def validate_sensitive_attributes(df: pd.DataFrame, attrs: Sequence[str], allow_autodetect: bool = False) -> List[str]:
    if not attrs:
        if allow_autodetect:
            return []
        raise ValidationError(
            "MISSING_SENSITIVE_ATTRIBUTES",
            "Sensitive attributes must be specified or auto-detected.",
            {"available_columns": df.columns.tolist()},
            status_code=422,
        )

    resolved_attrs = []
    for attr in attrs:
        if attr not in df.columns:
            raise ValidationError(
                "INVALID_SENSITIVE_ATTRIBUTE",
                f"Sensitive attribute '{attr}' does not exist in dataset.",
                {"available_columns": df.columns.tolist()},
                status_code=422,
            )

        if df[attr].dropna().empty:
            raise ValidationError(
                "INVALID_SENSITIVE_ATTRIBUTE",
                f"Sensitive attribute '{attr}' contains only missing values.",
                {"attribute": attr},
                status_code=422,
            )

        if df[attr].nunique(dropna=True) < 2:
            raise ValidationError(
                "INVALID_SENSITIVE_ATTRIBUTE",
                f"Sensitive attribute '{attr}' must contain at least two distinct groups.",
                {"attribute": attr, "unique_values": df[attr].dropna().unique().tolist()},
                status_code=422,
            )

        if not _is_categorical_usable(df[attr]):
            raise ValidationError(
                "INVALID_SENSITIVE_ATTRIBUTE",
                f"Sensitive attribute '{attr}' is not usable for categorical fairness analysis.",
                {
                    "attribute": attr,
                    "unique_values": df[attr].dropna().unique().tolist()[:10],
                    "unique_count": int(df[attr].nunique(dropna=True)),
                },
                status_code=422,
            )

        resolved_attrs.append(attr)

    return resolved_attrs


def validate_target_column(df: pd.DataFrame, target: Optional[str]) -> str:
    if not target:
        raise ValidationError(
            "MISSING_TARGET_COLUMN",
            "Target label column must be provided.",
            {"available_columns": df.columns.tolist()},
            status_code=422,
        )

    if target not in df.columns:
        raise ValidationError(
            "INVALID_TARGET_COLUMN",
            f"Target column '{target}' does not exist in dataset.",
            {"available_columns": df.columns.tolist()},
            status_code=422,
        )

    if df[target].dropna().empty:
        raise ValidationError(
            "INVALID_TARGET_COLUMN",
            f"Target column '{target}' contains only missing values.",
            {"attribute": target},
            status_code=422,
        )

    return target


def validate_binary_labels(df: pd.DataFrame, target: str) -> List[Any]:
    values = df[target].dropna().unique().tolist()

    if len(values) < 2:
        raise ValidationError(
            "INVALID_LABEL_DISTRIBUTION",
            f"Target column '{target}' must contain at least two classes.",
            {"unique_values": values},
            status_code=422,
        )

    if len(values) > 2:
        raise ValidationError(
            "NON_BINARY_TARGET",
            f"Target column '{target}' contains more than two labels. Only binary classification is supported.",
            {"unique_values": values[:10], "unique_count": len(values)},
            status_code=422,
        )

    return values


def _coerce_positive_label(series: pd.Series, positive_label: Any) -> Any:
    actual_values = series.dropna().unique().tolist()
    if positive_label in actual_values:
        return positive_label

    as_str = str(positive_label)
    for raw in actual_values:
        if str(raw) == as_str:
            return raw

    if pd.api.types.is_numeric_dtype(series):
        try:
            if "." in as_str:
                candidate = float(as_str)
            else:
                candidate = int(as_str)
            if candidate in actual_values:
                return candidate
        except Exception:
            pass

    raise ValidationError(
        "INVALID_POSITIVE_LABEL",
        f"Positive label '{positive_label}' does not exist in target column.",
        {"target_values": actual_values},
        status_code=422,
    )


def validate_positive_label(df: pd.DataFrame, target: str, positive_label: Any) -> Any:
    if positive_label is None:
        raise ValidationError(
            "MISSING_POSITIVE_LABEL",
            "Positive label is required for bias metric evaluation.",
            {"target_column": target},
            status_code=422,
        )
    return _coerce_positive_label(df[target], positive_label)


def is_json_record_safe(data: Any) -> bool:
    if isinstance(data, dict):
        return all(not isinstance(v, (dict, list)) for v in data.values())
    return False


def validate_json_payload(data: Any) -> None:
    if isinstance(data, list):
        if not data:
            raise ValidationError(
                "EMPTY_JSON_DATA",
                "Uploaded JSON array contains no records.",
                status_code=422,
            )
        if not all(isinstance(item, dict) for item in data):
            raise ValidationError(
                "INVALID_JSON_STRUCTURE",
                "JSON file must contain an array of objects.",
                status_code=422,
            )
        for item in data:
            if not is_json_record_safe(item):
                raise ValidationError(
                    "INVALID_JSON_STRUCTURE",
                    "Nested objects or arrays in JSON records are not supported. Flatten the file before upload.",
                    status_code=422,
                )
    elif isinstance(data, dict):
        if not data:
            raise ValidationError(
                "EMPTY_JSON_DATA",
                "Uploaded JSON object contains no records.",
                status_code=422,
            )
        if not all(isinstance(v, list) for v in data.values()):
            raise ValidationError(
                "INVALID_JSON_STRUCTURE",
                "JSON object must map field names to arrays of equal length.",
                status_code=422,
            )
        lengths = {len(v) for v in data.values()}
        if len(lengths) > 1:
            raise ValidationError(
                "INVALID_JSON_STRUCTURE",
                "JSON object arrays must all have the same length.",
                {"lengths": sorted(lengths)},
                status_code=422,
            )
    else:
        raise ValidationError(
            "INVALID_JSON_STRUCTURE",
            "JSON must be an object or an array of objects.",
            status_code=422,
        )
