import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import pytest

from utils.validation import (
    ValidationError,
    validate_duplicate_columns,
    validate_empty_dataset,
    validate_binary_labels,
    validate_positive_label,
    validate_sensitive_attributes,
    validate_json_payload,
)


def test_validate_duplicate_columns_raises():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValidationError) as exc:
        validate_duplicate_columns(df)
    assert exc.value.code == "DUPLICATE_COLUMNS"


def test_validate_empty_dataset_raises():
    df = pd.DataFrame(columns=["a", "target"])
    with pytest.raises(ValidationError) as exc:
        validate_empty_dataset(df)
    assert exc.value.code == "EMPTY_DATASET"


def test_validate_binary_labels_single_class():
    df = pd.DataFrame({"target": [1, 1, 1]})
    with pytest.raises(ValidationError) as exc:
        validate_binary_labels(df, "target")
    assert exc.value.code == "INVALID_LABEL_DISTRIBUTION"


def test_validate_positive_label_missing():
    df = pd.DataFrame({"target": [0, 1, 0]})
    with pytest.raises(ValidationError) as exc:
        validate_positive_label(df, "target", "2")
    assert exc.value.code == "INVALID_POSITIVE_LABEL"


def test_validate_sensitive_attributes_missing_attribute():
    df = pd.DataFrame({"target": [0, 1, 0], "age": [25, 35, 45]})
    with pytest.raises(ValidationError) as exc:
        validate_sensitive_attributes(df, ["gender"])
    assert exc.value.code == "INVALID_SENSITIVE_ATTRIBUTE"


def test_validate_sensitive_attributes_invalid_distribution():
    df = pd.DataFrame({"target": [0, 1, 0], "gender": ["male", "male", "male"]})
    with pytest.raises(ValidationError) as exc:
        validate_sensitive_attributes(df, ["gender"])
    assert exc.value.code == "INVALID_SENSITIVE_ATTRIBUTE"


def test_validate_json_payload_nested_structure_raises():
    payload = [{"a": 1, "b": {"c": 2}}]
    with pytest.raises(ValidationError) as exc:
        validate_json_payload(payload)
    assert exc.value.code == "INVALID_JSON_STRUCTURE"


def test_validate_json_payload_dict_length_mismatch():
    payload = {"a": [1, 2], "b": ["x"]}
    with pytest.raises(ValidationError) as exc:
        validate_json_payload(payload)
    assert exc.value.code == "INVALID_JSON_STRUCTURE"
