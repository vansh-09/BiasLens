import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_audit_endpoint_invalid_label_returns_structured_error():
    records = ["20,male", "30,female"] * 25
    csv_content = "age,gender\n" + "\n".join(records) + "\n"
    response = client.post(
        "/api/audit",
        files={"file": ("test.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] in {
        "MISSING_TARGET_COLUMN",
        "INVALID_TARGET_COLUMN",
        "NON_BINARY_TARGET",
        "INVALID_LABEL_DISTRIBUTION",
        "INVALID_POSITIVE_LABEL",
    }
    assert isinstance(data["error"]["message"], str)


def test_mitigate_endpoint_invalid_sensitive_returns_structured_error():
    records = ["0,20", "1,30"] * 25
    csv_content = "target,age\n" + "\n".join(records) + "\n"
    response = client.post(
        "/api/mitigate",
        data={
            "strategy_id": "mit_resampling",
            "label_column": "target",
            "sensitive_attributes": "gender",
        },
        files={"file": ("test.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "INVALID_SENSITIVE_ATTRIBUTE"
    assert "gender" in data["error"]["message"]


def test_unsupported_file_type_returns_structured_error():
    response = client.post(
        "/api/audit",
        files={"file": ("invalid.txt", "hello world", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "UNSUPPORTED_FILE_TYPE"
