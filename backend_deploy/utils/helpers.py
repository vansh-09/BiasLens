import uuid
import time
from datetime import datetime
from typing import Any


def generate_audit_id() -> str:
    return f"audit_{uuid.uuid4().hex[:12]}"


def get_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def score_to_grade(score: int) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 45: return "D"
    return "F"


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def round2(val: Any) -> float:
    try:
        return round(float(val), 4)
    except Exception:
        return 0.0


class Timer:
    def __init__(self):
        self._start = time.time()

    def elapsed(self) -> float:
        return round(time.time() - self._start, 3)