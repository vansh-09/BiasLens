"""
BiasLens — schemas.py (Debugged)
Pydantic models for strict data validation and API documentation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    PASS = "pass"
    INFO = "info"


class MetricResult(BaseModel):
    name: str
    value: float
    threshold: float
    severity: SeverityLevel
    description: str
    attribute: Optional[str] = None
    group: Optional[str] = None
    pass_fail: bool


class ProxyVariable(BaseModel):
    column: str
    sensitive_attr: str
    correlation: float
    severity: SeverityLevel
    description: str


class GroupOutcome(BaseModel):
    group_name: str
    attribute: str
    selection_rate: float
    count: int
    approved: int


class BiasIssue(BaseModel):
    id: str
    severity: SeverityLevel
    title: str
    description: str
    metric_value: str
    affected_attribute: str
    affected_group: Optional[str] = None
    legal_risk: Optional[str] = None
    recommendation: str


class MitigationStrategy(BaseModel):
    id: str
    title: str
    stage: str  # pre-processing, in-processing, post-processing
    description: str
    implementation: str
    effort: str   # Low, Medium, High
    impact: str   # Low, Medium, High
    library: str
    code_snippet: str


class DatasetInfo(BaseModel):
    total_records: int
    total_features: int
    sensitive_attributes: List[str]
    label_column: str
    missing_values: Dict[str, int]
    class_distribution: Dict[str, int]
    # Changed to Any to handle nested JSON objects from different datasets
    demographic_breakdown: Dict[str, Any] 


class AuditSummary(BaseModel):
    overall_score: int
    total_issues: int
    critical_count: int
    warning_count: int
    passed_count: int
    fairness_grade: str  # A, B, C, D, F
    analysis_time_seconds: float


class AuditResponse(BaseModel):
    audit_id: str
    filename: str
    dataset_info: DatasetInfo
    summary: AuditSummary
    metrics: List[MetricResult]
    issues: List[BiasIssue]
    proxy_variables: List[ProxyVariable]
    group_outcomes: List[GroupOutcome]
    mitigation_strategies: List[MitigationStrategy]
    # Crucial for Tier 1: ai_explanation can be a string or a status dict
    ai_explanation: Optional[Union[str, Dict[str, str]]] = None 
    created_at: str


class MitigationAuditResponse(BaseModel):
    original_audit: AuditResponse
    mitigated_audit: AuditResponse
    mitigation_applied: str
    improvement_score: float # Changed to float for precise math
    mitigated_filename: str


class HeatmapData(BaseModel):
    x_labels: List[str]
    y_labels: List[str]
    matrix: List[List[float]]


class AnalyzeRequest(BaseModel):
    label_column: Optional[str] = None
    sensitive_attributes: Optional[List[str]] = None
    # Any allows for "1", 1, "Approved", etc.
    positive_label: Optional[Any] = 1 


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    field: Optional[str] = None