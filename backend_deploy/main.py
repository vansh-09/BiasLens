"""
BiasLens API v2.6.0 - main.py (Full Tier 1 Edition)
========================================================
Run: uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import io
from typing import Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional, Dict, List
import traceback
import time
import hashlib
import asyncio
from collections import defaultdict
import pandas as pd
import numpy as np
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# Auth Imports
import firebase_admin
from firebase_admin import credentials, auth
from services.auth import is_admin, Token, TokenData

# Service & Utils Imports
from config import settings
from models.schemas import AuditResponse, MitigationAuditResponse, HeatmapData
from utils.file_parser import (
    parse_uploaded_file, auto_detect_label_column,
    auto_detect_sensitive_attributes, preprocess_dataframe, encode_dataframe
)
from utils.helpers import generate_audit_id, get_timestamp, Timer
from utils.validation import (
    ValidationError,
    validate_dataset_structure,
    validate_sensitive_attributes,
    validate_target_column,
    validate_binary_labels,
    validate_positive_label,
)
from services.metrics import run_all_metrics, detect_proxy_variables, compute_group_outcomes
from services.detector import build_dataset_info, build_issues_from_metrics, build_audit_summary
from services.mitigator import get_relevant_strategies, apply_mitigation
from services.explainer import get_ai_explanation
from services.reporter import generate_pdf_report
from services.export_service import (
    generate_json_export,
    generate_csv_export
)
from services.analyzer import compute_correlation_heatmap

try:
    firebase_admin.get_app()
except ValueError:
    # Initialize without a file by explicitly passing your Project ID!
    firebase_admin.initialize_app(options={
        'projectId': 'biaslens-3c517'
    })

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login", auto_error=False)

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)):
    if not token:
        return None
    try:
        decoded_token = auth.verify_id_token(token)
        return TokenData(email=decoded_token.get('email', ''), role="user", uid=decoded_token['uid'])
    except Exception as e:
        print(f"Auth bypass: {e}")
        return None

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="AI Fairness Auditor - detects bias in datasets and ML models using 9 fairness metrics.",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://main.d273set5uy7s7z.amplifyapp.com",
        "https://bias-lens-opal.vercel.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# == SECURITY MIDDLEWARE ======================================================
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return response

app.add_middleware(SecurityHeadersMiddleware)

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and exc.detail.get("error"):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
            },
        },
    )

# == RATE LIMITING ============================================================
_rate_limits: Dict[str, list] = defaultdict(list)
RATE_LIMIT_WINDOW = 60   # seconds
RATE_LIMIT_MAX = 30      # max requests per window
AUTH_RATE_LIMIT_MAX = 8   # stricter for auth endpoints

def check_rate_limit(ip: str, max_requests: int = RATE_LIMIT_MAX) -> bool:
    now = time.time()
    _rate_limits[ip] = [t for t in _rate_limits[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limits[ip]) >= max_requests:
        return False
    _rate_limits[ip].append(now)
    return True

# == GLOBAL CACHE FOR TIER 1 PERFORMANCE ======================================
ai_explanation_cache: Dict[str, str] = {}

# == Health ===================================================================
@app.get("/", tags=["health"])
async def root():
    return {"name": settings.APP_NAME, "version": settings.VERSION, "status": "running"}

@app.get("/health", tags=["health"])
async def health():
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "nvidia_configured": bool(settings.NVIDIA_API_KEY),
        "gemini_configured": bool(settings.GEMINI_API_KEY),
    }

# == CORE AUDIT LOGIC (Optimized) =============================================
async def _run_audit_pipeline(
    df: pd.DataFrame,
    filename: str,
    label_column: Optional[str] = None,
    sensitive_attributes: Optional[str] = None,
    positive_label: Optional[str] = "1",
    use_ai_explanation: bool = False
) -> AuditResponse:
    timer = Timer()
    df_raw = df.copy()

    validate_dataset_structure(df)

    # Tier 1 Optimization: Guard against massive datasets crashing the demo
    if len(df) > 50000:
        df = df.head(50000)

    # 1. Resolve label column
    if label_column:
        label_col = validate_target_column(df, label_column.strip())
    else:
        label_col = auto_detect_label_column(df)
        if not label_col:
            raise ValidationError(
                "MISSING_TARGET_COLUMN",
                f"Cannot detect target label column. Available columns: {df.columns.tolist()}.",
                {"available_columns": df.columns.tolist()},
            )
        validate_target_column(df, label_col)

    # 2. Resolve sensitive attributes
    if sensitive_attributes:
        requested = [a.strip() for a in sensitive_attributes.split(",") if a.strip()]
        s_attrs = validate_sensitive_attributes(df, requested)
    else:
        s_attrs = auto_detect_sensitive_attributes(df, label_col)
        if not s_attrs:
            raise ValidationError(
                "MISSING_SENSITIVE_ATTRIBUTES",
                "No sensitive attributes detected. Please specify manually.",
                {"available_columns": [c for c in df.columns if c != label_col]},
            )
        s_attrs = validate_sensitive_attributes(df, s_attrs, allow_autodetect=True)

    validate_binary_labels(df, label_col)
    pos_label = validate_positive_label(df, label_col, positive_label)

    # 3. Preprocess & Encode
    df_clean = preprocess_dataframe(df, label_col)
    df_encoded, _ = encode_dataframe(df_clean, label_col, s_attrs)

    # 4. Run Metrics asynchronously to preserve main thread
    metrics = await asyncio.to_thread(run_all_metrics, df_encoded, s_attrs, label_col, pos_label)
    if not metrics:
        raise HTTPException(status_code=500, detail="Metric computation failed.")

    # 5. Proxies, Outcomes, Issues, Strategies
    proxy_vars = await asyncio.to_thread(detect_proxy_variables, df_encoded, s_attrs, label_col)
    group_outcomes = await asyncio.to_thread(compute_group_outcomes, df_encoded, s_attrs, label_col, pos_label)
    issues = await asyncio.to_thread(build_issues_from_metrics, metrics, proxy_vars)
    strategies = await asyncio.to_thread(get_relevant_strategies, issues)
    dataset_info = await asyncio.to_thread(build_dataset_info, df_clean, s_attrs, label_col, df_raw)
    summary = await asyncio.to_thread(build_audit_summary, metrics, issues, timer)

    # 6. AI Explanation
    ai_explanation = None
    if use_ai_explanation:
        ai_explanation = await get_ai_explanation(filename, summary, issues, metrics)

    return AuditResponse(
        audit_id=generate_audit_id(),
        filename=filename,
        dataset_info=dataset_info,
        summary=summary,
        metrics=metrics,
        issues=issues,
        proxy_variables=proxy_vars,
        group_outcomes=group_outcomes,
        mitigation_strategies=strategies,
        ai_explanation=ai_explanation,
        created_at=get_timestamp()
    )


def build_success_response(payload: dict, validation: Optional[dict] = None) -> dict:
    response = {"success": True, "data": payload, **payload}
    if validation is not None:
        response["validation"] = validation
    return response

# == Main Audit Endpoint =======================================================
@app.post("/api/audit", tags=["audit"])
async def audit_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Dataset file (CSV, JSON, XLSX, Parquet)"),
    label_column: Optional[str]  = Form(None),
    sensitive_attributes: Optional[str] = Form(None),
    positive_label: Optional[str] = Form("1"),
    use_ai_explanation: bool = Form(True),
    user: Optional[TokenData] = Depends(get_current_user),
):
    try:
        df = await parse_uploaded_file(file)
        
        result = await _run_audit_pipeline(
            df=df,
            filename=file.filename,
            label_column=label_column,
            sensitive_attributes=sensitive_attributes,
            positive_label=positive_label,
            use_ai_explanation=False 
        )
        
        if use_ai_explanation:
            background_tasks.add_task(
                background_ai_worker, 
                result.audit_id, file.filename, result.summary, result.issues, result.metrics
            )

        return build_success_response(result.model_dump(), validation={
            "label_column": result.dataset_info.label_column,
            "sensitive_attributes": result.dataset_info.sensitive_attributes,
            "positive_label": positive_label,
        })
    except ValidationError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

async def background_ai_worker(audit_id, filename, summary, issues, metrics):
    try:
        explanation = await get_ai_explanation(filename, summary, issues, metrics)
        ai_explanation_cache[audit_id] = explanation
    except HTTPException as e:
        ai_explanation_cache[audit_id] = f"AI service unavailable ({e.status_code}): {e.detail}"
    except Exception as e:
        ai_explanation_cache[audit_id] = f"Error generating AI summary: {str(e)}"

@app.get("/api/audit/ai-status/{audit_id}", tags=["audit"])
async def get_ai_status(audit_id: str):
    if audit_id in ai_explanation_cache:
        return {"status": "ready", "explanation": ai_explanation_cache[audit_id]}
    return {"status": "processing"}


# == Mitigation Endpoint ======================================================
@app.post("/api/mitigate", tags=["audit"])
async def mitigate_bias(
    file: UploadFile = File(...),
    strategy_id: str = Form(...),
    label_column: str = Form(...),
    sensitive_attributes: str = Form(...),
    positive_label: Optional[str] = Form("1"),
):
    try:
        df_orig = await parse_uploaded_file(file)
        s_attrs = [a.strip() for a in sensitive_attributes.split(",") if a.strip()]
        validate_sensitive_attributes(df_orig, s_attrs)
        orig_audit = await _run_audit_pipeline(df_orig.copy(), file.filename, label_column, sensitive_attributes, positive_label)

        df_mitigated, desc = apply_mitigation(df_orig, strategy_id, label_column, s_attrs, positive_label)
        mitigated_audit = await _run_audit_pipeline(df_mitigated, f"mitigated_{file.filename}", label_column, sensitive_attributes, positive_label)

        payload = {
            "original_audit": orig_audit.model_dump(),
            "mitigated_audit": mitigated_audit.model_dump(),
            "mitigation_applied": desc,
            "improvement_score": mitigated_audit.summary.overall_score - orig_audit.summary.overall_score,
            "mitigated_filename": f"mitigated_{file.filename}",
        }
        return build_success_response(payload, validation={
            "strategy_id": strategy_id,
            "label_column": label_column,
            "sensitive_attributes": s_attrs,
            "positive_label": positive_label,
        })
    except ValidationError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Mitigation failed: {str(e)}")

# == Advanced Visualization ===================================================
@app.post("/api/analyze/heatmap", response_model=HeatmapData, tags=["utilities"])
async def analyze_heatmap(
    file: UploadFile = File(...),
    sensitive_attributes: str = Form(...),
    label_column: str = Form(...),
):
    try:
        df = await parse_uploaded_file(file)
        label_col = validate_target_column(df, label_column.strip())
        s_attrs = validate_sensitive_attributes(df, [a.strip() for a in sensitive_attributes.split(",") if a.strip()])
        data = compute_correlation_heatmap(df, s_attrs, label_col)
        return HeatmapData(**data)
    except ValidationError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# == PDF Report ===============================================================
@app.post("/api/audit/report", tags=["audit"])
async def audit_pdf(data: Dict[str, Any], user: Optional[TokenData] = Depends(get_current_user)): 
    try:
        print(f"[*] PDF Generation Started for: {data.get('filename', 'unknown')}")
        from models.schemas import AuditResponse
        # Use .parse_obj for more robust dict-to-model conversion
        try:
            audit_obj = AuditResponse.model_validate(data)
        except Exception as ve:
            print(f"[!] Validation Error: {ve}")
            # Fallback to manual object construction or improved error
            raise HTTPException(status_code=422, detail=f"Data validation failed: {str(ve)}")

        pdf = generate_pdf_report(audit_obj)
        print(f"[*] PDF Successfully Generated. Size: {pdf.getbuffer().nbytes} bytes")
        return StreamingResponse(
            pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="biaslens_{audit_obj.audit_id}.pdf"'}
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF Generation Error: {str(e)}")



    # == JSON Export =============================================================
@app.post("/api/audit/export/json", tags=["audit"])
async def export_json(
    data: Dict[str, Any],
    user: Optional[TokenData] = Depends(get_current_user)
):

    try:
        json_data = generate_json_export(data)

        return StreamingResponse(
            io.BytesIO(json_data.encode("utf-8")),
            media_type="application/json",
            headers={
                "Content-Disposition": "attachment; filename=biaslens_report.json"
            }
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"JSON export failed: {str(e)}")


# == CSV Export ==============================================================
@app.post("/api/audit/export/csv", tags=["audit"])
async def export_csv(
    data: Dict[str, Any],
    user: Optional[TokenData] = Depends(get_current_user)
):

    try:
        csv_file = generate_csv_export(data)

        return StreamingResponse(
            iter([csv_file.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=biaslens_report.csv"
            }
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")    

# == Column Detection =========================================================
@app.post("/api/detect-columns", tags=["utilities"])
async def detect_columns(file: UploadFile = File(...)):
    df = await parse_uploaded_file(file)
    label = auto_detect_label_column(df)
    sensitive = auto_detect_sensitive_attributes(df, label or "")
    return {
        "columns": df.columns.tolist(),
        "detected_label": label,
        "detected_sensitive": sensitive,
        "sample_data": df.head(3).to_dict(orient="records"),
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
    }

# == Metrics List =============================================================
@app.get("/api/metrics/list", tags=["utilities"])
async def list_metrics():
    return {"metrics": [
        {"name": "Disparate Impact", "threshold": 0.80, "ideal": ">= 0.80"},
        {"name": "Statistical Parity Difference","threshold": 0.10, "ideal": "0"},
        {"name": "Equal Opportunity Difference", "threshold": 0.10, "ideal": "0"},
        {"name": "Average Odds Difference", "threshold": 0.10, "ideal": "0"},
        {"name": "Predictive Parity", "threshold": 0.10, "ideal": "0"},
        {"name": "Individual Fairness", "threshold": 0.80, "ideal": ">= 0.80"},
        {"name": "Calibration Score", "threshold": 0.70, "ideal": ">= 0.70"},
        {"name": "Theil Index", "threshold": 0.10, "ideal": "0"},
        {"name": "Demographic Parity Ratio", "threshold": 0.80, "ideal": ">= 0.80"},
    ]}

# == Error Handlers ===========================================================
@app.exception_handler(404)
async def not_found(req, exc):
    return JSONResponse(status_code=404, content={"error": "Not found"})

if __name__ == "__main__":
    import uvicorn
    import os
    print(f"""
+------------------------------------------+
|  BIASLENS API  v{settings.VERSION}               |
|  Running on Render / Local               |
+------------------------------------------+
    """)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)