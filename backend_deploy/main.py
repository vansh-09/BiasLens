"""
BiasLens API v2.6.0 - main.py (Full Tier 1 Edition)
========================================================
Run: uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
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
from services.auth import (
    get_password_hash, verify_password, create_access_token, 
    decode_token, is_admin, Token, TokenData
)
from database import (
    get_db_connection, init_db, log_login_attempt,
    check_account_locked, increment_failed_attempts, reset_failed_attempts
)

# Service & Utils Imports
from config import settings
from models.schemas import AuditResponse, MitigationAuditResponse, HeatmapData
from utils.file_parser import (
    parse_uploaded_file, auto_detect_label_column,
    auto_detect_sensitive_attributes, preprocess_dataframe, encode_dataframe
)
from utils.helpers import generate_audit_id, get_timestamp, Timer
from services.metrics import run_all_metrics, detect_proxy_variables, compute_group_outcomes
from services.detector import build_dataset_info, build_issues_from_metrics, build_audit_summary
from services.mitigator import get_relevant_strategies, apply_mitigation
from services.explainer import get_ai_explanation
from services.reporter import generate_pdf_report
from services.analyzer import compute_correlation_heatmap

# Initialize DB on start
init_db()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = decode_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="AI Fairness Auditor - detects bias in datasets and ML models using 9 fairness metrics.",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    # User requested: CORS(app, resources={r"/*": {"origins": "*"}})
    allow_origins=["*"],
    allow_credentials=False,
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

    # Tier 1 Optimization: Guard against massive datasets crashing the demo
    if len(df) > 50000:
        df = df.head(50000)

    # 1. Resolve label column
    label_col = label_column or auto_detect_label_column(df)
    if not label_col or label_col not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot find label column. Available columns: {df.columns.tolist()}."
        )

    # 2. Resolve sensitive attributes
    s_attrs = None
    if sensitive_attributes:
        s_attrs = [a.strip() for a in sensitive_attributes.split(",")
                   if a.strip() in df.columns]

    if not s_attrs:
        s_attrs = auto_detect_sensitive_attributes(df, label_col)

    if not s_attrs:
        raise HTTPException(
            status_code=422,
            detail="No sensitive attributes detected. Please specify manually."
        )

    # 3. SAFETY FIX: Smart type matching for labels
    pos_label = positive_label
    if label_col in df.columns:
        actual_type = df[label_col].dtype
        try:
            if np.issubdtype(actual_type, np.integer):
                pos_label = int(float(positive_label))
            elif np.issubdtype(actual_type, np.floating):
                pos_label = float(positive_label)
        except Exception:
            pos_label = str(positive_label)

    # 4. Preprocess & Encode
    df_clean = preprocess_dataframe(df, label_col)
    df_encoded, _ = encode_dataframe(df_clean, label_col, s_attrs)

    # 5. Run Metrics asynchronously to preserve main thread 
    metrics = await asyncio.to_thread(run_all_metrics, df_encoded, s_attrs, label_col, pos_label)
    if not metrics:
        raise HTTPException(status_code=500, detail="Metric computation failed.")

    # 6. Proxies, Outcomes, Issues, Strategies 
    proxy_vars = await asyncio.to_thread(detect_proxy_variables, df_encoded, s_attrs, label_col)
    group_outcomes = await asyncio.to_thread(compute_group_outcomes, df_encoded, s_attrs, label_col, pos_label)
    issues = await asyncio.to_thread(build_issues_from_metrics, metrics, proxy_vars)
    strategies = await asyncio.to_thread(get_relevant_strategies, issues)
    dataset_info = await asyncio.to_thread(build_dataset_info, df_clean, s_attrs, label_col, df_raw)
    summary = await asyncio.to_thread(build_audit_summary, metrics, issues, timer)

    # 7. AI Explanation
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

# == Main Audit Endpoint =======================================================
@app.post("/api/audit", response_model=AuditResponse, tags=["audit"])
async def audit_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Dataset file (CSV, JSON, XLSX, Parquet)"),
    label_column: Optional[str]  = Form(None),
    sensitive_attributes: Optional[str] = Form(None),
    positive_label: Optional[str] = Form("1"),
    use_ai_explanation: bool = Form(True),
):
    try:
        df = await parse_uploaded_file(file)
        
        # Run math engine immediately
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
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

async def background_ai_worker(audit_id, filename, summary, issues, metrics):
    try:
        explanation = await get_ai_explanation(filename, summary, issues, metrics)
        ai_explanation_cache[audit_id] = explanation
    except Exception as e:
        ai_explanation_cache[audit_id] = f"Error generating AI summary: {str(e)}"

@app.get("/api/audit/ai-status/{audit_id}", tags=["audit"])
async def get_ai_status(audit_id: str):
    if audit_id in ai_explanation_cache:
        return {"status": "ready", "explanation": ai_explanation_cache[audit_id]}
    return {"status": "processing"}

# == Auth Endpoints ===========================================================
@app.post("/api/auth/register", tags=["auth"])
async def register(request: Request, email: str = Form(...), password: str = Form(...)):
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(f"auth:{client_ip}", AUTH_RATE_LIMIT_MAX):
        raise HTTPException(status_code=429, detail="Too many requests. Try again later.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if len(email) > 200 or '@' not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    
    hashed = get_password_hash(password)
    role = "admin" if is_admin(email) else "user"
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO users (email, hashed_password, role, provider) VALUES (?, ?, ?, ?)",
            (email, hashed, role, "email")
        )
        conn.commit()
        log_login_attempt(email, True, client_ip, "", "register")
        return {"message": "Registration successful"}
    except Exception:
        raise HTTPException(status_code=400, detail="Email already registered")
    finally:
        conn.close()

@app.post("/api/auth/login", response_model=Token, tags=["auth"])
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(f"auth:{client_ip}", AUTH_RATE_LIMIT_MAX):
        raise HTTPException(status_code=429, detail="Too many login attempts. Wait 60 seconds.")
    
    if check_account_locked(form_data.username):
        raise HTTPException(status_code=423, detail="Account locked due to too many failed attempts.")

    # MASTER BYPASS
    if is_admin(form_data.username) and form_data.password == "Ay@310807":
        access_token = create_access_token(data={"sub": form_data.username, "role": "admin"})
        log_login_attempt(form_data.username, True, client_ip, "", "admin")
        reset_failed_attempts(form_data.username)
        return {"access_token": access_token, "token_type": "bearer"}

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (form_data.username,)).fetchone()
    conn.close()
    
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        increment_failed_attempts(form_data.username)
        log_login_attempt(form_data.username, False, client_ip, "", "email")
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    reset_failed_attempts(form_data.username)
    access_token = create_access_token(data={"sub": user["email"], "role": user["role"]})
    log_login_attempt(user["email"], True, client_ip, "", "email")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me", tags=["auth"])
async def get_me(user: TokenData = Depends(get_current_user)):
    return user

@app.post("/api/auth/demo", response_model=Token, tags=["auth"])
async def demo_login(request: Request, provider: str = Form(...)):
    client_ip = request.client.host if request.client else "unknown"
    access_token = create_access_token(data={"sub": f"vip_{provider.lower()}@demo.local", "role": "user"})
    log_login_attempt(f"vip_{provider.lower()}@demo.local", True, client_ip, "", provider)
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/auth/social", response_model=Token, tags=["auth"])
async def social_login(
    request: Request,
    provider: str = Form(...),
    email: str = Form(...),
    name: str = Form("")
):
    """Social OAuth login — creates/finds user and issues JWT token."""
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(f"auth:{client_ip}", AUTH_RATE_LIMIT_MAX):
        raise HTTPException(status_code=429, detail="Too many requests.")
    
    email = email.strip()[:200]
    name = name.strip()[:100]
    provider = provider.strip()[:20]
    
    conn = get_db_connection()
    try:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            random_pw = get_password_hash(hashlib.sha256(f"{email}{time.time()}".encode()).hexdigest())
            role = "admin" if is_admin(email) else "user"
            conn.execute(
                "INSERT INTO users (email, hashed_password, role, provider, display_name) VALUES (?, ?, ?, ?, ?)",
                (email, random_pw, role, provider, name)
            )
            conn.commit()
        
        conn.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE email = ?", (email,))
        conn.commit()
        
        role = "admin" if is_admin(email) else (user["role"] if user else "user")
        access_token = create_access_token(data={"sub": email, "role": role, "name": name, "provider": provider})
        log_login_attempt(email, True, client_ip, "", provider)
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Social login failed: {str(e)}")
    finally:
        conn.close()


# == Mitigation Endpoint ======================================================
@app.post("/api/mitigate", response_model=MitigationAuditResponse, tags=["audit"])
async def mitigate_bias(
    file: UploadFile = File(...),
    strategy_id: str = Form(...),
    label_column: str = Form(...),
    sensitive_attributes: str = Form(...),
    positive_label: Optional[str] = Form("1"),
):
    try:
        df_orig = await parse_uploaded_file(file)
        orig_audit = await _run_audit_pipeline(df_orig.copy(), file.filename, label_column, sensitive_attributes, positive_label)
        s_attrs = [a.strip() for a in sensitive_attributes.split(",")]
        df_mitigated, desc = apply_mitigation(df_orig, strategy_id, label_column, s_attrs, positive_label)
        mitigated_audit = await _run_audit_pipeline(df_mitigated, f"mitigated_{file.filename}", label_column, sensitive_attributes, positive_label)

        return MitigationAuditResponse(
            original_audit=orig_audit,
            mitigated_audit=mitigated_audit,
            mitigation_applied=desc,
            improvement_score=mitigated_audit.summary.overall_score - orig_audit.summary.overall_score,
            mitigated_filename=f"mitigated_{file.filename}"
        )
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
        s_attrs = [a.strip() for a in sensitive_attributes.split(",") if a.strip() in df.columns]
        data = compute_correlation_heatmap(df, s_attrs, label_column)
        return HeatmapData(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# == PDF Report ===============================================================
@app.post("/api/audit/report", tags=["audit"])
async def audit_pdf(data: AuditResponse, user: TokenData = Depends(get_current_user)): 
    pdf = generate_pdf_report(data)
    return StreamingResponse(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="biaslens_{data.audit_id}.pdf"'}
    )

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
    print(f"""
+------------------------------------------+
|  BIASLENS API  v{settings.VERSION}               |
|  Running: http://127.0.0.1:8000          |
+------------------------------------------+
    """)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)