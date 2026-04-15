"""
BiasLens — services/explainer.py (Debugged)
AI-powered plain-English audit explanations via NVIDIA NIM API.
Falls back to Google Gemini, then rule-based explanation.
"""
from typing import Optional, List
from models.schemas import AuditSummary, BiasIssue, MetricResult
from config import settings
import traceback


async def get_ai_explanation(
    filename: str,
    summary: AuditSummary,
    issues: List[BiasIssue],
    metrics: List[MetricResult]
) -> Optional[str]:
    """
    Generate a plain-English audit explanation.
    Uses NVIDIA NIM API (llama-3.1-nemotron-ultra) if key available,
    else Gemini, else rule-based fallback.
    """
    # 1. Primary AI (NVIDIA)
    if getattr(settings, "NVIDIA_API_KEY", None):
        result = await _nvidia_explanation(filename, summary, issues, metrics)
        if result:
            return result

    # 2. Secondary AI (Gemini)
    if getattr(settings, "GEMINI_API_KEY", None):
        result = await _gemini_explanation(filename, summary, issues, metrics)
        if result:
            return result

    # 3. Fallback (No APIs available or both failed)
    return _fallback_explanation(filename, summary, issues)


def _build_prompt(filename: str, summary: AuditSummary, issues: List[BiasIssue], metrics: List[MetricResult]) -> str:
    """Builds a structured prompt for the LLMs without markdown syntax errors."""
    
    # FIXED: Severity is an Enum. We must use .value safely.
    crits = [i for i in issues if getattr(i.severity, "value", str(i.severity)).lower() == "critical"]
    warns = [i for i in issues if getattr(i.severity, "value", str(i.severity)).lower() == "warning"]

    issues_txt = ""
    for i in crits[:3]:
        issues_txt += f"\n- CRITICAL: {i.title} (value: {i.metric_value})"
    for i in warns[:2]:
        issues_txt += f"\n- WARNING: {i.title} (value: {i.metric_value})"

    if not issues_txt:
        issues_txt = "\n- None detected. Data is highly fair."

    metric_txt = ""
    for m in metrics[:6]:
        attr_str = m.attribute or 'overall'
        pass_str = 'PASS' if m.pass_fail else 'FAIL'
        metric_txt += f"\n- {m.name} [{attr_str}]: {m.value:.4f} ({pass_str})"

    # Calculate total metrics safely
    total_metrics = summary.passed_count + summary.critical_count + summary.warning_count

    return f"""You are an expert in AI fairness and bias detection. A dataset named '{filename}' has been audited.

AUDIT RESULTS:
- Overall Fairness Score: {summary.overall_score}/100 (Grade: {summary.fairness_grade})
- Critical Issues: {summary.critical_count}
- Warnings: {summary.warning_count}
- Metrics Passed: {summary.passed_count} out of {total_metrics}

TOP ISSUES:{issues_txt}

KEY METRICS:{metric_txt}

Write a clear, professional 3-paragraph executive summary (under 150 words total):
1. Overall assessment — what does the score mean in practice?
2. The most serious bias issue found and its real-world legal/ethical impact.
3. The single most important action to take immediately.

Be direct, specific, and use plain English. No bullet points. No markdown formatting."""


async def _nvidia_explanation(filename, summary, issues, metrics) -> Optional[str]:
    """Call NVIDIA NIM API asynchronously."""
    try:
        import httpx
        prompt = _build_prompt(filename, summary, issues, metrics)

        payload = {
            "model": "nvidia/llama-3.1-nemotron-ultra-253b", # FIXED: Standardized model name
            "messages": [
                {"role": "system", "content": "You are an AI fairness expert. Be concise and direct."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300,
            "temperature": 0.3,
            "top_p": 0.9,
        }

        # FIXED: Lowered timeout. If it takes >15s, it's better to switch to Gemini
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"NVIDIA API Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"NVIDIA integration failed: {e}")
    return None


async def _gemini_explanation(filename, summary, issues, metrics) -> Optional[str]:
    """Call Google Gemini API asynchronously using run_in_executor."""
    try:
        import google.generativeai as genai
        import asyncio
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = _build_prompt(filename, summary, issues, metrics)
        
        # FIXED: The genai library is synchronous. Calling it directly in an async function 
        # blocks the FastAPI event loop. We must wrap it in a thread.
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, model.generate_content, prompt)
        
        return response.text.strip()
    except Exception as e:
        print(f"Gemini integration failed: {e}")
    return None


def _fallback_explanation(filename, summary, issues) -> str:
    """Rule-based explanation when no API key is configured or both fail."""
    grade_map = {
        "A": "excellent — this dataset meets all major fairness standards",
        "B": "good — minor fairness improvements are recommended",
        "C": "concerning — significant bias issues were detected",
        "D": "poor — serious discrimination patterns require immediate attention",
        "F": "failing — critical bias detected, deployment would cause real harm",
    }
    desc = grade_map.get(summary.fairness_grade, "requires further investigation")
    
    # Safely get severities
    crits = [i for i in issues if getattr(i.severity, "value", str(i.severity)).lower() == "critical"]
    warns = [i for i in issues if getattr(i.severity, "value", str(i.severity)).lower() == "warning"]

    para1 = (
        f"The dataset '{filename}' received a fairness score of {summary.overall_score}/100 "
        f"(Grade {summary.fairness_grade}), which is {desc}. "
        f"{summary.critical_count} critical issue(s) and {summary.warning_count} warning(s) were detected."
    )

    para2 = ""
    if crits:
        top = crits[0]
        para2 = (
            f"The most serious finding is '{top.title}' with a value of {top.metric_value}. "
            f"This indicates that certain demographic groups are receiving significantly different outcomes. "
        )
        if top.legal_risk:
            para2 += f"{top.legal_risk} "
    else:
        para2 = "No critical demographic disparities were found across the evaluated metrics."

    para3 = (
        f"The immediate recommended action is to apply reweighing to the training data "
        f"and remove proxy variables before any model is trained or deployed."
    )
    if warns:
        para3 = f"Additionally, {len(warns)} warning(s) require attention. " + para3

    return f"{para1}\n\n{para2}\n\n{para3}".strip()