"""
BiasLens — services/reporter.py (Debugged)
Generates professional PDF audit reports.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.colors import HexColor
from io import BytesIO
from datetime import datetime
from typing import List
import html

from models.schemas import AuditResponse, SeverityLevel

# Color palette (Objects for ReportLab, Strings for HTML tags)
C_BG_STR      = "#020208"
C_CYAN_STR    = "#00f5ff"
C_PURPLE_STR  = "#7000ff"
C_PINK_STR    = "#ff0080"
C_GREEN_STR   = "#00ff88"
C_AMBER_STR   = "#fbbf24"
C_TEXT_STR    = "#1a1a2e"
C_MUTED_STR   = "#555577"

C_BG       = HexColor(C_BG_STR)
C_PURPLE   = HexColor(C_PURPLE_STR)
C_PINK     = HexColor(C_PINK_STR)
C_GREEN    = HexColor(C_GREEN_STR)
C_AMBER    = HexColor(C_AMBER_STR)
C_TEXT     = HexColor(C_TEXT_STR)
C_MUTED    = HexColor(C_MUTED_STR)
C_BORDER   = HexColor("#e0e0f0")
C_CRIT_BG  = HexColor("#fff0f3")
C_WARN_BG  = HexColor("#fffbeb")
C_PASS_BG  = HexColor("#f0fff8")


def generate_pdf_report(audit: AuditResponse) -> BytesIO:
    """Generate a full professional PDF audit report."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title=f"BiasLens Audit — {audit.filename}"
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Custom Styles ──
    title_style = ParagraphStyle("Title", parent=styles["Title"],
        fontSize=26, textColor=C_TEXT, spaceAfter=4, fontName="Helvetica-Bold")
    subtitle_style = ParagraphStyle("Sub", parent=styles["Normal"],
        fontSize=12, textColor=C_MUTED, spaceAfter=20)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"],
        fontSize=15, textColor=C_TEXT, spaceBefore=18, spaceAfter=8, fontName="Helvetica-Bold")
    h3_style = ParagraphStyle("H3", parent=styles["Heading3"],
        fontSize=12, textColor=C_MUTED, spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
        fontSize=10, textColor=C_TEXT, leading=16, spaceAfter=6)
    mono_style = ParagraphStyle("Mono", parent=styles["Code"],
        fontSize=9, textColor=HexColor("#444466"), leading=14,
        backColor=HexColor("#f5f5ff"), leftIndent=10, rightIndent=10)
    label_style = ParagraphStyle("Label", parent=styles["Normal"],
        fontSize=8, textColor=C_MUTED, fontName="Helvetica-Bold",
        spaceAfter=2, spaceBefore=6)

    # ── HEADER ──
    story.append(Paragraph("BIASLENS", ParagraphStyle("Brand",
        parent=styles["Normal"], fontSize=11, textColor=C_PURPLE,
        fontName="Helvetica-Bold", letterSpacing=4, spaceAfter=4)))
    
    # Safe escaping for filename in case it has XML breaking characters like < or &
    safe_filename = html.escape(audit.filename)
    story.append(Paragraph("AI Fairness Audit Report", title_style))
    story.append(Paragraph(
        f"File: <b>{safe_filename}</b> &nbsp;|&nbsp; "
        f"Audit ID: <font color='{C_PURPLE_STR}'>{audit.audit_id}</font> &nbsp;|&nbsp; "
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=C_PURPLE, spaceAfter=20))

    # ── EXECUTIVE SUMMARY BOX ──
    # FIXED: Type check for the AI Background Task logic
    if audit.ai_explanation and isinstance(audit.ai_explanation, str):
        story.append(Paragraph("Executive Summary", h2_style))
        summary_table = Table(
            [[Paragraph(audit.ai_explanation, ParagraphStyle("ExecSum",
                parent=styles["Normal"], fontSize=10, textColor=C_TEXT, leading=16))]],
            colWidths=["100%"]
        )
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), HexColor("#f0f0ff")),
            ("LEFTPADDING", (0,0), (-1,-1), 16),
            ("RIGHTPADDING", (0,0), (-1,-1), 16),
            ("TOPPADDING", (0,0), (-1,-1), 14),
            ("BOTTOMPADDING", (0,0), (-1,-1), 14),
            ("LINEAFTER", (0,0), (0,-1), 4, C_PURPLE),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 16))

    # ── SCORE CARDS ──
    story.append(Paragraph("Audit Overview", h2_style))
    score_color = (
        C_GREEN if audit.summary.overall_score >= 75 else
        C_AMBER if audit.summary.overall_score >= 50 else C_PINK
    )
    score_color_str = (
        C_GREEN_STR if audit.summary.overall_score >= 75 else
        C_AMBER_STR if audit.summary.overall_score >= 50 else C_PINK_STR
    )
    
    score_data = [
        [
            _score_cell("Fairness Score", f"{audit.summary.overall_score}/100",
                        f"Grade {audit.summary.fairness_grade}", score_color_str),
            _score_cell("Critical Issues", str(audit.summary.critical_count), "Immediate action needed", C_PINK_STR),
            _score_cell("Warnings",        str(audit.summary.warning_count),  "Review recommended",     C_AMBER_STR),
            _score_cell("Metrics Passed",  str(audit.summary.passed_count),   "Out of total metrics",   C_GREEN_STR),
        ]
    ]
    score_table = Table(score_data, colWidths=[4.1*cm]*4)
    score_table.setStyle(TableStyle([
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.5, C_BORDER),
        ("ROUNDEDCORNERS",(0,0),(-1,-1),4),
        ("TOPPADDING",  (0,0), (-1,-1), 14),
        ("BOTTOMPADDING",(0,0),(-1,-1), 14),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 20))

    # ── DATASET INFO ──
    story.append(Paragraph("Dataset Information", h2_style))
    ds = audit.dataset_info
    info_data = [
        ["Total Records", f"{ds.total_records:,}", "Total Features", str(ds.total_features)],
        ["Label Column", ds.label_column, "Sensitive Attributes", ", ".join(ds.sensitive_attributes)],
        ["Analysis Time", f"{audit.summary.analysis_time_seconds}s",
         "Missing Values", str(sum(ds.missing_values.values()))],
    ]
    _add_table(story, info_data, col_widths=[3.5*cm, 5.5*cm, 3.5*cm, 5.5*cm])
    story.append(Spacer(1, 20))

    # ── METRICS TABLE ──
    story.append(Paragraph("Fairness Metrics Results", h2_style))
    metrics_header = [["Metric", "Attribute", "Value", "Threshold", "Status"]]
    metrics_rows = []
    for m in audit.metrics:
        sev_color_str = {
            SeverityLevel.CRITICAL: C_PINK_STR,
            SeverityLevel.WARNING:  C_AMBER_STR,
            SeverityLevel.PASS:     C_GREEN_STR,
        }.get(m.severity, C_MUTED_STR)
        
        sev_label = {
            SeverityLevel.CRITICAL: "✗ CRITICAL",
            SeverityLevel.WARNING:  "⚠ WARNING",
            SeverityLevel.PASS:     "✓ PASS",
        }.get(m.severity, "?")
        
        metrics_rows.append([
            m.name,
            m.attribute or "—",
            f"{m.value:.4f}",
            f"{m.threshold:.2f}",
            Paragraph(f"<font color='{sev_color_str}'><b>{sev_label}</b></font>", body_style)
        ])

    all_rows = metrics_header + metrics_rows
    mt = Table(all_rows, colWidths=[5.5*cm, 2.5*cm, 2*cm, 2.2*cm, 2.5*cm])
    _style_metrics_table(mt, len(metrics_rows))
    story.append(mt)
    story.append(Spacer(1, 20))

    # ── ISSUES ──
    story.append(Paragraph(f"Issues Detected ({len(audit.issues)})", h2_style))
    for iss in audit.issues:
        sev = iss.severity
        bg = C_CRIT_BG if sev == SeverityLevel.CRITICAL else C_WARN_BG
        badge = "CRITICAL" if sev == SeverityLevel.CRITICAL else "WARNING"
        badge_color = C_PINK if sev == SeverityLevel.CRITICAL else C_AMBER
        badge_color_str = C_PINK_STR if sev == SeverityLevel.CRITICAL else C_AMBER_STR

        # Safely escape text to prevent ReportLab XML parser crashes
        safe_title = html.escape(iss.title)
        safe_desc = html.escape(iss.description)

        issue_content = [
            [Paragraph(
                f"<font color='{badge_color_str}'><b>[{badge}]</b></font>  "
                f"<b>{safe_title}</b>  "
                f"<font color='{C_MUTED_STR}'>(value: {iss.metric_value})</font>",
                body_style
            )],
            [Paragraph(safe_desc, ParagraphStyle("IssDesc", parent=body_style, fontSize=9, textColor=C_MUTED))],
        ]
        if iss.legal_risk:
            issue_content.append([Paragraph(f"⚖ Legal risk: {html.escape(iss.legal_risk)}",
                ParagraphStyle("Legal", parent=body_style, fontSize=9, textColor=HexColor("#993300")))])
        issue_content.append([Paragraph(f"→ Recommendation: {html.escape(iss.recommendation)}",
            ParagraphStyle("Rec", parent=body_style, fontSize=9, textColor=HexColor("#004400")))])

        it = Table(issue_content, colWidths=["100%"])
        it.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), bg),
            ("LEFTPADDING",   (0,0), (-1,-1), 14),
            ("RIGHTPADDING",  (0,0), (-1,-1), 14),
            ("TOPPADDING",    (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("LINEAFTER",     (0,0), (0,-1), 4, badge_color),
        ]))
        story.append(KeepTogether([it, Spacer(1, 8)]))

    story.append(Spacer(1, 10))

    # ── MITIGATION STRATEGIES ──
    story.append(Paragraph("Recommended Mitigation Strategies", h2_style))
    for strat in audit.mitigation_strategies[:4]:
        story.append(Paragraph(html.escape(strat.title), h3_style))
        story.append(Paragraph(
            f"<b>Stage:</b> {strat.stage} &nbsp;|&nbsp; "
            f"<b>Effort:</b> {strat.effort} &nbsp;|&nbsp; "
            f"<b>Impact:</b> {strat.impact} &nbsp;|&nbsp; "
            f"<b>Library:</b> {strat.library}",
            label_style
        ))
        story.append(Paragraph(html.escape(strat.description), body_style))
        
        # FIXED: Code snippet formatting for ReportLab
        # ReportLab ignores \n and requires <br/> for line breaks
        safe_code = html.escape(strat.code_snippet).replace('\n', '<br/>')
        story.append(Paragraph(safe_code, mono_style))
        story.append(Spacer(1, 10))

    # ── FOOTER ──
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BORDER))
    story.append(Paragraph(
        f"Generated by BiasLens v2.6.0 · {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} · "
        f"Audit ID: {audit.audit_id}",
        ParagraphStyle("Footer", parent=styles["Normal"],
            fontSize=8, textColor=C_MUTED, alignment=TA_CENTER, spaceBefore=8)
    ))

    doc.build(story)
    buf.seek(0)
    return buf


def _score_cell(label: str, value: str, sub: str, color_str: str) -> Table:
    """Helper using exact color string to prevent hexval index out-of-bounds"""
    data = [
        [Paragraph(label, ParagraphStyle("SL", fontSize=8, textColor=HexColor("#888888"), fontName="Helvetica-Bold"))],
        [Paragraph(f"<font color='{color_str}'><b>{value}</b></font>",
                   ParagraphStyle("SV", fontSize=20, alignment=TA_CENTER))],
        [Paragraph(sub, ParagraphStyle("SS", fontSize=7, textColor=HexColor("#aaaaaa"), alignment=TA_CENTER))],
    ]
    t = Table(data)
    t.setStyle(TableStyle([
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return t


def _add_table(story, data, col_widths):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (0,-1), HexColor("#f5f5ff")),
        ("BACKGROUND",   (2,0), (2,-1), HexColor("#f5f5ff")),
        ("FONTNAME",     (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",     (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("TEXTCOLOR",    (0,0), (0,-1), C_MUTED),
        ("TEXTCOLOR",    (2,0), (2,-1), C_MUTED),
        ("GRID",         (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",   (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",(0,0), (-1,-1), 7),
        ("LEFTPADDING",  (0,0), (-1,-1), 10),
    ]))
    story.append(t)


def _style_metrics_table(t, n_rows):
    style = [
        ("BACKGROUND",    (0,0), (-1,0), HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("GRID",          (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("ALIGN",         (2,0), (4,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, HexColor("#fafafa")]),
    ]
    t.setStyle(TableStyle(style))