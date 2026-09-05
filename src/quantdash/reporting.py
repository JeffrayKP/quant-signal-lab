from __future__ import annotations

import html
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY = colors.HexColor("#0A0E14")
SURFACE = colors.HexColor("#11161D")
GOLD = colors.HexColor("#BD9650")
TEXT = colors.HexColor("#F1EEE7")
MUTED = colors.HexColor("#AEB8C2")
LINE = colors.HexColor("#39434E")


def _safe(value: Any) -> str:
    return html.escape(str(value))


def _pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.{digits}%}" if np.isfinite(number) else "N/A"


def _page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
    canvas.setStrokeColor(GOLD)
    canvas.setLineWidth(1.2)
    canvas.line(0.55 * inch, letter[1] - 0.45 * inch, letter[0] - 0.55 * inch, letter[1] - 0.45 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(0.6 * inch, 0.38 * inch, "QUANT SIGNAL LAB  |  RESEARCH AND EDUCATION ONLY")
    canvas.drawRightString(letter[0] - 0.6 * inch, 0.38 * inch, f"PAGE {doc.page}")
    canvas.restoreState()


def _table(data: list[list[Any]], widths: list[float]) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), GOLD),
                ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), SURFACE),
                ("TEXTCOLOR", (0, 1), (-1, -1), TEXT),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_research_brief(
    leader: pd.Series,
    evidence: dict[str, str],
    regime: dict[str, str],
    holdings: pd.DataFrame,
    comparison: pd.DataFrame,
    top_signals: pd.DataFrame,
    horizon: int,
    target_return: float,
) -> bytes:
    """Create a compact, downloadable two-page investment-research brief."""
    output = BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.6 * inch,
        title=f"{leader.get('Ticker', 'Stock')} Research Brief",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleDark", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=25, textColor=TEXT, alignment=TA_LEFT, spaceAfter=8
    )
    kicker = ParagraphStyle("Kicker", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=GOLD, spaceBefore=6, spaceAfter=5)
    heading = ParagraphStyle(
        "Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=TEXT, spaceBefore=8, spaceAfter=6
    )
    body = ParagraphStyle("BodyDark", parent=styles["BodyText"], fontName="Helvetica", fontSize=9, leading=13, textColor=MUTED, spaceAfter=6)
    metric = ParagraphStyle("Metric", parent=body, fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=TEXT, alignment=TA_CENTER)
    ticker = _safe(leader.get("Ticker", "N/A"))
    story = [
        Paragraph("QUANTITATIVE INVESTMENT RESEARCH", kicker),
        Paragraph(f"{ticker} Decision Brief", title),
        Paragraph(
            f"A model-governed view of the highest-ranked {horizon}-trading-day research signal, "
            "portfolio context, and historical risk. This document is not personalized "
            "investment advice.",
            body,
        ),
    ]
    metric_data = [
        [
            Paragraph(f"Expected return<br/><font color='#BD9650'>{_pct(leader.get('Expected Return'))}</font>", metric),
            Paragraph(f"Probability positive<br/><font color='#BD9650'>{_pct(leader.get('Probability Positive'), 0)}</font>", metric),
            Paragraph(f"Standalone risk<br/><font color='#BD9650'>{_pct(leader.get('Risk Score'), 0)}</font>", metric),
            Paragraph(f"Target probability<br/><font color='#BD9650'>{_pct(leader.get('Probability Target'), 0)}</font>", metric),
        ]
    ]
    metrics_table = Table(metric_data, colWidths=[1.7 * inch] * 4)
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.extend(
        [
            metrics_table,
            Spacer(1, 9),
            Paragraph("EVIDENCE GATE", kicker),
            Paragraph(f"<b>{_safe(evidence.get('label', 'N/A'))}</b>. {_safe(evidence.get('detail', ''))}", body),
            Paragraph("MARKET REGIME", kicker),
            Paragraph(f"<b>{_safe(regime.get('label', 'N/A'))}</b>. {_safe(regime.get('detail', ''))}", body),
            Paragraph("Interpretation", heading),
            Paragraph(
                f"The point forecast is {_pct(leader.get('Expected Return'))}, with an uncertainty "
                f"band from {_pct(leader.get('Return Low'))} to {_pct(leader.get('Return High'))}. "
                f"The estimated probability of exceeding the selected {_pct(target_return, 0)} "
                f"target is {_pct(leader.get('Probability Target'), 0)}. Forecast uncertainty and "
                "model evidence must be considered before the ranking itself.",
                body,
            ),
        ]
    )

    if not top_signals.empty:
        signal_rows = [["Ticker", "Expected", "Prob. positive", "Risk", "Reliability"]]
        for _, row in top_signals.head(8).iterrows():
            signal_rows.append(
                [
                    _safe(row.get("Ticker")),
                    _pct(row.get("Expected Return")),
                    _pct(row.get("Probability Positive"), 0),
                    _pct(row.get("Risk Score"), 0),
                    _pct(row.get("Reliability"), 0),
                ]
            )
        story.extend([Paragraph("Top research signals", heading), _table(signal_rows, [1.1 * inch, 1.25 * inch, 1.5 * inch, 1.15 * inch, 1.25 * inch])])

    story.append(PageBreak())
    story.extend(
        [
            Paragraph("PORTFOLIO AND RISK CONTEXT", kicker),
            Paragraph("Diversified allocation", title),
            Paragraph(
                "The optimized portfolio is built separately from the standalone ranking. It "
                "considers covariance, downside risk, sector exposure, beta, and position limits.",
                body,
            ),
        ]
    )
    if not holdings.empty:
        holding_rows = [["Ticker", "Sector", "Weight", "Expected", "Role"]]
        for _, row in holdings.iterrows():
            holding_rows.append(
                [_safe(row.get("Ticker")), _safe(row.get("Sector")), _pct(row.get("Weight")), _pct(row.get("Expected Return")), _safe(row.get("Role"))]
            )
        story.extend([_table(holding_rows, [0.85 * inch, 1.35 * inch, 0.95 * inch, 1.1 * inch, 1.65 * inch]), Spacer(1, 10)])
    if not comparison.empty:
        comparison_rows = [["Strategy", "Hist. annual", "Volatility", "Max drawdown", "Daily VaR"]]
        for _, row in comparison.iterrows():
            comparison_rows.append(
                [
                    _safe(row.get("Strategy")),
                    _pct(row.get("Historical Annual Return")),
                    _pct(row.get("Annual Volatility")),
                    _pct(row.get("Maximum Drawdown")),
                    _pct(row.get("Daily VaR 95%")),
                ]
            )
        story.extend(
            [
                Paragraph("Historical risk comparison", heading),
                Paragraph(
                    "This uses current selections over past returns. It is descriptive and selection-biased, not an out-of-sample performance claim.", body
                ),
                _table(comparison_rows, [1.8 * inch, 1.15 * inch, 1.1 * inch, 1.2 * inch, 1.05 * inch]),
            ]
        )
    story.extend(
        [
            Spacer(1, 12),
            Paragraph("Limitations and controls", heading),
            Paragraph(
                "Market data may be delayed, incomplete, or wrong. The universe has survivorship "
                "and selection bias. Probability calibration can decay after regime changes. "
                "Historical covariance and stress estimates are not guarantees. Taxes, market "
                "impact, borrow constraints, and individual circumstances are not modeled. No "
                "output is a recommendation or promise of return.",
                body,
            ),
        ]
    )
    doc.build(story, onFirstPage=_page, onLaterPages=_page)
    return output.getvalue()
