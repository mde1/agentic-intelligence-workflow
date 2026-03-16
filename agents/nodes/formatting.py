from __future__ import annotations
import logging
from agents.state import GraphState
from agents.config import *

logger = logging.getLogger(__name__)

def format_response(state: GraphState) -> GraphState:
    print(">>> RUNNING format_response")
    analysis = state.get("fused_analysis", {}) or {}
    forecast = state.get("forecast_analysis", {}) or {}

    findings = analysis.get("top_findings", []) or []
    overall = analysis.get("overall_assessment", "No major developments detected.")

    parts = [overall.strip()]

    if findings:
        bullets = []
        for item in findings[:5]:
            title = item.get("title", "Unnamed finding")
            confidence = item.get("confidence", "unknown").upper()
            evidence = item.get("evidence", []) or []
            source_type = item.get("source_type", []) or []
            reason = item.get("reason", "")

            evidence_text = ", ".join(str(x) for x in evidence if x)

            bullet = f"• {title} ({confidence})"
            if evidence_text:
                bullet += f" — {evidence_text}"
            if reason:
                bullet += f". {reason}"
            if source_type:
                bullet += f" - {source_type}"

            bullets.append(bullet)

        parts.append("\nKey developments:\n" + "\n".join(bullets))
    else:
        parts.append("\nNo significant findings in the latest snapshot.")

    emerging_risks = forecast.get("emerging_risks", []) or []
    forecast_confidence = forecast.get("confidence", "unknown").upper()

    if emerging_risks:
        risk_lines = []
        for item in emerging_risks[:3]:
            risk = item.get("risk", "Unnamed risk")
            rationale = item.get("rationale", "")
            watch_for = item.get("watch_for", []) or []

            line = f"• {risk}"
            if rationale:
                line += f" — {rationale}"
            if watch_for:
                line += f" Watch for: {', '.join(watch_for[:3])}"
            risk_lines.append(line)

        parts.append(
            "\nEmerging risks if current activity continues:\n"
            + "\n".join(risk_lines)
            + f"\n\nAssessment confidence: {forecast_confidence}"
        )

    final_report = "\n".join(parts).strip()

    print(">>> RETURNING final_response")
    return {"final_report": final_report}
