from __future__ import annotations

import json
import logging
import re

from src.schemas import AgentResult

logger = logging.getLogger(__name__)

# Injected into every agent prompt — forces the exact output shape
STRUCTURED_OUTPUT_INSTRUCTION = """
Respond with ONLY a JSON object — no prose, no markdown fences. Schema:
{
  "Status": "Approved" | "Rejected" | "ConditionalApproval",
  "ConfidenceScore": <float 0.0-1.0>,
  "RiskScore": <int 1-10>,
  "Recommendations": ["<max 3 short bullets>"],
  "NextSteps": ["<max 3 short bullets>"],
  "Notes": "<one-liner for any unusual flag, or empty string>"
}
""".strip()


def parse_agent_output(
    raw: str,
    agent_name: str,
    model: str,
    complexity: str,
) -> AgentResult:
    """Extract the structured JSON from a model response and return an AgentResult.

    Falls back gracefully if the model returns malformed output.
    """
    # Strip markdown fences if the model ignored the instruction
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Find the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        data = json.loads(cleaned)
        return AgentResult(
            AgentName=agent_name,
            Model=model,
            Status=data.get("Status", "ConditionalApproval"),
            ConfidenceScore=float(data.get("ConfidenceScore", 0.5)),
            RiskScore=int(data.get("RiskScore", 5)),
            Recommendations=data.get("Recommendations", [])[:3],
            NextSteps=data.get("NextSteps", [])[:3],
            Notes=data.get("Notes", ""),
        )
    except Exception as exc:
        logger.warning("Could not parse structured output for %s: %s", agent_name, exc)
        # Graceful fallback — preserve raw text in Notes
        return AgentResult(
            AgentName=agent_name,
            Model=model,
            Status="ConditionalApproval",
            ConfidenceScore=0.5,
            RiskScore=5,
            Recommendations=["Review raw output — parsing failed"],
            NextSteps=["Manual underwriter review required"],
            Notes=raw[:300],
        )
