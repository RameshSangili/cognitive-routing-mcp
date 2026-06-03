from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP

from src.schemas import LoanAnalysis
from src.router import get_model_for_complexity, route_and_analyze
from src.clients.gemini_client import GEMINI_MODEL
from src.clients.claude_client import CLAUDE_MODEL

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP("cognitive-routing-mcp")


@mcp.tool()
def route_loan_application(
    analysis_json: str,
    task_prompt: str = "Generate a detailed loan proposal",
) -> dict:
    """Route a loan application through its full agent execution chain.

    Top-level model selection based on ComplexityLevel:
    - LOW      → Gemini (gemini-2.0-flash) — single ProposalAgent
    - MEDIUM   → Claude (claude-sonnet-4-6) + optional OpenAI DebateAgent
    - HIGH     → Claude (claude-sonnet-4-6) + full critic/debate/governance chain

    Each agent in ExecutionOrder runs sequentially; downstream agents receive
    accumulated context from all prior agents. Pega system agents
    (PegaValidation, FinalRecommendation) are acknowledged but not called.

    Args:
        analysis_json: JSON string containing the preliminary LoanAnalysis produced
                       by the orchestrating Claude agent.
        task_prompt:   The task instruction forwarded to each agent in the chain.

    Returns:
        A dict with keys: routed_to, complexity, complexity_score, confidence_score,
        escalation_required, agent_results (list), final_result (str).
    """
    try:
        data = json.loads(analysis_json)
        analysis = LoanAnalysis(**data)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to parse analysis_json: %s", exc)
        return {"error": f"Invalid analysis JSON: {exc}"}

    try:
        result = route_and_analyze(analysis, task_prompt)
    except Exception as exc:
        logger.error("Error during routing/analysis: %s", exc)
        return {"error": str(exc)}

    return result


@mcp.tool()
def get_routing_decision(analysis_json: str) -> dict:
    """Return the routing decision for a loan application without executing the analysis.

    Useful for previewing which model will be used before incurring API costs.

    Args:
        analysis_json: JSON string containing the preliminary LoanAnalysis.

    Returns:
        A dict with keys: complexity, model, reason.
    """
    try:
        data = json.loads(analysis_json)
        analysis = LoanAnalysis(**data)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to parse analysis_json: %s", exc)
        return {"error": f"Invalid analysis JSON: {exc}"}

    complexity = analysis.ComplexityLevel
    model = get_model_for_complexity(complexity)

    if complexity == "LOW":
        reason = (
            f"Complexity level is LOW (score={analysis.ComplexityScore}). "
            f"Routing to {GEMINI_MODEL} for efficient, cost-effective processing."
        )
    else:
        reason = (
            f"Complexity level is {complexity} (score={analysis.ComplexityScore}). "
            f"Routing to {CLAUDE_MODEL} for advanced reasoning and nuanced analysis."
        )

    return {
        "complexity": complexity,
        "model": model,
        "reason": reason,
    }


@mcp.tool()
def health_check() -> dict:
    """Return the health status of the cognitive-routing-mcp service."""
    return {"status": "ok", "service": "cognitive-routing-mcp"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info("Starting cognitive-routing-mcp on port %d", port)
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
