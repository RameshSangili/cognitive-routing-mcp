from __future__ import annotations

import json
import logging
import os

import anthropic
from dotenv import load_dotenv

from src.schemas import AgentResult, LoanAnalysis
from src.output_parser import STRUCTURED_OUTPUT_INSTRUCTION, parse_agent_output

load_dotenv()

logger = logging.getLogger(__name__)

CLAUDE_MODEL_FULL = "claude-sonnet-4-6"
CLAUDE_MODEL_LITE = "claude-haiku-4-5-20251001"
CLAUDE_MODEL = CLAUDE_MODEL_FULL  # exported for routing display

_LITE_AGENTS = {"CriticAgent", "GovernanceAgent"}

SYSTEM_PROMPT = (
    "You are a senior loan underwriting specialist. "
    "Analyse the loan data and respond in the exact JSON format requested."
)

_MAX_TOKENS: dict[str, int] = {
    "ProposalAgent": 512,
    "CriticAgent": 384,
    "DebateAgent": 512,
    "GovernanceAgent": 384,
    "FraudDetectionAgent": 384,
}
_DEFAULT_MAX_TOKENS = 512


def _slim_context(analysis: LoanAnalysis) -> str:
    data = analysis.model_dump(exclude={"ApplicationSummary"})
    return json.dumps(data, indent=None, separators=(",", ":"))


def analyze_with_claude(
    analysis: LoanAnalysis,
    prompt: str,
    agent_name: str = "",
    prior_context: str = "",
) -> AgentResult:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    client = anthropic.Anthropic(api_key=api_key)

    model = CLAUDE_MODEL_LITE if agent_name in _LITE_AGENTS else CLAUDE_MODEL_FULL
    max_tokens = _MAX_TOKENS.get(agent_name, _DEFAULT_MAX_TOKENS)

    slim_json = _slim_context(analysis)
    system_blocks = [
        {
            "type": "text",
            "text": f"{SYSTEM_PROMPT}\n\nLoan data:\n{slim_json}",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    trimmed_prior = prior_context[-600:] if len(prior_context) > 600 else prior_context
    user_content = f"{prompt}\n\n{STRUCTURED_OUTPUT_INSTRUCTION}"
    if trimmed_prior:
        user_content = f"{prompt}\n\nPrior output:\n{trimmed_prior}\n\n{STRUCTURED_OUTPUT_INSTRUCTION}"

    logger.info("Claude call: agent=%s model=%s max_tokens=%d", agent_name, model, max_tokens)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_blocks,
        messages=[{"role": "user", "content": user_content}],
    )

    raw = response.content[0].text if response.content else ""
    cache_read = getattr(response.usage, "cache_read_input_tokens", 0)
    logger.info(
        "Claude response: in=%d out=%d cache_read=%d",
        response.usage.input_tokens,
        response.usage.output_tokens,
        cache_read,
    )

    return parse_agent_output(raw, agent_name, model, analysis.ComplexityLevel)
