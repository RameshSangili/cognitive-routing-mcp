from __future__ import annotations

import json
import logging
import os

from openai import OpenAI
from dotenv import load_dotenv

from src.schemas import AgentResult, LoanAnalysis
from src.output_parser import STRUCTURED_OUTPUT_INSTRUCTION, parse_agent_output

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a credit risk analyst. "
    "Analyse the loan data and respond in the exact JSON format requested."
)
_MAX_TOKENS = 512


def _slim_context(analysis: LoanAnalysis) -> str:
    data = analysis.model_dump(exclude={"ApplicationSummary"})
    return json.dumps(data, indent=None, separators=(",", ":"))


def analyze_with_openai(
    analysis: LoanAnalysis,
    prompt: str,
    prior_context: str = "",
    agent_name: str = "DebateAgent",
) -> AgentResult:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)

    slim_json = _slim_context(analysis)
    trimmed_prior = prior_context[-600:] if len(prior_context) > 600 else prior_context
    context_block = f"\n\nPrior output:\n{trimmed_prior}" if trimmed_prior else ""

    user_content = (
        f"Loan data:\n{slim_json}{context_block}\n\n"
        f"Task: {prompt}\n\n{STRUCTURED_OUTPUT_INSTRUCTION}"
    )

    logger.info("OpenAI call: model=%s agent=%s", OPENAI_MODEL, agent_name)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=_MAX_TOKENS,
        response_format={"type": "json_object"},  # native JSON mode
    )

    raw = response.choices[0].message.content or ""
    logger.info("OpenAI response: total_tokens=%d", response.usage.total_tokens if response.usage else 0)

    return parse_agent_output(raw, agent_name, OPENAI_MODEL, analysis.ComplexityLevel)
