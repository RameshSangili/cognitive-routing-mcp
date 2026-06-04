from __future__ import annotations

import json
import logging
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

from src.schemas import AgentResult, LoanAnalysis
from src.output_parser import STRUCTURED_OUTPUT_INSTRUCTION, parse_agent_output

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
SYSTEM_INSTRUCTION = (
    "You are a loan proposal specialist. "
    "Analyse the loan data and respond in the exact JSON format requested."
)


def _slim_context(analysis: LoanAnalysis) -> str:
    data = analysis.model_dump(exclude={"ApplicationSummary"})
    return json.dumps(data, indent=None, separators=(",", ":"))


def analyze_with_gemini(analysis: LoanAnalysis, prompt: str) -> AgentResult:
    api_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_AI_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)

    slim_json = _slim_context(analysis)
    full_prompt = f"Loan data:\n{slim_json}\n\nTask: {prompt}\n\n{STRUCTURED_OUTPUT_INSTRUCTION}"

    logger.info("Gemini call: model=%s", GEMINI_MODEL)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=512,
        ),
    )

    raw = response.text or ""
    return parse_agent_output(raw, "ProposalAgent", GEMINI_MODEL, analysis.ComplexityLevel)
