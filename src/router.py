from __future__ import annotations

import logging
from typing import Any

from src.schemas import AgentResult, LoanAnalysis
from src.clients.gemini_client import GEMINI_MODEL, analyze_with_gemini
from src.clients.claude_client import CLAUDE_MODEL, analyze_with_claude
from src.clients.openai_client import OPENAI_MODEL, analyze_with_openai

logger = logging.getLogger(__name__)

PEGA_AGENTS = {"PegaValidation", "FinalRecommendation"}


def get_model_for_complexity(complexity_level: str) -> str:
    return GEMINI_MODEL if complexity_level.upper() == "LOW" else CLAUDE_MODEL


# Fallback chain: used when Model field is missing, empty, or unrecognised.
# Order matters — most capable first for each role.
FALLBACK_SINGLE = "openai"
FALLBACK_DEBATE = ("openai", "claude")

# Maps known tokens in the Model string → canonical participant key
_TOKEN_MAP = {
    "claude": "claude",
    "anthropic": "claude",
    "openai": "openai",
    "gpt": "openai",
    "chatgpt": "openai",
    "gemini": "gemini",
    "google": "gemini",
    "bard": "gemini",
}


# Complexity → preferred model key, in priority order.
# When a slash-list is given, the first option that matches the preferred key wins.
_COMPLEXITY_PREFERENCE: dict[str, list[str]] = {
    "LOW":    ["gemini", "openai", "claude"],
    "MEDIUM": ["claude", "openai", "gemini"],
    "HIGH":   ["claude", "openai", "gemini"],
}


def _resolve_token(text: str) -> str | None:
    """Return the canonical model key for a single token, or None if unrecognised."""
    lower = text.strip().lower()
    for token, canonical in _TOKEN_MAP.items():
        if token in lower:
            return canonical
    return None


def _select_from_options(options: list[str], complexity: str) -> str:
    """Pick the best model from a slash-list based on complexity preference.

    e.g. ["Gemini", "Claude", "OpenAI"] + LOW  → "gemini"
         ["Gemini", "Claude", "OpenAI"] + HIGH  → "claude"
    Falls back to the first recognisable option, then FALLBACK_SINGLE.
    """
    canonical_options = [_resolve_token(o) for o in options]
    preference = _COMPLEXITY_PREFERENCE.get(complexity.upper(), _COMPLEXITY_PREFERENCE["HIGH"])

    for preferred in preference:
        if preferred in canonical_options:
            logger.info(
                "Slash-list %s + complexity=%s → selected '%s'", options, complexity, preferred
            )
            return preferred

    # No preference match — take the first recognisable model in the list
    for c in canonical_options:
        if c is not None:
            logger.warning("No preference match — using first recognisable option '%s'", c)
            return c

    logger.warning("No recognisable model in %s — using fallback '%s'", options, FALLBACK_SINGLE)
    return FALLBACK_SINGLE


def _model_key(model_str: str, complexity: str = "HIGH") -> str:
    if not model_str or not model_str.strip():
        logger.warning("Empty Model field — falling back to '%s'", FALLBACK_SINGLE)
        return FALLBACK_SINGLE

    lower = model_str.lower()

    # Slash-list: "Gemini/Claude/OpenAI" — pick based on complexity
    if "/" in lower:
        options = model_str.split("/")
        return _select_from_options(options, complexity)

    known_count = sum(t in lower for t in _TOKEN_MAP)

    # Two-model debate: +, &, or vs with two known models
    if ("+" in lower or "&" in lower or "vs" in lower) and known_count >= 2:
        return "debate"
    if "gemini" in lower or "google" in lower or "bard" in lower:
        return "gemini"
    if "openai" in lower or "gpt" in lower or "chatgpt" in lower:
        return "openai"
    if "claude" in lower or "anthropic" in lower:
        return "claude"

    logger.warning("Unrecognised Model='%s' — falling back to '%s'", model_str, FALLBACK_SINGLE)
    return FALLBACK_SINGLE


def _debate_participants(model_str: str) -> tuple[str, str]:
    """Parse the two debate participants from a Model string.

    Handles all combinations: "Claude + OpenAI", "Gemini + OpenAI", "Claude + Gemini",
    aliases like "Google + GPT", separators +, &, vs.
    Falls back to FALLBACK_DEBATE if fewer than two models are recognised.
    """
    if not model_str or not model_str.strip():
        logger.warning("Empty Model field for debate — using fallback %s", FALLBACK_DEBATE)
        return FALLBACK_DEBATE

    lower = model_str.lower()
    seen: list[str] = []
    for token, canonical in _TOKEN_MAP.items():
        if token in lower and canonical not in seen:
            seen.append(canonical)

    if len(seen) >= 2:
        return seen[0], seen[1]

    logger.warning(
        "Could not parse two models from '%s' — using fallback %s", model_str, FALLBACK_DEBATE
    )
    return FALLBACK_DEBATE


def _single_call(
    participant: str,
    label: str,
    analysis: LoanAnalysis,
    prompt: str,
    prior_context: str,
    agent_name: str,
) -> AgentResult:
    """Dispatch one side of a debate to the named model."""
    if participant == "gemini":
        return analyze_with_gemini(analysis, f"{label}: {prompt}")
    if participant == "openai":
        return analyze_with_openai(analysis, f"{label}: {prompt}", prior_context, agent_name)
    return analyze_with_claude(
        analysis, f"{label}: {prompt}", agent_name=agent_name, prior_context=prior_context
    )


def _format_prior(result: AgentResult) -> str:
    """Format the dependency's result as context for the next agent.

    Passes Status, RiskScore, and Recommendations — enough for a critic or
    debate agent to act on without re-sending the full raw text.
    """
    recs = "; ".join(result.Recommendations)
    steps = "; ".join(result.NextSteps)
    return (
        f"[{result.AgentName}] Status={result.Status} | "
        f"Confidence={result.ConfidenceScore} | RiskScore={result.RiskScore}\n"
        f"Recommendations: {recs}\n"
        f"NextSteps: {steps}\n"
        f"Notes: {result.Notes}"
    )


def _call_agent(
    agent_name: str,
    model_str: str,
    analysis: LoanAnalysis,
    task_prompt: str,
    prior_context: str,
) -> AgentResult:
    agent_def = analysis.get_agent(agent_name)
    role_desc = agent_def.Role if agent_def else agent_name
    prompt = f"{role_desc}\n\n{task_prompt}"
    key = _model_key(model_str, complexity=analysis.ComplexityLevel)

    logger.info("Executing agent=%s depends_on_context_len=%d", agent_name, len(prior_context))

    if key == "gemini":
        return analyze_with_gemini(analysis, prompt)

    if key == "openai":
        return analyze_with_openai(analysis, prompt, prior_context, agent_name)

    if key == "debate":
        p1, p2 = _debate_participants(model_str)
        out_a = _single_call(p1, f"Perspective A ({p1})", analysis, prompt, prior_context, agent_name)
        out_b = _single_call(p2, f"Perspective B ({p2})", analysis, prompt, prior_context, agent_name)

        synthesis_prompt = (
            f"You are synthesising a structured debate between {p1.upper()} and {p2.upper()}.\n"
            f"{p1.upper()}: Status={out_a.Status}, Risk={out_a.RiskScore}, Recs={out_a.Recommendations}\n"
            f"{p2.upper()}: Status={out_b.Status}, Risk={out_b.RiskScore}, Recs={out_b.Recommendations}\n"
            f"Produce one unified recommendation resolving any disagreements."
        )
        # Claude always synthesises — enforces the AgentResult schema regardless of debate pair
        result = analyze_with_claude(analysis, synthesis_prompt, agent_name="DebateAgent")
        result.AgentName = agent_name
        result.Model = f"{out_a.Model} + {out_b.Model}"
        return result

    return analyze_with_claude(
        analysis, prompt, agent_name=agent_name, prior_context=prior_context
    )


def route_and_analyze(analysis: LoanAnalysis, task_prompt: str) -> dict[str, Any]:
    """Execute agents in ExecutionOrder sequence.

    Routing logic:
      1. ComplexityLevel decides the primary model (LOW → Gemini, MEDIUM/HIGH → Claude).
      2. Each agent's Model field in the Agents array overrides for that specific agent.
      3. Each agent's DependsOn field determines WHICH prior agent's result is passed
         as context — not just the immediate predecessor.
      4. Pega system agents are acknowledged but not called.
    """
    primary_model = get_model_for_complexity(analysis.ComplexityLevel)
    steps = sorted(analysis.ExecutionOrder, key=lambda s: s.Sequence)

    # Keyed store so any agent can look up any predecessor by name
    results_by_agent: dict[str, AgentResult] = {}
    agent_results: list[dict[str, Any]] = []
    last_ai_result: AgentResult | None = None

    for step in steps:
        agent_name = step.AgentName

        if agent_name in PEGA_AGENTS:
            logger.info("Skipping Pega agent: %s", agent_name)
            continue

        # Resolve context from the named dependency, not just the last agent
        prior_context = ""
        if step.DependsOn and step.DependsOn in results_by_agent:
            prior_context = _format_prior(results_by_agent[step.DependsOn])
            logger.info(
                "Agent %s using output of %s as context", agent_name, step.DependsOn
            )
        elif step.DependsOn:
            logger.warning(
                "Agent %s depends on %s but that result is not available",
                agent_name, step.DependsOn,
            )

        agent_def = analysis.get_agent(agent_name)
        if agent_def:
            model_str = agent_def.Model
        else:
            model_str = ""  # triggers fallback inside _model_key
            logger.warning("No Agent definition found for '%s' — will use fallback model", agent_name)

        result = _call_agent(
            agent_name=agent_name,
            model_str=model_str,
            analysis=analysis,
            task_prompt=task_prompt,
            prior_context=prior_context,
        )

        results_by_agent[agent_name] = result
        agent_results.append(result.model_dump())
        last_ai_result = result

    return {
        "routed_to": primary_model,
        "complexity": analysis.ComplexityLevel,
        "complexity_score": analysis.ComplexityScore,
        "confidence_score": analysis.ConfidenceScore,
        "escalation_required": analysis.EscalationRequired,
        "agent_results": agent_results,
        "final_agent": last_ai_result.AgentName if last_ai_result else "",
        "final_status": last_ai_result.Status if last_ai_result else "N/A",
    }
