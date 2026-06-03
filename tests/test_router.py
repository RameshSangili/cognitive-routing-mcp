from __future__ import annotations

import pytest

from src.schemas import LoanAnalysis
from src.router import get_model_for_complexity, _model_key, PEGA_AGENTS
from src.clients.gemini_client import GEMINI_MODEL
from src.clients.claude_client import CLAUDE_MODEL
from src.clients.openai_client import OPENAI_MODEL

LOW_SAMPLE = {
    "ComplexityScore": 3,
    "ComplexityLevel": "LOW",
    "ConfidenceScore": 0.85,
    "ApplicationSummary": "<p>Low complexity test</p>",
    "DtiCalculation": {
        "AnnualRateAssumed": 0.06,
        "MonthlyRate": 0.005,
        "MonthlyLoanPayment": 4833.2,
        "MonthlyMortgagePayment": 0.0,
        "MonthlyLiabilities": 0.0,
        "TotalMonthlyDebt": 4833.2,
        "MonthlyIncome": 7083.33,
        "DtiRatio": 0.29,
    },
    "Criteria": [{"Signal": "LoanAmount", "Value": "250000", "Points": 1}],
    "Agents": [{"Name": "ProposalAgent", "Role": "Generate proposal", "Model": "Gemini"}],
    "ExecutionOrder": [
        {"Sequence": 1, "AgentName": "ProposalAgent", "DependsOn": ""},
        {"Sequence": 2, "AgentName": "PegaValidation", "DependsOn": "ProposalAgent"},
        {"Sequence": 3, "AgentName": "FinalRecommendation", "DependsOn": "PegaValidation"},
    ],
    "EscalationRequired": False,
}

MEDIUM_SAMPLE = {
    "ComplexityScore": 4,
    "ComplexityLevel": "MEDIUM",
    "ConfidenceScore": 0.85,
    "ApplicationSummary": "<p>Medium complexity test</p>",
    "DtiCalculation": {
        "AnnualRateAssumed": 0.06,
        "MonthlyRate": 0.005,
        "MonthlyLoanPayment": 4851.62,
        "MonthlyMortgagePayment": 0.0,
        "MonthlyLiabilities": 0.0,
        "TotalMonthlyDebt": 4851.62,
        "MonthlyIncome": 6250.0,
        "DtiRatio": 0.78,
    },
    "Criteria": [{"Signal": "DtiRatio", "Value": "0.78", "Points": 3}],
    "Agents": [
        {"Name": "ProposalAgent", "Role": "Generate proposal", "Model": "Claude"},
        {"Name": "CriticAgent", "Role": "Critique proposal", "Model": "Claude"},
        {"Name": "DebateAgent", "Role": "Structured debate", "Model": "Claude + OpenAI"},
    ],
    "ExecutionOrder": [
        {"Sequence": 1, "AgentName": "ProposalAgent", "DependsOn": ""},
        {"Sequence": 2, "AgentName": "CriticAgent", "DependsOn": "ProposalAgent"},
        {"Sequence": 3, "AgentName": "DebateAgent", "DependsOn": "CriticAgent"},
        {"Sequence": 4, "AgentName": "PegaValidation", "DependsOn": "DebateAgent"},
        {"Sequence": 5, "AgentName": "FinalRecommendation", "DependsOn": "PegaValidation"},
    ],
    "EscalationRequired": False,
}


# --- get_model_for_complexity ---

def test_low_routes_to_gemini():
    assert get_model_for_complexity("LOW") == GEMINI_MODEL

def test_medium_routes_to_claude():
    assert get_model_for_complexity("MEDIUM") == CLAUDE_MODEL

def test_high_routes_to_claude():
    assert get_model_for_complexity("HIGH") == CLAUDE_MODEL

def test_complexity_case_insensitive():
    assert get_model_for_complexity("low") == GEMINI_MODEL
    assert get_model_for_complexity("high") == CLAUDE_MODEL


# --- _model_key ---

def test_model_key_gemini():
    assert _model_key("Gemini") == "gemini"

def test_model_key_claude():
    assert _model_key("Claude") == "claude"

def test_model_key_openai():
    assert _model_key("OpenAI") == "openai"

def test_model_key_debate():
    assert _model_key("Claude + OpenAI") == "debate"
    assert _model_key("claude + openai") == "debate"

def test_model_key_unknown_defaults_to_claude():
    assert _model_key("SomeFutureModel") == "claude"


# --- Schema parsing ---

def test_parse_low_sample():
    analysis = LoanAnalysis(**LOW_SAMPLE)
    assert analysis.ComplexityLevel == "LOW"
    assert analysis.DtiCalculation.DtiRatio == pytest.approx(0.29)
    assert len(analysis.Agents) == 1
    assert analysis.Agents[0].Model == "Gemini"

def test_parse_medium_sample():
    analysis = LoanAnalysis(**MEDIUM_SAMPLE)
    assert analysis.ComplexityLevel == "MEDIUM"
    assert analysis.DtiCalculation.DtiRatio == pytest.approx(0.78)
    assert len(analysis.Agents) == 3
    assert analysis.Agents[2].Model == "Claude + OpenAI"

def test_get_agent_helper():
    analysis = LoanAnalysis(**MEDIUM_SAMPLE)
    agent = analysis.get_agent("DebateAgent")
    assert agent is not None
    assert agent.Model == "Claude + OpenAI"

def test_get_agent_missing_returns_none():
    analysis = LoanAnalysis(**LOW_SAMPLE)
    assert analysis.get_agent("NonExistentAgent") is None

def test_invalid_complexity_raises():
    bad = {**LOW_SAMPLE, "ComplexityLevel": "VERY_HIGH"}
    with pytest.raises(Exception):
        LoanAnalysis(**bad)


# --- Pega agents ---

def test_pega_agents_not_ai_dispatched():
    assert "PegaValidation" in PEGA_AGENTS
    assert "FinalRecommendation" in PEGA_AGENTS
    assert "ProposalAgent" not in PEGA_AGENTS

def test_execution_order_sorted():
    analysis = LoanAnalysis(**MEDIUM_SAMPLE)
    sequences = [step.Sequence for step in analysis.ExecutionOrder]
    assert sequences == sorted(sequences)
