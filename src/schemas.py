from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel


class DtiCalculation(BaseModel):
    AnnualRateAssumed: float
    MonthlyRate: float
    MonthlyLoanPayment: float
    MonthlyMortgagePayment: float
    MonthlyLiabilities: float
    TotalMonthlyDebt: float
    MonthlyIncome: float
    DtiRatio: float


class Criterion(BaseModel):
    Signal: str
    Value: str
    Points: int


class Agent(BaseModel):
    Name: str
    Role: str
    Model: str


class ExecutionStep(BaseModel):
    Sequence: int
    AgentName: str
    DependsOn: str = ""


class LoanAnalysis(BaseModel):
    ComplexityScore: int
    ComplexityLevel: Literal["LOW", "MEDIUM", "HIGH"]
    ConfidenceScore: float
    ApplicationSummary: str
    DtiCalculation: DtiCalculation
    Criteria: List[Criterion]
    Agents: List[Agent]
    ExecutionOrder: List[ExecutionStep]
    EscalationRequired: bool

    def get_agent(self, name: str) -> Optional[Agent]:
        return next((a for a in self.Agents if a.Name == name), None)


# --- Structured agent output ---

class AgentResult(BaseModel):
    AgentName: str
    Model: str
    Status: Literal["Approved", "Rejected", "ConditionalApproval"]
    ConfidenceScore: float          # 0.0 – 1.0
    RiskScore: int                  # 1 (low) – 10 (high)
    Recommendations: List[str]      # ≤3 bullet points
    NextSteps: List[str]            # ≤3 bullet points
    Notes: str = ""                 # one-liner for unusual flags; empty if none
