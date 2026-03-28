from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Optional, Any
from enum import Enum


class ActionType(str, Enum):
    INSPECT_ARTIFACT = "inspect_artifact"
    FLAG_ISSUE = "flag_issue"
    REQUEST_MITIGATION = "request_mitigation"
    SET_RISK = "set_risk"
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    FINALIZE_REVIEW = "finalize_review"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GovernanceAction(BaseModel):
    action_type: ActionType
    target: Optional[str] = Field(None, max_length=100)
    severity: Optional[Severity] = None
    issue_code: Optional[str] = Field(None, max_length=50)
    note: Optional[str] = Field(None, max_length=500)


class GovernanceObservation(BaseModel):
    task_id: str
    feature_name: str
    feature_summary: str
    visible_artifacts: Dict[str, str]
    full_artifacts: Dict[str, str]
    inspected_artifacts: List[str]
    open_questions: List[str]
    flagged_issues: List[Dict]
    requested_mitigations: List[Dict]
    current_risk: Optional[str]
    available_actions: List[str]
    review_stage: str
    step_count: int
    max_steps: int
    message: str


class RewardBreakdown(BaseModel):
    total_reward: float
    inspection_reward: float = 0.0
    flagging_reward: float = 0.0
    mitigation_reward: float = 0.0
    risk_assessment_reward: float = 0.0
    decision_reward: float = 0.0
    false_positive_penalty: float = 0.0
    missed_issue_penalty: float = 0.0
    time_tax: float = 0.0
    reason: str = ""


class GraderScore(BaseModel):
    safety: float = Field(ge=0.0, le=1.0)
    compliance: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    mitigation_quality: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    num_violations: int
    expected_decision: Literal["approve", "reject", "escalate"]
    action_schema: Dict[str, Any]


# ─── Multi-Agent Models ───────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[int] = 1
    use_generated_task: bool = False
    reviewer_model: Optional[str] = None


class StepRequest(BaseModel):
    action: str


class GenerateRequest(BaseModel):
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    domain: Optional[str] = None
    num_violations: int = Field(default=2, ge=1, le=4)


class MutationRequest(BaseModel):
    base_task_id: int = 1
    num_extra_violations: int = Field(default=2, ge=1, le=3)
    difficulty_boost: bool = True


class GeneratedTask(BaseModel):
    task_id: str
    source: Literal["generated", "mutated"]
    feature_name: str
    feature_summary: str
    domain: str
    difficulty: str
    artifacts: Dict[str, str]
    ground_truth_violations: List[str]
    expected_risk: str
    expected_decision: str
    mutation_description: Optional[str] = None


class EpisodeStep(BaseModel):
    step: int
    action: Dict[str, Any]
    reward: float
    message: str


class EpisodeTranscript(BaseModel):
    task_id: str
    feature_name: str
    feature_summary: str
    artifacts: Dict[str, str]
    ground_truth_violations: List[str]
    expected_risk: str
    expected_decision: str
    steps: List[Any]
    final_flagged_issues: List[Dict]
    final_mitigations: List[Dict]
    final_risk: Optional[str]
    final_decision: Optional[str]
    total_reward: float
    total_steps: int
    reviewer_model: str


class JudgeDimension(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    examples: List[str] = []


class JudgeVerdict(BaseModel):
    task_id: str
    reviewer_model: str
    judge_model: str
    detection_accuracy: JudgeDimension
    reasoning_quality: JudgeDimension
    decision_appropriateness: JudgeDimension
    overall_score: float = Field(ge=0.0, le=1.0)
    overall_feedback: str
    grade: str


class JudgeRequest(BaseModel):
    transcript: EpisodeTranscript
