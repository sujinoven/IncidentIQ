
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=r"^EVAL-\d{3}$")

    category: Literal[
        "knowledge_qa",
        "incident_analysis",
        "mcp_operations",
        "security",
        "failure_handling",
    ]

    user_role: Literal[
        "employee",
        "it_support",
        "it_admin",
    ]

    query: str = Field(min_length=10)

    expected_behavior: str = Field(min_length=10)

    expected_document_ids: list[str] = Field(default_factory=list)

    expected_tool_calls: list[str] = Field(default_factory=list)

    must_not_disclose: list[str] = Field(default_factory=list)

    split: Literal["dev", "test"]
