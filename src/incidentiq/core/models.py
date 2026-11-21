#Implement shared data models
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# Supported enterprise user roles
UserRole = Literal[
    "employee",
    "it_support",
    "it_admin",
]

# Supported knowledge classifications
Classification = Literal[
    "employee_public",
    "it_internal",
    "admin_restricted",
]

# Supported agent implementations
FrameworkName = Literal[
    "langgraph",
    "crewai",
    "agno",
]


class UserContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(pattern=r"^USR-\d{3}$")
    role: UserRole


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=3,
        max_length=2000,
    )

    framework: FrameworkName

    # Supplied by trusted authentication code,
    # never inferred from the user's question.
    user: UserContext


class SourceReference(BaseModel):
    document_id: str
    title: str


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str

    framework: FrameworkName

    sources: list[SourceReference] = Field(
        default_factory=list
    )

    tools_used: list[str] = Field(
        default_factory=list
    )

    success: bool = True

    error: str | None = None

#Define a common interface for all AI frameworks(Langchain,CrewAI,Agno) to implement. This interface will be used by the agent to interact with the different frameworks in a consistent manner.

from typing import Protocol


class AgentFramework(Protocol):
    """Common execution contract for all agent frameworks."""

    def run(self, request: AgentRequest) -> AgentResponse:
        """Process a request and return a validated response."""
        ...
