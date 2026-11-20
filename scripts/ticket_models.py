
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Priority = Literal["P1", "P2", "P3", "P4"]
TicketStatus = Literal[
    "open", "in_progress", "resolved", "closed"
]
Classification = Literal[
    "employee_public", "it_internal", "admin_restricted"
]


class TicketCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_id: str = Field(pattern=r"^SVC-[A-Z]+$")
    summary: str = Field(min_length=10, max_length=160)
    description: str = Field(min_length=20, max_length=2000)
    priority: Priority


class TicketRecord(TicketCreateRequest):
    ticket_id: str = Field(pattern=r"^TKT-\d{4}$")
    requester_id: str = Field(pattern=r"^USR-\d{3}$")
    status: TicketStatus
    created_at: datetime
    assigned_team: str = Field(min_length=3)
    classification: Classification
