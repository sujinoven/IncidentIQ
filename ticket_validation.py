
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Ticket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(min_length=1)
    service_id: str = Field(min_length=1)

    priority: Literal["low", "medium", "high", "critical"]
    status: Literal["open", "in_progress", "resolved", "closed"]



ticket_data = {
    "ticket_id": "TKT-0021",
    "service_id": "SVC-VPN",
    "priority": "medium",
    "status": "open"
}

ticket = Ticket.model_validate(ticket_data)

print(ticket.model_dump())
