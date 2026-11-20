
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Incident(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str = Field(pattern=r"^INC-\d{4}$")
    service_id: str
    title: str = Field(min_length=5)
    severity: Literal["P1", "P2", "P3", "P4"]
    status: Literal["open", "investigating", "resolved"]
    started_at: datetime
    resolved_at: datetime | None = None
    symptoms: list[str] = Field(min_length=1)
    root_cause: str | None = None
    resolution: str | None = None
    runbook_ids: list[str] = Field(default_factory=list)
    classification: Literal[
        "employee_public",
        "it_internal",
        "admin_restricted",
    ]

    @model_validator(mode="after")
    def validate_lifecycle(self):
        if self.status == "resolved":
            if self.resolved_at is None:
                raise ValueError(
                    "Resolved incidents require resolved_at"
                )
            if not self.root_cause or not self.resolution:
                raise ValueError(
                    "Resolved incidents require root cause "
                    "and resolution"
                )

        if self.resolved_at is not None:
            if self.resolved_at < self.started_at:
                raise ValueError(
                    "Resolution time precedes start time"
                )

        return self
