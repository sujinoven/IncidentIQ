
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Asset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(pattern=r"^AST-\d{3}$")
    asset_name: str = Field(min_length=3)
    asset_type: Literal[
        "laptop",
        "server",
        "vpn_gateway",
        "wireless_access_point",
        "network_device",
        "storage_system",
    ]
    service_id: str = Field(pattern=r"^SVC-[A-Z]+$")
    assigned_to: str | None = None
    location: str = Field(min_length=2)
    status: Literal[
        "active",
        "maintenance",
        "retired",
    ]
    classification: Literal[
        "employee_public",
        "it_internal",
        "admin_restricted",
    ]
