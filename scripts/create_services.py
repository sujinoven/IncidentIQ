
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Service(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_id: str = Field(pattern=r"^SVC-[A-Z]+$")
    service_name: str = Field(min_length=3)
    owner_team: str = Field(min_length=3)
    status: Literal["operational", "degraded", "outage"]


service_definitions = [
    ("SVC-VPN", "Corporate VPN", "Network Operations"),
    ("SVC-IDENTITY", "Identity & SSO", "Identity Operations"),
    ("SVC-EMAIL", "Corporate Email", "Messaging Operations"),
    ("SVC-PAYROLL", "Payroll Application", "Business Applications"),
    ("SVC-HR", "HR Portal", "Business Applications"),
    ("SVC-ENDPOINT", "Employee Devices", "Endpoint Support"),
    ("SVC-WIFI", "Office Wi-Fi", "Network Operations"),
    ("SVC-FILES", "File Sharing", "Infrastructure Operations"),
    ("SVC-DEVOPS", "CI/CD Platform", "Platform Engineering"),
    ("SVC-MONITOR", "Monitoring Platform", "Site Reliability Engineering"),
]


services = [
    Service(
        service_id=service_id,
        service_name=service_name,
        owner_team=owner_team,
        status="operational",
    )
    for service_id, service_name, owner_team in service_definitions
]

# Validate uniqueness across records.
ids = [service.service_id for service in services]
if len(ids) != len(set(ids)):
    raise ValueError("Duplicate service IDs detected")

output_path = Path("data/operational/services.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

output_path.write_text(
    json.dumps(
        [service.model_dump() for service in services],
        indent=2,
    ),
    encoding="utf-8",
)

print(f"Created {len(services)} services: {output_path}")
