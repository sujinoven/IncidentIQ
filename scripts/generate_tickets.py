
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ticket_models import TicketRecord

DATA = Path("data")
OUTPUT = DATA / "operational" / "tickets.jsonl"

users = json.loads(
    (DATA / "operational" / "users.json").read_text(
        encoding="utf-8"
    )
)
services = json.loads(
    (DATA / "operational" / "services.json").read_text(
        encoding="utf-8"
    )
)

valid_users = {
    user["user_id"]
    for user in users
    if user["active"]
}

service_owners = {
    service["service_id"]: service["owner_team"]
    for service in services
}

# Service, summary, description
scenarios = [
    ("SVC-VPN", "VPN disconnects during meetings",
     "Employee reports repeated VPN disconnections during video meetings."),
    ("SVC-VPN", "VPN login rejected",
     "Employee cannot authenticate to the approved corporate VPN client."),
    ("SVC-VPN", "VPN connection timeout",
     "Employee reports that VPN connection attempts repeatedly time out."),
    ("SVC-VPN", "VPN certificate warning",
     "Employee reports a certificate warning when connecting to corporate VPN."),
    ("SVC-IDENTITY", "SSO authentication failure",
     "Employee cannot access applications through corporate single sign-on."),
    ("SVC-IDENTITY", "MFA device replacement request",
     "Employee replaced their phone and cannot complete MFA verification."),
    ("SVC-IDENTITY", "Account recovery assistance",
     "Employee cannot complete the approved account recovery workflow."),
    ("SVC-EMAIL", "Email delivery delay",
     "Employee reports that important messages arrive several hours late."),
    ("SVC-EMAIL", "External email rejected",
     "Employee receives delivery failure notices for external recipients."),
    ("SVC-EMAIL", "Email attachment blocked",
     "Employee cannot send a work document because the attachment is rejected."),
    ("SVC-PAYROLL", "Payroll portal unavailable",
     "Finance employee cannot open the payroll application."),
    ("SVC-PAYROLL", "Payroll login error",
     "Employee receives an authentication error in the payroll portal."),
    ("SVC-HR", "HR portal access denied",
     "Employee cannot access an authorized HR portal page."),
    ("SVC-HR", "HR application loading failure",
     "Employee reports that the HR portal fails to load."),
    ("SVC-ENDPOINT", "Laptop performance degradation",
     "Employee reports severe performance problems on a managed laptop."),
    ("SVC-ENDPOINT", "Endpoint security warning",
     "Employee receives an unexpected endpoint protection warning."),
    ("SVC-ENDPOINT", "Managed device update failure",
     "Employee reports that an approved security update repeatedly fails."),
    ("SVC-WIFI", "Office Wi-Fi connection failure",
     "Employee cannot connect to the approved office wireless network."),
    ("SVC-WIFI", "Office Wi-Fi instability",
     "Employee reports frequent wireless disconnections at the office."),
    ("SVC-FILES", "Shared folder access denied",
     "Employee cannot access a shared folder required for work."),
    ("SVC-FILES", "Shared files unavailable",
     "Employee reports that a shared file repository is unavailable."),
    ("SVC-DEVOPS", "CI pipeline build failure",
     "Engineering employee reports repeated CI pipeline build errors."),
    ("SVC-DEVOPS", "Deployment pipeline blocked",
     "Engineering employee reports a blocked approved deployment pipeline."),
    ("SVC-MONITOR", "Monitoring alert not delivered",
     "IT Support reports missing service monitoring notifications."),
    ("SVC-MONITOR", "Monitoring dashboard unavailable",
     "IT Support cannot access the approved monitoring dashboard."),
]

assert len(scenarios) == 25

# Historical dataset generation must not overwrite
# operational changes made by future MCP tools.
if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. "
        "Review or back up the file before regeneration."
    )

employee_ids = [
    user["user_id"]
    for user in users
    if user["role"] == "employee" and user["active"]
]

it_support_ids = [
    user["user_id"]
    for user in users
    if user["role"] == "it_support" and user["active"]
]

tickets = []

for index, (service_id, summary, description) in enumerate(
    scenarios, start=1
):
    if service_id not in service_owners:
        raise ValueError(f"Unknown service: {service_id}")

    requester_id = (
        it_support_ids[index % len(it_support_ids)]
        if service_id == "SVC-MONITOR"
        else employee_ids[(index - 1) % len(employee_ids)]
    )

    if requester_id not in valid_users:
        raise ValueError(f"Unknown requester: {requester_id}")

    status = ["open", "in_progress", "resolved", "closed"][
        (index - 1) % 4
    ]

    priority = (
        "P2" if service_id in {"SVC-PAYROLL", "SVC-MONITOR"}
        else "P3" if index % 3 == 0
        else "P4"
    )

    created_at = datetime(
        2026, 9, 1, 9, 0, tzinfo=timezone.utc
    ) + timedelta(days=index - 1)

    ticket = TicketRecord(
        ticket_id=f"TKT-{index:04d}",
        requester_id=requester_id,
        service_id=service_id,
        summary=summary,
        description=description,
        priority=priority,
        status=status,
        created_at=created_at,
        assigned_team=service_owners[service_id],
        classification="it_internal",
    )

    tickets.append(ticket.model_dump(mode="json"))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

OUTPUT.write_text(
    "\n".join(json.dumps(t) for t in tickets) + "\n",
    encoding="utf-8",
)

print(f"Generated {len(tickets)} historical tickets")
