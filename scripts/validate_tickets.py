
import json
from pathlib import Path

from ticket_models import TicketRecord

DATA = Path("data")

users = json.loads(
    (DATA / "operational/users.json").read_text(encoding="utf-8")
)
services = json.loads(
    (DATA / "operational/services.json").read_text(encoding="utf-8")
)

user_ids = {user["user_id"] for user in users}
service_ids = {service["service_id"] for service in services}

path = DATA / "operational/tickets.jsonl"
seen_ids = set()
count = 0

for line_number, line in enumerate(
    path.read_text(encoding="utf-8").splitlines(), start=1
):
    if not line.strip():
        continue

    ticket = TicketRecord.model_validate(json.loads(line))

    if ticket.ticket_id in seen_ids:
        raise ValueError(
            f"Duplicate ticket at line {line_number}"
        )

    seen_ids.add(ticket.ticket_id)

    if ticket.requester_id not in user_ids:
        raise ValueError(
            f"Unknown requester: {ticket.requester_id}"
        )

    if ticket.service_id not in service_ids:
        raise ValueError(
            f"Unknown service: {ticket.service_id}"
        )

    count += 1

assert count == 25, f"Expected 25 tickets, found {count}"

print(f"Validated {count} tickets")
