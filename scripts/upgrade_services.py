
import json
from datetime import datetime, timezone
from pathlib import Path

path = Path("data/operational/services.json")

services = json.loads(path.read_text(encoding="utf-8"))

# Synthetic snapshot timestamp, not a live health check.
snapshot_time = datetime.now(timezone.utc).isoformat()

descriptions = {
    "SVC-VPN": "Secure remote access for employees",
    "SVC-IDENTITY": "Corporate identity and single sign-on",
    "SVC-EMAIL": "Corporate email and messaging",
    "SVC-PAYROLL": "Employee payroll application",
    "SVC-HR": "Human resources portal",
    "SVC-ENDPOINT": "Managed employee devices",
    "SVC-WIFI": "Office wireless connectivity",
    "SVC-FILES": "Enterprise file sharing and storage",
    "SVC-DEVOPS": "Continuous integration and deployment",
    "SVC-MONITOR": "Infrastructure and service monitoring",
}

for service in services:
    service_id = service["service_id"]

    if service_id not in descriptions:
        raise ValueError(f"Unknown service: {service_id}")

    service.setdefault(
        "description",
        descriptions[service_id]
    )

    service.setdefault("checked_at", snapshot_time)
    service.setdefault("active_alerts", [])

# Back up the original data before changing it.
backup_path = path.with_suffix(".json.bak")

if not backup_path.exists():
    backup_path.write_text(
        path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

path.write_text(
    json.dumps(services, indent=2),
    encoding="utf-8",
)

print(f"Upgraded {len(services)} services")
print(f"Backup: {backup_path}")
