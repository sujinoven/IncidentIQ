
import csv
import json
from pathlib import Path

from asset_models import Asset

DATA = Path("data")
OUTPUT = DATA / "operational" / "assets.csv"

asset_specs = [
    ("AST-001", "VPN Gateway Primary", "vpn_gateway",
     "SVC-VPN", None, "Primary Data Center", "admin_restricted"),
    ("AST-002", "VPN Gateway Secondary", "vpn_gateway",
     "SVC-VPN", None, "Secondary Data Center", "admin_restricted"),
    ("AST-003", "Identity Server 01", "server",
     "SVC-IDENTITY", None, "Primary Data Center", "admin_restricted"),
    ("AST-004", "Email Transport Server", "server",
     "SVC-EMAIL", None, "Primary Data Center", "it_internal"),
    ("AST-005", "Payroll Application Server", "server",
     "SVC-PAYROLL", None, "Primary Data Center", "admin_restricted"),
    ("AST-006", "HR Application Server", "server",
     "SVC-HR", None, "Primary Data Center", "it_internal"),
    ("AST-007", "Employee Laptop 01", "laptop",
     "SVC-ENDPOINT", "USR-001", "Finance Office", "it_internal"),
    ("AST-008", "Employee Laptop 02", "laptop",
     "SVC-ENDPOINT", "USR-002", "Sales Office", "it_internal"),
    ("AST-009", "Employee Laptop 03", "laptop",
     "SVC-ENDPOINT", "USR-004", "Engineering Office", "it_internal"),
    ("AST-010", "Office Wireless AP 01", "wireless_access_point",
     "SVC-WIFI", None, "Main Office Floor 1", "it_internal"),
    ("AST-011", "Office Wireless AP 02", "wireless_access_point",
     "SVC-WIFI", None, "Main Office Floor 2", "it_internal"),
    ("AST-012", "File Storage Server", "storage_system",
     "SVC-FILES", None, "Primary Data Center", "admin_restricted"),
    ("AST-013", "CI Runner 01", "server",
     "SVC-DEVOPS", None, "Engineering Infrastructure", "it_internal"),
    ("AST-014", "Monitoring Server", "server",
     "SVC-MONITOR", None, "Primary Data Center", "it_internal"),
    ("AST-015", "Backup Storage System", "storage_system",
     "SVC-FILES", None, "Secondary Data Center", "admin_restricted"),
]

services = json.loads(
    (DATA / "operational" / "services.json").read_text(
        encoding="utf-8"
    )
)
users = json.loads(
    (DATA / "operational" / "users.json").read_text(
        encoding="utf-8"
    )
)

valid_services = {s["service_id"] for s in services}
valid_users = {u["user_id"] for u in users}

assets = []

for (
    asset_id, name, asset_type, service_id,
    assigned_to, location, classification
) in asset_specs:

    asset = Asset(
        asset_id=asset_id,
        asset_name=name,
        asset_type=asset_type,
        service_id=service_id,
        assigned_to=assigned_to,
        location=location,
        status="active",
        classification=classification,
    )

    if asset.service_id not in valid_services:
        raise ValueError(
            f"Unknown service: {asset.service_id}"
        )

    if asset.assigned_to and asset.assigned_to not in valid_users:
        raise ValueError(
            f"Unknown user: {asset.assigned_to}"
        )

    assets.append(asset)

ids = [asset.asset_id for asset in assets]
if len(ids) != len(set(ids)):
    raise ValueError("Duplicate asset IDs")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Protect existing operational data from accidental replacement.
if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Review it before regenerating."
    )

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(Asset.model_fields),
    )
    writer.writeheader()

    for asset in assets:
        writer.writerow(asset.model_dump())

print(f"Generated {len(assets)} assets")
