
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from incident_models import Incident

DATA = Path("data")
OUTPUT = DATA / "operational" / "incidents.jsonl"

# service, title, symptoms, root cause, resolution, runbook
scenarios = [
    (
        "SVC-VPN",
        "VPN connection timeout",
        ["VPN client times out", "Remote connection unavailable"],
        "VPN gateway connectivity degradation",
        "Network Operations restored gateway connectivity",
        "RB-VPN-001",
    ),
    (
        "SVC-VPN",
        "VPN authentication failures",
        ["VPN login rejected", "Repeated authentication errors"],
        "Identity service authentication disruption",
        "Identity Operations restored authentication service",
        "RB-VPN-002",
    ),
    (
        "SVC-VPN",
        "VPN intermittent disconnections",
        ["VPN disconnects repeatedly"],
        "Unstable gateway network path",
        "Network Operations corrected the affected network path",
        "RB-VPN-001",
    ),
    (
        "SVC-VPN",
        "VPN gateway availability issue",
        ["Multiple users cannot connect to VPN"],
        "VPN gateway service failure",
        "Network Operations restored the gateway service",
        "RB-VPN-001",
    ),
    (
        "SVC-IDENTITY",
        "SSO authentication outage",
        ["Multiple users cannot sign in"],
        "Identity provider service disruption",
        "Identity Operations restored the identity provider",
        "RB-ID-004",
    ),
    (
        "SVC-IDENTITY",
        "MFA enrollment failures",
        ["Users cannot complete MFA enrollment"],
        "MFA enrollment service degradation",
        "Identity Operations restored enrollment processing",
        "RB-ID-005",
    ),
    (
        "SVC-IDENTITY",
        "SSO session validation errors",
        ["SSO sessions rejected unexpectedly"],
        "Identity provider session validation issue",
        "Identity Operations corrected session configuration",
        "RB-ID-004",
    ),
    (
        "SVC-IDENTITY",
        "Account recovery disruption",
        ["Approved account recovery requests fail"],
        "Identity recovery workflow outage",
        "Identity Operations restored recovery workflow",
        "RB-ID-005",
    ),
    (
        "SVC-EMAIL",
        "Email delivery delays",
        ["Messages delayed across departments"],
        "Mail transport queue congestion",
        "Messaging Operations restored normal queue processing",
        "RB-EMAIL-006",
    ),
    (
        "SVC-EMAIL",
        "Email routing failures",
        ["External messages rejected"],
        "Mail routing configuration error",
        "Messaging Operations corrected approved routing settings",
        "RB-EMAIL-006",
    ),
    (
        "SVC-EMAIL",
        "Corporate email service degradation",
        ["Users cannot send messages"],
        "Mail transport service failure",
        "Messaging Operations restored transport services",
        "RB-EMAIL-006",
    ),
    (
        "SVC-PAYROLL",
        "Payroll portal outage",
        ["Payroll portal unavailable"],
        "Payroll application service failure",
        "Business Applications restored the application",
        "RB-PAY-007",
    ),
    (
        "SVC-PAYROLL",
        "Payroll application errors",
        ["Payroll users receive server errors"],
        "Application dependency failure",
        "Business Applications restored the affected dependency",
        "RB-PAY-007",
    ),
    (
        "SVC-PAYROLL",
        "Payroll login disruption",
        ["Payroll application login fails"],
        "Payroll authentication integration failure",
        "Business Applications restored the identity integration",
        "RB-PAY-007",
    ),
    (
        "SVC-HR",
        "HR portal authentication errors",
        ["HR portal access denied"],
        "HR portal identity integration issue",
        "Business Applications restored the integration",
        "RB-ID-004",
    ),
    (
        "SVC-HR",
        "HR portal unavailable",
        ["Employees cannot access HR portal"],
        "HR application service failure",
        "Business Applications restored the HR application",
        None,
    ),
    (
        "SVC-ENDPOINT",
        "Endpoint malware alert",
        ["Security software detects suspicious activity"],
        "Malicious file detected on a managed device",
        "Security Operations contained and remediated the device",
        "RB-END-009",
    ),
    (
        "SVC-ENDPOINT",
        "Endpoint security agent failure",
        ["Endpoint protection agent reports unhealthy status"],
        "Endpoint security agent malfunction",
        "Endpoint Support restored approved protection software",
        "RB-END-009",
    ),
    (
        "SVC-ENDPOINT",
        "Managed laptop performance degradation",
        ["Laptop performance is severely degraded"],
        "Resource exhaustion on managed device",
        "Endpoint Support resolved the resource issue",
        None,
    ),
    (
        "SVC-WIFI",
        "Office Wi-Fi outage",
        ["Employees cannot connect to office Wi-Fi"],
        "Wireless access point controller disruption",
        "Network Operations restored wireless service",
        "RB-WIFI-008",
    ),
    (
        "SVC-WIFI",
        "Wi-Fi authentication failures",
        ["Corporate Wi-Fi login rejected"],
        "Wireless authentication integration failure",
        "Network Operations restored authentication connectivity",
        "RB-WIFI-008",
    ),
    (
        "SVC-WIFI",
        "Intermittent office Wi-Fi",
        ["Wireless connections drop frequently"],
        "Access point instability",
        "Network Operations restored affected access points",
        "RB-WIFI-008",
    ),
    (
        "SVC-FILES",
        "Shared folder access failures",
        ["Users receive access denied errors"],
        "File share permission configuration issue",
        "Infrastructure Operations restored approved permissions",
        "RB-FILE-010",
    ),
    (
        "SVC-FILES",
        "File sharing service outage",
        ["Shared folders unavailable"],
        "File server service failure",
        "Infrastructure Operations restored file sharing service",
        "RB-FILE-010",
    ),
    (
        "SVC-FILES",
        "Backup job failures",
        ["Backup monitoring reports failed jobs"],
        "Backup storage availability issue",
        "Infrastructure Operations restored backup processing",
        "RB-BKP-012",
    ),
    (
        "SVC-DEVOPS",
        "CI pipeline build failures",
        ["Multiple build jobs fail"],
        "CI runner configuration issue",
        "Platform Engineering restored runner configuration",
        "RB-CICD-011",
    ),
    (
        "SVC-DEVOPS",
        "Deployment pipeline disruption",
        ["Deployment jobs cannot complete"],
        "Pipeline dependency service failure",
        "Platform Engineering restored the dependency",
        "RB-CICD-011",
    ),
    (
        "SVC-MONITOR",
        "Monitoring alert delivery failure",
        ["Service alerts not delivered"],
        "Monitoring notification service disruption",
        "Site Reliability Engineering restored notifications",
        None,
    ),
    (
        "SVC-MONITOR",
        "Monitoring dashboard outage",
        ["Monitoring dashboards unavailable"],
        "Monitoring frontend service failure",
        "Site Reliability Engineering restored dashboard service",
        None,
    ),
]

assert len(scenarios) == 29

# Check the approved service and runbook references.
services = json.loads(
    (DATA / "operational" / "services.json").read_text(
        encoding="utf-8"
    )
)
valid_services = {s["service_id"] for s in services}

documents = json.loads(
    (DATA / "metadata" / "documents.json").read_text(
        encoding="utf-8"
    )
)
valid_runbooks = {
    d["document_id"]
    for d in documents
    if d["document_type"] == "runbook"
}

for service, _, _, _, _, runbook in scenarios:
    if service not in valid_services:
        raise ValueError(f"Unknown service: {service}")
    if runbook and runbook not in valid_runbooks:
        raise ValueError(f"Unknown runbook: {runbook}")

# Preserve the existing canonical VPN certificate incident.
if not OUTPUT.exists():
    raise FileNotFoundError(
        "Create and validate INC-0012 before generating incidents"
    )

existing = [
    json.loads(line)
    for line in OUTPUT.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

existing_by_id = {r["incident_id"]: r for r in existing}
if len(existing_by_id) != len(existing):
    raise ValueError("Duplicate incident IDs in existing file")

if set(existing_by_id) != {"INC-0012"}:
    raise ValueError(
        "Expected only INC-0012 before initial generation. "
        "Review existing data to avoid overwriting it."
    )

generated = []
numbers = [n for n in range(1, 31) if n != 12]

for index, (number, scenario) in enumerate(
    zip(numbers, scenarios)
):
    service, title, symptoms, root_cause, resolution, runbook = scenario

    started = datetime(
        2026, 7, 1, 8, 0, tzinfo=timezone.utc
    ) + timedelta(days=index * 3)

    resolved = started + timedelta(hours=2 + index % 5)

    # P1 is reserved for scenarios with substantial impact.
    severity = (
        "P1"
        if title in {
            "SSO authentication outage",
            "Payroll portal outage",
            "Office Wi-Fi outage",
        }
        else "P2" if index % 3 == 0
        else "P3"
    )

    classification = (
        "admin_restricted"
        if runbook in {"RB-END-009", "RB-BKP-012"}
        else "it_internal"
    )

    incident = Incident(
        incident_id=f"INC-{number:04d}",
        service_id=service,
        title=title,
        severity=severity,
        status="resolved",
        started_at=started,
        resolved_at=resolved,
        symptoms=symptoms,
        root_cause=root_cause,
        resolution=resolution,
        runbook_ids=[runbook] if runbook else [],
        classification=classification,
    )

    generated.append(incident.model_dump(mode="json"))

all_records = existing + generated

if len(all_records) != 30:
    raise ValueError("Expected exactly 30 incidents")

OUTPUT.write_text(
    "\n".join(json.dumps(r) for r in all_records) + "\n",
    encoding="utf-8",
)

print(f"Generated {len(generated)} new incidents")
print(f"Total historical incidents: {len(all_records)}")
print("By service:", dict(Counter(
    record["service_id"] for record in all_records
)))
