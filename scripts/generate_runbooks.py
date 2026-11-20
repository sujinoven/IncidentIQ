#Python dataset generator instead of manually creating 11 separate files.
import json
from pathlib import Path

DATA = Path("data")
RUNBOOK_DIR = DATA / "knowledge" / "runbooks"
METADATA_FILE = DATA / "metadata" / "documents.json"

RUNBOOK_DIR.mkdir(parents=True, exist_ok=True)
METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

runbooks = [
    {
        "id": "RB-VPN-001",
        "title": "VPN Connectivity Troubleshooting",
        "service": "SVC-VPN",
        "owner": "network_operations",
        "classification": "it_internal",
        "symptoms": [
            "VPN client cannot establish a connection.",
            "Users experience connection timeouts.",
        ],
        "steps": [
            "Confirm whether the issue affects one user or multiple users.",
            "Check the corporate VPN service status.",
            "Verify the user's network connectivity.",
            "Inspect approved VPN client diagnostic logs.",
            "Compare symptoms with historical VPN incidents.",
        ],
        "escalation": "Escalate persistent gateway issues to Network Operations.",
    },
    {
        "id": "RB-VPN-002",
        "title": "VPN Authentication Failure",
        "service": "SVC-VPN",
        "owner": "network_operations",
        "classification": "it_internal",
        "symptoms": [
            "VPN login is rejected.",
            "MFA verification fails.",
        ],
        "steps": [
            "Confirm the affected account and failure timestamp.",
            "Check whether the identity service is operational.",
            "Review approved authentication error logs.",
            "Check account lockout status using authorized tools.",
            "Refer the user to approved account recovery procedures.",
        ],
        "escalation": "Escalate identity-related failures to Identity Operations.",
    },
    {
        "id": "RB-ID-004",
        "title": "SSO Login Failure",
        "service": "SVC-IDENTITY",
        "owner": "identity_operations",
        "classification": "it_internal",
        "symptoms": [
            "Users cannot sign in through corporate SSO.",
            "Applications display authentication errors.",
        ],
        "steps": [
            "Identify affected applications and users.",
            "Check the identity provider service status.",
            "Review authentication failures in approved logs.",
            "Check for expired sessions or configuration changes.",
            "Escalate suspected identity provider outages.",
        ],
        "escalation": "Escalate widespread failures to Identity Operations.",
    },
    {
        "id": "RB-ID-005",
        "title": "MFA Device Recovery",
        "service": "SVC-IDENTITY",
        "owner": "identity_operations",
        "classification": "it_internal",
        "symptoms": [
            "User has lost access to their MFA device.",
            "User cannot complete MFA verification.",
        ],
        "steps": [
            "Create or locate the support ticket.",
            "Verify the user's identity using approved procedures.",
            "Check account recovery eligibility.",
            "Request authorization for MFA factor reset.",
            "Direct the user through approved MFA re-enrollment.",
        ],
        "escalation": "Escalate suspicious recovery requests to Security Operations.",
    },
    {
        "id": "RB-EMAIL-006",
        "title": "Email Delivery Failure",
        "service": "SVC-EMAIL",
        "owner": "messaging_operations",
        "classification": "it_internal",
        "symptoms": [
            "Email messages are delayed.",
            "Users receive delivery failure notifications.",
        ],
        "steps": [
            "Identify affected senders and recipients.",
            "Check email service health.",
            "Review authorized message trace records.",
            "Inspect delivery error codes.",
            "Determine whether the issue is isolated or widespread.",
        ],
        "escalation": "Escalate transport failures to Messaging Operations.",
    },
    {
        "id": "RB-PAY-007",
        "title": "Payroll Application Outage",
        "service": "SVC-PAYROLL",
        "owner": "business_applications",
        "classification": "it_internal",
        "symptoms": [
            "Payroll application is unavailable.",
            "Users receive server errors.",
        ],
        "steps": [
            "Confirm the business impact and affected users.",
            "Check payroll service status.",
            "Review application monitoring alerts.",
            "Check recent approved deployment records.",
            "Escalate before attempting production changes.",
        ],
        "escalation": "Escalate outages to Business Applications.",
    },
    {
        "id": "RB-WIFI-008",
        "title": "Office Wi-Fi Diagnostics",
        "service": "SVC-WIFI",
        "owner": "network_operations",
        "classification": "it_internal",
        "symptoms": [
            "Users cannot connect to office Wi-Fi.",
            "Wireless connections are unstable.",
        ],
        "steps": [
            "Identify the affected office location.",
            "Determine whether multiple devices are affected.",
            "Check wireless service monitoring.",
            "Review approved access point health metrics.",
            "Document signal or authentication errors.",
        ],
        "escalation": "Escalate access point failures to Network Operations.",
    },
    {
        "id": "RB-END-009",
        "title": "Endpoint Malware Investigation",
        "service": "SVC-ENDPOINT",
        "owner": "security_operations",
        "classification": "admin_restricted",
        "symptoms": [
            "Endpoint protection reports a threat.",
            "Suspicious processes or alerts are detected.",
        ],
        "steps": [
            "Confirm the alert using approved security tools.",
            "Record the affected asset identifier.",
            "Preserve relevant evidence according to policy.",
            "Request authorization for containment actions.",
            "Coordinate investigation with Security Operations.",
        ],
        "escalation": "Escalate suspected compromise to Security Operations.",
    },
    {
        "id": "RB-FILE-010",
        "title": "File Sharing Access Failure",
        "service": "SVC-FILES",
        "owner": "infrastructure_operations",
        "classification": "it_internal",
        "symptoms": [
            "Users cannot access shared folders.",
            "File access requests are denied.",
        ],
        "steps": [
            "Confirm the affected share and user.",
            "Check file sharing service status.",
            "Review authorized access permissions.",
            "Inspect relevant authentication failures.",
            "Refer permission changes for approval.",
        ],
        "escalation": "Escalate infrastructure failures to Infrastructure Operations.",
    },
    {
        "id": "RB-CICD-011",
        "title": "CI/CD Pipeline Failure",
        "service": "SVC-DEVOPS",
        "owner": "platform_engineering",
        "classification": "it_internal",
        "symptoms": [
            "Build pipelines fail.",
            "Deployment jobs cannot complete.",
        ],
        "steps": [
            "Identify the failed pipeline and job.",
            "Check CI/CD platform health.",
            "Review build logs and error messages.",
            "Check recent approved configuration changes.",
            "Escalate production deployment issues.",
        ],
        "escalation": "Escalate platform failures to Platform Engineering.",
    },
    {
        "id": "RB-BKP-012",
        "title": "Backup Job Failure and Recovery",
        "service": "SVC-FILES",
        "owner": "infrastructure_operations",
        "classification": "admin_restricted",
        "symptoms": [
            "Scheduled backup job fails.",
            "Backup monitoring reports errors.",
        ],
        "steps": [
            "Identify the failed backup job and affected system.",
            "Review authorized backup monitoring alerts.",
            "Check backup job failure records.",
            "Assess recovery requirements with the service owner.",
            "Obtain authorization before restoration.",
        ],
        "escalation": "Escalate recovery failures to Infrastructure Operations.",
    },
]


def render_runbook(item):
    def bullets(values):
        return "\n".join(f"- {value}" for value in values)

    steps = "\n".join(
        f"{index}. {step}"
        for index, step in enumerate(item["steps"], start=1)
    )

    return f"""# {item['title']}

Document ID: {item['id']}
Organization: NovaTech Solutions
Version: 1.0
Owner: {item['owner']}
Effective Date: 2026-10-01

## 1. Purpose
Provide an approved diagnostic procedure for {item['title'].lower()}.

## 2. Symptoms
{bullets(item['symptoms'])}

## 3. Prerequisites
- Active incident or support ticket.
- Authorized access to relevant diagnostic systems.
- Compliance with NovaTech security policies.

## 4. Diagnostic Procedure
{steps}

## 5. Safety Restrictions
- Do not bypass security controls.
- Do not disclose credentials or restricted information.
- Do not modify production systems without authorization.

## 6. Verification
- Confirm the reported symptoms have been resolved.
- Verify service health using approved monitoring.
- Document findings and actions in the incident record.

## 7. Escalation
{item['escalation']}
"""


existing = (
    json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    if METADATA_FILE.exists()
    else []
)

metadata_by_id = {
    item["document_id"]: item
    for item in existing
}

if len(metadata_by_id) != len(existing):
    raise ValueError("Duplicate document IDs in existing metadata")

service_file = DATA / "operational" / "services.json"
services = json.loads(service_file.read_text(encoding="utf-8"))
valid_service_ids = {service["service_id"] for service in services}

for item in runbooks:
    if item["service"] not in valid_service_ids:
        raise ValueError(f"Unknown service: {item['service']}")

    file_path = RUNBOOK_DIR / f"{item['id']}.md"

    # Do not silently overwrite manually edited documents.
    if not file_path.exists():
        file_path.write_text(
            render_runbook(item),
            encoding="utf-8",
        )

    metadata_by_id[item["id"]] = {
        "document_id": item["id"],
        "title": item["title"],
        "document_type": "runbook",
        "service_id": item["service"],
        "classification": item["classification"],
        "version": "1.0",
        "status": "approved",
        "updated_at": "2026-10-01",
        "owner_team": item["owner"],
        "content_file": f"knowledge/runbooks/{item['id']}.md",
    }

METADATA_FILE.write_text(
    json.dumps(
        list(metadata_by_id.values()),
        indent=2,
    ),
    encoding="utf-8",
)

print(f"Prepared {len(runbooks)} runbooks")
print(f"Metadata registry contains {len(metadata_by_id)} documents")
