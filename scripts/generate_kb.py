
import json
from pathlib import Path

DATA = Path("data")
KB_DIR = DATA / "knowledge" / "kb_articles"
METADATA_FILE = DATA / "metadata" / "documents.json"

KB_DIR.mkdir(parents=True, exist_ok=True)

# ID, title, service, classification, symptoms, causes, actions
articles = [
    (
        "KB-VPN-002", "VPN Disconnects Frequently", "SVC-VPN",
        "employee_public",
        ["VPN disconnects during work", "Frequent reconnection prompts"],
        ["Unstable internet connection", "VPN service degradation"],
        ["Check internet stability", "Reconnect using the approved VPN client",
         "Report repeated disconnections to IT Support"],
    ),
    (
        "KB-ID-003", "Cannot Sign In to Corporate SSO", "SVC-IDENTITY",
        "employee_public",
        ["SSO login fails", "Authentication error appears"],
        ["Account access problem", "Identity service disruption"],
        ["Check the service status", "Use approved account recovery",
         "Contact IT Support if access remains blocked"],
    ),
    (
        "KB-ID-004", "Lost or Replaced MFA Device", "SVC-IDENTITY",
        "employee_public",
        ["MFA prompt cannot be completed"],
        ["Registered MFA device is unavailable"],
        ["Contact IT Support", "Complete approved identity verification",
         "Follow authorized MFA re-enrollment instructions"],
    ),
    (
        "KB-EMAIL-005", "Email Not Received", "SVC-EMAIL",
        "employee_public",
        ["Expected email has not arrived"],
        ["Delivery delay", "Incorrect recipient address"],
        ["Check spam and quarantine notifications",
         "Confirm sender details", "Report persistent delivery issues"],
    ),
    (
        "KB-EMAIL-006", "Email Attachment Rejected", "SVC-EMAIL",
        "employee_public",
        ["Attachment is blocked or rejected"],
        ["File type restriction", "Security scanning policy"],
        ["Check the rejection notice", "Use approved file-sharing methods",
         "Contact IT Support for assistance"],
    ),
    (
        "KB-PAY-007", "Payroll Portal Unavailable", "SVC-PAYROLL",
        "employee_public",
        ["Payroll portal does not load"],
        ["Application outage", "Authentication failure"],
        ["Check published service status", "Retry later if degraded",
         "Report the issue without sharing payroll data"],
    ),
    (
        "KB-HR-008", "HR Portal Access Problems", "SVC-HR",
        "employee_public",
        ["HR portal access denied"],
        ["Permission issue", "Expired session"],
        ["Sign in through approved SSO", "Check service status",
         "Contact IT Support for access review"],
    ),
    (
        "KB-WIFI-009", "Office Wi-Fi Connection Problems", "SVC-WIFI",
        "employee_public",
        ["Office Wi-Fi fails to connect"],
        ["Wireless network degradation", "Authentication failure"],
        ["Verify the approved network name", "Reconnect normally",
         "Report the affected office location"],
    ),
    (
        "KB-END-010", "Laptop Running Slowly", "SVC-ENDPOINT",
        "employee_public",
        ["Laptop performance is slow"],
        ["Resource pressure", "Pending approved updates"],
        ["Restart the laptop", "Check available storage",
         "Contact Endpoint Support if the issue persists"],
    ),
    (
        "KB-END-011", "Endpoint Security Alert", "SVC-ENDPOINT",
        "employee_public",
        ["Security warning appears on the device"],
        ["Endpoint protection detected suspicious activity"],
        ["Do not dismiss or disable protection",
         "Record the alert details", "Contact IT Support immediately"],
    ),
    (
        "KB-FILE-012", "Shared Folder Access Denied", "SVC-FILES",
        "employee_public",
        ["Access denied when opening a shared folder"],
        ["Missing permission", "Authentication issue"],
        ["Verify the intended folder", "Sign in using the approved account",
         "Request an authorized access review"],
    ),
    (
        "KB-DEV-013", "CI/CD Pipeline Error Guide", "SVC-DEVOPS",
        "it_internal",
        ["Pipeline execution fails"],
        ["Build error", "Configuration issue"],
        ["Review approved job logs", "Check platform status",
         "Escalate deployment-impacting failures"],
    ),
    (
        "KB-MON-014", "Understanding Service Health Alerts", "SVC-MONITOR",
        "it_internal",
        ["Monitoring dashboard shows degraded service"],
        ["Health-check failure", "Service dependency issue"],
        ["Review alert timestamps", "Check affected services",
         "Correlate with active incidents"],
    ),
    (
        "KB-BKP-015", "Backup Failure Notification Guide", "SVC-FILES",
        "it_internal",
        ["Backup monitoring reports failure"],
        ["Backup job error", "Storage availability issue"],
        ["Review approved backup alert details",
         "Record the affected system", "Escalate to Infrastructure Operations"],
    ),
]


def bullets(values):
    return "\n".join(f"- {value}" for value in values)


def numbered(values):
    return "\n".join(
        f"{i}. {value}" for i, value in enumerate(values, 1)
    )


metadata = json.loads(
    METADATA_FILE.read_text(encoding="utf-8")
)
metadata_by_id = {
    doc["document_id"]: doc for doc in metadata
}

if len(metadata_by_id) != len(metadata):
    raise ValueError("Duplicate document IDs")

services = json.loads(
    (DATA / "operational" / "services.json").read_text(
        encoding="utf-8"
    )
)
valid_services = {service["service_id"] for service in services}

for (
    doc_id, title, service_id, classification,
    symptoms, causes, actions
) in articles:

    if service_id not in valid_services:
        raise ValueError(f"Unknown service: {service_id}")

    content = f"""# {title}

Document ID: {doc_id}
Organization: NovaTech Solutions
Version: 1.0
Effective Date: 2026-10-01

## Overview
This article explains {title.lower()} and provides
approved initial troubleshooting guidance.

## Common Symptoms
{bullets(symptoms)}

## Possible Causes
{bullets(causes)}

## Safe Troubleshooting Steps
{numbered(actions)}

## Safety Guidance
- Never share passwords or MFA codes.
- Never disable company security controls.
- Do not perform unauthorized system changes.

## Escalation
Contact the responsible IT service team if the
problem persists or affects multiple users.
"""

    file_path = KB_DIR / f"{doc_id}.md"

    if not file_path.exists():
        file_path.write_text(content, encoding="utf-8")

    metadata_by_id[doc_id] = {
        "document_id": doc_id,
        "title": title,
        "document_type": "kb_article",
        "service_id": service_id,
        "classification": classification,
        "version": "1.0",
        "status": "approved",
        "updated_at": "2026-10-01",
        "owner_team": "it_knowledge_management",
        "content_file": f"knowledge/kb_articles/{doc_id}.md",
    }

METADATA_FILE.write_text(
    json.dumps(
        list(metadata_by_id.values()),
        indent=2,
    ),
    encoding="utf-8",
)

print(f"Prepared {len(articles)} KB articles")
print(f"Total registered documents: {len(metadata_by_id)}")
