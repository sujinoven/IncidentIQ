
import json
from pathlib import Path

path = Path("data/operational/incidents.jsonl")

content = path.read_text(encoding="utf-8")

# Parse the entire file as JSON.
records = json.loads(content)

if not isinstance(records, list):
    raise ValueError("Expected a JSON array of incidents")

if not all(isinstance(record, dict) for record in records):
    raise ValueError("Every incident must be a JSON object")

# Save the original file before modifying anything.
backup = path.with_suffix(".jsonl.bak")

if backup.exists():
    raise FileExistsError(
        f"Backup already exists: {backup}"
    )

backup.write_text(content, encoding="utf-8")

# Convert to one JSON object per line.
path.write_text(
    "\n".join(
        json.dumps(record, ensure_ascii=False)
        for record in records
    ) + "\n",
    encoding="utf-8",
)

print(f"Converted {len(records)} incidents to JSONL")
print(f"Original file backed up to {backup}")
