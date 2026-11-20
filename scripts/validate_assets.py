
import csv
import json
from pathlib import Path

from asset_models import Asset

DATA = Path("data")

services = json.loads(
    (DATA / "operational/services.json").read_text(
        encoding="utf-8"
    )
)
users = json.loads(
    (DATA / "operational/users.json").read_text(
        encoding="utf-8"
    )
)

service_ids = {s["service_id"] for s in services}
user_ids = {u["user_id"] for u in users}

asset_ids = set()

with (DATA / "operational/assets.csv").open(
    newline="", encoding="utf-8"
) as file:
    reader = csv.DictReader(file)

    for row in reader:
        # Empty CSV fields represent null assignments.
        row["assigned_to"] = row["assigned_to"] or None

        asset = Asset.model_validate(row)

        if asset.asset_id in asset_ids:
            raise ValueError(
                f"Duplicate asset: {asset.asset_id}"
            )

        asset_ids.add(asset.asset_id)

        if asset.service_id not in service_ids:
            raise ValueError(
                f"Unknown service: {asset.service_id}"
            )

        if asset.assigned_to and asset.assigned_to not in user_ids:
            raise ValueError(
                f"Unknown assigned user: {asset.assigned_to}"
            )

assert len(asset_ids) == 15, (
    f"Expected 15 assets, found {len(asset_ids)}"
)

print(f"Validated {len(asset_ids)} assets")
