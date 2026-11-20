
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class User(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(pattern=r"^USR-\d{3}$")
    display_name: str = Field(min_length=2)
    role: Literal["employee", "it_support", "it_admin"]
    department: str = Field(min_length=2)
    active: bool


user_specs = [
    ("USR-001", "Asha Raman", "employee", "Finance"),
    ("USR-002", "Daniel Reed", "employee", "Sales"),
    ("USR-003", "Meera Shah", "employee", "Human Resources"),
    ("USR-004", "Noah Chen", "employee", "Engineering"),
    ("USR-005", "Priya Nair", "employee", "Operations"),
    ("USR-006", "Olivia Brooks", "employee", "Finance"),
    ("USR-007", "Arjun Patel", "employee", "Engineering"),
    ("USR-008", "Emma Wilson", "employee", "Sales"),
    ("USR-009", "Sofia Martin", "it_support", "IT Support"),
    ("USR-010", "Liam Walker", "it_support", "IT Support"),
    ("USR-011", "Ethan Clark", "it_admin", "Infrastructure"),
    ("USR-012", "Maya Iyer", "it_admin", "Security Operations"),
]

users = [
    User(
        user_id=user_id,
        display_name=name,
        role=role,
        department=department,
        active=True,
    )
    for user_id, name, role, department in user_specs
]

ids = [user.user_id for user in users]
if len(ids) != len(set(ids)):
    raise ValueError("Duplicate user IDs")

path = Path("data/operational/users.json")
path.parent.mkdir(parents=True, exist_ok=True)

path.write_text(
    json.dumps([u.model_dump() for u in users], indent=2),
    encoding="utf-8",
)

print(f"Created {len(users)} users")
