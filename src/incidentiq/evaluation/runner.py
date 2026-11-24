
import json
from typing import Literal

from incidentiq.core.config import settings

#It loads our existing evaluation cases and converts each JSONL record into a Python dictionary.
def load_evaluation_dataset(
    split: Literal["dev", "test"] = "dev", 
) -> list[dict]:#The split parameter allows us to switch between development and test datasets without changing the evaluation logic.
    """
    Load evaluation examples from the selected dataset.

    dev:
        Used during development and improvement.

    test:
        Reserved for final framework benchmarking.
    """

    if split not in {"dev", "test"}:
        raise ValueError("Invalid evaluation split.")

    path = (
        settings.incidentiq_data_dir
        / "evaluation"
        / f"{split}.jsonl"
    )

    examples = []

    with path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                example = json.loads(line.rstrip(","))

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid evaluation record at "
                    f"line {line_number}"
                ) from exc

            if not isinstance(example, dict):
                raise ValueError(
                    f"Expected an object at line {line_number}"
                )

            examples.append(example)

    if not examples:
        raise ValueError("Evaluation dataset is empty.")

    return examples


#These evaluators measure exact set matching, precision, and recall. They don't require any OpenAI API calls
def evaluate_tool_usage(
    expected_tools: list[str],
    actual_tools: list[str],
) -> dict:
    """
    Compare expected tools against actual tool usage.
    """

    expected = set(expected_tools)
    actual = set(actual_tools)

    matched = expected.intersection(actual)

    missing = expected - actual
    unexpected = actual - expected

    recall = (
        len(matched) / len(expected)
        if expected
        else None
    )

    precision = (
        len(matched) / len(actual)
        if actual
        else None
    )

    return {
        "score": int(expected == actual),
        "precision": precision,
        "recall": recall,
        "matched_tools": sorted(matched),
        "missing_tools": sorted(missing),
        "unexpected_tools": sorted(unexpected),
    }


#citation is not automatically a fabricated citation; it may be another valid source. We'll assess semantic support separately.
def evaluate_citations(
    expected_ids: list[str],
    actual_ids: list[str],
) -> dict:
    """
    Measure overlap between expected and actual
    document citations.
    """

    expected = set(expected_ids)
    actual = set(actual_ids)

    matched = expected.intersection(actual)

    precision = (
        len(matched) / len(actual)
        if actual
        else None
    )

    recall = (
        len(matched) / len(expected)
        if expected
        else None
    )

    return {
        "score": int(expected == actual),
        "precision": precision,
        "recall": recall,
        "matched_ids": sorted(matched),
        "missing_ids": sorted(expected - actual),
        "unexpected_ids": sorted(actual - expected),
    }

