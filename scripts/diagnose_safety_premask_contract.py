#!/usr/bin/env python3
"""Sanitized diagnosis for local-model versus deterministic premask failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety_gateway.model import LocalQwenAnnotator
from safety_gateway.redaction import (
    INTERNAL_TOKEN_RE,
    premask_text,
    resolve_redaction,
)
from safety_gateway.settings import GatewaySettings


DEFAULT_CHALLENGE = (
    PROJECT_ROOT
    / "data"
    / "local"
    / "safety-eval"
    / "challenge_v3"
    / "challenge.jsonl"
)
CASE_IDS = {
    "challenge_v3_0001",  # failed multi-PII/email
    "challenge_v3_0004",  # failed phone
    "challenge_v3_0005",  # failed DOB
    "challenge_v3_0003",  # successful address control
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", type=Path, default=DEFAULT_CHALLENGE)
    args = parser.parse_args()
    settings = GatewaySettings()
    annotator = LocalQwenAnnotator.load(
        model_dir=settings.model_dir,
        adapter_dir=settings.adapter_dir,
        backend=settings.model_backend,
        max_input_tokens=settings.max_input_tokens,
        max_new_tokens=settings.max_new_tokens,
    )
    rows = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in args.challenge.resolve().read_text(encoding="utf-8").splitlines()
        )
        if row["id"] in CASE_IDS
    }
    for case_id in sorted(CASE_IDS):
        row = rows[case_id]
        premasked = premask_text(row["text"])
        annotation = annotator._annotate_sync(premasked.text)
        reported = [
            {
                "label": entity.label.value,
                "reserved_internal_token": bool(INTERNAL_TOKEN_RE.search(entity.value)),
                "value_present_in_model_input": entity.value in premasked.text,
            }
            for entity in annotation.entities
        ]
        try:
            resolve_redaction(row["text"], premasked, annotation)
            resolution = "valid"
        except Exception as error:
            resolution = f"{type(error).__name__}: {error}"
        print(
            json.dumps(
                {
                    "id": case_id,
                    "gold_labels": sorted(
                        entity["label"] for entity in row["target"].get("entities", [])
                    ),
                    "premasked_labels": sorted(
                        mapping.label.value for mapping in premasked.mappings
                    ),
                    "reported_entities": reported,
                    "resolution": resolution,
                },
                ensure_ascii=True,
            )
        )


if __name__ == "__main__":
    main()
