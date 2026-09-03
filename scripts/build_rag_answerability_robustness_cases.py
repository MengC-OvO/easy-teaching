#!/usr/bin/env python3
"""Create typo, telegraphic and adjacent-distractor answerability variants."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path


STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "in",
    "is", "of", "on", "the", "to", "under", "what", "when", "which",
    "why", "with",
}


def typo_noise(query: str) -> str:
    words = query.split()
    changed = 0
    output = []
    for word in words:
        bare = re.sub(r"[^A-Za-z]", "", word)
        if len(bare) >= 7:
            changed += 1
            if changed % 2 == 0:
                position = max(2, len(word) - 2)
                word = word[:position] + word[position + 1 :]
        output.append(word)
    return " ".join(output).lower()


def telegraphic(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9.:%-]+", query.casefold())
    return "keywords only: " + " / ".join(
        token for token in tokens if token not in STOPWORDS
    )


def adjacent_distractor(query: str) -> str:
    return (
        "I may be mixing this up with a neighbouring outcome, quality area or "
        "regulation. Use only the requested source and verify the exact claim: "
        + query
    )


def build_variants(cases: list[dict]) -> list[dict]:
    transforms = {
        "typo": typo_noise,
        "telegraphic": telegraphic,
        "distractor": adjacent_distractor,
    }
    variants = []
    for source in cases:
        for label, transform in transforms.items():
            case = deepcopy(source)
            case["id"] = f"{source['id']}-{label}"
            case["turns"][0]["message"] = transform(
                source["turns"][0]["message"]
            )
            case["tags"] = [
                *source.get("tags", []),
                "answerability-robustness",
                label,
            ]
            variants.append(case)
    return variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.source.read_text(encoding="utf-8"))
    variants = build_variants(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(variants, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(variants)} answerability robustness cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
