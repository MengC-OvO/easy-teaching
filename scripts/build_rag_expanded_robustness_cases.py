#!/usr/bin/env python3
"""Create three labelled robustness variants for every expanded gold question."""

import argparse
import json
import re
from pathlib import Path


STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "in",
    "is", "of", "on", "the", "to", "under", "what", "when", "which", "why",
    "with",
}


def typo_noise(query: str) -> str:
    words = query.split()
    eligible = 0
    output = []
    for word in words:
        bare = re.sub(r"[^A-Za-z]", "", word)
        if len(bare) >= 7:
            eligible += 1
            if eligible % 2 == 0:
                position = max(2, len(word) - 2)
                word = word[:position] + word[position + 1 :]
        output.append(word)
    return " ".join(output).lower()


def telegraphic(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9.%-]+", query.lower())
    kept = [token for token in tokens if token not in STOPWORDS]
    return "keywords only: " + " / ".join(kept)


def adjacent_distractor(query: str) -> str:
    return (
        "I may be mixing this up with a neighbouring outcome, quality area or "
        "regulation. Use only the requested source and identify the exact evidence: "
        + query
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_cases = json.loads(args.source.read_text(encoding="utf-8"))
    transforms = {
        "typo": typo_noise,
        "telegraphic": telegraphic,
        "distractor": adjacent_distractor,
    }
    cases = []
    for source in source_cases:
        for label, transform in transforms.items():
            cases.append(
                {
                    **source,
                    "case_id": f"{source['case_id']}-{label}",
                    "query": transform(source["query"]),
                    "tags": [
                        *source.get("tags", []),
                        "expanded-robustness-variant",
                        label,
                    ],
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} robustness variants to {args.output}")


if __name__ == "__main__":
    main()
