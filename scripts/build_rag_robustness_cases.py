#!/usr/bin/env python3
"""Build frozen query-robustness variants from the independently labelled RAG set.

These are perturbations of existing questions, not new independent knowledge
labels. Reports must keep them separate from the clean 40-case baseline.
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "evals" / "rag_final_cases.json"
DEFAULT_OUTPUT = ROOT / "data" / "evals" / "rag_robustness_cases.json"


VARIANTS = {
    "final-eylf-02-pedagogy": ("eylf pedagogy definishn?? exact meaning pls", ["typo", "telegraphic"]),
    "final-eylf-03-principles": ("8 principles — what are they actually for in practice (not outcomes)?", ["distractor", "contrast"]),
    "final-eylf-05-diversity": ("EYLF 对儿童不同文化和知识方式，教师应该怎样尊重？", ["chinese", "cross-lingual"]),
    "final-eylf-07-sustainability": ("Not just recycling: what THREE dimensions of sustainability does EYLF describe?", ["hard", "distractor"]),
    "final-eylf-08-critical-reflection": ("who gets included / excluded / advantaged — educator reflection prompts?", ["telegraphic", "paraphrase"]),
    "final-eylf-09-play-intentionality": ("In play-based learning, intentionality belongs only to adults, right? Correct the premise.", ["false-premise", "hard"]),
    "final-eylf-10-cultural-responsiveness": ("culturaly responzive practice: respect + act on what?", ["typo", "compressed"]),
    "final-eylf-11-transitions": ("如何利用儿童家庭已有的知识经验，支持学习连续性和过渡？", ["chinese", "cross-lingual"]),
    "final-eylf-12-assessment-evaluation": ("assessment vs evaluation: how do both feed the learning cycle rather than end it?", ["contrast", "hard"]),
    "final-eylf-14-outcome-one": ("identity outcome capabilities? EYLF O1, concise", ["telegraphic", "abbreviation"]),
    "final-eylf-16-outcome-three": ("EYLF结果3具体覆盖哪些健康、福祉和个人安全内容？", ["chinese", "cross-lingual"]),
    "final-eylf-18-outcome-five": ("Outcome five isn't only spoken language—what communication modes are covered?", ["false-premise", "hard"]),
    "final-nqs-01-cultural-responsiveness": ("NQS cultural responsiveness + discrimination: definition and required action", ["compressed", "multi-part"]),
    "final-nqs-03-approved-framework": ("elemnt 1.1.1 curriculum decisions contribute to wot exactly?", ["typo", "telegraphic"]),
    "final-nqs-04-curriculum-decisions": ("观察记录和家庭意见，应如何真正影响课程决策？", ["chinese", "cross-lingual"]),
    "final-nqs-06-program-opportunities": ("Do only planned lessons count? Explain learning in routines, events and interactions.", ["false-premise", "distractor"]),
    "final-nqs-07-intentionality": ("1.2.1 intentionality ≠ teacher control. What does it mean?", ["contrast", "hard"]),
    "final-nqs-08-responsive-scaffolding": ("responsive teachng/scafolding — extend kids ideas + play how?", ["typo", "compressed"]),
    "final-nqs-10-assessment-cycle": ("NQS 1.3.1持续评估与规划循环，包括哪些阶段？", ["chinese", "cross-lingual"]),
    "final-nqs-12-documentation": ("documentation isn't paperwork for its own sake: whose learning does it make visible, and to whom?", ["distractor", "multi-part"]),
    "final-nqs-14-family-progress": ("Element 1.3.3: families get what about program AND their child's progress?", ["multi-part", "exact-element"]),
    "final-nqs-18-parent-request": ("家长提出要求时，服务机构必须提供哪些教育项目资料？", ["chinese", "cross-lingual"]),
    "final-policy-02-outdoor-play": ("Outdoor play checks: supervision + allergy + environment; don't give generic activity ideas.", ["constraint", "multi-part"]),
    "final-policy-03-learning-records": ("Demo records: PII handling and diagnostic claims—what are the limits?", ["compressed", "multi-part"]),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    originals = {
        item["case_id"]: item
        for item in json.loads(SOURCE.read_text(encoding="utf-8"))
    }
    cases = []
    for source_id, (query, tags) in VARIANTS.items():
        source = originals[source_id]
        cases.append(
            {
                **source,
                "case_id": source_id.replace("final-", "robust-"),
                "query": query,
                "tags": [*source.get("tags", []), "robustness-variant", *tags],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
