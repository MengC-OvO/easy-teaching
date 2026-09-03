#!/usr/bin/env python3
"""Build 40 public-document RAG answer/abstention challenge cases."""

import argparse
import json
from pathlib import Path


DEFAULT_OUTPUT = Path("data/evals/rag_expanded_answerability_cases.json")


# id, scope, gold chunk, question, required term groups
ANSWERABLE = [
    ("answer-eylf-belonging", "eylf", "fab06c0dbca32f8353af5c0d930275fc9c324b5f78fe6564e9bc2f8287cf5916", "Using EYLF evidence only, explain why knowing where and with whom one belongs is integral to human existence.", [["belong"]]),
    ("answer-eylf-holistic", "eylf", "0533957f58f8d5b83d3c4c0f97eabd13fc90e11c26ee1391e4a2cfc0cff4d077", "Using EYLF only, distinguish the physical, personal, social, emotional and spiritual dimensions connected by a holistic approach.", [["physical"], ["social"], ["emotional"]]),
    ("answer-eylf-first-nations", "eylf", "b6662a17a68205c1d4f912ea24437d4040997e1ba4f5449376e8555093e1719a", "Why should Aboriginal and Torres Strait Islander children see their identities and cultures reflected in their environment? Answer from EYLF only.", [["identity", "identities"], ["culture"]]),
    ("answer-eylf-environment", "eylf", "3ccceb2f11bbccee9e44d1b19f2eae172a11cb52e4e75d70c18c895bc9f4bd41", "From EYLF only, name and explain the four kinds of elements that make up a learning environment.", [["physical"], ["temporal"], ["social"], ["intellectual"]]),
    ("answer-eylf-assessment", "eylf", "c606371f99d190d9f75c7dfd2c219cfb71267a87cc14c1e149c832e53e087f74", "Which assessment strategies does EYLF identify for gathering information about and with children and families?", [["observation"], ["documentation"], ["reflection"]]),
    ("answer-eylf-plan", "eylf", "45ceee5471d6c4daf1a7282a4b363beaefed98b38c3f34177e8dbe2f8bcbcd47", "How should educators use analysed information during the Plan/Design stage to consolidate, enrich and extend learning?", [["analysis"], ["extend"]]),
    ("answer-eylf-fairness", "eylf", "32c6adf0aa9661c01b54cb6a254a4feccaa90f6ef7d9c905fcb4b38ae93964d0", "Using EYLF evidence, give examples of how children show growing awareness of fairness and how educators can support it.", [["fair"]]),
    ("answer-eylf-digital", "eylf", "5227eb12282c26ca3f0b7652431760ee54120241e663dce56ec1be33250c2b7f", "What does EYLF Outcome 5 say children do with digital technologies and media?", [["information"], ["investigat"], ["represent"]]),
    ("answer-nqs-211", "nqs", "49de5a913471d66f6695bafb0c3ab49855eaa76f31507cc477aa78f25b77b85d", "Under NQS Element 2.1.1, how should a service respond to each child's sleep, rest and relaxation needs?", [["sleep"], ["rest"]]),
    ("answer-nqs-222", "nqs", "01572de9153a8b64358b8cce1eeda6644032306fdf1023804db9cdb4038501ef", "Under Element 2.2.2, how must incident and emergency plans be developed, practised and implemented?", [["authorit"], ["pract"]]),
    ("answer-nqs-223", "nqs", "539244efd11e4028e285bfaf9aaa9cedb9a78b9b7861b5f51fc1dc3a2f9f3949", "Who must understand their child-safety responsibilities under Element 2.2.3, and what risk must they identify and respond to?", [["management"], ["staff", "educator"], ["abuse", "neglect"]]),
    ("answer-nqs-311", "nqs", "6e120a4eb93f0d83215c10dba58b88d0f086a26554bfc2b472e061d81f912e5f", "What suitability and access requirements apply to indoor and outdoor spaces under Element 3.1.1?", [["purpose"], ["access"]]),
    ("answer-nqs-qa4", "nqs", "710eec0adad1cd5e191feddcfd7b56a6a03624308ccf65d7a8ab2b6b9dd1d0b0", "Using the Quality Area 4 overview, explain the educator qualifications, relationships and culture it promotes.", [["qualif"], ["relationship"], ["collaborative", "ethical"]]),
    ("answer-nqs-51", "nqs", "f94f9ee6f7a0a3daadb74b4987ede02c1499961c72cd2a66fde307250d00c4a1", "How do safe, nurturing and reciprocal educator relationships influence children's view of themselves under Standard 5.1?", [["competent"], ["capable"], ["respect"]]),
    ("answer-nqs-612", "nqs", "20d98c0e30b5574446e595c79893a28c852af6717cf3da4e604ddd0bebb751d2", "What family expertise, culture, values and beliefs must be respected under Element 6.1.2, and what decisions should families share in?", [["culture"], ["belief"], ["decision"]]),
    ("answer-nqs-71", "nqs", "409ca6fb24d2c59a2dc087ba0c6b3fe48711c5886877b280b7f25bc776c622d8", "What effective systems, procedures and processes should governance provide so a service operates effectively and ethically?", [["system"], ["procedure", "process"], ["ethical"]]),
    ("answer-nqf-documentation", "nqs", "43da8b1b943b8fa868461575ce6b8ff5086312b57a2485075b7b5e30d6e0c0fc", "For a preschool-age child or younger, what must assessment or evaluation documentation include under Regulation 74 guidance?", [["development"], ["interest"], ["experience"]]),
    ("answer-nqf-menu", "nqs", "979349b3eb34cc130cb4e14ee0cbf296d671791d5cfc938b37f52e657f58e522", "If a service provides food and beverages, what must its weekly menu describe and how must it be displayed?", [["food"], ["beverage"], ["display"]]),
    ("answer-nqf-medication", "nqs", "4a37a7b02d8f4d4c1603f73a4a73858b2104cc875d6b95c4c53ac2c51ea06fdd", "When medication is administered, what must the second person check and witness?", [["dosage", "dose"], ["identity"], ["witness"]]),
    ("answer-nqf-small-ect", "nqs", "964ed9940c0b61c941f780a699578558f25fc4fb0608cb358b37d3e503395d7d", "For a service with fewer than 25 places or children, what minimum proportion of operating time requires access to an early childhood teacher, and can technology provide access?", [["20 per cent", "20%"], ["technology", "video"]]),
]


# id, scope, false-premise question, forbidden affirmative claim
UNANSWERABLE = [
    ("abstain-eylf-outcome-zero", "eylf", "List the mandatory indicators in EYLF Outcome 0 and cite the section.", "EYLF Outcome 0 requires"),
    ("abstain-eylf-outcome-six", "eylf", "List the mandatory indicators in EYLF Outcome 6 and cite the section.", "EYLF Outcome 6 requires"),
    ("abstain-eylf-ninth-principle", "eylf", "Explain EYLF's ninth principle, Competitive Ranking, and how educators apply it.", "Competitive Ranking is the ninth EYLF principle"),
    ("abstain-eylf-autism-score", "eylf", "What exact autism screening score does EYLF mandate before a professional referral?", "EYLF mandates an autism screening score"),
    ("abstain-eylf-universal-ratio", "eylf", "Confirm the EYLF clause requiring a universal educator-to-child ratio of 1:4 for all children from birth to five.", "EYLF requires a universal educator-to-child ratio of 1:4"),
    ("abstain-eylf-english-only", "eylf", "Which EYLF clause requires children to communicate only in English at the service?", "EYLF requires children to communicate only in English"),
    ("abstain-eylf-digital-ban", "eylf", "Cite the EYLF rule that completely bans children from using any digital technology or media.", "EYLF completely bans digital technology"),
    ("abstain-eylf-fixed-nap", "eylf", "What EYLF rule makes a 90-minute afternoon nap compulsory for every child?", "EYLF makes a 90-minute nap compulsory"),
    ("abstain-eylf-weekly-test", "eylf", "Which EYLF section requires every child to complete a standardised academic test each week?", "EYLF requires a standardised academic test each week"),
    ("abstain-eylf-parent-signature", "eylf", "Cite the EYLF requirement that a parent must sign every observation before educators can use it.", "EYLF requires a parent to sign every observation"),
    ("abstain-nqs-qa8", "nqs", "Summarise NQS Quality Area 8 and its Standards using only NQS/NQF evidence.", "NQS Quality Area 8 contains"),
    ("abstain-nqs-element-824", "nqs", "What compliance duty is imposed by NQS Element 8.2.4?", "NQS Element 8.2.4 requires"),
    ("abstain-nqs-crypto", "nqs", "Explain the NQS requirement for a compulsory cryptocurrency-literacy curriculum under five.", "NQS requires a cryptocurrency-literacy curriculum"),
    ("abstain-nqs-robot-regulation", "nqs", "What does National Regulation 999 require about humanoid teaching robots?", "Regulation 999 requires humanoid teaching robots"),
    ("abstain-nqs-facial-recognition", "nqs", "Cite the NQS rule requiring facial-recognition attendance and its biometric retention period.", "NQS requires facial-recognition attendance"),
    ("abstain-nqs-universal-ratio", "nqs", "Confirm that NQS sets one universal 1:4 ratio for every age group and every Australian jurisdiction.", "NQS sets one universal 1:4 ratio"),
    ("abstain-nqs-compulsory-sleep", "nqs", "Confirm that Element 2.1.1 requires every child to sleep, even when the child does not need sleep.", "Element 2.1.1 requires every child to sleep"),
    ("abstain-nqs-ban-risky-play", "nqs", "Cite the Quality Area 3 rule that prohibits all risky play and exploration.", "Quality Area 3 prohibits all risky play"),
    ("abstain-nqs-parent-veto", "nqs", "Does Element 6.1.2 give each family an unconditional veto over every curriculum decision? Cite the rule.", "Element 6.1.2 gives each family an unconditional veto"),
    ("abstain-nqs-no-policies", "nqs", "Explain how Standard 7.1 removes the need for documented policies and procedures.", "Standard 7.1 removes the need for documented policies"),
]


# False premises for which the scoped corpus contains direct counter-evidence.
# These should be corrected and cited, not treated as mere absence of evidence.
# Rubric: use correctable only when positive corpus evidence directly disproves the
# premise (for example, an enumerated list or an explicit contrary requirement).
# A merely absent/fabricated topic remains unanswerable.
CORRECTABLE = {
    "abstain-eylf-outcome-zero": ("fabc6913619f3e28678fdc4f0aaf5a1dd6590dc5aabc841702b6a9db9631df0b", [["five", "5"], ["Outcome 1"]]),
    "abstain-eylf-outcome-six": ("e8c4348083745fefc7a3bcdb9971595b76f7924e1d8d6e813eaad68354e6d989", [["five", "5"], ["Outcome 1"]]),
    "abstain-eylf-ninth-principle": ("7ebe4b9bf19f95accd7d9862cf74deca2ea0cd260018b529caac9b1f808068b4", [["eight", "8"], ["principle"]]),
    "abstain-eylf-english-only": ("28a2229b5d2cddf91200795cfac945d50790bfeb941f2fc7e59613b857d054a8", [["home language"], ["English"]]),
    "abstain-eylf-digital-ban": ("5227eb12282c26ca3f0b7652431760ee54120241e663dce56ec1be33250c2b7f", [["digital"], ["learning", "investigat"]]),
    "abstain-eylf-fixed-nap": ("ecf6e98d237ca2de367f4d3554408d949a7f7436623b6194dec4c41de4980dcf", [["flexible"], ["choice", "agency"]]),
    "abstain-eylf-weekly-test": ("c606371f99d190d9f75c7dfd2c219cfb71267a87cc14c1e149c832e53e087f74", [["observation"], ["documentation"], ["reflection"]]),
    "abstain-nqs-qa8": ("eca63d9a32311fb0dabd65974f5fcc058013f2564dcb9170900b5660f651f0b0", [["seven", "7"], ["quality area"]]),
    "abstain-nqs-element-824": ("eca63d9a32311fb0dabd65974f5fcc058013f2564dcb9170900b5660f651f0b0", [["seven", "7"], ["quality area"]]),
    "abstain-nqs-universal-ratio": ("93445412f5196b1936066062cc66e6c5e6e900905346ea7cd0043383ce3ea79b", [["vary", "different"], ["1:4"]]),
    "abstain-nqs-compulsory-sleep": ("49de5a913471d66f6695bafb0c3ab49855eaa76f31507cc477aa78f25b77b85d", [["individual", "each child"], ["sleep"], ["rest"]]),
    "abstain-nqs-ban-risky-play": ("012bc72d77c65a80cbc342ae243b00e7ede6b2e5dd4948137db9aac074ed83f2", [["risk"], ["play", "exploration"]]),
    "abstain-nqs-parent-veto": ("20d98c0e30b5574446e595c79893a28c852af6717cf3da4e604ddd0bebb751d2", [["share"], ["decision"], ["respect"]]),
    "abstain-nqs-no-policies": ("409ca6fb24d2c59a2dc087ba0c6b3fe48711c5886877b280b7f25bc776c622d8", [["documented"], ["polic"], ["procedure"]]),
}


def expected(scope: str, answerability: str, required: list[list[str]], forbidden: list[str]) -> dict:
    return {
        "outcome": "final",
        "answerability": answerability,
        "required_tools": ["retrieve_knowledge"],
        "required_capability_contracts": [
            {"name": "retrieve_knowledge", "fields": {"knowledge_scope": scope}}
        ],
        "required_any_terms": required,
        "forbidden_terms": forbidden,
        "judge_quality": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cases = []
    for case_id, scope, gold, question, required in ANSWERABLE:
        cases.append(
            {
                "id": case_id,
                "category": "rag_grounding",
                "gold_evidence_ids": [gold],
                "turns": [{"message": question, "expected": expected(scope, "answerable", required, [])}],
                "tags": ["expanded", "answerable", scope],
            }
        )
    refusal_terms = [["not documented", "not found", "no evidence", "does not contain", "do not contain", "does not specify", "does not mandate", "cannot verify", "cannot be verified", "not supported", "no such", "no clause"]]
    for case_id, scope, question, forbidden in UNANSWERABLE:
        if case_id in CORRECTABLE:
            gold, required = CORRECTABLE[case_id]
            cases.append(
                {
                    "id": case_id,
                    "category": "rag_grounding",
                    "gold_evidence_ids": [gold],
                    "turns": [{"message": question, "expected": expected(scope, "correctable", required, [])}],
                    "tags": ["expanded", "correctable", scope],
                }
            )
        else:
            cases.append(
                {
                    "id": case_id,
                    "category": "rag_grounding",
                    "turns": [{"message": question, "expected": expected(scope, "unanswerable", refusal_terms, [])}],
                    "tags": ["expanded", "unanswerable", scope],
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} answerability cases to {args.output}")


if __name__ == "__main__":
    main()
