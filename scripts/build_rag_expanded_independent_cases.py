#!/usr/bin/env python3
"""Build a manually curated, corpus-backed independent RAG test set.

Every question was written after inspecting the referenced chunk in the current
processed corpus. These are new labels, not paraphrases of the original 40 cases.
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "knowledge" / "processed" / "chunks.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "evals" / "rag_expanded_independent_cases.json"


# (case_id, scope, chunk_id, query, tags)
LABELS = [
    ("expanded-eylf-belonging", "eylf", "fab06c0dbca32f8353af5c0d930275fc9c324b5f78fe6564e9bc2f8287cf5916", "Why is knowing where and with whom one belongs described as integral to human existence?", ["concept", "belonging"]),
    ("expanded-eylf-being", "eylf", "b0b7be1aeba9d133c9b672118a1ea16235d26f4769997dca47377f28c46b25e2", "How does the EYLF explain Being as valuing childhood in the present rather than only preparing for the future?", ["concept", "being"]),
    ("expanded-eylf-becoming", "eylf", "ed49b26eddb23a4fbbf2c1858856bc952402c073787a415df5cffe3cfe6d625a", "What changes during childhood are captured by the EYLF idea of Becoming?", ["concept", "becoming"]),
    ("expanded-eylf-interdependent-elements", "eylf", "a3e819efe8cf3b25ba2f60c96d259e5dfd58c15d5c6cc855a10559f623e34bf2", "Which interdependent framework elements put children's learning at the core of EYLF?", ["framework-structure"]),
    ("expanded-eylf-diverse-experience", "eylf", "c7b4a4cae05ff43cee126e75f39c64a170359fce942a89070bb343aa9a045d10", "What diverse experiences and cultural resources do children bring from family and community life into learning?", ["diversity", "home-language"]),
    ("expanded-eylf-theoretical-perspectives", "eylf", "0662c2fd2d4a21ee90ffc782f54d661d0c1b7b54a75f29056dbee175eb7de070", "Why do educators draw on multiple theories, world views and knowledge systems in early childhood pedagogy?", ["pedagogy", "theory"]),
    ("expanded-eylf-attuned-relationships", "eylf", "f4f6f648149fe7f07a0e69b8b2bebe61aca988d138727add79d05baba8bda75c", "How do educators who are attuned to children's thoughts and feelings support learning and wellbeing?", ["relationships"]),
    ("expanded-eylf-first-nations-identity", "eylf", "b6662a17a68205c1d4f912ea24437d4040997e1ba4f5449376e8555093e1719a", "Why should Aboriginal and Torres Strait Islander children see their identities and cultures reflected in the learning environment?", ["first-nations", "identity"]),
    ("expanded-eylf-collaborative-leadership", "eylf", "617b17e02274f68591ded2c60d4c9ec3a75fdce61d12f01dd31e85a965da7566", "How does EYLF describe the everyday leadership responsibilities exercised by all educators?", ["leadership", "ethics"]),
    ("expanded-eylf-holistic-dimensions", "eylf", "0533957f58f8d5b83d3c4c0f97eabd13fc90e11c26ee1391e4a2cfc0cff4d077", "Which dimensions of learning, development and wellbeing must a holistic approach connect?", ["holistic", "multi-part"]),
    ("expanded-eylf-responsive-strengths", "eylf", "156542ce083df4e6f0af1703db6251ecf031b79aba7d2a0826a0121d67ccb943", "How does building on each child's strengths, capabilities and curiosity affect motivation and engagement?", ["responsiveness", "strengths"]),
    ("expanded-eylf-environment-elements", "eylf", "3ccceb2f11bbccee9e44d1b19f2eae172a11cb52e4e75d70c18c895bc9f4bd41", "What physical, temporal, social and intellectual elements make up an EYLF learning environment?", ["environment", "enumeration"]),
    ("expanded-eylf-transition-agency", "eylf", "c098cac76a493b479142e0a7926c7c41bcf03fa8366ad5f26e697ce9843f125b", "How should educators give children an active role in preparing for transitions to a new setting?", ["transition", "agency"]),
    ("expanded-eylf-assessment-strategies", "eylf", "c606371f99d190d9f75c7dfd2c219cfb71267a87cc14c1e149c832e53e087f74", "Which assessment strategies can educators select for different purposes under EYLF?", ["assessment", "enumeration"]),
    ("expanded-eylf-evaluation-effectiveness", "eylf", "71c0bfdc9ffaac2bfbaa0e30ad321b41286cad8360f36cc91750326bfaf8ed0b", "What do EYLF evaluation practices critically reflect on within the planning cycle?", ["evaluation", "planning-cycle"]),
    ("expanded-eylf-plan-stage", "eylf", "45ceee5471d6c4daf1a7282a4b363beaefed98b38c3f34177e8dbe2f8bcbcd47", "How does analysis of collected information shape the Plan/Design stage of the EYLF planning cycle?", ["planning-cycle", "plan"]),
    ("expanded-eylf-implement-stage", "eylf", "328e71d251f8ec5074292c0e52f1e8e0d0327a24b502e2746ed35cc359201d1f", "What happens during the Implement/Enact stage of the EYLF planning cycle?", ["planning-cycle", "implement"]),
    ("expanded-eylf-fairness", "eylf", "32c6adf0aa9661c01b54cb6a254a4feccaa90f6ef7d9c905fcb4b38ae93964d0", "What behaviours show that children are becoming aware of fairness, and how can educators respond?", ["outcome-2", "fairness"]),
    ("expanded-eylf-flexible-rest", "eylf", "ecf6e98d237ca2de367f4d3554408d949a7f7436623b6194dec4c41de4980dcf", "What flexible approach does EYLF suggest for helping children understand sleep, rest and relaxation?", ["outcome-3", "rest"]),
    ("expanded-eylf-digital-thinking", "eylf", "5227eb12282c26ca3f0b7652431760ee54120241e663dce56ec1be33250c2b7f", "How does Outcome 5 describe children using digital technologies to investigate ideas and represent thinking?", ["outcome-5", "digital"]),
    ("expanded-nqs-qa2-purpose", "nqs", "e8b2019ac11523cf52c9e697c3c1d0ac1965dc51772fb848cc47fc58ed5bae63", "What right of children is reinforced by NQS Quality Area 2?", ["qa2", "overview"]),
    ("expanded-nqs-qa2-two-standards", "nqs", "7b7b11dce7c09173b869984b2c6fee7a1bb5b6f1162be62dcc06874091053194", "What do the two Standards in Quality Area 2 focus on, and why are they important?", ["qa2", "standards"]),
    ("expanded-nqs-211-sleep-rest", "nqs", "49de5a913471d66f6695bafb0c3ab49855eaa76f31507cc477aa78f25b77b85d", "What does Element 2.1.1 require for each child's sleep, rest and relaxation needs?", ["qa2", "element-2.1.1"]),
    ("expanded-nqs-212-health-practice", "nqs", "9433cae31b6f32f2b57d5b33e35f85521aef081bd4cfcb8cfc47c10bc31966b3", "What core health, illness, injury and hygiene expectation is stated in Element 2.1.2?", ["qa2", "element-2.1.2"]),
    ("expanded-nqs-infection-reduction", "nqs", "123f95d97d78bb9eb8e77b80a0b4c41c01a6263cd1464de14cdb3b0751918afd", "Why can effective illness management and high hygiene standards reduce but not eliminate infection?", ["qa2", "infection"]),
    ("expanded-nqs-221-supervision", "nqs", "128dea31668c57e2338036a7ee6b5b9af6769411cdce4d336633ec85f4eab7dd", "What must adequate supervision and reasonable precautions protect children from under Element 2.2.1?", ["qa2", "element-2.2.1"]),
    ("expanded-nqs-222-emergency", "nqs", "01572de9153a8b64358b8cce1eeda6644032306fdf1023804db9cdb4038501ef", "What must services do with incident and emergency plans under Element 2.2.2?", ["qa2", "element-2.2.2"]),
    ("expanded-nqs-223-child-protection", "nqs", "539244efd11e4028e285bfaf9aaa9cedb9a78b9b7861b5f51fc1dc3a2f9f3949", "Whose roles and responsibilities must include identifying and responding to children at risk of abuse or neglect?", ["qa2", "element-2.2.3"]),
    ("expanded-nqs-qa3-purpose", "nqs", "23ad6e261c6d53c0d01d3d527c25c286233bf0b564c82f8dfe95d259f154660e", "How does the physical environment in Quality Area 3 contribute to wellbeing, creativity and independence?", ["qa3", "overview"]),
    ("expanded-nqs-311-access", "nqs", "6e120a4eb93f0d83215c10dba58b88d0f086a26554bfc2b472e061d81f912e5f", "What must indoor and outdoor spaces, buildings, fixtures and fittings support under Element 3.1.1?", ["qa3", "element-3.1.1"]),
    ("expanded-nqs-312-maintenance", "nqs", "0a4c540ed7aa0c37aa8a5f6624aafa7fcf0df34958ecae418238269f8ffa2c6d", "What condition must premises, furniture and equipment meet under Element 3.1.2?", ["qa3", "element-3.1.2"]),
    ("expanded-nqs-32-inclusive-environment", "nqs", "84b4fc97f32b2ebb2eaefb89c09647f0598ede589227eb3454f4af3b1db1acb9", "How can an inclusive and flexible service environment support competence, exploration and play-based learning?", ["qa3", "standard-3.2"]),
    ("expanded-nqs-qa4-purpose", "nqs", "710eec0adad1cd5e191feddcfd7b56a6a03624308ccf65d7a8ab2b6b9dd1d0b0", "What educator qualities and relationships are emphasised by Quality Area 4?", ["qa4", "overview"]),
    ("expanded-nqs-41-sufficient-staff", "nqs", "b6f86b1f43c05adac66c0d2b8b58108406e5cc16b2d3fd022f458a256120c17f", "Why does having sufficient educators available matter for learning, development and wellbeing?", ["qa4", "standard-4.1"]),
    ("expanded-nqs-roster-continuity", "nqs", "3c14f174958392f212512c7015da6149b2f0b64c5b85c6a53dc7feb618a355e2", "What rostering consideration may assessors discuss when examining familiarity and continuity for children and families?", ["qa4", "rostering"]),
    ("expanded-nqs-qa5-purpose", "nqs", "a596fa82a7972ecedf69cc596ae964c6ffe4d151b02cef82c347952e74dde7a5", "What kinds of educator-child relationships are the focus of Quality Area 5, and what do they promote?", ["qa5", "overview"]),
    ("expanded-nqs-51-relationships", "nqs", "f94f9ee6f7a0a3daadb74b4987ede02c1499961c72cd2a66fde307250d00c4a1", "How do safe, nurturing and reciprocal relationships help children see themselves?", ["qa5", "standard-5.1"]),
    ("expanded-nqs-qa6-purpose", "nqs", "e93573d06f952c13ad479d8f990a2688dc5843e8188c28525e25fd3702e6ab60", "Why are respectful family relationships and community partnerships central to Quality Area 6?", ["qa6", "overview"]),
    ("expanded-nqs-61-family-role", "nqs", "0ed18511a97231c18af449154ec07796055e3cec745c86befc21e9b9c46767a7", "What relationship with families and support for their parenting role is required by Standard 6.1?", ["qa6", "standard-6.1"]),
    ("expanded-nqs-612-parent-views", "nqs", "20d98c0e30b5574446e595c79893a28c852af6717cf3da4e604ddd0bebb751d2", "What family expertise and participation in decision-making must Element 6.1.2 respect?", ["qa6", "element-6.1.2"]),
    ("expanded-nqs-partnership-privacy", "nqs", "ac75f627c821f8e0c9d954be0d2ecbc1c15da91efdccf72673cb1e8a7048a516", "What privacy and confidentiality foundation should collaborative partnerships recognise?", ["qa6", "privacy"]),
    ("expanded-nqs-qa7-purpose", "nqs", "74bcbde0165b09d44dc8ef23c1b1ab7e044a5768849de6ecc4ff90b85d946cc8", "What does effective leadership and governance establish under Quality Area 7?", ["qa7", "overview"]),
    ("expanded-nqs-71-governance", "nqs", "409ca6fb24d2c59a2dc087ba0c6b3fe48711c5886877b280b7f25bc776c622d8", "What systems must an approved provider maintain so the service operates effectively and ethically?", ["qa7", "standard-7.1"]),
    ("expanded-nqs-713-roles", "nqs", "1a044f5f6d533254340c2dea06886666f9bc564bdd117b9cf2b7f1983564cc9e", "What does Element 7.1.3 require about roles, responsibilities and decision-making?", ["qa7", "element-7.1.3"]),
    ("expanded-nqs-educational-leader-capacity", "nqs", "11d3d20239fa500ce5053bff5ff51c106bb1a699de881b6678f529efae2002b8", "How may an educational leader build educators' capacity in pedagogy, assessment, reflection and planning?", ["qa7", "educational-leader"]),
    ("expanded-nqf-approved-framework-law", "nqs", "e47e8064d782ce4740bb9f7d43c158b9d7f37e717d3739a7d270b2e34cb95a57", "What must the approved learning-framework-based program delivered to children achieve under Section 168?", ["operational", "program"]),
    ("expanded-nqf-assessment-documentation", "nqs", "43da8b1b943b8fa868461575ce6b8ff5086312b57a2485075b7b5e30d6e0c0fc", "What must assessment or evaluation documentation include for a preschool-age child or younger?", ["operational", "documentation"]),
    ("expanded-nqf-supervision-law", "nqs", "771f180b12b1e68d503218e11df9b437af0276c0ff7401866b91e7bd11d8acd3", "Under Section 165, when must children be adequately supervised, including excursions and transport?", ["operational", "supervision"]),
    ("expanded-nqf-individual-sleep-needs", "nqs", "6a15eb12c4b5a241136b9c63202e2501facaf098035586341eec9936b15e6c4e", "Which individual, health-care and cultural factors must be considered when meeting children's sleep and rest needs?", ["operational", "sleep"]),
    ("expanded-nqf-weekly-menu", "nqs", "979349b3eb34cc130cb4e14ee0cbf296d671791d5cfc938b37f52e657f58e522", "What must a service-provided weekly food and beverage menu accurately describe and where must it be displayed?", ["operational", "nutrition"]),
    ("expanded-nqf-medication-double-check", "nqs", "4a37a7b02d8f4d4c1603f73a4a73858b2104cc875d6b95c4c53ac2c51ea06fdd", "What three things must a second person check or witness when medication is administered?", ["operational", "medication"]),
    ("expanded-nqf-fdc-risk-assessment", "nqs", "22aaf60dcfc891b7acdaaf02e23d9a067e4d6d19bc58857b95cc56199a5b4aa7", "What assessment must be completed before education and care begins at a proposed family day care residence or venue?", ["operational", "family-day-care"]),
    ("expanded-nqf-ect-access", "nqs", "9241e7019444e9bb48e9df6936eeb8c085072a77a4a0d85fff7797ba4d2b165d", "What early-childhood-teacher access, attendance or engagement requirement applies to centre-based services with preschool-age children?", ["operational", "qualification"]),
    ("expanded-nqf-small-service-ect", "nqs", "964ed9940c0b61c941f780a699578558f25fc4fb0608cb358b37d3e503395d7d", "For services with fewer than 25 places or children, what minimum proportion of operating time requires access to an early childhood teacher?", ["operational", "qualification", "exact-number"]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    catalog = {
        row["chunk_id"]: row
        for line in CHUNKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    original_path = ROOT / "data" / "evals" / "rag_final_cases.json"
    original_ids = {
        evidence["chunk_id"]
        for case in json.loads(original_path.read_text(encoding="utf-8"))
        for evidence in case["relevant_evidence"]
    }
    cases = []
    for case_id, scope, chunk_id, query, tags in LABELS:
        if chunk_id not in catalog:
            raise ValueError(f"Missing labelled chunk: {chunk_id}")
        if chunk_id in original_ids:
            raise ValueError(f"Expanded label overlaps original gold: {chunk_id}")
        chunk = catalog[chunk_id]
        cases.append(
            {
                "case_id": case_id,
                "split": "test",
                "query": query,
                "scope": scope,
                "tags": ["expanded-independent", *tags],
                "relevant_evidence": [
                    {
                        "source_id": chunk["document"]["source_id"],
                        "chunk_id": chunk_id,
                        "page": chunk.get("page"),
                        "section_contains": chunk.get("section"),
                        "relevance": 3,
                    }
                ],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} independent cases to {args.output}")


if __name__ == "__main__":
    main()
