# Activity Planning Skill

Create a teacher-reviewable early childhood activity plan using only synthetic or
de-identified context.

## Workflow

1. Confirm the requested activity and ask for clarification if essential class or
   activity information is missing.
2. Load the synthetic class profile before selecting learning goals, materials,
   activity steps, or observation points.
3. Design age-appropriate learning goals, materials, ordered steps, and observable
   indicators. Do not diagnose a child.
4. Retrieve and align relevant EYLF outcomes. Use only returned evidence and
   preserve its evidence IDs and citations.
5. Check risk guidance and activity safety when the experience introduces physical
   or environmental risk.
6. Return a complete `ActivityPlan` and keep `is_draft=true`.

## Output requirements

- Include a title and class-profile summary.
- Include at least one learning goal and material.
- Number activity steps from 1 without gaps.
- Use observable, non-diagnostic observation points.
- Include at least one evidence-grounded EYLF alignment with citations.
- Never claim the draft has been approved, saved, exported, or sent.
