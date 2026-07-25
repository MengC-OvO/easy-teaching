from scripts.run_week2_evals import DEFAULT_CASES_PATH, load_cases, run


def test_week2_evaluation_set_has_twenty_cases_and_passes() -> None:
    cases = load_cases(DEFAULT_CASES_PATH)
    results = run(DEFAULT_CASES_PATH)

    assert len(cases) == 20
    assert len(results) == 20
    assert all(result["passed"] for result in results)
