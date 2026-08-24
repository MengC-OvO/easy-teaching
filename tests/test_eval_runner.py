from evals import EvalCategory, EvalMode, load_eval_cases, run_eval_suite


def test_offline_suite_runs_all_fixed_cases_without_model_calls() -> None:
    report = run_eval_suite()

    assert report.mode is EvalMode.OFFLINE
    assert report.summary.total == 30
    assert report.summary.token_usage.model_calls == 0
    assert set(report.summary.categories) == set(EvalCategory)
    assert all(result.error_code is None for result in report.results)


def test_runner_can_filter_to_one_category() -> None:
    report = run_eval_suite(categories=[EvalCategory.SAFETY])

    assert report.summary.total == 5
    assert report.summary.passed == 5
    assert set(report.summary.categories) == {EvalCategory.SAFETY}


def test_rag_fixture_suite_runs_against_persistent_lexical_index() -> None:
    case_ids = {case.id for case in load_eval_cases()}
    report = run_eval_suite(categories=[EvalCategory.RAG])

    assert {result.case_id for result in report.results} <= case_ids
    assert report.summary.total == 5
    assert report.summary.failed == 0
