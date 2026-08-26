"""Suite model and analysis tests."""

import json

import pytest

from itemwise import Item, RunResult, Suite, analyze, to_html, to_text

ITEMS = [
    {"id": "always-pass", "prompt": "2+2?", "tags": ["trivial"]},
    {"id": "never-pass", "prompt": "unsolved problem", "tags": ["frontier"]},
    {"id": "splits-clean", "prompt": "hard but fair"},
    {"id": "only-best", "prompt": "very hard"},
    {"id": "most-pass", "prompt": "moderate"},
]

SCORES = {
    "strong-model": [1, 0, 1, 1, 1],
    "good-model":   [1, 0, 1, 0, 1],
    "weak-model":   [1, 0, 0, 0, 1],
    "poor-model":   [1, 0, 0, 0, 0],
}


def build() -> RunResult:
    return RunResult(Suite.from_dicts(ITEMS), dict(SCORES))


# --------------------------------------------------------------- suite model

def test_suite_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate item ids"):
        Suite.from_dicts([{"id": "a"}, {"id": "a"}])


def test_item_requires_an_id():
    with pytest.raises(ValueError, match="missing required field 'id'"):
        Item.from_dict({"prompt": "no id here"})


def test_unknown_fields_are_kept_as_meta():
    item = Item.from_dict({"id": "x", "prompt": "p", "expected": "42", "owner": "jude"})
    assert item.meta == {"expected": "42", "owner": "jude"}


def test_suite_round_trips_through_jsonl(tmp_path):
    path = tmp_path / "evals.jsonl"
    path.write_text("\n".join(json.dumps(i) for i in ITEMS), encoding="utf-8")
    suite = Suite.from_jsonl(path)
    assert suite.ids == [i["id"] for i in ITEMS]
    assert len(suite) == 5


def test_jsonl_error_names_the_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "ok"}\nnot json at all\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        Suite.from_jsonl(path)


def test_runresult_rejects_wrong_score_count():
    with pytest.raises(ValueError, match="has 2 scores but the suite has 5"):
        RunResult(Suite.from_dicts(ITEMS), {"m": [1, 0]})


def test_runresult_rejects_non_binary_scores():
    with pytest.raises(ValueError, match="non-binary"):
        RunResult(Suite.from_dicts(ITEMS), {"m": [1, 0, 1, 0, 7]})


def test_from_records_builds_the_same_thing():
    suite = Suite.from_dicts(ITEMS)
    records = [
        {"model": m, "item_id": iid, "passed": bool(s)}
        for m, row in SCORES.items()
        for iid, s in zip(suite.ids, row)
    ]
    assert RunResult.from_records(suite, records).scores == SCORES


def test_from_records_reports_missing_results():
    suite = Suite.from_dicts(ITEMS)
    records = [{"model": "m", "item_id": "always-pass", "passed": True}]
    with pytest.raises(ValueError, match="no result for 4 item"):
        RunResult.from_records(suite, records)


def test_from_records_rejects_unknown_item_id():
    suite = Suite.from_dicts(ITEMS)
    with pytest.raises(ValueError, match="unknown item id"):
        RunResult.from_records(suite, [{"model": "m", "item_id": "ghost", "passed": True}])


# ------------------------------------------------------------------ analysis

def test_analysis_finds_both_kinds_of_dead_item():
    report = analyze(build())
    dead = {s.item.id for s in report.dead_items()}
    assert dead == {"always-pass", "never-pass"}


def test_dead_items_get_different_diagnoses():
    report = analyze(build())
    by_id = {s.item.id: s for s in report.stats}
    assert "Every model passes" in by_id["always-pass"].diagnosis
    assert "grader" in by_id["never-pass"].diagnosis


def test_wasted_fraction_is_two_of_five():
    assert analyze(build()).wasted_fraction == pytest.approx(0.4)


def test_signal_ratio_is_the_complement():
    assert analyze(build()).signal_ratio == pytest.approx(0.6)


def test_best_item_ranks_first():
    report = analyze(build())
    assert report.ranked()[0].item.id == "splits-clean"


def test_backwards_is_not_claimed_when_four_models_cannot_establish_it():
    """With 4 models the most extreme split is 1 of C(4,2)=6, so p >= 1/6.

    No item can clear a 0.05 bar. The honest output is an empty list plus a
    warning that the cohort is too small - not a confident accusation.
    """
    scores = dict(SCORES)
    scores["strong-model"] = [1, 0, 1, 1, 0]
    scores["poor-model"] = [1, 0, 0, 0, 1]
    report = analyze(RunResult(Suite.from_dicts(ITEMS), scores))

    assert report.backwards_items() == []
    assert report.backwards_detectable() is False

    # ...but the item is still on the shortlist, pointing the wrong way.
    suspicious = [s.item.id for s in report.suspicious_items()]
    assert "most-pass" in suspicious
    stat = next(s for s in report.stats if s.item.id == "most-pass")
    assert stat.point_biserial < 0.0
    assert stat.backwards_p is not None and stat.backwards_p > 0.05
    assert "not by more than noise" in stat.diagnosis


def test_backwards_item_is_detected_once_there_are_enough_models():
    """The same broken item, with a cohort big enough to convict it."""
    n = 12
    items = [{"id": f"fair-{i:02d}"} for i in range(20)] + [{"id": "broken"}]
    models = [f"m{i:02d}" for i in range(n)]  # m00 strongest
    scores = {}
    for rank, m in enumerate(models):
        fair = [1 if rank <= i else 0 for i in range(20)]
        scores[m] = fair + [1 if rank >= n // 2 else 0]  # only the weak half pass

    report = analyze(RunResult(Suite.from_dicts(items), scores))
    assert report.backwards_detectable() is True
    assert [s.item.id for s in report.backwards_items()] == ["broken"]

    stat = next(s for s in report.stats if s.item.id == "broken")
    assert stat.backwards_significant
    assert stat.backwards_p < 0.01
    assert "grader" in stat.diagnosis


def test_analysis_refuses_a_single_model():
    with pytest.raises(ValueError, match="at least 2 models"):
        analyze(RunResult(Suite.from_dicts(ITEMS), {"solo": [1, 0, 1, 0, 1]}))


def test_separates_distinguishes_real_gaps_from_noise():
    report = analyze(build())
    # strong-model (4) vs poor-model (1): gap of 3, comfortably above the noise
    assert report.separates("strong-model", "poor-model") is True
    # good-model (3) vs weak-model (2): gap of 1, inside the measurement error
    assert report.separates("good-model", "weak-model") is False


def test_separates_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        analyze(build()).separates("strong-model", "imaginary")


def test_min_real_gap_rejects_unsupported_confidence():
    with pytest.raises(ValueError, match="confidence must be one of"):
        analyze(build()).min_real_gap(confidence=0.75)


def test_higher_confidence_demands_a_bigger_gap():
    report = analyze(build())
    assert report.min_real_gap(0.99) > report.min_real_gap(0.95) > report.min_real_gap(0.80)


def test_model_totals_are_correct():
    report = analyze(build())
    assert report.model_totals == {
        "strong-model": 4, "good-model": 3, "weak-model": 2, "poor-model": 1,
    }


# ------------------------------------------------------------------- reports

def test_text_report_leads_with_what_to_act_on():
    out = to_text(analyze(build()))
    assert "ITEMWISE SUITE REPORT" in out
    assert "DEAD ITEMS" in out
    assert "always-pass" in out
    assert "NOT DISTINGUISHABLE" in out


def test_html_report_is_self_contained_and_escaped():
    items = list(ITEMS)
    items[0] = {"id": "<script>alert(1)</script>", "prompt": "x"}
    scores = {m: list(r) for m, r in SCORES.items()}
    report = analyze(RunResult(Suite.from_dicts(items), scores))
    out = to_html(report, title="Injection <test>")

    assert out.startswith("<!doctype html>")
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out
    assert "Injection &lt;test&gt;" in out
    # no external requests at all
    for scheme in ("http://", "https://", "//cdn"):
        assert scheme not in out
