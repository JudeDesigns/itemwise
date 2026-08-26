# itemwise

**Find the eval cases that actually tell you something.**

Most LLM eval suites are mostly dead weight. Cases every model passes. Cases every
model fails. Cases where the grader is quietly broken and the *weaker* model scores
higher. You pay to run all of them on every commit, and they move no decision.

`itemwise` applies **classical test theory** — the item analysis used to build
professional certification exams — to your eval suite, and tells you which cases
carry signal and which are costing you money to learn nothing.

No dependencies. Bring your own harness.

---

## The problem, concretely

Here is `itemwise` on a realistic 40-item suite across 5 models:

```
  items                40
  models compared      12
  Cronbach's alpha     0.860  (good)
  std error of meas.   1.93 items
  carrying signal      48%
  dead weight          52% of every run

BACKWARDS ITEMS (3) - fix these first
  Weaker models pass these more often than stronger ones, by more
  than chance explains. That is nearly always a broken grader.
  Tested against the null that the item is unrelated to ability,
  corrected across all 40 items at FDR 0.05.
  format-00                        r_pb=-0.845  passed=0.42  p=0.00253
  format-01                        r_pb=-0.845  passed=0.42  p=0.00253
  format-02                        r_pb=-0.845  passed=0.42  p=0.00253

DEAD ITEMS (18) - retire or rewrite
  every model passes (13): smoke-00 ... reasoning-01
  no model passes (5) - check the grader before the prompt: frontier-00 ...

MODEL TOTALS
  frontier-a    32 / 40
  frontier-b    32 / 40
  mid-tier      27 / 40
  small-open    23 / 40
  tiny-open     17 / 40

  A gap must exceed 5.43 items to be real (95% conf).
    frontier-b (32) vs mid-tier (27):  gap 5  ->  NOT DISTINGUISHABLE
    small-open (23) vs tiny-open (17):  gap 6  ->  real
```

Three things fell out of that, none of which a pass-rate dashboard would show you:

1. **55% of every run is wasted.** Nineteen items carry zero information and three
   are actively misleading. Drop them and you cut inference cost by more than half
   with no loss of conclusion.
2. **Three graders are broken.** The `format-*` items reward a shortcut that smaller
   models take more often. Someone would have shipped on that signal.
3. **Your leaderboard is lying.** `frontier-b` sits five points above `mid-tier`, and
   that gap is *inside the measurement error*. On this suite those models are tied.
   Ordering them is storytelling.

---

## Install

```bash
pip install itemwise
```

Python 3.10+. Zero runtime dependencies.

---

## Quickstart

`itemwise` doesn't run your models. It reads pass/fail outcomes you already have,
from whatever harness you already use.

```python
from itemwise import Suite, RunResult, analyze, to_text

suite = Suite.from_jsonl("evals.jsonl")   # {"id": "...", "prompt": "..."} per line

result = RunResult(suite, {
    "gpt-x":     [1, 0, 1, 1, 0],    # 1 = passed, in suite order
    "claude-y":  [1, 0, 1, 0, 0],
    "llama-z":   [1, 0, 0, 0, 0],
})

report = analyze(result)
print(to_text(report))
```

Already have flat records? That works too:

```python
result = RunResult.from_records(suite, [
    {"model": "gpt-x", "item_id": "arith-001", "passed": True},
])
```

Then act on it:

```python
report.dead_items()        # zero information - retire or rewrite
report.backwards_items()   # weak models beat strong ones, beyond chance
report.suspicious_items()  # point the wrong way, not yet provable
report.backwards_detectable()  # do you have enough models to even ask?
report.ranked()            # every item, most informative first
report.wasted_fraction     # what share of each run informs nothing
report.separates("a", "b") # is the gap between two models real?
```

And for a shareable artefact:

```python
from itemwise import to_html
open("report.html", "w").write(to_html(report, title="Nightly suite"))
```

Self-contained HTML — no CDN, no network, works offline and in air-gapped CI.

---

## Trimming a suite in CI

```python
keep = [s.item.id for s in report.stats
        if s.verdict in ("acceptable", "strong")]
```

Run the full suite weekly to recompute item statistics; run only `keep` on every
commit. Same conclusions, a fraction of the tokens.

Guard against decay:

```python
def test_suite_is_still_healthy():
    report = analyze(load_last_full_run())
    assert not report.backwards_items(), "a grader has broken"
    assert report.wasted_fraction < 0.30, "suite is filling up with dead items"
```

---

## What the statistics mean

| Statistic | Symbol | Reading |
|---|---|---|
| **Difficulty** | `p` | Proportion of models that passed. `1.0` = everyone passed, `0.0` = nobody did. Both extremes carry zero information. |
| **Discrimination index** | `D` | Top 27% pass rate minus bottom 27%. `≥0.40` excellent, `0.20–0.29` marginal, negative means the item is backwards. |
| **Point-biserial** | `r_pb` | Correlation between passing this item and overall score. More trustworthy than `D` because it uses every model, not just the extremes. **Corrected by default** — the item's own contribution is removed from the total, since otherwise every item correlates with itself. |
| **Backwards p-value** | `p` | For each item: if it were unrelated to model ability, how often would the models that passed it score this badly on the rest of the suite? Counted exactly where possible. Corrected across the suite with Benjamini–Hochberg before anything is called backwards. |
| **Cronbach's alpha** | `α` | Internal consistency. `0.80–0.89` good; `≥0.90` may mean near-duplicate items; `<0.60` means the suite is measuring several unrelated things and should be split. |
| **Standard error** | `SEM` | How much of a score is noise. Drives `min_real_gap()` and `separates()`. |

Verdicts assigned per item: `dead` · `backwards` · `weak` · `acceptable` · `strong`.

## Run on a real benchmark

[`analysis/swebench/`](analysis/swebench/) applies itemwise to SWE-bench Verified,
using the per-instance results all 134 public leaderboard submissions publish.

```bash
cd analysis/swebench && python3 analyse.py
```

Among the top 25 systems, 270 of the 500 instances are dead — every one of them
solves it, or none do. Among the top 5, 394 are. The 230 that still carry
information reproduce the full 500-instance ranking exactly. And all 24 adjacent
pairs in the top 25 sit inside the suite's own 95% error bar: a bootstrap over
instances puts three different systems at 36%, 34% and 23% likely to be
genuinely first.

`validate.py` in that directory re-derives both claims by methods that share none
of classical test theory's assumptions. It is also what killed this library's
first published finding — see [CHANGELOG.md](CHANGELOG.md).

---

### On calling an item backwards

A negative `r_pb` is not evidence of a broken grader. On a suite of 200 items
compared across 20 models with nothing wrong anywhere, dozens of items will
correlate negatively with ability by chance alone — and a fixed threshold like
`r_pb < -0.05` will report every one of them as a finding.

So itemwise tests it. The item's rest-score excludes the item itself, so
permuting which models passed cannot change what it is correlated against, and
the null distribution can be counted exactly: of the `C(n, k)` ways `k` of your
`n` models could have passed, how many look at least this backwards? The
resulting p-values are then corrected across the whole suite.

This has a consequence worth knowing before you rely on it. With `n` models the
smallest reachable p-value is `1 / C(n, k)`, so **below about eight models no
item can be called backwards at all**, however inverted it looks. `report.backwards_detectable()`
tells you whether you are in that regime; when it returns `False`, an empty
backwards list means *cannot tell*, not *nothing wrong*. `report.suspicious_items()`
gives you the shortlist to revisit once you have run more models.

Raise or lower the bar with `analyze(result, backwards_fdr=0.10)`.

---

## Why classical test theory

Because the problem is not new. Psychometricians have spent a century on exactly this
question — *which questions on this test actually distinguish people who know the
material from people who don't* — and the answers are well established, cheap to
compute, and directly transferable. An eval suite is a test. Models are the examinees.
Everything else follows.

What the field figured out long ago and LLM evaluation is rediscovering the hard way:

- An item everyone passes measures nothing, however good it looks in a dashboard.
- Aggregate scores hide item-level pathology. Suites rot silently.
- **Every measurement has error, and a difference smaller than that error is not a
  difference.** Most model leaderboards report rankings well inside their own noise.

---

## What this is not

- **Not a harness.** It doesn't call models, grade outputs, or manage prompts. Use
  promptfoo, braintrust, inspect, a pytest loop — anything. `itemwise` reads results.
- **Not a benchmark.** It has no opinion on what you should test.
- **Not IRT.** Classical test theory only, for now — the statistics are sample-dependent
  and need ≥3 models to be meaningful (≥5 is comfortable). Item response theory would
  lift that limit and is on the roadmap.

---

## Roadmap

- [ ] Distractor analysis for multiple-choice eval formats
- [ ] Item response theory (2PL) for sample-independent difficulty
- [ ] Suite trimming that preserves construct coverage, not just top-`r_pb` items
- [ ] Drift detection: flag items whose statistics move between runs
- [ ] CLI: `itemwise report results.jsonl`

See [docs/why.md](docs/why.md) for the reasoning behind the approach.

Issues and PRs welcome, particularly from anyone with real psychometrics training.

---

## Development

```bash
git clone https://github.com/JudeDesigns/itemwise
cd itemwise
pip install -e ".[dev]"
pytest                              # 44 tests, no network, runs in under a second
python examples/demo.py             # end-to-end on synthetic data
```

Every statistic is unit-tested against values computed by hand, written out in the
test file so you can check the arithmetic yourself rather than trusting it.

---

## License

MIT — see [LICENSE](LICENSE).
