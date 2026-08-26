# Why itemwise exists

*An eval suite is a test. Models are the people taking it. Almost everything that
follows is a consequence of taking that seriously.*

---

## The problem

You have an eval suite. Three hundred cases, run on every commit, graded pass/fail,
rolled up into a number that goes on a dashboard.

That number is the average of three hundred questions you have never individually
audited. So consider what is probably in there:

- **Cases every model passes.** Smoke tests from the first week that nothing has
  outgrown. They cost tokens on every run and change no decision.
- **Cases no model passes.** Sometimes genuinely hard. More often a broken grader,
  a malformed expected answer, or a prompt that asks something impossible.
- **Cases where the grader is subtly wrong** — rewarding a format, a refusal, or a
  shortcut. These are the dangerous ones, because they look like normal cases and
  quietly push your aggregate the wrong way.
- **Near-duplicates** that triple-count one capability and make your suite look
  more reliable than it is.

Your dashboard cannot show you any of this. A pass rate is an average, and averages
hide the distribution they came from.

## The observation

This problem is not new. It is not even close to new.

Anyone who has built a professional certification exam — AWS, CompTIA, a medical
board, a university final — has faced exactly this question: *which of these
questions actually distinguish people who know the material from people who
don't?* The field that answers it is **psychometrics**, and the specific toolkit is
**classical test theory**. It has been refined since the 1930s, the mathematics is
elementary, and it transfers to LLM evaluation without modification.

The translation is direct:

| Psychometrics | LLM evaluation |
|---|---|
| examinee | a model, or a model configuration |
| test item | an eval case |
| item score | 1 if the model passed, 0 if it failed |
| total score | how many cases that model passed |
| item analysis | **what this library does** |

That is the entire conceptual leap. Everything below is standard, century-old
statistics applied to a new population of examinees.

---

## The four statistics

### Difficulty — `p`

The proportion of models that passed an item.

```
p = 1.00   every model passed
p = 0.00   no model passed
p = 0.50   half did
```

Confusingly but conventionally, **higher `p` means an easier item**.

An item at `p = 1.00` or `p = 0.00` has **zero variance**, and an item with zero
variance cannot correlate with anything. It contributes literally no information
about relative model quality, no matter how thoughtfully it was written. It is a
line item on your inference bill and nothing else.

### Discrimination index — `D`

Rank the models by total score. Take the top 27% and the bottom 27%. Subtract the
bottom group's pass rate from the top group's.

```
D = p_upper - p_lower        range [-1, +1]
```

The conventional reading (Ebel, 1965):

| `D` | Reading |
|---|---|
| >= 0.40 | excellent |
| 0.30 – 0.39 | good |
| 0.20 – 0.29 | marginal, worth revising |
| < 0.20 | poor — not separating strong from weak |
| **negative** | **backwards — investigate immediately** |

Why 27%? Kelley (1939) showed it maximises the separation between groups relative
to sampling error under a normal distribution. It is a genuine optimum, not a
convention someone picked.

### Point-biserial correlation — `r_pb`

The more trustworthy statistic, because it uses **every** model rather than only
the extremes.

```
r_pb = (M1 - M0) / SD_total * sqrt(p * q)
```

where `M1` and `M0` are the mean total scores of models that passed and failed the
item, and `q = 1 - p`.

**itemwise corrects this by default**, subtracting the item's own contribution from
the total before correlating. Without that correction every item is partly
correlated with itself, which inflates the statistic — severely on short suites.
Many tools skip the correction. It makes results look better and it is wrong.

### Reliability — Cronbach's alpha

```
alpha = k/(k-1) * (1 - sum(item variances) / total variance)
```

How internally consistent the suite is — whether its items agree with each other
about which models are good.

| `alpha` | Reading |
|---|---|
| >= 0.90 | very high — check for near-duplicate items |
| 0.80 – 0.89 | good |
| 0.70 – 0.79 | acceptable |
| 0.60 – 0.69 | questionable |
| < 0.60 | measuring several unrelated things |

**A low alpha is not automatically a fault.** A suite that deliberately spans
reasoning, code generation and safety *should* score low — those are different
constructs. That is a signal to split it into sub-suites and report them
separately, not to delete items. Alpha is a question, not a verdict.

---

## The part most people get wrong

Cronbach's alpha gives you the **standard error of measurement**:

```
SEM = SD_total * sqrt(1 - alpha)
```

This is how much of an observed score is plausibly noise. And it has a
consequence that is uncomfortable and unavoidable:

> **A score difference smaller than the measurement error is not a difference.**

To compare two models you need the standard error of the *difference*, which is
`SEM * sqrt(2)`, scaled by your confidence level:

```python
report.min_real_gap(confidence=0.95)   # 5.43 items on the demo suite
report.separates("model-a", "model-b") # True / False
```

On the 40-item demo suite, a model scoring **32** and a model scoring **27** are
**statistically indistinguishable**. A five-point gap. A leaderboard would print
them in order, someone would screenshot it, and a purchasing decision would follow.

This is not pedantry. Published model leaderboards routinely report orderings
well inside their own noise, and nobody computes the interval, because reporting a
rank is easy and reporting an interval is not.

---

## Worked example

Five eval cases, four models. Small enough to check by hand.

```python
from itemwise import Suite, RunResult, analyze

suite = Suite.from_dicts([
    {"id": "json-format"},   {"id": "obscure-api"},  {"id": "multi-step"},
    {"id": "edge-case"},     {"id": "basic-logic"},
])

result = RunResult(suite, {
    "big-model":    [1, 0, 1, 1, 1],   # 4/5
    "medium-model": [1, 0, 1, 0, 1],   # 3/5
    "small-model":  [1, 0, 0, 0, 1],   # 2/5
    "tiny-model":   [1, 0, 0, 0, 0],   # 1/5
})

report = analyze(result)
```

```
eval case          p    r_pb   verdict
----------------------------------------------
json-format     1.00  +0.000   dead
obscure-api     0.00  +0.000   dead
multi-step      0.50  +0.707   strong
edge-case       0.25  +0.522   strong
basic-logic     0.75  +0.522   strong

dead weight        40%
reliability        0.625
min. real gap      1.90 items
```

Two of five cases inform nothing — 40% of every run. And `medium-model` (3) versus
`small-model` (2) is a gap of 1 against a minimum real gap of 1.90: **those models
are tied.** The ranking between them is an artefact.

On a realistic 40-item suite in `examples/demo.py`, the dead weight is **55%** and
three items have backwards graders.

---

## Objections

**"My eval cases are all deliberately chosen. None are dead."**
Then the analysis costs you nothing and confirms it. In practice, suites accumulate:
smoke tests from week one survive years past the point anything fails them, and
nobody deletes a passing test.

**"Everything passing is good news — it means our model works."**
It means your *suite* has stopped measuring. A test where everyone scores 100% has
no measurement left in it. That is the moment to write harder items, not to
celebrate.

**"We only have one model."**
Then item analysis cannot help you, and itemwise will refuse rather than pretend.
Discrimination is inherently about *variation between examinees*. With one model,
every item looks identical. You need at least three; five is comfortable. Model
versions, checkpoints, temperature settings and prompt variants all count.

**"Isn't this just correlation analysis?"**
Structurally, yes. The contribution is not novel mathematics — it is the
**vocabulary, the interpretation bands, and the diagnostics**: knowing that a
negative point-biserial means *check your grader before you trust the finding*, and
that `alpha >= 0.90` means *look for duplicates* rather than *celebrate*. That
interpretive layer took the psychometrics field decades to establish and it is
what makes the numbers actionable.

**"Why not item response theory?"**
Because IRT needs far more data and would put a barrier in front of the first
useful result. Classical test theory is sample-dependent — your statistics describe
*your* model pool, not absolute item properties — which is a real limitation,
stated plainly. 2PL IRT is on the roadmap.

---

## What to actually do with it

1. **Fix backwards items first.** A negative `r_pb` is nearly always a broken
   grader, and a broken grader is actively misleading you right now.
2. **Retire or rewrite dead items.** Immediate, permanent cost reduction.
3. **Split the suite if alpha is low.** You are averaging unrelated constructs into
   one meaningless number.
4. **Run the full suite weekly, the informative subset per commit.**
   ```python
   keep = [s.item.id for s in report.stats
           if s.verdict in ("acceptable", "strong")]
   ```
5. **Report intervals, not ranks.** When someone asks whether the new model is
   better, answer with `separates()`. Sometimes the honest answer is "we cannot
   tell yet," and that is a far more valuable thing to be able to say than a
   confident number you cannot defend.

---

## Further reading

- Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests.
  *Psychometrika*, 16(3), 297–334.
- Kelley, T. L. (1939). The selection of upper and lower groups for the validation
  of test items. *Journal of Educational Psychology*, 30(1), 17–24.
- Ebel, R. L. (1965). *Measuring Educational Achievement.* Prentice-Hall.
- Lord, F. M., & Novick, M. R. (1968). *Statistical Theories of Mental Test
  Scores.* Addison-Wesley.

---

*Built by someone who spent three years writing technical certification assessments
before noticing that AI evaluation had the same problem and none of the tools.*
