# The top of SWE-bench Verified is a statistical tie

Three systems have a defensible claim to first place on SWE-bench Verified. Below
them, the ordering of the top 25 is noise: all 24 adjacent pairs are statistical
ties.

And 54% of the benchmark carries no information about those systems at all.

Here is the arithmetic, and [the code to reproduce
it](https://github.com/JudeDesigns/itemwise/tree/main/analysis/swebench).

*Every number below reruns in about fifteen seconds from `analysis/swebench/`,
standard library only.*

---

## Where the numbers come from

Every submission to the SWE-bench leaderboard publishes, in the public
[`swe-bench/experiments`](https://github.com/swe-bench/experiments) repository,
exactly which of the 500 Verified instances it resolved. There are 134 of them.

That is a per-system, per-item pass/fail matrix over a benchmark the industry
watches closely — and there are very few of those published anywhere. It is also,
structurally, an exam: the instances are questions, the systems are the candidates
sitting it. Which means about a century of psychometrics applies directly, and
almost none of it has been applied.

Two details, both handled in `fetch.py`. `no_generation` and `no_logs` count as
unresolved, matching how the leaderboard scores a submission. And instances that
nobody has ever solved appear in no `resolved` list at all, so a naive union finds
468 items rather than 500 — those 32 are the purest dead items in the suite, and
dropping them would understate the problem rather than overstate it.

## Finding 1: over half the benchmark has stopped measuring

An instance that every system solves tells you nothing about which system is
better. Neither does one that nobody solves. In test theory these have zero
variance and therefore zero information — they cannot correlate with ability,
however carefully they were written.

Among the top 25 submissions:

- **230 of 500** instances are solved by every one of them
- **40** are solved by none
- **270 of 500 — 54% — carry no information** about the relative quality of these
  systems

It gets sharper as the cohort narrows:

| cohort  | dead items | %     | everyone solves | nobody solves |
|---------|-----------:|------:|----------------:|--------------:|
| top 5   |    394/500 | 78.8% |             334 |            60 |
| top 10  |    336/500 | 67.2% |             285 |            51 |
| top 25  |    270/500 | 54.0% |             230 |            40 |
| top 50  |    181/500 | 36.2% |             146 |            35 |
| top 100 |     53/500 | 10.6% |              21 |            32 |
| all 134 |     32/500 |  6.4% |               0 |            32 |

The benchmark still discriminates fine across the full field — it separates a 2023
RAG baseline from a 2025 agent without difficulty. What it has largely stopped
doing is separating the frontier from itself, which is the only comparison anyone
making a purchasing decision actually cares about.

And the informative half is doing all the work. Restricting to the 230 non-dead
items reproduces the full-500 top-25 ranking **exactly** — Spearman +1.000. Not
approximately. Half of SWE-bench Verified could be deleted tomorrow and the top-25
leaderboard would not move by a single position.

That is not a statistical inference. It is arithmetic over published results.

## Finding 2: the top of the leaderboard cannot be ordered

Every measurement has error, and a difference smaller than that error is not a
difference. Classical test theory gives you the standard error of measurement from
the suite's internal consistency, and from that the smallest gap between two
scores that is not plausibly noise.

For the top 25, that threshold is **15.24 instances — 3.05 percentage points** at
95% confidence.

**All 24 of the 24 adjacent pairs fall inside it.** Eight systems are statistically
tied with first place.

Because that conclusion depends on a model, I re-derived it a second way that
shares none of the same assumptions — a bootstrap over instances, asking how often
each system comes out on top when you resample which problems were in the
benchmark:

```
P(system is truly #1)
  35.9%   sonar-foundation-agent + claude-opus-4-5    396/500   79.2%
  33.8%   livesweagent + claude-opus-4-5              396/500   79.2%
  25.5%   trae + doubao-seed-code                     394/500   78.8%
   2.0%   livesweagent + gemini-3-pro-preview         387/500   77.4%
  <1%     everything else
```

Two methods, same answer. Three systems have a real claim to the top and the gaps
below them are not measurable with this instrument.

This matters commercially. If you are choosing a coding agent on leaderboard rank,
and the top eight are tied, you are choosing on noise.

## The finding I had to throw away

I went looking for broken graders — instances where weaker systems succeed more
often than stronger ones, which usually means the grading logic rewards something
other than the intended capability.

The first version of my tool found 37 of them in the top-25 cohort. It was the
result I most wanted, because a concrete list of broken items is a far more
actionable finding than "your benchmark is saturating."

So I built a null model to check it: a curveball randomisation that shuffles the
matrix while holding both item difficulties and system totals fixed, so the only
thing destroyed is any genuine relationship between an item and ability. If 37 is
a real signal, the null should rarely produce that many.

The null produced **44.7 ± 4.0** of them, across a range of 35 to 56.

The observed count of 37 sits *below* chance. `p(null ≥ observed) = 0.99`. All 37
were noise, and my tool had a bug.

The bug was a fixed threshold — flag anything whose corrected point-biserial falls
below −0.05 — whose false-positive rate climbs as a cohort becomes more
homogeneous. Which is exactly the regime you are in when comparing frontier
systems to each other. The threshold is the kind of thing that looks principled,
is easy to write, and is wrong in precisely the case you most want to use it.

The fix, shipped as 0.2.0, tests each item against the null that it is unrelated
to ability. Because the rest-score excludes the item itself, permuting which
systems passed cannot move what the item is being correlated against, so the
p-value is exact: of the C(n,k) ways that k of n systems could have passed, how
many look at least this backwards. Benjamini–Hochberg then corrects across the
suite.

Re-run under 0.2.0: **zero** backwards items at every cohort size, agreeing with
the independent randomisation. Three deliberately broken graders planted in the
test suite are still caught at p = 0.0025, so the test has power and not merely
calibration.

There is a real limit, documented rather than hidden: the smallest reachable
p-value is 1/C(n,k), so below about 8 systems no item can be flagged at all. The
function says so instead of guessing, and the top-5 SWE-bench cohort correctly
returns "cannot answer."

**So: I have no evidence that SWE-bench Verified has broken graders.** I am
including this because it was my headline for about a day, and because findings 1
and 2 should be read by someone who knows I killed the one I liked most.

## What this does not mean

- **SWE-bench Verified is not a bad benchmark.** It discriminates well across the
  full field. It has been saturated at the top by rapid progress, which is what
  success looks like.
- **These are single runs.** The standard error here bounds measurement error over
  the item universe, not run-to-run variance. Both matter and only the first is
  bounded — the true uncertainty is *larger* than what I report, not smaller.
- **The submissions are not one population.** They span October 2023 to December
  2025, across different harness versions and different contamination exposure.
- **The instances are not independent.** 231 of the 500 come from `django/django`
  alone, so the effective number of independent items is well below 500.
- **Cronbach's alpha across all 134 systems is 0.994**, which is inflated by the
  sheer spread of ability. The top-25 figure of 0.804 is the honest one.

## What I think should change

**Report intervals, not ranks.** A leaderboard that prints positions without a
resolution is presenting noise as information. If eight systems are tied, say so.

**Retire saturated items, or split them out.** An instance solved by all 25 leading
systems belongs in a regression suite, not a discrimination benchmark. Keeping it
costs everyone compute on every evaluation and changes no conclusion.

**Publish per-item results.** SWE-bench does this and it is why this analysis was
possible at all. Almost no other major benchmark does. Aggregate scores hide
item-level pathology completely.

## The tool

The analysis uses [itemwise](https://github.com/JudeDesigns/itemwise), a small
zero-dependency Python library that applies classical test theory to evaluation
suites. It reads a pass/fail matrix from whatever harness you already use and
reports which cases carry signal, which are dead, which are backwards, and whether
two systems are actually distinguishable.

Everything above reruns from
[`analysis/swebench/`](https://github.com/JudeDesigns/itemwise/tree/main/analysis/swebench):
`analyse.py` for every number in the post, `validate.py` for the independent
checks including the randomisation that killed my own finding. Standard library
only, about fifteen seconds.

If you run your own evals and have three or more systems' results, it will tell you
what fraction of your suite has stopped measuring. In my experience the number is
higher than people expect.

I would genuinely like to be shown wrong on any of this. The code is there to make
that easy.
