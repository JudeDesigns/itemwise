# itemwise on SWE-bench Verified — findings
Run 2026-08-26. Re-run against itemwise 0.2.0 after the backwards-detector fix.
Data: `swe-bench/experiments`, `evaluation/verified/*/results/results.json` —
134 public submissions, per-instance resolved/unresolved over all 500 items.

## Method
- Item = one SWE-bench Verified instance. Score = 1 if in the submission's `resolved` list.
- `no_generation` / `no_logs` count as unresolved, matching leaderboard scoring.
- Union of resolved across all 134 submissions = 468 instances. The remaining 32 are never
  solved by anyone; per-repo denominators from `resolved_by_repo.json` (mode across
  submissions; sums to exactly 500) place them without needing the HF dataset.
- Analysis: itemwise 0.2.0, `analysis/swebench/analyse.py`.
- Validation: `analysis/swebench/validate.py` - curveball randomisation preserving
  item difficulties and system totals, plus a bootstrap over instances. Both are
  standard library only and rerun in about fifteen seconds.

## Finding 1 — the suite has mostly stopped measuring anything  [SOLID]
Dead = every system in the cohort solves it, or none do. Zero information either way.

| cohort  | dead       | %     | everyone solves | nobody solves | SEM  | min real gap (95%) |
|---------|------------|-------|-----------------|---------------|------|--------------------|
| top 5   | 394/500    | 78.8% | 334             | 60            | 4.54 | 2.52 pp            |
| top 10  | 336/500    | 67.2% | 285             | 51            | 5.16 | 2.86 pp            |
| top 25  | 270/500    | 54.0% | 230             | 40            | 5.50 | 3.05 pp            |
| top 50  | 181/500    | 36.2% | 146             | 35            | 6.01 | 3.33 pp            |
| top 100 |  53/500    | 10.6% |  21             | 32            | 7.13 | 3.95 pp            |
| all 134 |  32/500    |  6.4% |   0             | 32            | 7.47 | 4.14 pp            |

Among the top 25, the 230 non-dead items reproduce the full-500 ranking **exactly**
(Spearman +1.000). The 88 informative items (r_pb >= 0.2) give +0.942.

This is arithmetic over published results, not inference. It cannot be wrong.

## Finding 2 — the top of the leaderboard is a coin flip  [SOLID, two methods]
Classical test theory: all 24 of 24 adjacent pairs in the top 25 fall inside the 95%
minimum real gap of 15.24 items (3.05 pp). Eight systems are tied with #1.

Bootstrap over items, P(system is truly #1):
  35.9%  sonar-foundation-agent + claude-opus-4-5   (396/500, 79.2%)
  33.8%  livesweagent + claude-opus-4-5             (396/500, 79.2%)
  25.5%  trae + doubao-seed-code                    (394/500, 78.8%)
   2.0%  livesweagent + gemini-3-pro-preview        (387/500, 77.4%)
  <1%    everything else

Two independent methods, same conclusion.

## Finding 3 — NO evidence of broken graders  [was the original hypothesis; killed]
itemwise 0.1 flagged 37 backwards items in the top-25 cohort.

Curveball null, preserving item difficulties AND system totals (200 reps,
`validate.py`): null count 44.7 +/- 4.0, range 35-56. Observed 37.
p(null >= observed) = 0.99.

The observed count sits *below* chance. All 37 were noise.

### The fix, shipped as 0.2.0
The bug was a fixed threshold: `point_biserial < -0.05`, whose false-positive rate rises
with cohort homogeneity — exactly the regime a team comparing its own frontier models is
in. (itemwise 0.1 already used the *corrected*, item-removed point-biserial; that part
was right.)

0.2.0 tests each item against the null that it is unrelated to ability. Because the
rest-score excludes the item itself, permuting which models passed cannot move what the
item is correlated against, so the p-value is exact: of the C(n,k) ways k of n models
could have passed, how many look at least this backwards. Benjamini-Hochberg then
corrects across the suite.

Re-run under 0.2.0: **zero** backwards items at every cohort size (5/10/25/50/134),
agreeing with the independent randomisation. The demo's three planted broken graders are
still caught at p = 0.0025, so the test has power, not just calibration.

Known limitation, stated in the docs rather than hidden: the smallest reachable p-value
is 1/C(n,k), so below ~8 models no item can be called backwards at all. `backwards_detectable()`
reports that, and the top-5 SWE-bench cohort correctly returns False.

## Caveats for any public writeup
- Submissions are single runs. SEM bounds measurement error over the item universe, not
  run-to-run variance. Both matter; this only bounds the first.
- Submissions span Oct 2023 - Dec 2025, different harness versions, different
  contamination exposure. Not independent draws from one population.
- Items are not independent: 231 of 500 instances come from django/django.
- Cronbach's alpha over all 134 systems is 0.994, inflated by the ability spread. The
  top-25 alpha of 0.804 is the honest number.
