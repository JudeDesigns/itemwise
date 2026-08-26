# itemwise on SWE-bench Verified

Everything behind the claims in [FINDINGS.md](FINDINGS.md), as code you can run.

```bash
python3 analyse.py     # every number in FINDINGS.md          (~1 second)
python3 validate.py    # the independent checks               (~15 seconds)
python3 fetch.py       # refresh matrix.json from upstream    (needs network)
```

Standard library only, same as itemwise itself. Run them from this directory.

## Where the data comes from

Every submission to the SWE-bench leaderboard publishes, in the public
[`swe-bench/experiments`](https://github.com/swe-bench/experiments) repository,
exactly which of the 500 SWE-bench Verified instances it resolved. That is a
per-model, per-item pass/fail matrix over a benchmark the industry watches
closely — and there are very few of those published anywhere.

`fetch.py` does a sparse, blobless clone (about 5 MB, not the full 2 GB), reads
`evaluation/verified/*/results/results.json` for all 134 submissions, and writes
`matrix.json`. That file is committed, so `analyse.py` and `validate.py` run
offline.

Two details worth knowing, both handled in `fetch.py`:

- **`no_generation` and `no_logs` count as unresolved**, which is how the
  leaderboard itself scores a submission.
- **Instances nobody ever solved appear in no `resolved` list at all**, so a
  naive union finds 468 items, not 500. They are also the purest dead items in
  the suite, and dropping them would understate the waste. The per-repository
  denominators in `resolved_by_repo.json` say how many are missing from each
  repository; they sum to exactly 500, and `fetch.py` fails loudly if they ever
  don't.

## Why validate.py exists

itemwise's statistics come from classical test theory. A conclusion that only
holds under that model is not worth publishing, so both headline claims are
re-derived by methods sharing none of its assumptions: a randomisation that
holds item difficulties and system totals fixed, and a bootstrap over instances.

The first of those is how the original "37 broken graders" finding was killed —
before launch rather than after. `validate.py` reproduces that kill in fifteen
seconds, and the fix it forced is in [`../../CHANGELOG.md`](../../CHANGELOG.md).

## Caveats

These are restated in FINDINGS.md and belong in anything written from this.

- Submissions are single runs. The standard error here bounds measurement error
  over the item universe, not run-to-run variance. Both matter; only the first
  is bounded.
- Submissions span October 2023 to December 2025, across different harness
  versions and different contamination exposure. They are not independent draws
  from one population.
- The instances are not independent: 231 of the 500 come from `django/django`.
- Cronbach's alpha across all 134 systems is 0.994, inflated by the sheer
  ability spread. The top-25 figure of 0.804 is the honest one.
