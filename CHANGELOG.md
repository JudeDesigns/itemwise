# Changelog

## 0.2.0

**Breaking:** an item is no longer called backwards on a threshold. It has to
survive a significance test.

- `stats.backwards_p_value()` — exact one-tailed p-value for "this item is
  unrelated to model ability", counted over the `C(n, k)` ways the passing set
  could have fallen. Falls back to a normal approximation only when exact
  counting is too expensive; `backwards_p_method()` reports which was used.
- `stats.benjamini_hochberg()` — false-discovery-rate correction, applied across
  the whole suite before any item is reported as backwards.
- `analyze(result, backwards_fdr=0.05)` — the bar is now a parameter you set,
  not a constant hidden in the source.
- `ItemStats.backwards_p`, `ItemStats.backwards_significant` — new fields.
- `Report.backwards_detectable()` — whether the cohort is large enough for the
  question to be answerable at all. Below roughly eight models it is not, and an
  empty backwards list means "cannot tell" rather than "nothing wrong".
- `Report.suspicious_items()` — items pointing the wrong way that the evidence
  cannot yet convict. A shortlist, explicitly not a finding.
- Reports say which of those two situations you are in, rather than printing an
  empty section either way.
- The demo runs 12 models instead of 5, because 5 is below the floor above.

Why: the old `r_pb < -0.05` rule has a false-positive rate that rises with
cohort homogeneity. On 134 public SWE-bench Verified submissions it flagged 37
"broken graders" among the top 25 systems, where a difficulty- and
total-preserving null produces 44.7 +/- 4.0 by chance. Every one was noise. See
`docs/why.md`.

## 0.1.0

Initial release. Difficulty, discrimination index, corrected point-biserial,
Cronbach's alpha, standard error of measurement, minimum real gap between
models. Text and HTML reports. Zero dependencies.
