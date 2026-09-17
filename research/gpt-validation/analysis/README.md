# GPT validation analysis

`compute_validation_stats.py` turns the four admin CSV exports of the review
service (see `../DESIGN.md`) into inter-rater reliability and GPT validity
statistics, as JSON and as a Markdown report with a draft results paragraph.
It needs only numpy and pandas; every statistic (kappa, ICC, F distribution,
Wilson interval, bootstrap) is implemented and unit-tested here.

## Usage

```bash
uv run --no-project --with pandas --with numpy python compute_validation_stats.py \
  --items items.csv --criteria criteria.csv --cues cues.csv --overall overall.csv \
  [--experts expert-1,expert-2] [--out report.json] [--markdown report.md] \
  [--bootstrap 2000 --seed 7]
```

- CSVs are read as UTF-8 with an optional BOM (`utf-8-sig`).
- `--experts`: the two expert ids. If omitted, the script uses the ids in the
  rating exports other than `admin` and stops with an error unless there are
  exactly two.
- If you pass neither `--out` nor `--markdown`, the Markdown goes to stdout.
- `--bootstrap 0` turns the bootstrap intervals off. The same seed gives the
  same output.
- Exit code 2 means bad input: missing columns, duplicate rows, or not exactly
  two experts.

Tests:

```bash
uv run --no-project --with pandas --with numpy --with pytest pytest -q
```

## Which items are analyzed

An item is included only when **all** of these hold:

1. `items.csv` `eligible` is true (status `ready`, coaching from OpenAI).
2. Each of the two experts has an `overall.csv` row with `completed` true and
   an integer `overall_score` from 0 to 3.
3. Each expert has a 0/1 step-1 answer for every criterion listed for that
   item in `criteria.csv`.
4. Each expert has a valid rating (`correct`/`partial`/`incorrect`) for every
   cue from 1 to `n_cues`.

The report lists excluded items with all their reasons, counts them by first
failing reason (`not_eligible`, `not_rated_by:<expert>`,
`not_completed_by:<expert>`, `invalid_overall_score:<expert>`,
`incomplete_step1:<expert>`, `incomplete_cue_ratings:<expert>`), and splits
ineligible items by status and coaching source. Source filenames never appear
in the output. The analysis uses `cue_rating` (`cue_score` is only checked
against it). GPT positives come from `criteria.csv` `gpt_flagged`, and the
script warns if that disagrees with `items.csv` `gpt_flagged_criteria`.

## What is reported

### 1. Inter-rater reliability

| Target | Unit | Statistics |
|---|---|---|
| Step 1 "needs improvement" | item × criterion | percent agreement, Cohen's κ, Fleiss' κ, PABAK, each expert's prevalence (Wilson CI); per criterion within skill, per skill, pooled |
| Cue rating (incorrect 0 / partial 1 / correct 2) | item × cue | percent agreement, within-one agreement, unweighted, linear- and quadratic-weighted Cohen's κ, Fleiss' κ (nominal); per skill, pooled |
| Overall score 0–3 | item | percent agreement, quadratic-weighted κ (plus unweighted and linear in the JSON), ICC(2,1) and ICC(3,1) with 95% CIs, Fleiss' κ, each expert's mean; per skill, pooled |

### 2. GPT validity

- **Detection.** Each item × criterion is one case. GPT is positive when it
  raised a cue for the criterion. Expert truth is judged under two rules:
  `both` (strict, primary: both experts ticked the criterion) and `either`
  (lenient, secondary: at least one did). For each rule the report gives
  TP/FP/FN/TN, precision, recall, specificity and accuracy with Wilson CIs, and
  F1 with a percentile bootstrap CI that resamples items. It does this pooled,
  per skill and per criterion. The adjudication block counts expert
  disagreements, which are exactly the cases where `both` and `either` differ,
  and splits them by whether GPT flagged the criterion. The JSON also has
  cluster-bootstrap CIs for precision and recall.
- **Cues.** Shares of cues rated correct by both experts, correct by at least
  one, at least partial by both, incorrect by either, and incorrect by both
  (Wilson CIs). Also the mean cue score, where each cue's score is the mean of
  the two experts on a 0–2 scale, with SD and an item-cluster bootstrap CI.
- **Overall.** Mean ± SD, median and IQR, and the distribution of each item's
  mean expert score, plus each expert's score distribution. Rates with Wilson
  CIs: acceptable (item mean ≥ 2), strictly acceptable (both ≥ 2), both gave 3,
  harmful (any expert gave 0), and both gave 0.
- **Time burden.** For each expert, the n, median, IQR, mean and SD of
  `step1_seconds`, `step2_seconds` and their sum, over included items.

### 3. Results paragraph

A draft paragraph in the Markdown and in the JSON `results_paragraph`. It is a
starting point: check it against the tables before using it.

## Formulas

- **Cohen's κ (weighted).**
  κ_w = 1 − Σ w_ij o_ij / Σ w_ij e_ij.
  Here o is the observed proportion table and e_ij is rater 1's marginal for i
  times rater 2's marginal for j. The disagreement weights are 1[i≠j] for
  unweighted, |i−j|/(k−1) for linear, and ((i−j)/(k−1))² for quadratic, with
  categories fixed at 0/1, 0–2 or 0–3. Unweighted, this is the familiar
  (p_o − p_e)/(1 − p_e). When the expected disagreement is 0 (p_e = 1, meaning
  both raters used a single category), κ is reported as `null` with a note.
- **Fleiss' κ.** For m raters per subject:
  P_i = (Σ_j n_ij² − m)/(m(m−1)), P̄ = mean P_i, p_j = Σ_i n_ij/(N m),
  P_e = Σ p_j², and κ = (P̄ − P_e)/(1 − P_e). With m = 2 this equals Scott's π,
  which is tested.
- **PABAK** = 2 p_o − 1 (binary).
- **ICC** (Shrout & Fleiss 1979; McGraw & Wong 1996). From the two-way ANOVA
  on n items × k raters, with mean squares MSR (items), MSC (raters) and MSE
  (residual):
  - ICC(2,1) = (MSR − MSE) / (MSR + (k−1)MSE + k(MSC − MSE)/n): two-way
    random effects, absolute agreement.
  - ICC(3,1) = (MSR − MSE) / (MSR + (k−1)MSE): two-way mixed, consistency.
  - ICC(3,1) CI: F = MSR/MSE, F_L = F / F_{.975}(n−1, (n−1)(k−1)),
    F_U = F · F_{.975}((n−1)(k−1), n−1), and the bounds are
    (F_{L,U} − 1)/(F_{L,U} + k − 1).
  - ICC(2,1) CI: McGraw & Wong's Satterthwaite approximation with non-integer
    df ν, the same method as R `psych::ICC`.
  - F quantiles come from bisection on a regularized incomplete beta function
    (Lentz continued fraction) implemented in the script; no bootstrap is
    involved.
  - The code handles k raters in general. It is tested with k = 4 against the
    Shrout & Fleiss 6×4 example (ICC(2,1) = .29 [.019, .76],
    ICC(3,1) = .71 [.34, .95]) and used here with k = 2.
- **Wilson interval.**
  (p̂ + z²/2n ± z √(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n), with z = 1.95996.
- **Detection metrics.** Precision = TP/(TP+FP), recall = TP/(TP+FN),
  specificity = TN/(TN+FP), F1 = 2TP/(2TP+FP+FN).
- **Cluster bootstrap.** Resample items with replacement, recompute the pooled
  confusion matrix, and take the 2.5th and 97.5th percentiles. Resamples where
  F1 is undefined are dropped; the JSON field `bootstrap_valid_resamples`
  gives how many were kept. Each statistic has its own random stream derived
  from `--seed` and a stable label, so results don't change when the order of
  analyses changes.

## Statistical choices to be aware of

- **Non-independence.** The pooled step-1 κ and the Wilson CIs for detection
  and cue shares treat each item × criterion (or cue) as independent, although
  cases from the same video are correlated. The Wilson CIs are therefore
  somewhat too narrow. The F1 CI and the precision/recall bootstrap CIs in the
  JSON resample whole items, so prefer those for the pooled detection results.
- **Kappa paradox.** When "needs improvement" is very common or very rare for
  a criterion, κ can be low or undefined even though agreement is high. Read κ
  together with percent agreement, PABAK and prevalence.
- **Fleiss' κ with two raters** is Scott's π: it pools the two experts'
  marginals. Cohen's κ keeps them separate. The two differ when the experts
  use categories at different rates.
- **ICC on a 0–3 ordinal score** treats the score as interval-scaled. The
  quadratic-weighted κ is asymptotically equivalent and is reported alongside.
  ICC(2,1) counts systematic leniency differences between experts against
  agreement; ICC(3,1) does not.
- **Consensus rules.** `both` is primary: it is conservative about what counts
  as a real problem, which raises recall and lowers precision. `either` is
  secondary. No adjudicated consensus is built; the disagreement count shows
  how many cases adjudication would affect.
- **What detection measures.** GPT's cue list is a selection of the checkpoints
  it chose to comment on, not an exhaustive list of problems. Lower recall can
  reflect deliberate brevity (for example, a limit on the number of cues)
  rather than missed errors.
- **Small cells.** Per-criterion results rest on n = the number of items for
  that skill. With few positives their CIs are very wide, and bootstrap F1
  intervals can reach 0 or 1.
- **"Harmful/incorrect".** Any expert giving 0 is a deliberately sensitive
  (upper-bound) definition; `harmful_both_0` is the strict counterpart.
- **Percentiles** use numpy's default linear interpolation, and SD uses ddof = 1.
