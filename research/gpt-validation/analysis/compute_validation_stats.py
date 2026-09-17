#!/usr/bin/env python3
"""Reliability and validity statistics for the GPT feedback validation study.

Reads the four admin CSV exports described in ../DESIGN.md (items.csv,
criteria.csv, cues.csv, overall.csv) and reports:

1. Inter-rater reliability between the two experts
   - step 1 checkpoint detection (binary): percent agreement, Cohen's kappa,
     Fleiss' kappa, PABAK, prevalence per expert; per criterion, per skill,
     pooled
   - cue ratings (ordinal 0/1/2): percent agreement, unweighted / linear /
     quadratic weighted Cohen's kappa, Fleiss' kappa
   - overall score (0-3): quadratic weighted kappa, ICC(2,1), ICC(3,1) with
     F-distribution 95% CIs, Fleiss' kappa
2. GPT validity against expert consensus
   - checkpoint detection (strict "both" and lenient "either" consensus):
     confusion matrix, precision / recall / specificity (Wilson CIs), F1
     (item-cluster percentile bootstrap CI)
   - cue correctness shares (Wilson CIs) and mean cue score
   - overall acceptability / harmfulness rates (Wilson CIs)
   - time burden per expert
3. A results paragraph (Markdown output)

Only numpy and pandas are required; all statistics are implemented here.

Usage:
    python compute_validation_stats.py --items items.csv --criteria criteria.csv \
        --cues cues.csv --overall overall.csv [--experts expert-1,expert-2] \
        [--out report.json] [--markdown report.md] [--bootstrap 2000 --seed 7]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

Z95 = 1.959963984540054  # standard normal 0.975 quantile
ALPHA = 0.05
EPS = 1e-12

CUE_SCORES = {"correct": 2, "partial": 1, "incorrect": 0}
CUE_LABELS = {v: k for k, v in CUE_SCORES.items()}
OVERALL_CATEGORIES = [0, 1, 2, 3]
CUE_CATEGORIES = [0, 1, 2]
BINARY_CATEGORIES = [0, 1]

REQUIRED_COLUMNS = {
    "items": [
        "item_id", "display_code", "skill", "source_file", "workbook_id", "status", "eligible",
        "coaching_source", "coaching_model", "handedness", "total_grade", "n_cues",
        "gpt_flagged_criteria",
    ],
    "criteria": [
        "item_id", "skill", "expert_id", "criterion_id", "criterion_name",
        "expert_needs_improvement", "gpt_flagged",
    ],
    "cues": ["item_id", "skill", "expert_id", "cue_index", "criterion_id", "cue_rating", "cue_score"],
    "overall": [
        "item_id", "skill", "expert_id", "overall_score", "comment", "step1_seconds",
        "step2_seconds", "completed",
    ],
}

TRUE_STRINGS = {"1", "true", "t", "yes", "y"}
FALSE_STRINGS = {"0", "false", "f", "no", "n"}


class DataError(Exception):
    """Input data is unusable (wrong columns, duplicates, wrong number of experts...)."""


# ---------------------------------------------------------------------------
# Special functions: regularized incomplete beta, F distribution
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float, max_iter: int = 10_000, tol: float = 1e-15) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            return h
    raise ArithmeticError(f"incomplete beta continued fraction did not converge (a={a}, b={b}, x={x})")


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) for a, b > 0."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def f_cdf(x: float, d1: float, d2: float) -> float:
    """CDF of the F(d1, d2) distribution."""
    if x <= 0:
        return 0.0
    if math.isinf(x):
        return 1.0
    u = d1 * x / (d1 * x + d2)
    return betainc_reg(d1 / 2.0, d2 / 2.0, u)


def f_ppf(p: float, d1: float, d2: float) -> float:
    """Quantile of F(d1, d2) by bisection on the beta variable u in (0, 1).

    F = d2*u / (d1*(1-u)) with u ~ Beta(d1/2, d2/2); the CDF of u is monotone,
    so bisection is unconditionally stable. Degrees of freedom may be
    non-integer (needed for the ICC(2,1) confidence interval).
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if d1 <= 0 or d2 <= 0:
        raise ValueError("degrees of freedom must be positive")
    a, b = d1 / 2.0, d2 / 2.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc_reg(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    u = 0.5 * (lo + hi)
    return d2 * u / (d1 * (1.0 - u))


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------


def _clean_float(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def contingency_table(a: Sequence, b: Sequence, categories: Sequence) -> np.ndarray:
    index = {c: i for i, c in enumerate(categories)}
    table = np.zeros((len(categories), len(categories)), dtype=float)
    for x, y in zip(a, b):
        if x not in index or y not in index:
            raise ValueError(f"rating outside categories {list(categories)}: {x!r}, {y!r}")
        table[index[x], index[y]] += 1
    return table


def cohen_kappa(
    a: Sequence, b: Sequence, categories: Sequence | None = None, weights: str | None = None
) -> dict[str, Any]:
    """Cohen's kappa for two raters, optionally linear/quadratic weighted.

    kappa_w = 1 - sum(w_ij * o_ij) / sum(w_ij * e_ij), with disagreement
    weights w_ij = 1[i != j] (unweighted), |i-j|/(k-1) (linear) or
    ((i-j)/(k-1))^2 (quadratic); o = observed proportions, e = product of
    the raters' marginal proportions. Unweighted this equals
    (p_o - p_e) / (1 - p_e). Categories must be listed in ordinal order for
    weighted kappa.
    """
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError("rating vectors differ in length")
    n = len(a)
    result: dict[str, Any] = {
        "kappa": None, "n": n, "weights": weights or "none",
        "percent_agreement": None, "expected_agreement": None, "note": None,
    }
    if n == 0:
        result["note"] = "no cases"
        return result
    if categories is None:
        categories = sorted(set(a) | set(b))
    k = len(categories)
    obs = contingency_table(a, b, categories) / n
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights in ("linear", "quadratic"):
        if k < 2:
            w = np.zeros((k, k))
        else:
            w = np.abs(i - j) / (k - 1)
            if weights == "quadratic":
                w = w**2
    else:
        raise ValueError(f"unknown weights {weights!r}")
    p_o = float(np.trace(obs))
    p_e = float(np.trace(exp))
    result["percent_agreement"] = p_o
    result["expected_agreement"] = p_e
    observed_dis = float((w * obs).sum())
    expected_dis = float((w * exp).sum())
    if expected_dis <= EPS:
        result["note"] = (
            "kappa undefined: chance-expected disagreement is 0 (p_e = 1; "
            "both raters used a single category)"
        )
        return result
    result["kappa"] = 1.0 - observed_dis / expected_dis
    return result


def fleiss_kappa_from_counts(counts: np.ndarray) -> dict[str, Any]:
    """Fleiss' kappa from an N x K matrix of rater counts per subject/category.

    P_i = (sum_j n_ij^2 - m) / (m (m-1)), P_bar = mean P_i,
    p_j = sum_i n_ij / (N m), P_e = sum_j p_j^2,
    kappa = (P_bar - P_e) / (1 - P_e). With m = 2 raters this is Scott's pi.
    """
    counts = np.asarray(counts, dtype=float)
    result: dict[str, Any] = {"kappa": None, "n": int(counts.shape[0]) if counts.ndim == 2 else 0,
                              "observed_agreement": None, "expected_agreement": None, "note": None}
    if counts.ndim != 2 or counts.shape[0] == 0:
        result["note"] = "no cases"
        return result
    m_per_subject = counts.sum(axis=1)
    m = m_per_subject[0]
    if not np.allclose(m_per_subject, m) or m < 2:
        raise ValueError("Fleiss' kappa needs the same number (>= 2) of raters per subject")
    n_subjects = counts.shape[0]
    p_j = counts.sum(axis=0) / (n_subjects * m)
    p_i = ((counts**2).sum(axis=1) - m) / (m * (m - 1))
    p_bar = float(p_i.mean())
    p_e = float((p_j**2).sum())
    result["observed_agreement"] = p_bar
    result["expected_agreement"] = p_e
    if 1.0 - p_e <= EPS:
        result["note"] = "kappa undefined: expected agreement is 1 (a single category was used)"
        return result
    result["kappa"] = (p_bar - p_e) / (1.0 - p_e)
    return result


def fleiss_kappa(ratings: Sequence[Sequence], categories: Sequence | None = None) -> dict[str, Any]:
    """Fleiss' kappa from an N x m matrix of ratings (rows = subjects, cols = raters)."""
    rows = [list(r) for r in ratings]
    if categories is None:
        categories = sorted({x for r in rows for x in r})
    index = {c: i for i, c in enumerate(categories)}
    counts = np.zeros((len(rows), len(categories)))
    for s, r in enumerate(rows):
        for x in r:
            counts[s, index[x]] += 1
    return fleiss_kappa_from_counts(counts)


def icc(data: np.ndarray | Sequence[Sequence[float]], alpha: float = ALPHA) -> dict[str, Any]:
    """Shrout & Fleiss (1979) / McGraw & Wong (1996) intraclass correlations.

    data: n subjects x k raters, complete. Two-way ANOVA mean squares
    MSR (rows/subjects), MSC (columns/raters), MSE (residual) and one-way
    MSW (within subjects).

    ICC(1,1) = (MSR - MSW) / (MSR + (k-1) MSW)
    ICC(2,1) = (MSR - MSE) / (MSR + (k-1) MSE + k (MSC - MSE) / n)   absolute agreement
    ICC(3,1) = (MSR - MSE) / (MSR + (k-1) MSE)                        consistency
    plus the average-measures forms ICC(1,k), ICC(2,k), ICC(3,k).

    Confidence intervals use the exact F-based limits (ICC(1), ICC(3)) and the
    Satterthwaite approximation of McGraw & Wong (1996) for ICC(2), as in
    R's psych::ICC.
    """
    y = np.asarray(data, dtype=float)
    if y.ndim != 2:
        raise ValueError("ICC needs a 2-D subjects x raters array")
    n, k = y.shape
    out: dict[str, Any] = {"n_subjects": int(n), "n_raters": int(k), "note": None}
    if n < 2 or k < 2:
        out["note"] = "ICC needs at least 2 subjects and 2 raters"
        return out
    if np.isnan(y).any():
        raise ValueError("ICC requires complete data")
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())
    ss_rows = float(k * ((y.mean(axis=1) - grand) ** 2).sum())
    ss_cols = float(n * ((y.mean(axis=0) - grand) ** 2).sum())
    ss_err = ss_total - ss_rows - ss_cols
    ss_within = ss_total - ss_rows
    df_r, df_c, df_e, df_w = n - 1, k - 1, (n - 1) * (k - 1), n * (k - 1)
    msr, msc = ss_rows / df_r, ss_cols / df_c
    mse, msw = max(ss_err, 0.0) / df_e, max(ss_within, 0.0) / df_w
    out["mean_squares"] = {"MSR": msr, "MSC": msc, "MSE": mse, "MSW": msw}
    out["df"] = {"rows": df_r, "cols": df_c, "error": df_e, "within": df_w}

    def div(num: float, den: float) -> float | None:
        if abs(den) <= EPS:
            return None
        return num / den

    q = 1.0 - alpha / 2.0
    notes: list[str] = []

    # ICC(1)
    icc1 = div(msr - msw, msr + (k - 1) * msw)
    icc1k = div(msr - msw, msr)
    ci1 = ci1k = (None, None)
    f1 = div(msr, msw)
    p1 = None
    if f1 is not None:
        p1 = 1.0 - f_cdf(f1, df_r, df_w)
        fl = f1 / f_ppf(q, df_r, df_w)
        fu = f1 * f_ppf(q, df_w, df_r)
        ci1 = ((fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1))
        ci1k = (1 - 1 / fl, 1 - 1 / fu)

    # ICC(3)
    icc3 = div(msr - mse, msr + (k - 1) * mse)
    icc3k = div(msr - mse, msr)
    ci3 = ci3k = (None, None)
    f3 = div(msr, mse)
    p3 = None
    if f3 is not None:
        p3 = 1.0 - f_cdf(f3, df_r, df_e)
        fl = f3 / f_ppf(q, df_r, df_e)
        fu = f3 * f_ppf(q, df_e, df_r)
        ci3 = ((fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1))
        ci3k = (1 - 1 / fl, 1 - 1 / fu)
    else:
        notes.append("residual mean square is 0; F-based CIs undefined")

    # ICC(2)
    icc2 = div(msr - mse, msr + (k - 1) * mse + k * (msc - mse) / n)
    icc2k = div(msr - mse, msr + (msc - mse) / n)
    ci2 = ci2k = (None, None)
    if icc2 is not None and f3 is not None and abs(1.0 - icc2) > EPS:
        a = k * icc2 / (n * (1.0 - icc2))
        b = 1.0 + k * icc2 * (n - 1) / (n * (1.0 - icc2))
        num = (a * msc + b * mse) ** 2
        den = (a * msc) ** 2 / df_c + (b * mse) ** 2 / df_e
        if den > EPS and num > EPS:
            v = num / den
            f_star = f_ppf(q, df_r, v)
            f_2star = f_ppf(q, v, df_r)
            lower = n * (msr - f_star * mse) / (f_star * (k * msc + (k * n - k - n) * mse) + n * msr)
            upper = n * (f_2star * msr - mse) / (k * msc + (k * n - k - n) * mse + n * f_2star * msr)
            ci2 = (lower, upper)
            ci2k = (
                lower * k / (1 + lower * (k - 1)),
                upper * k / (1 + upper * (k - 1)),
            )
            out["icc2_ci_df"] = v

    def entry(value, ci, f, dfs, p):
        return {
            "value": _clean_float(value),
            "ci95": [_clean_float(ci[0]), _clean_float(ci[1])],
            "F": _clean_float(f), "df1": dfs[0], "df2": dfs[1], "p_value": _clean_float(p),
        }

    out["ICC1_1"] = entry(icc1, ci1, f1, (df_r, df_w), p1)
    out["ICC2_1"] = entry(icc2, ci2, f3, (df_r, df_e), p3)
    out["ICC3_1"] = entry(icc3, ci3, f3, (df_r, df_e), p3)
    out["ICC1_k"] = entry(icc1k, ci1k, f1, (df_r, df_w), p1)
    out["ICC2_k"] = entry(icc2k, ci2k, f3, (df_r, df_e), p3)
    out["ICC3_k"] = entry(icc3k, ci3k, f3, (df_r, df_e), p3)
    if icc2 is None or icc3 is None:
        notes.append("ICC undefined: no between-subject or residual variance (all scores identical)")
    out["note"] = "; ".join(notes) or None
    return out


# ---------------------------------------------------------------------------
# Proportions and classification metrics
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float | None, float | None]:
    """Wilson score interval for k successes out of n."""
    if n <= 0:
        return (None, None)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def proportion(k: int, n: int) -> dict[str, Any]:
    k, n = int(k), int(n)
    lo, hi = wilson_ci(k, n)
    return {"value": (k / n) if n else None, "k": k, "n": n, "ci95": [lo, hi], "ci_method": "wilson"}


def f1_from_counts(tp: float, fp: float, fn: float) -> float | None:
    den = 2 * tp + fp + fn
    return (2 * tp / den) if den > 0 else None


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    """precision = TP/(TP+FP), recall = TP/(TP+FN), specificity = TN/(TN+FP),
    F1 = 2TP/(2TP+FP+FN), accuracy = (TP+TN)/N; Wilson CIs for proportions."""
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    return {
        "counts": {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "N": tp + fp + fn + tn},
        "precision": proportion(tp, tp + fp),
        "recall": proportion(tp, tp + fn),
        "specificity": proportion(tn, tn + fp),
        "accuracy": proportion(tp + tn, tp + fp + fn + tn),
        "f1": {"value": f1_from_counts(tp, fp, fn), "ci95": [None, None], "ci_method": None},
        "prevalence_expert_positive": proportion(tp + fn, tp + fp + fn + tn),
        "gpt_positive_rate": proportion(tp + fp, tp + fp + fn + tn),
    }


def _rng(seed: int, label: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(label.encode("utf-8"))])


def cluster_bootstrap_confusion(
    per_cluster: np.ndarray, n_boot: int, seed: int, label: str, chunk: int = 250
) -> dict[str, Any]:
    """Percentile bootstrap resampling clusters (items) with replacement.

    per_cluster: C x 4 array of (TP, FP, FN, TN) counts per item. Returns 95%
    percentile CIs for F1, precision and recall; resamples where a metric is
    undefined are dropped (count reported).
    """
    per_cluster = np.asarray(per_cluster, dtype=float)
    c = per_cluster.shape[0]
    res: dict[str, Any] = {"n_boot": int(n_boot), "n_clusters": int(c)}
    if n_boot <= 0 or c == 0:
        for name in ("f1", "precision", "recall"):
            res[name] = {"ci95": [None, None], "n_valid": 0}
        return res
    rng = _rng(seed, label)
    sums = np.empty((n_boot, 4))
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, c, size=(b, c))
        sums[done : done + b] = per_cluster[idx].sum(axis=1)
        done += b
    tp, fp, fn = sums[:, 0], sums[:, 1], sums[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics = {
            "f1": np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan),
            "precision": np.where(tp + fp > 0, tp / (tp + fp), np.nan),
            "recall": np.where(tp + fn > 0, tp / (tp + fn), np.nan),
        }
    for name, values in metrics.items():
        valid = values[~np.isnan(values)]
        if valid.size == 0:
            res[name] = {"ci95": [None, None], "n_valid": 0}
        else:
            lo, hi = np.percentile(valid, [2.5, 97.5])
            res[name] = {"ci95": [float(lo), float(hi)], "n_valid": int(valid.size)}
    return res


def cluster_bootstrap_mean(
    values: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int, label: str
) -> list[float | None]:
    """Percentile bootstrap CI for a mean of observation-level values, resampling clusters."""
    values = np.asarray(values, dtype=float)
    if n_boot <= 0 or values.size == 0:
        return [None, None]
    codes, uniq = pd.factorize(pd.Series(clusters))
    sums = np.bincount(codes, weights=values)
    counts = np.bincount(codes).astype(float)
    rng = _rng(seed, label)
    c = len(uniq)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 250):
        b = min(250, n_boot - start)
        idx = rng.integers(0, c, size=(b, c))
        means[start : start + b] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def describe(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and not pd.isna(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None, "q1": None, "q3": None,
                "iqr": None, "min": None, "max": None}
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
        "min": float(arr.min()), "max": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# Loading and filtering
# ---------------------------------------------------------------------------


def read_export(path: str | Path, name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False, na_values=[""])
    except FileNotFoundError as exc:
        raise DataError(f"{name}: file not found: {path}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS[name] if c not in df.columns]
    if missing:
        raise DataError(f"{name}.csv ({path}) is missing columns: {', '.join(missing)}")
    for col in df.columns:
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        df[col] = df[col].replace("", np.nan)
    return df


def parse_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    s = str(value).strip().lower()
    if s in TRUE_STRINGS:
        return True
    if s in FALSE_STRINGS:
        return False
    return None


def parse_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        f = float(str(value).strip())
    except ValueError:
        return None
    if math.isnan(f) or f != int(f):
        return None
    return int(f)


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(str(value).strip())
    except ValueError:
        return None
    return None if math.isnan(f) else f


def resolve_experts(frames: dict[str, pd.DataFrame], experts_arg: str | None) -> tuple[list[str], list[str]]:
    """Return the two expert ids and warnings. Raises DataError unless exactly two."""
    warnings: list[str] = []
    present = set()
    for name in ("criteria", "cues", "overall"):
        present |= set(frames[name]["expert_id"].dropna().astype(str))
    if experts_arg:
        experts = [e.strip() for e in experts_arg.split(",") if e.strip()]
        if len(experts) != 2 or len(set(experts)) != 2:
            raise DataError(f"--experts must name exactly two distinct expert ids, got {experts!r}")
        for e in experts:
            if e not in present:
                raise DataError(
                    f"expert {e!r} not found in criteria/cues/overall exports "
                    f"(ids present: {sorted(present)})"
                )
        others = sorted(present - set(experts))
        if others:
            warnings.append(f"ignoring ratings from other ids: {others}")
        return experts, warnings
    candidates = sorted(present - {"admin"})
    if len(candidates) != 2:
        raise DataError(
            f"exactly two experts are expected but the exports contain {len(candidates)}: "
            f"{candidates}; pass --experts id1,id2 to choose"
        )
    return candidates, warnings


def _check_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    dup = df[df.duplicated(keys, keep=False)]
    if not dup.empty:
        sample = dup[keys].drop_duplicates().head(5).to_dict("records")
        raise DataError(f"{name}.csv has duplicate rows for {keys}: {sample}")


def filter_items(
    items: pd.DataFrame,
    criteria: pd.DataFrame,
    cues: pd.DataFrame,
    overall: pd.DataFrame,
    experts: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    """Keep items that are eligible and fully completed by both experts.

    An item is complete for an expert when overall.csv has completed = true and
    a valid overall_score (0-3), criteria.csv has a 0/1 step-1 answer for every
    criterion listed for that item, and cues.csv has a valid rating for every
    cue 1..n_cues. Data-completeness checks run only for completed ratings.
    Returns (included item ids in items.csv order, exclusion summary).
    """
    _check_unique(items, ["item_id"], "items")
    crit = criteria[criteria["expert_id"].isin(experts)]
    cue = cues[cues["expert_id"].isin(experts)]
    ov = overall[overall["expert_id"].isin(experts)]
    _check_unique(crit, ["item_id", "expert_id", "criterion_id"], "criteria")
    _check_unique(cue, ["item_id", "expert_id", "cue_index"], "cues")
    _check_unique(ov, ["item_id", "expert_id"], "overall")

    crit_groups = {k: g for k, g in crit.groupby(["item_id", "expert_id"], sort=False)}
    crit_ids_by_item = crit.groupby("item_id")["criterion_id"].agg(lambda s: set(s.dropna()))
    cue_groups = {k: g for k, g in cue.groupby(["item_id", "expert_id"], sort=False)}
    cue_idx_by_item = cue.groupby("item_id")["cue_index"].agg(lambda s: {parse_int(x) for x in s} - {None})
    ov_rows = {(r.item_id, r.expert_id): r for r in ov.itertuples(index=False)}

    included: list[str] = []
    excluded: list[dict[str, Any]] = []
    for row in items.itertuples(index=False):
        item_id = row.item_id
        reasons: list[str] = []
        if parse_bool(row.eligible) is not True:
            reasons.append("not_eligible")
        for e in experts:
            o = ov_rows.get((item_id, e))
            if o is None:
                reasons.append(f"not_rated_by:{e}")
                continue
            if parse_bool(o.completed) is not True:
                reasons.append(f"not_completed_by:{e}")
                continue
            score = parse_int(o.overall_score)
            if score is None or score not in OVERALL_CATEGORIES:
                reasons.append(f"invalid_overall_score:{e}")
            expected_crit = crit_ids_by_item.get(item_id, set())
            g = crit_groups.get((item_id, e))
            step1_ok = bool(expected_crit) and g is not None
            if step1_ok:
                answers = {cid: parse_int(v) for cid, v in zip(g["criterion_id"], g["expert_needs_improvement"])}
                step1_ok = set(answers) == expected_crit and all(v in (0, 1) for v in answers.values())
            if not step1_ok:
                reasons.append(f"incomplete_step1:{e}")
            n_cues = parse_int(row.n_cues)
            expected_cues = set(range(1, n_cues + 1)) if n_cues is not None else cue_idx_by_item.get(item_id, set())
            if expected_cues:
                cg = cue_groups.get((item_id, e))
                rated = set()
                if cg is not None:
                    rated = {
                        parse_int(i) for i, r in zip(cg["cue_index"], cg["cue_rating"])
                        if isinstance(r, str) and r.strip().lower() in CUE_SCORES
                    }
                if not expected_cues <= rated:
                    reasons.append(f"incomplete_cue_ratings:{e}")
        if reasons:
            excluded.append({
                "item_id": item_id, "display_code": row.display_code, "skill": row.skill,
                "status": row.status, "coaching_source": row.coaching_source, "reasons": reasons,
            })
        else:
            included.append(item_id)

    primary: dict[str, int] = {}
    any_reason: dict[str, int] = {}
    for ex in excluded:
        primary[ex["reasons"][0]] = primary.get(ex["reasons"][0], 0) + 1
        for r in dict.fromkeys(ex["reasons"]):
            any_reason[r] = any_reason.get(r, 0) + 1
    not_eligible = [ex for ex in excluded if "not_eligible" in ex["reasons"]]
    ne_breakdown: dict[str, int] = {}
    for ex in not_eligible:
        key = ", ".join(
            f"{field}={ex[field] if isinstance(ex[field], str) else '(blank)'}" for field in ("status", "coaching_source")
        )
        ne_breakdown[key] = ne_breakdown.get(key, 0) + 1
    eligible_incomplete = [ex for ex in excluded if "not_eligible" not in ex["reasons"]]

    known = set(items["item_id"])
    unknown = sorted((set(crit["item_id"]) | set(cue["item_id"]) | set(ov["item_id"])) - known)

    summary = {
        "n_items_in_export": int(len(items)),
        "n_eligible": int(len(items) - len(not_eligible)),
        "n_included": len(included),
        "n_excluded": len(excluded),
        "n_excluded_not_eligible": len(not_eligible),
        "n_excluded_eligible_but_incomplete": len(eligible_incomplete),
        "excluded_by_primary_reason": dict(sorted(primary.items())),
        "excluded_by_any_reason": dict(sorted(any_reason.items())),
        "not_eligible_breakdown": dict(sorted(ne_breakdown.items())),
        "reason_legend": {
            "not_eligible": "items.csv eligible is not true (failed, or coaching not from OpenAI)",
            "not_rated_by": "no overall.csv row for that expert",
            "not_completed_by": "overall.csv completed is not true",
            "invalid_overall_score": "completed but overall_score is not an integer 0-3",
            "incomplete_step1": "missing/invalid step-1 answer for at least one criterion",
            "incomplete_cue_ratings": "missing/invalid rating for at least one GPT cue",
        },
        "excluded_items": [
            {"item_id": ex["item_id"], "display_code": ex["display_code"], "skill": ex["skill"],
             "reasons": ex["reasons"]}
            for ex in excluded
        ],
        "rating_rows_for_unknown_items": unknown,
    }
    return included, summary


# ---------------------------------------------------------------------------
# Wide analysis tables
# ---------------------------------------------------------------------------


def build_tables(
    items: pd.DataFrame,
    criteria: pd.DataFrame,
    cues: pd.DataFrame,
    overall: pd.DataFrame,
    experts: Sequence[str],
    included: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    e1, e2 = experts
    warnings: list[str] = []
    inc = set(included)
    item_info = items[items["item_id"].isin(inc)].set_index("item_id")
    order = {iid: i for i, iid in enumerate(included)}

    # criteria: one row per item x criterion
    c = criteria[criteria["item_id"].isin(inc) & criteria["expert_id"].isin(experts)].copy()
    c["_crit_order"] = np.arange(len(c))
    c["needs"] = c["expert_needs_improvement"].map(parse_int)
    c["gflag"] = c["gpt_flagged"].map(parse_int)
    base = (
        c.sort_values("_crit_order")
        .groupby(["item_id", "criterion_id"], sort=False)
        .agg(criterion_name=("criterion_name", "first"), crit_order=("_crit_order", "min"))
        .reset_index()
    )
    wide = base.copy()
    for tag, e in (("r1", e1), ("r2", e2)):
        sub = c[c["expert_id"] == e][["item_id", "criterion_id", "needs", "gflag"]]
        sub = sub.rename(columns={"needs": tag, "gflag": f"gflag_{tag}"})
        wide = wide.merge(sub, on=["item_id", "criterion_id"], how="left")
    wide["skill"] = wide["item_id"].map(item_info["skill"])

    flagged_lists = item_info["gpt_flagged_criteria"].map(
        lambda s: {x.strip() for x in str(s).split(";") if x.strip()} if isinstance(s, str) else set()
    )
    from_items = np.array([
        int(cid in flagged_lists.get(iid, set())) for iid, cid in zip(wide["item_id"], wide["criterion_id"])
    ])
    g1, g2 = wide["gflag_r1"], wide["gflag_r2"]
    gpt = g1.where(g1.notna(), g2)
    if ((g1.notna() & g2.notna()) & (g1 != g2)).any():
        warnings.append("criteria.csv gpt_flagged differs between the two experts' rows for some cases; using expert 1's row")
    gpt = gpt.where(gpt.notna(), pd.Series(from_items, index=wide.index))
    mismatch = int((gpt.astype(int).to_numpy() != from_items).sum())
    if mismatch:
        warnings.append(
            f"{mismatch} item x criterion cases where criteria.csv gpt_flagged disagrees with "
            "items.csv gpt_flagged_criteria; criteria.csv gpt_flagged was used"
        )
    wide["gpt"] = gpt.astype(int)
    crit_known = wide.groupby("item_id")["criterion_id"].agg(set)
    unknown_flags = sum(
        len(flagged_lists.get(iid, set()) - crit_known.get(iid, set())) for iid in item_info.index
    )
    if unknown_flags:
        warnings.append(
            f"{unknown_flags} GPT-flagged criterion ids in items.csv do not match any rubric criterion "
            "in criteria.csv and are not counted in detection metrics"
        )
    wide["r1"] = wide["r1"].astype(int)
    wide["r2"] = wide["r2"].astype(int)
    wide["_item_order"] = wide["item_id"].map(order)
    wide = wide.sort_values(["_item_order", "crit_order"]).reset_index(drop=True)

    # cues: one row per item x cue
    q = cues[cues["item_id"].isin(inc) & cues["expert_id"].isin(experts)].copy()
    q["cue_index_int"] = q["cue_index"].map(parse_int)
    q["score"] = q["cue_rating"].map(lambda r: CUE_SCORES.get(str(r).strip().lower()) if isinstance(r, str) else None)
    given = q["cue_score"].map(parse_int)
    bad = int(((given.notna()) & (q["score"].notna()) & (given != q["score"])).sum())
    if bad:
        warnings.append(f"{bad} cues.csv rows where cue_score does not match cue_rating; cue_rating was used")
    n_cues = item_info["n_cues"].map(parse_int)
    q = q[q["score"].notna()]
    # keep cues 1..n_cues (extra rows beyond n_cues are ignored)
    q = q[[(nc is None) or (1 <= ci <= nc) for ci, nc in zip(q["cue_index_int"], q["item_id"].map(n_cues))]]
    cue_base = (
        q.groupby(["item_id", "cue_index_int"], sort=False)
        .agg(criterion_id=("criterion_id", "first"))
        .reset_index()
    )
    cue_wide = cue_base
    for tag, e in (("s1", e1), ("s2", e2)):
        sub = q[q["expert_id"] == e][["item_id", "cue_index_int", "score"]].rename(columns={"score": tag})
        cue_wide = cue_wide.merge(sub, on=["item_id", "cue_index_int"], how="left")
    cue_wide = cue_wide.dropna(subset=["s1", "s2"])
    cue_wide["s1"] = cue_wide["s1"].astype(int)
    cue_wide["s2"] = cue_wide["s2"].astype(int)
    cue_wide["skill"] = cue_wide["item_id"].map(item_info["skill"])
    cue_wide["_item_order"] = cue_wide["item_id"].map(order)
    cue_wide = cue_wide.sort_values(["_item_order", "cue_index_int"]).reset_index(drop=True)

    # overall: one row per item
    o = overall[overall["item_id"].isin(inc) & overall["expert_id"].isin(experts)]
    ov = pd.DataFrame({"item_id": list(included)})
    ov["skill"] = ov["item_id"].map(item_info["skill"])
    for tag, e in (("o1", e1), ("o2", e2)):
        sub = o[o["expert_id"] == e].set_index("item_id")
        ov[tag] = ov["item_id"].map(sub["overall_score"]).map(parse_int).astype(int)
        ov[f"step1_seconds_{tag}"] = ov["item_id"].map(sub["step1_seconds"]).map(parse_float)
        ov[f"step2_seconds_{tag}"] = ov["item_id"].map(sub["step2_seconds"]).map(parse_float)
    return wide, cue_wide, ov, warnings


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def _counts(values: Iterable[int], categories: Sequence[int]) -> dict[str, int]:
    vals = list(values)
    return {str(c): int(sum(1 for v in vals if v == c)) for c in categories}


def binary_reliability(r1: Sequence[int], r2: Sequence[int], experts: Sequence[str]) -> dict[str, Any]:
    r1, r2 = [int(x) for x in r1], [int(x) for x in r2]
    n = len(r1)
    ck = cohen_kappa(r1, r2, BINARY_CATEGORIES)
    fk = fleiss_kappa(list(zip(r1, r2)), BINARY_CATEGORIES)
    both_pos = sum(1 for a, b in zip(r1, r2) if a == 1 and b == 1)
    both_neg = sum(1 for a, b in zip(r1, r2) if a == 0 and b == 0)
    notes = [x for x in (ck["note"], fk["note"]) if x]
    return {
        "n": n,
        "percent_agreement": ck["percent_agreement"],
        "cohen_kappa": ck["kappa"],
        "fleiss_kappa": fk["kappa"],
        "pabak": (2 * ck["percent_agreement"] - 1) if n else None,
        "expected_agreement_cohen": ck["expected_agreement"],
        "prevalence_needs_improvement": {
            experts[0]: proportion(sum(r1), n),
            experts[1]: proportion(sum(r2), n),
        },
        "cells": {
            "both_yes": both_pos, "both_no": both_neg,
            f"only_{experts[0]}": sum(1 for a, b in zip(r1, r2) if a == 1 and b == 0),
            f"only_{experts[1]}": sum(1 for a, b in zip(r1, r2) if a == 0 and b == 1),
        },
        "note": "; ".join(dict.fromkeys(notes)) or None,
    }


def ordinal_reliability(s1: Sequence[int], s2: Sequence[int], categories: Sequence[int]) -> dict[str, Any]:
    s1, s2 = [int(x) for x in s1], [int(x) for x in s2]
    unw = cohen_kappa(s1, s2, categories)
    lin = cohen_kappa(s1, s2, categories, "linear")
    quad = cohen_kappa(s1, s2, categories, "quadratic")
    fk = fleiss_kappa(list(zip(s1, s2)), categories)
    notes = [x for x in (unw["note"], lin["note"], quad["note"], fk["note"]) if x]
    n = len(s1)
    return {
        "n": n,
        "percent_agreement": unw["percent_agreement"],
        "within_one_category": (sum(1 for a, b in zip(s1, s2) if abs(a - b) <= 1) / n) if n else None,
        "cohen_kappa_unweighted": unw["kappa"],
        "cohen_kappa_linear": lin["kappa"],
        "cohen_kappa_quadratic": quad["kappa"],
        "fleiss_kappa": fk["kappa"],
        "note": "; ".join(dict.fromkeys(notes)) or None,
    }


def reliability_section(crit: pd.DataFrame, cue: pd.DataFrame, ov: pd.DataFrame, experts: Sequence[str]) -> dict[str, Any]:
    step1: dict[str, Any] = {
        "unit": "item x criterion",
        "pooled": binary_reliability(crit["r1"], crit["r2"], experts),
        "by_skill": {},
    }
    for skill, g in crit.groupby("skill", sort=True):
        per_criterion = []
        for cid, gc in g.groupby("criterion_id", sort=False):
            entry = binary_reliability(gc["r1"], gc["r2"], experts)
            entry = {"criterion_id": cid, "criterion_name": gc["criterion_name"].iloc[0], **entry}
            per_criterion.append(entry)
        step1["by_skill"][skill] = {
            "pooled": binary_reliability(g["r1"], g["r2"], experts),
            "n_items": int(g["item_id"].nunique()),
            "criteria": per_criterion,
        }
    cues_rel: dict[str, Any] = {
        "unit": "item x GPT cue", "scale": "incorrect=0, partial=1, correct=2",
        "n_items": int(cue["item_id"].nunique()),
        "pooled": ordinal_reliability(cue["s1"], cue["s2"], CUE_CATEGORIES),
        "by_skill": {s: ordinal_reliability(g["s1"], g["s2"], CUE_CATEGORIES) for s, g in cue.groupby("skill")},
    }

    def overall_rel(g: pd.DataFrame) -> dict[str, Any]:
        rel = ordinal_reliability(g["o1"], g["o2"], OVERALL_CATEGORIES)
        icc_res = icc(g[["o1", "o2"]].to_numpy(dtype=float))
        rel["ICC2_1"] = icc_res.get("ICC2_1")
        rel["ICC3_1"] = icc_res.get("ICC3_1")
        rel["icc_note"] = icc_res.get("note")
        rel["icc_mean_squares"] = icc_res.get("mean_squares")
        rel["mean_score"] = {experts[0]: describe(g["o1"]), experts[1]: describe(g["o2"])}
        return rel

    return {
        "step1_detection": step1,
        "cue_ratings": cues_rel,
        "overall_score": {
            "unit": "item", "scale": "0-3",
            "pooled": overall_rel(ov),
            "by_skill": {s: overall_rel(g) for s, g in ov.groupby("skill")},
        },
    }


def detection_block(g: pd.DataFrame, truth_col: str, n_boot: int, seed: int, label: str) -> dict[str, Any]:
    t = g[truth_col].astype(bool)
    p = g["gpt"].astype(bool)
    frame = pd.DataFrame({
        "item_id": g["item_id"],
        "TP": (p & t).astype(int), "FP": (p & ~t).astype(int),
        "FN": (~p & t).astype(int), "TN": (~p & ~t).astype(int),
    })
    per_item = frame.groupby("item_id", sort=False)[["TP", "FP", "FN", "TN"]].sum()
    tot = per_item.sum()
    res = binary_metrics(tot["TP"], tot["FP"], tot["FN"], tot["TN"])
    res["n_items"] = int(len(per_item))
    boot = cluster_bootstrap_confusion(per_item.to_numpy(), n_boot, seed, label)
    res["f1"]["ci95"] = boot["f1"]["ci95"]
    res["f1"]["ci_method"] = "percentile bootstrap resampling items" if n_boot > 0 else None
    res["f1"]["bootstrap_valid_resamples"] = boot["f1"]["n_valid"]
    res["precision"]["ci95_cluster_bootstrap"] = boot["precision"]["ci95"]
    res["recall"]["ci95_cluster_bootstrap"] = boot["recall"]["ci95"]
    return res


def detection_section(crit: pd.DataFrame, n_boot: int, seed: int) -> dict[str, Any]:
    df = crit.copy()
    df["truth_both"] = (df["r1"] == 1) & (df["r2"] == 1)
    df["truth_either"] = (df["r1"] == 1) | (df["r2"] == 1)
    disagree = df["r1"] != df["r2"]
    out: dict[str, Any] = {
        "unit": "item x criterion; GPT positive = criterion has a GPT cue (gpt_flagged = 1)",
        "rules": {
            "both": "expert positive when both experts ticked needs improvement (strict; primary)",
            "either": "expert positive when at least one expert ticked it (lenient; secondary)",
        },
        "adjudication": {
            "n_cases": int(len(df)),
            "n_expert_disagreements": int(disagree.sum()),
            "share": proportion(int(disagree.sum()), len(df)),
            "disagreements_gpt_flagged": int((disagree & (df["gpt"] == 1)).sum()),
            "disagreements_gpt_not_flagged": int((disagree & (df["gpt"] == 0)).sum()),
            "n_items_with_any_disagreement": int(df.loc[disagree, "item_id"].nunique()),
            "note": "cases where 'both' and 'either' differ; adjudication would decide them",
        },
    }
    for rule in ("both", "either"):
        col = f"truth_{rule}"
        block: dict[str, Any] = {
            "pooled": detection_block(df, col, n_boot, seed, f"det/{rule}/all"),
            "by_skill": {},
        }
        for skill, g in df.groupby("skill", sort=True):
            per_crit = []
            for cid, gc in g.groupby("criterion_id", sort=False):
                m = detection_block(gc, col, n_boot, seed, f"det/{rule}/{skill}/{cid}")
                per_crit.append({"criterion_id": cid, "criterion_name": gc["criterion_name"].iloc[0], **m})
            block["by_skill"][skill] = {
                "pooled": detection_block(g, col, n_boot, seed, f"det/{rule}/{skill}"),
                "criteria": per_crit,
            }
        out[rule] = block
    return out


def cue_validity_block(g: pd.DataFrame, n_boot: int, seed: int, label: str) -> dict[str, Any]:
    s1, s2 = g["s1"].to_numpy(), g["s2"].to_numpy()
    n = len(g)
    mean_scores = (s1 + s2) / 2.0
    return {
        "n_cues": n,
        "n_items": int(g["item_id"].nunique()),
        "correct_both": proportion(int(((s1 == 2) & (s2 == 2)).sum()), n),
        "correct_at_least_one": proportion(int(((s1 == 2) | (s2 == 2)).sum()), n),
        "at_least_partial_both": proportion(int(((s1 >= 1) & (s2 >= 1)).sum()), n),
        "incorrect_either": proportion(int(((s1 == 0) | (s2 == 0)).sum()), n),
        "incorrect_both": proportion(int(((s1 == 0) & (s2 == 0)).sum()), n),
        "mean_cue_score": {
            **describe(mean_scores),
            "scale": "0-2, mean of both experts per cue",
            "ci95_cluster_bootstrap": cluster_bootstrap_mean(mean_scores, g["item_id"].to_numpy(), n_boot, seed, label),
        },
        "rating_distribution": {
            "expert_1": {CUE_LABELS[c]: int((s1 == c).sum()) for c in CUE_CATEGORIES},
            "expert_2": {CUE_LABELS[c]: int((s2 == c).sum()) for c in CUE_CATEGORIES},
        },
    }


def overall_validity_block(g: pd.DataFrame) -> dict[str, Any]:
    o1, o2 = g["o1"].to_numpy(), g["o2"].to_numpy()
    n = len(g)
    m = (o1 + o2) / 2.0
    dist = {f"{v:g}": int((m == v).sum()) for v in np.arange(0, 3.5, 0.5)}
    return {
        "n_items": n,
        "item_mean_score": describe(m),
        "item_mean_distribution": dist,
        "score_distribution": {"expert_1": _counts(o1, OVERALL_CATEGORIES), "expert_2": _counts(o2, OVERALL_CATEGORIES)},
        "acceptable_mean_ge_2": proportion(int((m >= 2).sum()), n),
        "acceptable_both_ge_2": proportion(int(((o1 >= 2) & (o2 >= 2)).sum()), n),
        "useful_both_eq_3": proportion(int(((o1 == 3) & (o2 == 3)).sum()), n),
        "harmful_any_0": proportion(int(((o1 == 0) | (o2 == 0)).sum()), n),
        "harmful_both_0": proportion(int(((o1 == 0) & (o2 == 0)).sum()), n),
    }


def validity_section(crit: pd.DataFrame, cue: pd.DataFrame, ov: pd.DataFrame, experts: Sequence[str],
                     n_boot: int, seed: int) -> dict[str, Any]:
    time_burden = {}
    for tag, e in (("o1", experts[0]), ("o2", experts[1])):
        time_burden[e] = {
            "step1_seconds": describe(ov[f"step1_seconds_{tag}"]),
            "step2_seconds": describe(ov[f"step2_seconds_{tag}"]),
            "total_seconds": describe(
                ov[f"step1_seconds_{tag}"].astype(float) + ov[f"step2_seconds_{tag}"].astype(float)
            ),
        }
    return {
        "detection": detection_section(crit, n_boot, seed),
        "cues": {
            "note": "expert_1/expert_2 in rating_distribution refer to the experts in the order listed under 'experts'",
            "pooled": cue_validity_block(cue, n_boot, seed, "cue/all"),
            "by_skill": {s: cue_validity_block(g, n_boot, seed, f"cue/{s}") for s, g in cue.groupby("skill")},
        },
        "overall": {
            "pooled": overall_validity_block(ov),
            "by_skill": {s: overall_validity_block(g) for s, g in ov.groupby("skill")},
        },
        "time_burden": time_burden,
    }


# ---------------------------------------------------------------------------
# Report assembly and rendering
# ---------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def analyze(
    items: pd.DataFrame, criteria: pd.DataFrame, cues: pd.DataFrame, overall: pd.DataFrame,
    experts_arg: str | None = None, n_boot: int = 2000, seed: int = 7,
) -> dict[str, Any]:
    frames = {"items": items, "criteria": criteria, "cues": cues, "overall": overall}
    experts, warnings = resolve_experts(frames, experts_arg)
    included, sample = filter_items(items, criteria, cues, overall, experts)
    report: dict[str, Any] = {
        "experts": experts,
        "settings": {"bootstrap_resamples": n_boot, "seed": seed, "confidence_level": 0.95},
        "sample": sample,
        "warnings": warnings,
    }
    if sample["rating_rows_for_unknown_items"]:
        warnings.append(
            f"{len(sample['rating_rows_for_unknown_items'])} item ids appear in rating exports but not in items.csv; ignored"
        )
    if not included:
        warnings.append("no items are eligible and completed by both experts; nothing to analyze")
        report["reliability"] = None
        report["validity"] = None
        return to_jsonable(report)
    crit, cue, ov, table_warnings = build_tables(items, criteria, cues, overall, experts, included)
    warnings.extend(table_warnings)
    report["sample"]["included_by_skill"] = {s: int(c) for s, c in ov["skill"].value_counts().sort_index().items()}
    report["sample"]["n_item_criterion_cases"] = int(len(crit))
    report["sample"]["n_cues"] = int(len(cue))
    report["reliability"] = reliability_section(crit, cue, ov, experts)
    report["validity"] = validity_section(crit, cue, ov, experts, n_boot, seed)
    report["results_paragraph"] = results_paragraph(report)
    return to_jsonable(report)


def f2(x: Any, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:.{digits}f}"


def pct(x: Any, digits: int = 1) -> str:
    if x is None:
        return "NA"
    return f"{100 * x:.{digits}f}%"


def ci_str(ci: Sequence | None, digits: int = 2, as_pct: bool = False) -> str:
    if not ci or ci[0] is None or ci[1] is None:
        return "NA"
    if as_pct:
        return f"{pct(ci[0])} to {pct(ci[1])}"
    return f"{f2(ci[0], digits)} to {f2(ci[1], digits)}"


def prop_str(p: dict[str, Any]) -> str:
    return f"{pct(p['value'])} ({p['k']}/{p['n']}; 95% CI {ci_str(p['ci95'], as_pct=True)})"


def results_paragraph(r: dict[str, Any]) -> str:
    s = r["sample"]
    rel, val = r["reliability"], r["validity"]
    by_skill = s.get("included_by_skill", {})
    skills = ", ".join(f"{n} {k}" for k, n in by_skill.items())
    ovr = rel["overall_score"]["pooled"]
    st1 = rel["step1_detection"]["pooled"]
    cue_rel = rel["cue_ratings"]["pooled"]
    ovv = val["overall"]["pooled"]
    cv = val["cues"]["pooled"]
    det_b = val["detection"]["both"]["pooled"]
    det_e = val["detection"]["either"]["pooled"]
    icc2, icc3 = ovr["ICC2_1"], ovr["ICC3_1"]
    excl = ""
    if s["n_excluded"]:
        excl = (
            f" ({s['n_excluded']} of {s['n_items_in_export']} exported items were excluded: "
            f"{s['n_excluded_not_eligible']} not eligible and {s['n_excluded_eligible_but_incomplete']} "
            "not completed by both experts)"
        )
    return (
        f"Two badminton experts independently evaluated N = {s['n_included']} GPT-generated feedback responses "
        f"({skills}){excl}, first marking which rubric checkpoints needed improvement without seeing GPT's output "
        f"and then rating each GPT cue and the response as a whole. "
        f"Agreement on the overall score (0–3) was ICC(2,1) = {f2(icc2['value'])} (95% CI {ci_str(icc2['ci95'])}; "
        f"ICC(3,1) = {f2(icc3['value'])}, {ci_str(icc3['ci95'])}) with quadratic-weighted Cohen's κ = "
        f"{f2(ovr['cohen_kappa_quadratic'])}; agreement on step-1 checkpoint detection was "
        f"{pct(st1['percent_agreement'])} (Cohen's κ = {f2(st1['cohen_kappa'])}, n = {st1['n']} item × checkpoint cases), "
        f"and on cue ratings {pct(cue_rel['percent_agreement'])} (linear-weighted κ = {f2(cue_rel['cohen_kappa_linear'])}, "
        f"n = {cue_rel['n']} cues). "
        f"{pct(ovv['acceptable_mean_ge_2']['value'])} of responses were judged acceptable (mean expert score ≥ 2; "
        f"95% CI {ci_str(ovv['acceptable_mean_ge_2']['ci95'], as_pct=True)}; both experts ≥ 2: "
        f"{pct(ovv['acceptable_both_ge_2']['value'])}), with a mean score of {f2(ovv['item_mean_score']['mean'])} "
        f"± {f2(ovv['item_mean_score']['sd'])}, and {pct(ovv['harmful_any_0']['value'])} were rated incorrect or "
        f"potentially harmful by at least one expert. Of {cv['n_cues']} individual cues, "
        f"{pct(cv['correct_both']['value'])} were rated correct by both experts "
        f"({pct(cv['correct_at_least_one']['value'])} by at least one) and "
        f"{pct(cv['incorrect_either']['value'])} were rated incorrect by at least one expert. "
        f"Against the strict expert consensus (both experts flagged the checkpoint), GPT's checkpoint selection "
        f"achieved precision = {f2(det_b['precision']['value'])} ({ci_str(det_b['precision']['ci95'])}), "
        f"recall = {f2(det_b['recall']['value'])} ({ci_str(det_b['recall']['ci95'])}), and "
        f"F1 = {f2(det_b['f1']['value'])} ({ci_str(det_b['f1']['ci95'])}); against the lenient consensus "
        f"(at least one expert) precision = {f2(det_e['precision']['value'])}, recall = "
        f"{f2(det_e['recall']['value'])}, F1 = {f2(det_e['f1']['value'])}."
    )


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def render_markdown(r: dict[str, Any]) -> str:
    e1, e2 = r["experts"]
    s = r["sample"]
    out: list[str] = ["# GPT feedback validation — reliability and validity", ""]
    out.append(f"Experts: `{e1}` (expert 1), `{e2}` (expert 2). "
               f"Bootstrap resamples: {r['settings']['bootstrap_resamples']}, seed {r['settings']['seed']}. "
               "All intervals are 95%.")
    out.append("")
    if r.get("results_paragraph"):
        out += ["## Results paragraph", "", r["results_paragraph"], ""]

    out += ["## Sample", ""]
    out.append(_md_table(["", "n"], [
        ["Items in export", s["n_items_in_export"]],
        ["Eligible", s["n_eligible"]],
        ["Included (eligible and completed by both experts)", s["n_included"]],
        ["Excluded: not eligible", s["n_excluded_not_eligible"]],
        ["Excluded: eligible but not completed by both", s["n_excluded_eligible_but_incomplete"]],
    ]))
    out.append("")
    if s.get("included_by_skill"):
        out.append("Included by skill: " + ", ".join(f"{k} n = {v}" for k, v in s["included_by_skill"].items())
                   + f"; item × criterion cases n = {s.get('n_item_criterion_cases')}; cues n = {s.get('n_cues')}.")
        out.append("")
    if s["excluded_by_primary_reason"]:
        out.append("Exclusions by primary reason (first failing check):")
        out.append("")
        out.append(_md_table(["Reason", "n items"], s["excluded_by_primary_reason"].items()))
        out.append("")
    if s["not_eligible_breakdown"]:
        out.append(_md_table(["Not eligible because", "n items"], s["not_eligible_breakdown"].items()))
        out.append("")
    if r["warnings"]:
        out += ["**Warnings**", ""] + [f"- {w}" for w in r["warnings"]] + [""]
    if r["reliability"] is None:
        return "\n".join(out)

    rel, val = r["reliability"], r["validity"]

    # --- reliability
    out += ["## 1. Inter-rater reliability", "", "### Step 1 checkpoint detection (needs improvement, binary)", ""]
    hdr = ["Skill", "Criterion", "n", "Agreement", "Cohen κ", "Fleiss κ", "PABAK",
           f"Prev. {e1}", f"Prev. {e2}", "Note"]
    rows = []

    def brow(skill, name, b):
        p = b["prevalence_needs_improvement"]
        return [skill, name, b["n"], pct(b["percent_agreement"]), f2(b["cohen_kappa"]), f2(b["fleiss_kappa"]),
                f2(b["pabak"]), pct(p[e1]["value"]), pct(p[e2]["value"]), b["note"] or ""]

    st1 = rel["step1_detection"]
    for skill, block in st1["by_skill"].items():
        for c in block["criteria"]:
            rows.append(brow(skill, f"{c['criterion_name']} (`{c['criterion_id']}`)", c))
        rows.append(brow(skill, "**all criteria**", block["pooled"]))
    rows.append(brow("**all**", "**all criteria**", st1["pooled"]))
    out += [_md_table(hdr, rows), ""]

    out += ["### Cue ratings (incorrect = 0, partial = 1, correct = 2)", ""]
    hdr = ["Skill", "n cues", "Agreement", "Within 1", "κ", "κ linear", "κ quadratic", "Fleiss κ", "Note"]

    def orow(label, b):
        return [label, b["n"], pct(b["percent_agreement"]), pct(b["within_one_category"]),
                f2(b["cohen_kappa_unweighted"]), f2(b["cohen_kappa_linear"]), f2(b["cohen_kappa_quadratic"]),
                f2(b["fleiss_kappa"]), b["note"] or ""]

    cr = rel["cue_ratings"]
    rows = [orow(k, v) for k, v in cr["by_skill"].items()] + [orow("**all**", cr["pooled"])]
    out += [_md_table(hdr, rows), ""]

    out += ["### Overall score (0–3)", ""]
    hdr = ["Skill", "n items", f"Mean {e1}", f"Mean {e2}", "Agreement", "κ quadratic", "ICC(2,1) [95% CI]",
            "ICC(3,1) [95% CI]", "Fleiss κ", "Note"]

    def icc_str(x):
        if not x:
            return "NA"
        return f"{f2(x['value'])} [{ci_str(x['ci95'])}]"

    def ovrow(label, b):
        note = "; ".join(x for x in (b["note"], b["icc_note"]) if x)
        return [label, b["n"], f2(b["mean_score"][e1]["mean"]), f2(b["mean_score"][e2]["mean"]),
                pct(b["percent_agreement"]), f2(b["cohen_kappa_quadratic"]), icc_str(b["ICC2_1"]),
                icc_str(b["ICC3_1"]), f2(b["fleiss_kappa"]), note]

    osec = rel["overall_score"]
    rows = [ovrow(k, v) for k, v in osec["by_skill"].items()] + [ovrow("**all**", osec["pooled"])]
    out += [_md_table(hdr, rows), ""]

    # --- validity
    det = val["detection"]
    adj = det["adjudication"]
    out += ["## 2. GPT validity against expert consensus", "", "### Checkpoint detection", ""]
    out.append(
        f"Unit: item × criterion (n = {adj['n_cases']}). GPT positive = GPT raised a cue for the criterion. "
        f"Experts disagreed on {adj['n_expert_disagreements']} cases ({pct(adj['share']['value'])}; "
        f"{adj['n_items_with_any_disagreement']} items) that adjudication would have to settle "
        f"({adj['disagreements_gpt_flagged']} flagged by GPT, {adj['disagreements_gpt_not_flagged']} not)."
    )
    out.append("")
    hdr = ["Rule", "Skill", "Criterion", "n", "TP", "FP", "FN", "TN", "Precision [CI]", "Recall [CI]",
           "Specificity [CI]", "F1 [bootstrap CI]"]

    def drow(rule, skill, name, m):
        c = m["counts"]
        return [rule, skill, name, c["N"], c["TP"], c["FP"], c["FN"], c["TN"],
                f"{f2(m['precision']['value'])} [{ci_str(m['precision']['ci95'])}]",
                f"{f2(m['recall']['value'])} [{ci_str(m['recall']['ci95'])}]",
                f"{f2(m['specificity']['value'])} [{ci_str(m['specificity']['ci95'])}]",
                f"{f2(m['f1']['value'])} [{ci_str(m['f1']['ci95'])}]"]

    for rule, label in (("both", "both (strict, primary)"), ("either", "either (lenient)")):
        rows = [drow(label, "**all**", "**all**", det[rule]["pooled"])]
        for skill, block in det[rule]["by_skill"].items():
            rows.append(drow(label, skill, "**all**", block["pooled"]))
            for c in block["criteria"]:
                rows.append(drow(label, skill, f"{c['criterion_name']} (`{c['criterion_id']}`)", c))
        out += [_md_table(hdr, rows), ""]
    out.append("Precision/recall/specificity CIs: Wilson score intervals (cases treated as independent). "
               "F1 CI: percentile bootstrap resampling items. Cluster-bootstrap CIs for precision and recall "
               "are in the JSON report.")
    out.append("")

    out += ["### Cue correctness", ""]
    hdr = ["Skill", "n cues (items)", "Correct (both)", "Correct (≥1 expert)", "Incorrect (≥1 expert)",
           "Mean cue score 0–2 ± SD [bootstrap CI]"]

    def crow(label, b):
        m = b["mean_cue_score"]
        return [label, f"{b['n_cues']} ({b['n_items']})", prop_str(b["correct_both"]),
                prop_str(b["correct_at_least_one"]), prop_str(b["incorrect_either"]),
                f"{f2(m['mean'])} ± {f2(m['sd'])} [{ci_str(m['ci95_cluster_bootstrap'])}]"]

    rows = [crow(k, v) for k, v in val["cues"]["by_skill"].items()] + [crow("**all**", val["cues"]["pooled"])]
    out += [_md_table(hdr, rows), ""]

    out += ["### Overall response quality", ""]
    hdr = ["Skill", "n items", "Mean ± SD (per-item mean)", "Median [IQR]", "Acceptable (mean ≥ 2)",
           "Acceptable (both ≥ 2)", "Harmful/incorrect (any 0)"]

    def vrow(label, b):
        d = b["item_mean_score"]
        return [label, b["n_items"], f"{f2(d['mean'])} ± {f2(d['sd'])}",
                f"{f2(d['median'])} [{f2(d['q1'])}–{f2(d['q3'])}]", prop_str(b["acceptable_mean_ge_2"]),
                prop_str(b["acceptable_both_ge_2"]), prop_str(b["harmful_any_0"])]

    ov = val["overall"]
    rows = [vrow(k, v) for k, v in ov["by_skill"].items()] + [vrow("**all**", ov["pooled"])]
    out += [_md_table(hdr, rows), ""]
    dist = ov["pooled"]["item_mean_distribution"]
    out.append("Distribution of per-item mean score (all skills):")
    out.append("")
    out.append(_md_table(["Mean score"] + list(dist.keys()), [["n items"] + list(dist.values())]))
    out.append("")
    sd = ov["pooled"]["score_distribution"]
    out.append(_md_table(["Expert", "0", "1", "2", "3"],
                         [[e1] + list(sd["expert_1"].values()), [e2] + list(sd["expert_2"].values())]))
    out.append("")

    out += ["### Time burden (included items)", ""]
    hdr = ["Expert", "Step", "n", "Median (s)", "IQR (s)"]
    rows = []
    for e, t in val["time_burden"].items():
        for step in ("step1_seconds", "step2_seconds", "total_seconds"):
            d = t[step]
            rows.append([e, step.replace("_seconds", ""), d["n"], f2(d["median"], 1),
                         f"{f2(d['q1'], 1)}–{f2(d['q3'], 1)}"])
    out += [_md_table(hdr, rows), ""]
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inter-rater reliability and GPT validity statistics for the GPT feedback validation study.",
    )
    p.add_argument("--items", required=True, help="items.csv export")
    p.add_argument("--criteria", required=True, help="criteria.csv export")
    p.add_argument("--cues", required=True, help="cues.csv export")
    p.add_argument("--overall", required=True, help="overall.csv export")
    p.add_argument("--experts", help="comma-separated ids of the two experts (default: the two non-admin ids found)")
    p.add_argument("--out", help="write the JSON report here")
    p.add_argument("--markdown", help="write the Markdown report here")
    p.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples (0 disables; default 2000)")
    p.add_argument("--seed", type=int, default=7, help="bootstrap seed (default 7)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap < 0:
        print("error: --bootstrap must be >= 0", file=sys.stderr)
        return 2
    try:
        report = analyze(
            read_export(args.items, "items"),
            read_export(args.criteria, "criteria"),
            read_export(args.cues, "cues"),
            read_export(args.overall, "overall"),
            experts_arg=args.experts, n_boot=args.bootstrap, seed=args.seed,
        )
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    markdown = render_markdown(report)
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(markdown + "\n", encoding="utf-8")
    if not args.out and not args.markdown:
        print(markdown)
    else:
        s = report["sample"]
        print(f"included {s['n_included']} of {s['n_items_in_export']} items; "
              + ", ".join(x for x in (f"JSON -> {args.out}" if args.out else "",
                                       f"Markdown -> {args.markdown}" if args.markdown else "") if x))
    for w in report["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
