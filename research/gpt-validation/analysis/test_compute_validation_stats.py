"""Tests for compute_validation_stats.py.

Run from this directory:
    uv run --no-project --with pandas --with numpy --with pytest pytest -q
"""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import compute_validation_stats as cvs

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Special functions
# ---------------------------------------------------------------------------


def test_betainc_closed_forms():
    for x in (0.1, 0.37, 0.5, 0.93):
        assert cvs.betainc_reg(1, 1, x) == pytest.approx(x, abs=1e-12)
        assert cvs.betainc_reg(3, 1, x) == pytest.approx(x**3, abs=1e-12)
        assert cvs.betainc_reg(1, 4, x) == pytest.approx(1 - (1 - x) ** 4, abs=1e-12)
        # symmetry I_x(a,b) = 1 - I_{1-x}(b,a)
        assert cvs.betainc_reg(2.5, 7.3, x) == pytest.approx(1 - cvs.betainc_reg(7.3, 2.5, 1 - x), abs=1e-12)


def test_f_quantiles_match_tables():
    # F_{0.95}(1, 10) = t_{0.975,10}^2 = 2.228139^2
    assert cvs.f_ppf(0.95, 1, 10) == pytest.approx(4.9646, abs=1e-4)
    assert cvs.f_ppf(0.975, 5, 15) == pytest.approx(3.5764, abs=1e-4)
    assert cvs.f_ppf(0.95, 3, 20) == pytest.approx(3.0984, abs=1e-4)
    # round trip, including non-integer df
    for p, d1, d2 in ((0.975, 5, 7.43), (0.025, 2.2, 40), (0.5, 12, 3)):
        assert cvs.f_cdf(cvs.f_ppf(p, d1, d2), d1, d2) == pytest.approx(p, abs=1e-10)


# ---------------------------------------------------------------------------
# Kappa
# ---------------------------------------------------------------------------


def _from_table(table):
    """Expand a 2-rater contingency table (rows = rater A) into rating vectors."""
    a, b = [], []
    for i, row in enumerate(table):
        for j, count in enumerate(row):
            a += [i] * count
            b += [j] * count
    return a, b


def test_cohen_kappa_textbook_example():
    # Wikipedia "Cohen's kappa": 50 proposals, Yes/Yes 20, A-yes/B-no 5, A-no/B-yes 10, No/No 15
    # p_o = 0.70, p_e = 0.5*0.6 + 0.5*0.4 = 0.50, kappa = 0.40
    a, b = _from_table([[15, 10], [5, 20]])
    res = cvs.cohen_kappa(a, b, [0, 1])
    assert res["n"] == 50
    assert res["percent_agreement"] == pytest.approx(0.70)
    assert res["expected_agreement"] == pytest.approx(0.50)
    assert res["kappa"] == pytest.approx(0.40)


def test_cohen_kappa_small_hand_example_unweighted_and_weighted():
    a = [0, 0, 1, 2]
    b = [0, 1, 1, 2]
    # marginals A = (.5,.25,.25), B = (.25,.5,.25); p_o = .75; p_e = .3125
    assert cvs.cohen_kappa(a, b, [0, 1, 2])["kappa"] == pytest.approx((0.75 - 0.3125) / 0.6875)  # 7/11
    # linear weights 0,.5,1: observed disagreement .125, expected .4375 -> 1 - .125/.4375 = 5/7
    assert cvs.cohen_kappa(a, b, [0, 1, 2], "linear")["kappa"] == pytest.approx(5 / 7)
    # quadratic weights 0,.25,1: observed .0625, expected .3125 -> 0.8
    assert cvs.cohen_kappa(a, b, [0, 1, 2], "quadratic")["kappa"] == pytest.approx(0.8)


def test_weighted_kappa_with_two_categories_equals_unweighted():
    a, b = _from_table([[15, 10], [5, 20]])
    for w in ("linear", "quadratic"):
        assert cvs.cohen_kappa(a, b, [0, 1], w)["kappa"] == pytest.approx(0.40)


def test_perfect_agreement_and_degenerate_kappa():
    assert cvs.cohen_kappa([0, 1, 1], [0, 1, 1], [0, 1])["kappa"] == pytest.approx(1.0)
    res = cvs.cohen_kappa([1, 1, 1], [1, 1, 1], [0, 1])
    assert res["kappa"] is None
    assert res["percent_agreement"] == 1.0
    assert "p_e = 1" in res["note"]
    fk = cvs.fleiss_kappa([[1, 1], [1, 1]], [0, 1])
    assert fk["kappa"] is None and fk["note"]
    # quadratic weighted kappa with a single used category is also undefined
    assert cvs.cohen_kappa([2, 2], [2, 2], [0, 1, 2, 3], "quadratic")["kappa"] is None


def test_fleiss_two_raters_equals_scotts_pi():
    a = [0, 0, 1, 2, 2, 1, 0, 2, 1, 1]
    b = [0, 1, 1, 2, 0, 1, 0, 2, 2, 1]
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    pooled = [(a.count(c) + b.count(c)) / (2 * n) for c in (0, 1, 2)]
    p_e = sum(p * p for p in pooled)
    scott_pi = (p_o - p_e) / (1 - p_e)
    assert cvs.fleiss_kappa(list(zip(a, b)), [0, 1, 2])["kappa"] == pytest.approx(scott_pi)
    # and the hand example above: pooled (.375,.375,.25) -> p_e = .34375 -> pi = 13/21
    assert cvs.fleiss_kappa(list(zip([0, 0, 1, 2], [0, 1, 1, 2])), [0, 1, 2])["kappa"] == pytest.approx(13 / 21)


def test_fleiss_kappa_textbook_example():
    # Wikipedia "Fleiss' kappa" worked example: 10 subjects, 14 raters, 5 categories -> 0.210
    counts = np.array([
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
    ])
    res = cvs.fleiss_kappa_from_counts(counts)
    assert res["observed_agreement"] == pytest.approx(0.378, abs=1e-3)
    assert res["expected_agreement"] == pytest.approx(0.213, abs=1e-3)
    assert res["kappa"] == pytest.approx(0.210, abs=1e-3)


# ---------------------------------------------------------------------------
# ICC
# ---------------------------------------------------------------------------

SHROUT_FLEISS = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]


def test_icc_shrout_fleiss_1979_k4():
    res = cvs.icc(SHROUT_FLEISS)
    ms = res["mean_squares"]
    # Shrout & Fleiss (1979) Table 2 mean squares
    assert ms["MSR"] == pytest.approx(11.24, abs=0.01)
    assert ms["MSW"] == pytest.approx(6.26, abs=0.01)
    assert ms["MSC"] == pytest.approx(32.49, abs=0.01)
    assert ms["MSE"] == pytest.approx(1.02, abs=0.01)
    # Table 4 point estimates
    assert res["ICC1_1"]["value"] == pytest.approx(0.17, abs=0.005)
    assert res["ICC2_1"]["value"] == pytest.approx(0.29, abs=0.005)
    assert res["ICC3_1"]["value"] == pytest.approx(0.71, abs=0.005)
    assert res["ICC1_k"]["value"] == pytest.approx(0.44, abs=0.005)
    assert res["ICC2_k"]["value"] == pytest.approx(0.62, abs=0.005)
    assert res["ICC3_k"]["value"] == pytest.approx(0.91, abs=0.005)
    # 95% CIs as reported by R psych::ICC for this dataset
    assert res["ICC2_1"]["ci95"] == pytest.approx([0.019, 0.76], abs=0.005)
    assert res["ICC3_1"]["ci95"] == pytest.approx([0.34, 0.95], abs=0.005)
    assert res["ICC1_1"]["ci95"] == pytest.approx([-0.13, 0.72], abs=0.005)
    assert res["ICC2_k"]["ci95"] == pytest.approx([0.071, 0.93], abs=0.005)
    assert res["ICC3_k"]["ci95"] == pytest.approx([0.68, 0.99], abs=0.01)
    assert res["ICC3_1"]["F"] == pytest.approx(11.03, abs=0.01)
    assert res["ICC3_1"]["p_value"] < 0.001


def test_icc_two_raters_hand_computed():
    y = np.array([[1, 2], [2, 2], [3, 3], [0, 1], [2, 3]], dtype=float)
    n, k = y.shape
    g = y.mean()
    msr = k * ((y.mean(1) - g) ** 2).sum() / (n - 1)
    msc = n * ((y.mean(0) - g) ** 2).sum() / (k - 1)
    sse = ((y - g) ** 2).sum() - msr * (n - 1) - msc * (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    res = cvs.icc(y)
    assert res["ICC3_1"]["value"] == pytest.approx((msr - mse) / (msr + mse))
    assert res["ICC2_1"]["value"] == pytest.approx((msr - mse) / (msr + mse + 2 * (msc - mse) / n))
    lo, hi = res["ICC2_1"]["ci95"]
    assert lo < res["ICC2_1"]["value"] < hi
    # rater 2 is systematically higher -> absolute agreement below consistency
    assert res["ICC2_1"]["value"] < res["ICC3_1"]["value"]


def test_icc_constant_scores_is_null():
    res = cvs.icc([[2, 2], [2, 2], [2, 2]])
    assert res["ICC2_1"]["value"] is None and res["ICC3_1"]["value"] is None
    assert res["note"]


# ---------------------------------------------------------------------------
# Proportions and metrics
# ---------------------------------------------------------------------------


def test_wilson_ci_known_values():
    assert cvs.wilson_ci(8, 10) == pytest.approx((0.4902, 0.9433), abs=1e-4)
    assert cvs.wilson_ci(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert cvs.wilson_ci(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)
    assert cvs.wilson_ci(50, 100) == pytest.approx((0.4038, 0.5962), abs=1e-4)
    assert cvs.wilson_ci(0, 0) == (None, None)


def test_binary_metrics():
    m = cvs.binary_metrics(tp=8, fp=2, fn=4, tn=6)
    assert m["precision"]["value"] == pytest.approx(0.8)
    assert m["recall"]["value"] == pytest.approx(8 / 12)
    assert m["specificity"]["value"] == pytest.approx(6 / 8)
    assert m["accuracy"]["value"] == pytest.approx(14 / 20)
    assert m["f1"]["value"] == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))
    assert m["precision"]["ci95"] == pytest.approx([0.4902, 0.9433], abs=1e-4)
    empty = cvs.binary_metrics(0, 0, 0, 5)
    assert empty["precision"]["value"] is None and empty["f1"]["value"] is None


def test_cluster_bootstrap_confusion_is_reproducible_and_brackets_estimate():
    rng = np.random.default_rng(1)
    per_item = rng.integers(0, 3, size=(40, 4))
    a = cvs.cluster_bootstrap_confusion(per_item, 500, 7, "x")
    b = cvs.cluster_bootstrap_confusion(per_item, 500, 7, "x")
    assert a == b
    tp, fp, fn, _ = per_item.sum(0)
    f1 = 2 * tp / (2 * tp + fp + fn)
    lo, hi = a["f1"]["ci95"]
    assert lo < f1 < hi


# ---------------------------------------------------------------------------
# Synthetic exports
# ---------------------------------------------------------------------------

CRITERIA = {
    "serve": [("sv_c1", "握拍"), ("sv_c2", "站位"), ("sv_c3", "引拍"), ("sv_c4", "擊球點"), ("sv_c5", "隨揮"), ("sv_c6", "重心")],
    "smash": [("sm_c1", "側身"), ("sm_c2", "引拍"), ("sm_c3", "擊球點"), ("sm_c4", "轉體"), ("sm_c5", "手腕"), ("sm_c6", "落地")],
}
EXPERTS = ("expert-1", "expert-2")


def _write(path: Path, header: list[str], rows: list[list]):
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def make_exports(tmp_path: Path, n_items: int = 24, seed: int = 3):
    """Synthetic exports with columns exactly as in DESIGN.md.

    Items 0..n-1 are eligible and complete, plus:
      - one failed item (not eligible, no ratings)
      - one deterministic_fallback item (not eligible)
      - one eligible item expert-2 never opened
      - one eligible item expert-2 did step 1 only (completed = false)
      - one eligible item where expert-1 is missing a cue rating
    """
    rng = np.random.default_rng(seed)
    items, crit_rows, cue_rows, ov_rows = [], [], [], []
    truth = {}  # (item_id, criterion_id) -> (r1, r2, gpt)

    def add_item(idx, skill, *, status="ready", source="openai", ratings="complete"):
        item_id = f"{skill}-{idx:010x}"
        crits = CRITERIA[skill]
        eligible = status == "ready" and source == "openai"
        gpt_flags = [cid for cid, _ in crits if rng.random() < 0.4] if eligible else []
        if eligible and not gpt_flags:
            gpt_flags = [crits[0][0]]
        items.append([
            item_id, f"{'SV' if skill == 'serve' else 'SM'}-{idx:03d}", skill, f"student{idx}_{skill}.mp4",
            f"CG{idx:02d}", status, "true" if eligible else "false", source, "gpt-5.6-terra" if eligible else "",
            "right", round(float(rng.uniform(2, 9)), 1), len(gpt_flags), ";".join(gpt_flags),
        ])
        if ratings == "none":
            return item_id
        for e_i, expert in enumerate(EXPERTS):
            mode = ratings if isinstance(ratings, str) else ratings.get(expert, "complete")
            if mode == "none":
                continue
            for cid, name in crits:
                g = int(cid in gpt_flags)
                # experts mostly agree with each other and with GPT
                base = g if rng.random() < 0.8 else 1 - g
                val = base if rng.random() < 0.85 else 1 - base
                key = (item_id, cid)
                truth.setdefault(key, [None, None, g])[e_i] = val
                crit_rows.append([item_id, skill, expert, cid, name, val, g])
            if mode == "step1_only":
                ov_rows.append([item_id, skill, expert, "", "", round(float(rng.uniform(20, 60)), 1), "", "false"])
                continue
            for ci, cid in enumerate(gpt_flags, start=1):
                if mode == "missing_cue" and ci == 1:
                    continue
                rating = rng.choice(["correct", "correct", "partial", "incorrect"])
                cue_rows.append([item_id, skill, expert, ci, cid, rating, cvs.CUE_SCORES[rating]])
            score = int(rng.choice([0, 1, 2, 2, 3, 3]))
            ov_rows.append([
                item_id, skill, expert, score, "ok" if rng.random() < 0.3 else "",
                round(float(rng.uniform(20, 60)), 1), round(float(rng.uniform(15, 50)), 1), "true",
            ])
        return item_id

    for i in range(n_items):
        add_item(i, "serve" if i % 2 == 0 else "smash")
    add_item(100, "serve", status="failed", source="", ratings="none")
    add_item(101, "smash", source="deterministic_fallback", ratings="none")
    add_item(102, "serve", ratings={"expert-2": "none"})
    add_item(103, "smash", ratings={"expert-2": "step1_only"})
    add_item(104, "serve", ratings={"expert-1": "missing_cue"})

    paths = {k: tmp_path / f"{k}.csv" for k in ("items", "criteria", "cues", "overall")}
    _write(paths["items"], cvs.REQUIRED_COLUMNS["items"], items)
    _write(paths["criteria"], cvs.REQUIRED_COLUMNS["criteria"], crit_rows)
    _write(paths["cues"], cvs.REQUIRED_COLUMNS["cues"], cue_rows)
    _write(paths["overall"], cvs.REQUIRED_COLUMNS["overall"], ov_rows)
    return paths, truth


def _load(paths):
    return [cvs.read_export(paths[k], k) for k in ("items", "criteria", "cues", "overall")]


def test_synthetic_columns_match_design_doc(tmp_path):
    design = (HERE.parent / "DESIGN.md").read_text(encoding="utf-8")
    for name, cols in cvs.REQUIRED_COLUMNS.items():
        line = next(l for l in design.splitlines() if l.strip().startswith(f"- `{name}.csv`"))
        spec = re.sub(r"\([^)]*\)", "", line.split(":", 1)[1]).replace("`", "")
        documented = [c.strip() for c in spec.split(",") if c.strip()]
        assert documented == cols, name


def test_filtering_eligibility_and_completion(tmp_path):
    paths, _ = make_exports(tmp_path, n_items=6)
    items, criteria, cues, overall = _load(paths)
    included, summary = cvs.filter_items(items, criteria, cues, overall, EXPERTS)
    assert len(included) == 6
    assert summary["n_items_in_export"] == 11
    assert summary["n_excluded"] == 5
    assert summary["n_excluded_not_eligible"] == 2
    assert summary["n_excluded_eligible_but_incomplete"] == 3
    assert summary["excluded_by_primary_reason"] == {
        "not_eligible": 2,
        "not_rated_by:expert-2": 1,
        "not_completed_by:expert-2": 1,
        "incomplete_cue_ratings:expert-1": 1,
    }
    assert summary["not_eligible_breakdown"] == {
        "status=failed, coaching_source=(blank)": 1,
        "status=ready, coaching_source=deterministic_fallback": 1,
    }
    # source filenames never leak into the summary
    assert "student" not in json.dumps(summary)


def test_filtering_detects_missing_step1_answer(tmp_path):
    paths, _ = make_exports(tmp_path, n_items=4)
    items, criteria, cues, overall = _load(paths)
    first = items["item_id"].iloc[0]
    mask = (criteria["item_id"] == first) & (criteria["expert_id"] == "expert-1")
    criteria.loc[criteria[mask].index[0], "expert_needs_improvement"] = np.nan
    included, summary = cvs.filter_items(items, criteria, cues, overall, EXPERTS)
    assert first not in included
    assert summary["excluded_by_any_reason"]["incomplete_step1:expert-1"] == 1
    # dropping a whole criterion row for one expert is also incomplete
    criteria2 = _load(paths)[1]
    criteria2 = criteria2.drop(criteria2[(criteria2["item_id"] == first) & (criteria2["expert_id"] == "expert-2")].index[:1])
    included2, _ = cvs.filter_items(items, criteria2, cues, overall, EXPERTS)
    assert first not in included2


def test_expert_resolution_errors(tmp_path):
    paths, _ = make_exports(tmp_path, n_items=4)
    items, criteria, cues, overall = _load(paths)
    frames = {"items": items, "criteria": criteria, "cues": cues, "overall": overall}
    assert cvs.resolve_experts(frames, None)[0] == list(EXPERTS)
    with pytest.raises(cvs.DataError, match="exactly two"):
        cvs.resolve_experts(frames, "expert-1")
    with pytest.raises(cvs.DataError, match="not found"):
        cvs.resolve_experts(frames, "expert-1,expert-9")
    extra = overall.iloc[:1].copy()
    extra["expert_id"] = "expert-3"
    frames3 = {**frames, "overall": pd.concat([overall, extra])}
    with pytest.raises(cvs.DataError, match="exactly two experts"):
        cvs.resolve_experts(frames3, None)
    experts, warnings = cvs.resolve_experts(frames3, "expert-1,expert-2")
    assert experts == ["expert-1", "expert-2"] and warnings
    # admin id is ignored in auto-detection
    admin = overall.iloc[:1].copy()
    admin["expert_id"] = "admin"
    assert cvs.resolve_experts({**frames, "overall": pd.concat([overall, admin])}, None)[0] == list(EXPERTS)


def test_end_to_end(tmp_path):
    paths, truth = make_exports(tmp_path, n_items=24)
    out_json, out_md = tmp_path / "report.json", tmp_path / "report.md"
    rc = cvs.main([
        "--items", str(paths["items"]), "--criteria", str(paths["criteria"]), "--cues", str(paths["cues"]),
        "--overall", str(paths["overall"]), "--out", str(out_json), "--markdown", str(out_md),
        "--bootstrap", "300", "--seed", "7",
    ])
    assert rc == 0
    report = json.loads(out_json.read_text(encoding="utf-8"))
    md = out_md.read_text(encoding="utf-8")

    assert report["experts"] == list(EXPERTS)
    s = report["sample"]
    assert s["n_included"] == 24 and s["n_excluded"] == 5
    assert s["included_by_skill"] == {"serve": 12, "smash": 12}
    assert s["n_item_criterion_cases"] == 24 * 6

    # detection counts match an independent computation from the generator's truth
    included_truth = [v for (iid, _), v in truth.items()
                      if not any(iid.endswith(f"{x:010x}") for x in (102, 103, 104))]
    assert len(included_truth) == 144
    for rule, fn in (("both", lambda a, b: a and b), ("either", lambda a, b: a or b)):
        tp = sum(1 for a, b, g in included_truth if g and fn(a, b))
        fp = sum(1 for a, b, g in included_truth if g and not fn(a, b))
        fn_ = sum(1 for a, b, g in included_truth if not g and fn(a, b))
        tn = sum(1 for a, b, g in included_truth if not g and not fn(a, b))
        pooled = report["validity"]["detection"][rule]["pooled"]
        assert pooled["counts"] == {"TP": tp, "FP": fp, "FN": fn_, "TN": tn, "N": 144}
        lo, hi = pooled["f1"]["ci95"]
        assert lo <= pooled["f1"]["value"] <= hi
        per_skill = report["validity"]["detection"][rule]["by_skill"]
        assert sum(b["pooled"]["counts"]["N"] for b in per_skill.values()) == 144
        assert all(len(b["criteria"]) == 6 for b in per_skill.values())
    n_dis = sum(1 for a, b, _ in included_truth if a != b)
    assert report["validity"]["detection"]["adjudication"]["n_expert_disagreements"] == n_dis
    both, either = (report["validity"]["detection"][r]["pooled"]["counts"] for r in ("both", "either"))
    assert (either["TP"] + either["FN"]) - (both["TP"] + both["FN"]) == n_dis

    # step 1 reliability pooled equals direct computation
    r1 = [a for a, _, _ in included_truth]
    r2 = [b for _, b, _ in included_truth]
    st1 = report["reliability"]["step1_detection"]["pooled"]
    assert st1["n"] == 144
    assert st1["cohen_kappa"] == pytest.approx(cvs.cohen_kappa(r1, r2, [0, 1])["kappa"])
    assert set(report["reliability"]["step1_detection"]["by_skill"]) == {"serve", "smash"}

    ovr = report["reliability"]["overall_score"]["pooled"]
    assert ovr["n"] == 24
    assert ovr["ICC2_1"]["value"] is not None
    ovv = report["validity"]["overall"]["pooled"]
    assert ovv["n_items"] == 24
    assert sum(ovv["item_mean_distribution"].values()) == 24
    assert ovv["acceptable_both_ge_2"]["k"] <= ovv["acceptable_mean_ge_2"]["k"]

    cues = report["validity"]["cues"]["pooled"]
    assert cues["n_cues"] == report["reliability"]["cue_ratings"]["pooled"]["n"] == s["n_cues"]
    assert cues["correct_both"]["k"] <= cues["correct_at_least_one"]["k"]

    tb = report["validity"]["time_burden"]
    assert tb["expert-1"]["step1_seconds"]["n"] == 24

    assert "Two badminton experts independently evaluated N = 24" in md
    assert "ICC(2,1)" in md and "Cohen" in md and "F1" in md
    assert "student" not in md and "student" not in out_json.read_text(encoding="utf-8")

    # reproducible with the same seed
    out2 = tmp_path / "report2.json"
    cvs.main([
        "--items", str(paths["items"]), "--criteria", str(paths["criteria"]), "--cues", str(paths["cues"]),
        "--overall", str(paths["overall"]), "--out", str(out2), "--bootstrap", "300", "--seed", "7",
    ])
    assert json.loads(out2.read_text(encoding="utf-8")) == report


def test_cli_subprocess_and_error_exit(tmp_path):
    paths, _ = make_exports(tmp_path, n_items=8)
    base = [sys.executable, str(HERE / "compute_validation_stats.py"), "--items", str(paths["items"]),
            "--criteria", str(paths["criteria"]), "--cues", str(paths["cues"]), "--overall", str(paths["overall"])]
    ok = subprocess.run(base + ["--bootstrap", "50"], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert "# GPT feedback validation" in ok.stdout
    bad = subprocess.run(base + ["--experts", "expert-1,expert-2,expert-3"], capture_output=True, text=True)
    assert bad.returncode == 2
    assert "exactly two" in bad.stderr


def test_bootstrap_disabled(tmp_path):
    paths, _ = make_exports(tmp_path, n_items=6)
    report = cvs.analyze(*_load(paths), n_boot=0)
    f1 = report["validity"]["detection"]["both"]["pooled"]["f1"]
    assert f1["ci95"] == [None, None]
    assert not math.isnan(report["reliability"]["step1_detection"]["pooled"]["percent_agreement"])
