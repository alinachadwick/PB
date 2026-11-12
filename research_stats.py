"""
Pre-registered-style analysis plan for the three research questions:

  RQ1. Are people's preferences well-described by a relatively small set of priorities?
  RQ2. Can the survey elicit those priorities in a way that predicts later choices?
  RQ3. Does either of these outcomes vary systematically across demographic groups?

Design decisions baked in here (confirmed with Alina 2026-08-05):
  - RQ1's "small set" is whatever priorities a respondent actually reports (no separate
    top-K-vs-everything comparison, since the survey no longer has a top-7 ranking step).
    RQ1 is evaluated via (a) a descriptive count of reported priorities and (b) reusing
    RQ2's predictive-validity test as the evidentiary basis.
  - RQ2's primary metric is the existing naive consistency rate (conflicted/indifferent
    counts as consistent with either choice) — strong-prediction accuracy is reported
    alongside as a secondary, stricter diagnostic.
  - Each respondent contributes 5 correlated pairwise comparisons, not 5 independent
    draws. All significance tests here aggregate to the RESPONDENT level first (one
    proportion per respondent) rather than pooling raw pairs, to respect that clustering
    without requiring a mixed-effects/GEE model (statsmodels isn't installed).
  - RQ3 groups on the fields the simulated pipeline currently produces: demographic_area
    (urban/rural) and age bracket.

Core functions operate on a plain pandas DataFrame with one row per respondent:
    respondent_id, group_area, group_age, n_priorities,
    cardinal_n_correct, cardinal_n_total, lex_n_correct, lex_n_total

That schema is the contract for "immediately analyze pilot data when we get it" — build
a DataFrame with these columns from wherever the real survey exports land, and every
function below runs unmodified. load_results_file() is just today's adapter from the
validation_v1_testing.py text output to that schema.
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "results"
N_PAIRS_PER_PERSONA = 5


# ---------------------------------------------------------------------------
# Loading — adapter from today's results_*.txt format to the respondent schema.
# Swap this out (or hand-build a DataFrame) once real pilot data exists; nothing
# below this section needs to change.
# ---------------------------------------------------------------------------

def latest_results_file():
    files = glob.glob(os.path.join(RESULTS_DIR, "results_*.txt"))
    if not files:
        raise SystemExit(f"No results_*.txt files found in {RESULTS_DIR}/")
    return max(files, key=os.path.getmtime)


def load_results_file(path):
    text = open(path).read()
    blocks = re.split(r"\n(?=PERSONA: )", text)
    rows = []

    for block in blocks:
        if not block.startswith("PERSONA:"):
            continue

        demo_m = re.search(r"DEMOGRAPHICS: area=(\S+) age=(\S+)", block)
        n_pri_m = re.search(r"TOTAL ELICITED PRIORITIES:\s+(\d+)", block)
        cons_m = re.search(r"Consistency rate:\s+(\d+)%", block)
        lex_m = re.search(r"Lexicographic consistency rate:\s+(\d+)%", block)
        n_pairs = len(re.findall(r"^\s*Pair \d+:", block, re.MULTILINE))

        # Incomplete block (run still in progress when the file was read) — skip it
        # rather than guess at missing fields.
        if demo_m is None or n_pri_m is None or cons_m is None or lex_m is None:
            continue

        n_total = n_pairs or N_PAIRS_PER_PERSONA
        rows.append({
            "respondent_id": len(rows),
            "group_area": demo_m.group(1),
            "group_age": demo_m.group(2),
            "n_priorities": int(n_pri_m.group(1)),
            "cardinal_n_correct": round(int(cons_m.group(1)) / 100 * n_total),
            "cardinal_n_total": n_total,
            "lex_n_correct": round(int(lex_m.group(1)) / 100 * n_total),
            "lex_n_total": n_total,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RQ1 — small set of priorities
# ---------------------------------------------------------------------------

def describe_priority_counts(df):
    """Descriptive half of RQ1: how many priorities do respondents actually report?"""
    n = df["n_priorities"]
    return {
        "n_respondents": len(df),
        "mean": n.mean(),
        "median": n.median(),
        "min": int(n.min()),
        "max": int(n.max()),
        "std": n.std(),
    }


# ---------------------------------------------------------------------------
# RQ2 — predictive validity (and the evidentiary half of RQ1)
# ---------------------------------------------------------------------------

def _bootstrap_ci_on_mean_proportion(proportions, n_boot=10000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng()
    proportions = np.asarray(proportions)
    boot_means = rng.choice(proportions, size=(n_boot, len(proportions)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def test_predictive_validity(df, n_correct_col, n_total_col, label, chance=0.5, rng=None):
    """
    One-sample test at the respondent level: is mean per-respondent consistency rate
    significantly different from chance? Wilcoxon signed-rank on (proportion - chance)
    is the significance test; a respondent-level bootstrap gives the CI to actually report.
    """
    proportions = (df[n_correct_col] / df[n_total_col]).to_numpy()
    mean_prop = proportions.mean()
    lo, hi = _bootstrap_ci_on_mean_proportion(proportions, rng=rng)

    diffs = proportions - chance
    if np.allclose(diffs, 0):
        wilcoxon_p = float("nan")
        note = "all respondents identical — Wilcoxon undefined (no variation to test)"
    elif np.all(diffs == diffs[0]):
        # constant nonzero offset: no variance for wilcoxon either, but direction is clear
        wilcoxon_p = float("nan")
        note = "all respondents identical (nonzero offset) — Wilcoxon undefined"
    else:
        try:
            _, wilcoxon_p = stats.wilcoxon(diffs)
            note = ""
        except ValueError as e:
            wilcoxon_p = float("nan")
            note = f"Wilcoxon failed: {e}"

    return {
        "label": label,
        "n_respondents": len(df),
        "mean_consistency": mean_prop,
        "ci95": (lo, hi),
        "wilcoxon_p_vs_chance": wilcoxon_p,
        "note": note,
    }


def compare_cardinal_vs_lex(df, rng=None):
    """Paired comparison: does the cardinal model outperform the lexicographic one (or
    vice versa) on the SAME respondents? Wilcoxon signed-rank on the per-respondent
    difference in proportions."""
    cardinal_prop = (df["cardinal_n_correct"] / df["cardinal_n_total"]).to_numpy()
    lex_prop = (df["lex_n_correct"] / df["lex_n_total"]).to_numpy()
    diff = cardinal_prop - lex_prop

    if np.allclose(diff, 0):
        p = float("nan")
        note = "cardinal and lexicographic identical for every respondent — nothing to test"
    else:
        try:
            _, p = stats.wilcoxon(diff)
            note = ""
        except ValueError as e:
            p, note = float("nan"), f"Wilcoxon failed: {e}"

    return {
        "mean_cardinal": cardinal_prop.mean(),
        "mean_lex": lex_prop.mean(),
        "mean_diff": diff.mean(),
        "wilcoxon_p": p,
        "note": note,
    }


# ---------------------------------------------------------------------------
# RQ3 — demographic variation
# ---------------------------------------------------------------------------

def test_group_difference(df, group_col, n_correct_col, n_total_col):
    """
    Compares respondent-level consistency proportions across groups.
    2 groups -> Mann-Whitney U. 3+ groups -> Kruskal-Wallis omnibus test.
    """
    df = df.copy()
    df["_prop"] = df[n_correct_col] / df[n_total_col]
    groups = df.groupby(group_col)["_prop"].apply(list)
    group_names = list(groups.index)

    if len(group_names) < 2:
        return {"group_col": group_col, "note": "fewer than 2 groups present — nothing to compare",
                "group_ns": {name: len(vals) for name, vals in groups.items()}}

    group_ns = {name: len(vals) for name, vals in groups.items()}
    group_means = {name: float(np.mean(vals)) for name, vals in groups.items()}

    if len(group_names) == 2:
        a, b = groups[group_names[0]], groups[group_names[1]]
        try:
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            test_name = "Mann-Whitney U"
        except ValueError as e:
            stat, p, test_name = float("nan"), float("nan"), f"Mann-Whitney U failed: {e}"
    else:
        try:
            stat, p = stats.kruskal(*groups.values)
            test_name = "Kruskal-Wallis"
        except ValueError as e:
            stat, p, test_name = float("nan"), float("nan"), f"Kruskal-Wallis failed: {e}"

    return {
        "group_col": group_col, "test": test_name, "statistic": stat, "p_value": p,
        "group_ns": group_ns, "group_means": group_means,
    }


def power_simulation(n_per_group, p1, p2, n_pairs=N_PAIRS_PER_PERSONA,
                      alpha=0.05, n_sims=3000, rng=None):
    """
    Monte Carlo power estimate for detecting a true difference (p1 vs p2) in per-pair
    accuracy between two groups of n_per_group respondents each, via Mann-Whitney U on
    the resulting respondent-level proportions. Use this ahead of the pilot to gauge
    whether a planned sample size could plausibly detect a demographic difference you'd
    care about.
    """
    rng = rng or np.random.default_rng()
    rejections = 0
    for _ in range(n_sims):
        group_a = rng.binomial(n_pairs, p1, size=n_per_group) / n_pairs
        group_b = rng.binomial(n_pairs, p2, size=n_per_group) / n_pairs
        if np.all(group_a == group_a[0]) and np.all(group_b == group_b[0]) and group_a[0] == group_b[0]:
            continue  # degenerate draw, skip
        try:
            _, p = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
        except ValueError:
            continue
        if p < alpha:
            rejections += 1
    return rejections / n_sims


# ---------------------------------------------------------------------------
# Demo / report
# ---------------------------------------------------------------------------

def run_report(path):
    df = load_results_file(path)
    print(f"Analyzing: {path}")
    print(f"Complete respondents: {len(df)}\n")
    if df.empty:
        print("No complete respondent records found.")
        return

    print("=" * 78)
    print("RQ1 — Are preferences well-described by a small set of priorities?")
    print("=" * 78)
    desc = describe_priority_counts(df)
    print(f"Priorities reported per respondent: mean={desc['mean']:.1f}, "
          f"median={desc['median']:.0f}, range=[{desc['min']}, {desc['max']}], n={desc['n_respondents']}")
    print("(Evidentiary test — does this reported set predict later choices? see RQ2 below.)\n")

    print("=" * 78)
    print("RQ2 — Does elicitation predict later choices?")
    print("=" * 78)
    for label, ncol, tcol in [("Cardinal", "cardinal_n_correct", "cardinal_n_total"),
                               ("Lexicographic", "lex_n_correct", "lex_n_total")]:
        r = test_predictive_validity(df, ncol, tcol, label)
        p_str = f"p={r['wilcoxon_p_vs_chance']:.4f}" if r["wilcoxon_p_vs_chance"] == r["wilcoxon_p_vs_chance"] else r["note"]
        print(f"{label} consistency: mean={r['mean_consistency']:.0%}  "
              f"95% CI [{r['ci95'][0]:.0%}, {r['ci95'][1]:.0%}]  vs. chance: {p_str}  (n={r['n_respondents']})")

    cmp = compare_cardinal_vs_lex(df)
    p_str = f"p={cmp['wilcoxon_p']:.4f}" if cmp["wilcoxon_p"] == cmp["wilcoxon_p"] else cmp["note"]
    print(f"\nCardinal vs. lexicographic (paired): mean diff = {cmp['mean_diff']:+.0%}  {p_str}\n")

    print("=" * 78)
    print("RQ3 — Does this vary across demographic groups?")
    print("=" * 78)
    for group_col in ["group_area", "group_age"]:
        r = test_group_difference(df, group_col, "cardinal_n_correct", "cardinal_n_total")
        if "test" not in r:
            print(f"[{group_col}] {r['note']}  (group sizes: {r['group_ns']})")
            continue
        p_str = f"p={r['p_value']:.4f}" if r["p_value"] == r["p_value"] else "n/a"
        print(f"[{group_col}] {r['test']}: {p_str}")
        print(f"    group sizes: {r['group_ns']}")
        print(f"    group means: {{{', '.join(f'{k}: {v:.0%}' for k, v in r['group_means'].items())}}}")

    print("\n" + "=" * 78)
    print("Power check: could a pilot this size even detect a group difference?")
    print("=" * 78)
    for n_per_group in [5, 10, 20, 40]:
        for effect in [0.15, 0.25]:
            base = 0.75
            power = power_simulation(n_per_group, base, base - effect)
            print(f"  n={n_per_group}/group, true gap={effect:.0%} (e.g. {base:.0%} vs {base-effect:.0%}): "
                  f"power ~= {power:.0%}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else latest_results_file()
    run_report(path)
