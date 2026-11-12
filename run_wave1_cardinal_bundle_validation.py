"""
Wave-1 validation: predicts each INDIVIDUAL bundle's Likert-cardinal rating (Very
Unhappy .. Very Happy) from the model's own utility functions, rather than just the
pairwise A-vs-B choice run_wave1_real_validation.py validates.

For each respondent and each priority, omega_j ranges over that priority's own
Likert-tier box [lo_j, hi_j] (LIKERT_TO_OMEGA) -- the exact same bounds the pairwise
cardinal model already uses (compute_delta_range/priority_contributions). Since every
u_j(bundle) is now bounded to [0, 1] by construction (the utility-function-generation
prompt enforces this -- see validation_v1_testing.py's rule 4), U(bundle) is a RANGE,
not a point:
    U_min(bundle) = sum_j lo_j * u_j(bundle)
    U_max(bundle) = sum_j hi_j * u_j(bundle)
with U_mid = (U_min + U_max) / 2 used as the point estimate below.

IMPORTANT: bounding each INDIVIDUAL u_j to [0, 1] does NOT make the aggregate U(bundle)
land in [0, 1] -- it's a sum across however many priorities a respondent has (roughly
5-16 in this data), so someone with more priorities has a higher achievable ceiling
than someone with fewer, even if both are equally satisfied. To make U(bundle)
genuinely comparable to a fixed 0-1 scale, everything here is normalized by each
respondent's own ACHIEVABLE MAXIMUM:
    U_max_achievable = sum_j hi_j * (that priority's own best-case u_j across the
                        REAL bundles this respondent was actually shown -- see
                        priority_achievable_max)
    U_norm(bundle) = U(bundle) / U_max_achievable
(An earlier version of this normalizer used a hypothetical unconstrained bundle --
e.g. one containing every available project -- as the ceiling. That badly inflates the
denominator for count/diminishing-returns-style functions, since real survey bundles
are always a fixed size and can never approach an "every project" bundle; it showed up
as U_norm never reaching the top of the scale at all, an entirely empty predicted-VH
column in the confusion matrix, even though the underlying ranking (Spearman) was
fine. Normalizing against bundles that were actually possible fixes this.)
U_norm(bundle) is then a genuine 0-1 fraction of "how close to this respondent's own
personal best case is this bundle" -- comparable across respondents despite each
person having a different number of priorities and an arbitrary raw utility scale.

This is what makes it meaningful to compare U_norm directly against a fixed VU..VH
scale (evenly spaced: VU=0, SU=0.25, N=0.5, SH=0.75, VH=1.0) -- an ABSOLUTE comparison,
in contrast to the earlier version of this file (see git history / prior conversation),
which used purely rank-based quantile matching because individual u_j outputs used to
be on an arbitrary 0-3 scale with no natural zero-point. Metrics:
  1. Spearman rank correlation between U_norm(bundle) and the respondent's own stated
     rating, computed per respondent (scale-free regardless of normalization, kept as
     a companion sanity check).
  2. Mean absolute error between U_norm(bundle) and the target reference point for the
     respondent's ACTUAL rating -- meaningful now that both sides are on the same 0-1
     scale.
  3. A discrete VU..VH prediction: whichever of the 5 reference points (0/.25/.5/.75/1)
     is closest to U_norm(bundle).

Reuses results/llm_model_cache_wave1.json (run_wave1_real_validation.py's cache) and
the priority-reuse registry, exactly like run_wave1_full_analysis.py -- run that
first so this is free, or just run this script (it populates the same cache/registry
files, so whichever runs first pays the LLM cost).

Run locally:
    python3 run_wave1_cardinal_bundle_validation.py
"""
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from format_wave1_results import load_all
from priority_dedup import PriorityRegistry
from run_wave0_real_validation import (
    RESULTS_DIR, run_one, load_cache, save_cache,
    PRIORITY_REGISTRY_PATH, PRIORITY_SIMILARITY_THRESHOLD, CRITERION_SIGN_BY_TAG,
)
from run_wave1_real_validation import CACHE_PATH
from validation_v1_testing import likert_to_omega_ranges, normalize

RATING_ORDER = ["Very Unhappy", "Somewhat Unhappy", "Neutral", "Somewhat Happy", "Very Happy"]
RATING_ABBREV = {
    "Very Happy": "VH", "Somewhat Happy": "SH", "Neutral": "N",
    "Somewhat Unhappy": "SU", "Very Unhappy": "VU",
}
RATING_TO_NUM = {r: i - 2 for i, r in enumerate(RATING_ORDER)}
# Evenly-spaced targets on the same [0, 1] scale u_j is now bounded to -- see module
# docstring for why this is only meaningful after the achievable-max normalization.
RATING_TARGET = {r: i / 4 for i, r in enumerate(RATING_ORDER)}


def respondent_achievable_max(utility_functions, omega_ranges, candidate_bundles):
    """U_max_achievable = the highest U_MID(bundle) achieved by any SINGLE bundle among
    the real, fixed-size bundles this respondent was actually shown (candidate_bundles)
    -- the per-respondent normalizing denominator described in the module docstring.

    Deliberately NOT the sum of each priority's own independently-best value across
    those bundles (an earlier version did this, and an even earlier version summed each
    priority's best over a hypothetical unconstrained bundle): different priorities can
    have different "ideal" bundles that no single real bundle satisfies simultaneously
    (e.g. wanting project A present AND wanting project B present, when no shown bundle
    contains both), so summing independent per-priority maxima produces a ceiling no
    real bundle can ever approach.

    Also deliberately U_MID, not U_MAX: an intermediate version of this used the best
    U_max(bundle) (the omega upper-bound) as the denominator, but then U_norm itself is
    the average of u_min_norm and u_max_norm -- and u_min_norm (using the omega LOWER
    bound) is always strictly less than 1.0 even for the objectively best bundle, so
    that midpoint could mathematically never reach 1.0 for ANY bundle. Both of these
    bugs showed up identically: an entirely empty "predicted VH" column in the
    confusion matrix, even though the underlying ranking (Spearman) was fine the whole
    time. Normalizing U_MID by the best actual U_MID over real bundles fixes this: it's
    the best a single real bundle could possibly have scored on the same point estimate
    being predicted, so it's achievable by construction (it's literally one of the
    bundles being compared)."""
    def u_mid(bundle):
        u_min, u_max = bundle_utility_range(bundle, utility_functions, omega_ranges)
        return (u_min + u_max) / 2
    return max(u_mid(b) for b in candidate_bundles)


def bundle_utility_range(bundle, utility_functions, omega_ranges):
    u_min = u_max = 0.0
    for priority, (lo, hi) in omega_ranges.items():
        u_j = utility_functions.get(priority)
        if u_j is None:
            continue
        u_val = u_j(bundle)  # always in [0, 1] by construction
        u_min += lo * u_val
        u_max += hi * u_val
    return u_min, u_max


def priority_signs(run):
    """{normalized priority text: "positive"/"negative"} for every ORIGINALLY elicited
    criterion this respondent gave, from which survey question it came from (see
    CRITERION_SIGN_BY_TAG) -- the same sign build_utility_functions_v1 was actually
    called with. Priorities folded in later from "Other:" validation citations aren't
    covered (their sign was inferred by a one-off LLM call at generation time, not
    recoverable here without another call) -- callers should treat a missing lookup as
    unknown, not assume positive."""
    return {
        normalize(c["text"]): CRITERION_SIGN_BY_TAG.get(c["question"], "positive")
        for c in run["criteria"]
    }


def bundle_priority_contributions(bundle, utility_functions, omega_ranges, achievable_max):
    """EVERY priority's contribution to this bundle's U_mid, normalized by the same
    respondent achievable-max denominator as everything else here, sorted
    highest-first -- i.e. the full breakdown of WHY this bundle scored the way it did.
    A priority contributing ~0 simply didn't distinguish this bundle from an empty
    one -- it's still listed so the full picture is visible."""
    contributions = []
    for priority, (lo, hi) in omega_ranges.items():
        u_j = utility_functions.get(priority)
        if u_j is None:
            continue
        contrib = (lo + hi) / 2 * u_j(bundle)
        contributions.append((priority, contrib / achievable_max if achievable_max > 0 else 0.0))
    contributions.sort(key=lambda pc: -pc[1])
    return contributions


def collect_bundle_points(run):
    """(bundle, [U_min, U_max] normalized by this respondent's own achievable maximum,
    raw_rating) for every rated bundle this respondent saw."""
    utility_functions = run["result"]["utility_functions"]
    omega_ranges = likert_to_omega_ranges(run["result"]["importance"], utility_functions)
    signs = priority_signs(run)
    candidate_bundles = [b for pair in run["bundle_pairs"] for b in pair]
    achievable_max = respondent_achievable_max(utility_functions, omega_ranges, candidate_bundles)
    points = []
    for code, (bundle_a, bundle_b), ans in zip(run["pair_codes"], run["bundle_pairs"], run["validation_answers"]):
        for side, bundle, raw_rating in [
            ("A", bundle_a, ans["rating_a_raw"]), ("B", bundle_b, ans["rating_b_raw"])
        ]:
            if raw_rating not in RATING_TO_NUM:
                continue
            u_min, u_max = bundle_utility_range(bundle, utility_functions, omega_ranges)
            u_min_norm = u_min / achievable_max if achievable_max > 0 else 0.0
            u_max_norm = u_max / achievable_max if achievable_max > 0 else 0.0
            contributions = bundle_priority_contributions(bundle, utility_functions, omega_ranges, achievable_max)
            points.append({
                "code": code, "side": side, "bundle": bundle,
                "u_min": u_min, "u_max": u_max,
                "u_min_norm": u_min_norm, "u_max_norm": u_max_norm,
                "u_norm": (u_min_norm + u_max_norm) / 2,
                "raw_rating": raw_rating,
                "priority_contributions": contributions, "priority_signs": signs,
                "top_priority": contributions[0][0] if contributions else None,
                "top_priority_contrib": contributions[0][1] if contributions else -1.0,
            })
    return points, achievable_max


def nearest_reference_predict(points):
    """Assigns each point a predicted label: whichever of the 5 evenly-spaced
    reference points (VU=0, SU=.25, N=.5, SH=.75, VH=1) is closest to U_norm(bundle) --
    an ABSOLUTE comparison, meaningful now that U_norm is a genuine 0-1 fraction of
    this respondent's own achievable best (see module docstring). Also attaches
    "mae" (|U_norm - target for the actual rating|) and "severity" (tiers off) for the
    worst-misses breakdown, and "u_percentile" (rank position among this respondent's
    own bundles) purely for the plot's x-axis spread. Mutates and returns `points`."""
    n = len(points)
    by_u = sorted(range(n), key=lambda i: points[i]["u_norm"])
    for rank, i in enumerate(by_u):
        points[i]["u_percentile"] = rank / (n - 1) if n > 1 else 0.5
    for p in points:
        predicted = min(RATING_ORDER, key=lambda r: abs(RATING_TARGET[r] - p["u_norm"]))
        p["predicted_rating"] = predicted
        p["mae"] = abs(p["u_norm"] - RATING_TARGET[p["raw_rating"]])
        p["severity"] = abs(RATING_TO_NUM[predicted] - RATING_TO_NUM[p["raw_rating"]])
    return points


def evaluate_respondent(points):
    """None if there are fewer than 2 rated bundles (can't rank-correlate with only
    one point)."""
    if len(points) < 2:
        return None
    u = [p["u_norm"] for p in points]
    actual = [RATING_TO_NUM[p["raw_rating"]] for p in points]
    # Spearman r is undefined (NaN) if EITHER array has zero variance -- most often
    # because our predicted U(bundle) came out identical for every bundle this
    # respondent saw (none of their priorities' utility functions discriminate between
    # the specific bundles they were shown), occasionally because they gave the same
    # rating to every bundle. Report this explicitly (None, not NaN) rather than
    # letting NaN silently poison the aggregate mean.
    u_constant = len(set(u)) == 1
    actual_constant = len(set(actual)) == 1
    rho = None if (u_constant or actual_constant) else spearmanr(u, actual)[0]
    nearest_reference_predict(points)
    exact = sum(1 for p in points if p["predicted_rating"] == p["raw_rating"])
    within_one = sum(
        1 for p in points
        if abs(RATING_TO_NUM[p["predicted_rating"]] - RATING_TO_NUM[p["raw_rating"]]) <= 1
    )
    mean_mae = sum(p["mae"] for p in points) / len(points)
    return {
        "spearman_r": rho, "u_constant": u_constant, "actual_constant": actual_constant,
        "n": len(points), "mean_mae": mean_mae,
        "exact_rate": exact / len(points), "within_one_rate": within_one / len(points),
        "points": points,
    }


def render_report(all_results):
    out = ["# Wave 1 -- Cardinal Bundle-Rating Validation\n"]
    out.append(
        "Predicts each individual bundle's Likert-cardinal rating (VU/SU/N/SH/VH) from "
        "the model's own utility functions, using each priority's own Likert-tier box "
        "[lo, hi] (the exact bounds the pairwise cardinal model already uses). Each "
        "u_j(bundle) is bounded to [0, 1] by the generation prompt, and U(bundle) is "
        "further normalized by this respondent's own ACHIEVABLE MAXIMUM (their best "
        "case across every priority) so it's a genuine 0-1 fraction comparable to a "
        "fixed VU=0/SU=.25/N=.5/SH=.75/VH=1 scale -- an ABSOLUTE comparison, not the "
        "rank-based quantile matching an earlier version of this script used (see "
        "module docstring for why the normalization is necessary and what changed).\n"
    )

    valid = [r for r in all_results if r[3] is not None]
    out.append(f"## Aggregate ({len(valid)} respondents with >=2 rated bundles)\n")
    if valid:
        rho_defined = [r for r in valid if r[3]["spearman_r"] is not None]
        rho_excluded = [r for r in valid if r[3]["spearman_r"] is None]
        mean_exact = sum(r[3]["exact_rate"] for r in valid) / len(valid)
        mean_within_one = sum(r[3]["within_one_rate"] for r in valid) / len(valid)
        mean_mae = sum(r[3]["mean_mae"] for r in valid) / len(valid)
        if rho_defined:
            mean_rho = sum(r[3]["spearman_r"] for r in rho_defined) / len(rho_defined)
            out.append(
                f"- Mean per-respondent Spearman r (predicted U_norm vs. stated rating): "
                f"{mean_rho:.2f} (n={len(rho_defined)}/{len(valid)}"
                + (f"; {len(rho_excluded)} excluded, see below" if rho_excluded else "") + ")"
            )
        if rho_excluded:
            out.append(
                f"- Excluded from the Spearman mean (undefined -- zero variance in predicted U, "
                f"actual rating, or both): "
                + ", ".join(
                    f"{r[0]} ({'constant U' if r[3]['u_constant'] else ''}"
                    f"{' + ' if r[3]['u_constant'] and r[3]['actual_constant'] else ''}"
                    f"{'constant rating' if r[3]['actual_constant'] else ''})"
                    for r in rho_excluded
                )
            )
        out.append(f"- Mean absolute error (U_norm vs. target reference point for actual rating): {mean_mae:.3f}")
        out.append(f"- Mean exact-label accuracy (nearest reference point): {mean_exact:.0%}")
        out.append(f"- Mean within-one-tier accuracy: {mean_within_one:.0%}")

        confusion = {a: {p: 0 for p in RATING_ORDER} for a in RATING_ORDER}
        for _, _, _, result in valid:
            for pt in result["points"]:
                confusion[pt["raw_rating"]][pt["predicted_rating"]] += 1
        out.append("\n**Confusion matrix** (rows = actual, columns = predicted, all respondents pooled):\n")
        header = "| Actual \\ Predicted | " + " | ".join(RATING_ABBREV[r] for r in RATING_ORDER) + " |"
        out.append(header)
        out.append("|" + "---|" * (len(RATING_ORDER) + 1))
        for actual in RATING_ORDER:
            row = [str(confusion[actual][pred]) for pred in RATING_ORDER]
            out.append(f"| {RATING_ABBREV[actual]} | " + " | ".join(row) + " |")
        out.append("")

        worst = [
            (respid, pt) for respid, _, _, result in valid for pt in result["points"]
            if pt["severity"] >= 2
        ]
        worst.sort(key=lambda rp: -rp[1]["severity"])
        out.append(
            f"**Worst misses** ({len(worst)} bundles predicted 2+ tiers off actual). Each "
            "has a full breakdown below of every priority's normalized contribution to "
            "this bundle's U_norm, highest first -- [NEG] marks a priority elicited as "
            "something the respondent wants to AVOID (from the \"what would make a bad "
            "bundle\" question), where a HIGH contribution means the bundle successfully "
            "avoids the bad property, not that it contains something good. [sign unknown] "
            "marks a priority folded in later from an \"Other:\" citation, whose sign "
            "isn't recoverable without a fresh LLM call.\n"
        )
        if worst:
            out.append("| Respondent | Pair | Bundle | U_norm range | Actual | Predicted |")
            out.append("|---|---|---|---|---|---|")
            for respid, pt in worst:
                bundle_str = ", ".join(sorted(pt["bundle"]))
                out.append(
                    f"| {respid} | {pt['code']}.{pt['side']} | {bundle_str} | "
                    f"[{pt['u_min_norm']:.2f}, {pt['u_max_norm']:.2f}] | {RATING_ABBREV[pt['raw_rating']]} | "
                    f"{RATING_ABBREV[pt['predicted_rating']]} |"
                )
            out.append("")
            for respid, pt in worst:
                bundle_str = ", ".join(sorted(pt["bundle"]))
                out.append(
                    f"<details><summary>{respid} {pt['code']}.{pt['side']} ({bundle_str}) -- "
                    f"actual {RATING_ABBREV[pt['raw_rating']]}, predicted "
                    f"{RATING_ABBREV[pt['predicted_rating']]}</summary>\n"
                )
                for priority, contrib in pt["priority_contributions"]:
                    sign = pt["priority_signs"].get(priority)
                    tag = "[NEG] " if sign == "negative" else ("" if sign == "positive" else "[sign unknown] ")
                    out.append(f"- {tag}{priority}: {contrib:.3f}")
                out.append("\n</details>\n")
        else:
            out.append("_None -- no bundle was predicted more than one tier off._")
        out.append("")

    out.append("## Per respondent\n")
    out.append("| Respondent | Area | Variant | n bundles | Spearman r | MAE | Exact | Within-1 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for respid, area, variant, result in all_results:
        if result is None:
            out.append(f"| {respid} | {area} | {variant} | - | - | - | - | - |")
            continue
        rho_str = f"{result['spearman_r']:.2f}" if result["spearman_r"] is not None else "n/a"
        out.append(
            f"| {respid} | {area} | {variant} | {result['n']} | {rho_str} | {result['mean_mae']:.3f} | "
            f"{result['exact_rate']:.0%} | {result['within_one_rate']:.0%} |"
        )
    out.append("")

    for respid, area, variant, result in all_results:
        if result is None:
            continue
        out.append(f"<details><summary><b>{respid}</b> ({area}, {variant}) -- bundle-level detail</summary>\n")
        out.append("| Pair | Side | Bundle | U_norm range [min, max] | U_norm | Actual | Predicted (nearest reference) |")
        out.append("|---|---|---|---|---|---|---|")
        for pt in sorted(result["points"], key=lambda p: -p["u_norm"]):
            bundle_str = ", ".join(sorted(pt["bundle"]))
            match = "✓" if pt["predicted_rating"] == pt["raw_rating"] else ""
            out.append(
                f"| {pt['code']} | {pt['side']} | {bundle_str} | [{pt['u_min_norm']:.2f}, {pt['u_max_norm']:.2f}] | "
                f"{pt['u_norm']:.2f} | {RATING_ABBREV[pt['raw_rating']]} | {RATING_ABBREV[pt['predicted_rating']]} {match} |"
            )
        out.append("\n</details>\n")

    return "\n".join(out)


SEVERITY_COLORS = {0: "#9ecae1", 1: "#fd8d3c", 2: "#de2d26", 3: "#67000d"}


def plot_results(all_results, path):
    """Four panels, all computed straight from the same all_results the markdown
    report uses -- no separate computation path to drift out of sync:
      1. Pooled confusion matrix (actual vs. nearest-reference-point predicted label).
      2. Distribution of per-respondent Spearman r (predicted U_norm vs. stated
         rating), excluding the handful where it's undefined (zero variance on one
         side).
      3. Per-respondent exact-match rate vs. within-one-tier rate, one point per
         respondent, colored by variant -- shows whether good aggregate accuracy is
         spread evenly across respondents or driven by a few standouts.
      4. WHICH bundles we got far off on and WHY: U_norm (an absolute 0-1 fraction of
         this respondent's own achievable best, now directly comparable across
         respondents) vs. actual rating, with vertical reference lines at the 5
         VU..VH target points so it's visually obvious which reference a point is
         closest to, colored by miss severity, with the worst few annotated by the
         single priority that contributed most to that bundle's (wrong) predicted
         score.
    """
    valid = [r for r in all_results if r[3] is not None]

    confusion = np.zeros((len(RATING_ORDER), len(RATING_ORDER)), dtype=int)
    idx = {r: i for i, r in enumerate(RATING_ORDER)}
    for _, _, _, result in valid:
        for pt in result["points"]:
            confusion[idx[pt["raw_rating"]], idx[pt["predicted_rating"]]] += 1

    rhos = [r[3]["spearman_r"] for r in valid if r[3]["spearman_r"] is not None]
    n_excluded = len(valid) - len(rhos)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # --- Panel 1: confusion matrix heatmap ---
    ax = axes[0][0]
    im = ax.imshow(confusion, cmap="Blues")
    abbrevs = [RATING_ABBREV[r] for r in RATING_ORDER]
    ax.set_xticks(range(len(RATING_ORDER))); ax.set_xticklabels(abbrevs)
    ax.set_yticks(range(len(RATING_ORDER))); ax.set_yticklabels(abbrevs)
    ax.set_xlabel("Predicted (nearest reference point)")
    ax.set_ylabel("Actual (stated rating)")
    vmax = confusion.max()
    for i in range(len(RATING_ORDER)):
        for j in range(len(RATING_ORDER)):
            count = confusion[i, j]
            color = "white" if count > vmax * 0.6 else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=color, fontsize=10)
    ax.set_title("Confusion matrix (pooled)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Panel 2: distribution of per-respondent Spearman r ---
    ax = axes[0][1]
    ax.hist(rhos, bins=np.arange(-1.0, 1.05, 0.1), color="#4C72B0", edgecolor="white")
    mean_rho = sum(rhos) / len(rhos) if rhos else 0.0
    ax.axvline(mean_rho, color="#C44E52", linewidth=2, label=f"mean = {mean_rho:.2f}")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Per-respondent Spearman r (predicted U_norm vs. stated rating)")
    ax.set_ylabel("Number of respondents")
    title = f"Distribution across {len(rhos)} respondents"
    if n_excluded:
        title += f"\n({n_excluded} excluded -- zero variance on one side)"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)

    # --- Panel 3: per-respondent exact vs. within-one accuracy ---
    ax = axes[1][0]
    colors = {"V1": "#4C72B0", "SimProjects": "#DD8452"}
    for variant in ("V1", "SimProjects"):
        xs = [r[3]["exact_rate"] for r in valid if r[2] == variant]
        ys = [r[3]["within_one_rate"] for r in valid if r[2] == variant]
        ax.scatter(xs, ys, alpha=0.7, s=40, color=colors[variant], label=variant, edgecolor="black", linewidth=0.4)
    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Exact-match rate")
    ax.set_ylabel("Within-one-tier rate")
    ax.set_title("Per-respondent accuracy")
    ax.legend(loc="lower right", fontsize=9)

    # --- Panel 4: which bundles missed, and why ---
    ax = axes[1][1]
    rng = np.random.default_rng(0)
    all_points = [(respid, pt) for respid, _, _, result in valid for pt in result["points"]]
    by_severity = {}
    for respid, pt in all_points:
        by_severity.setdefault(pt["severity"], []).append((respid, pt))
    for ref_label, ref_val in RATING_TARGET.items():
        ax.axvline(ref_val, color="gray", linewidth=0.7, linestyle=":", zorder=1)
    for severity in sorted(by_severity):
        pts = by_severity[severity]
        xs = [pt["u_norm"] for _, pt in pts]
        ys = [RATING_TO_NUM[pt["raw_rating"]] + rng.uniform(-0.12, 0.12) for _, pt in pts]
        label = "exact" if severity == 0 else ("within-1" if severity == 1 else f"{severity}+ tiers off")
        ax.scatter(xs, ys, s=28 + severity * 30, alpha=0.75 if severity else 0.35,
                   color=SEVERITY_COLORS.get(severity, "black"), label=label,
                   edgecolor="black" if severity >= 2 else "none", linewidth=0.6, zorder=2 + severity)

    worst = sorted(all_points, key=lambda rp: -rp[1]["severity"])[:6]
    for respid, pt in worst:
        if pt["severity"] < 2:
            break
        sign = pt["priority_signs"].get(pt["top_priority"]) if pt["top_priority"] else None
        tag = "[NEG] " if sign == "negative" else ""
        label = (tag + (pt["top_priority"] or "?"))[:30]
        ax.annotate(
            label, (pt["u_norm"], RATING_TO_NUM[pt["raw_rating"]]),
            fontsize=6.5, color="#333333",
            xytext=(4, 6), textcoords="offset points",
        )

    ax.set_yticks([-2, -1, 0, 1, 2]); ax.set_yticklabels(RATING_ORDER, fontsize=8)
    ax.set_xticks(list(RATING_TARGET.values()))
    ax.set_xticklabels([f"{v:.2f}\n({RATING_ABBREV[k]})" for k, v in RATING_TARGET.items()], fontsize=7)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("U_norm (fraction of this respondent's own achievable best)")
    ax.set_ylabel("Actual rating")
    ax.set_title("Which bundles missed, and why\n(labels = top-contributing priority for 2+ tier misses)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, title="severity")

    fig.suptitle(
        f"Wave 1 cardinal bundle-rating validation -- {len(valid)} respondents, "
        f"{sum(r[3]['n'] for r in valid)} rated bundles",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("saved", path)


def plot_cardinal_vs_likert(all_results, path):
    """A clean, dedicated scatter of predicted cardinal utility (U_norm) against the
    actual VU..VH rating, pooled across ALL respondents -- one point per rated bundle.
    This pooling is now valid (it wasn't in the earlier plot_wave1_cardinal_vs_rating.py,
    which pooled raw, un-normalized U values across respondents on arbitrary individual
    scales and only got r=0.22): U_norm is a per-respondent-normalized 0-1 fraction of
    each person's own achievable best, so U_norm=0.8 means roughly the same thing for
    everyone, and a single pooled trend line is now a meaningful thing to fit.

    x = U_norm (with error bars showing [U_min_norm, U_max_norm]), y = actual rating
    (numeric, jittered for visibility) with tick labels VU..VH. Vertical/horizontal
    reference lines mark the 5 canonical target points (0/.25/.5/.75/1 <-> VU/SU/N/SH/VH)
    so it's visually obvious how far off the diagonal points fall. A pooled linear fit
    (with Pearson r) is drawn for reference.
    """
    points = [pt for _, _, _, result in all_results if result for pt in result["points"]]
    x = np.array([p["u_norm"] for p in points])
    y = np.array([RATING_TO_NUM[p["raw_rating"]] for p in points])
    xerr_lo = x - np.array([p["u_min_norm"] for p in points])
    xerr_hi = np.array([p["u_max_norm"] for p in points]) - x

    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for target in RATING_TARGET.values():
        ax.axvline(target, color="gray", linewidth=0.6, linestyle=":", zorder=0)
    for num in RATING_TO_NUM.values():
        ax.axhline(num, color="gray", linewidth=0.6, linestyle=":", zorder=0)

    rng = np.random.default_rng(0)
    y_jittered = y + rng.uniform(-0.1, 0.1, size=len(y))
    ax.errorbar(x, y_jittered, xerr=[xerr_lo, xerr_hi], fmt="o", markersize=4, alpha=0.45,
                ecolor="gray", elinewidth=0.6, capsize=0, color="#4C72B0", label="bundle ratings", zorder=2)

    xs = np.linspace(0, 1, 100)
    ax.plot(xs, slope * xs + intercept, color="#C44E52", linewidth=2,
             label=f"pooled fit: rating = {slope:.2f}·U_norm + {intercept:.2f}  (r={r:.2f}, R²={r**2:.2f})",
             zorder=3)

    ax.set_xticks(list(RATING_TARGET.values()))
    ax.set_xticklabels([f"{v:.2f}\n({RATING_ABBREV[k]})" for k, v in RATING_TARGET.items()], fontsize=8)
    ax.set_yticks([-2, -1, 0, 1, 2]); ax.set_yticklabels(RATING_ORDER)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("U_norm -- predicted cardinal utility (fraction of this respondent's own achievable best)")
    ax.set_ylabel("Actual stated rating")
    ax.set_title(f"Wave 1: predicted cardinal utility vs. stated Likert rating ({len(points)} bundle ratings, pooled)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("saved", path)
    print(f"Pooled Pearson r = {r:.3f}, R² = {r**2:.3f}, slope = {slope:.3f}, intercept = {intercept:.3f}")


def plot_cardinal_vs_likert_by_pair(all_results, path):
    """Per-PAIR view (not per independent bundle, unlike plot_cardinal_vs_likert):
    for each of the ~240 real pairwise comparisons, draws a line connecting bundle A's
    (U_norm, actual rating) to bundle B's -- colored green if the utility ordering
    agrees with the actual rating ordering for THAT SPECIFIC comparison (whichever
    bundle scored higher U_norm also got the higher/equal stated rating), red if it's
    reversed, gray if the two bundles got the same actual rating (no ordering to
    agree/disagree with). This directly shows, per pairwise choice, whether higher
    predicted utility tracks higher stated happiness -- the pooled scatter in
    plot_cardinal_vs_likert loses this pairwise structure entirely, lumping all
    bundles into one undifferentiated cloud.
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for target in RATING_TARGET.values():
        ax.axvline(target, color="gray", linewidth=0.6, linestyle=":", zorder=0)
    for num in RATING_TO_NUM.values():
        ax.axhline(num, color="gray", linewidth=0.6, linestyle=":", zorder=0)

    rng = np.random.default_rng(0)
    n_consistent = n_inconsistent = n_tied = 0
    for respid, _, _, result in all_results:
        if not result:
            continue
        by_pair = {}
        for p in result["points"]:
            by_pair.setdefault(p["code"], {})[p["side"]] = p
        for code, sides in by_pair.items():
            if "A" not in sides or "B" not in sides:
                continue
            pa, pb = sides["A"], sides["B"]
            u_a, u_b = pa["u_norm"], pb["u_norm"]
            r_a, r_b = RATING_TO_NUM[pa["raw_rating"]], RATING_TO_NUM[pb["raw_rating"]]
            du, dr = u_a - u_b, r_a - r_b
            if dr == 0:
                color = "#999999"
                n_tied += 1
            elif (du > 0) == (dr > 0):
                color = "#2ca25f"
                n_consistent += 1
            else:
                color = "#de2d26"
                n_inconsistent += 1
            jitter = rng.uniform(-0.08, 0.08)
            ax.plot([u_a, u_b], [r_a + jitter, r_b + jitter], color=color, alpha=0.55, linewidth=1.2, zorder=1)
            ax.scatter([u_a, u_b], [r_a + jitter, r_b + jitter], color=color, s=16, alpha=0.8, zorder=2)

    total = n_consistent + n_inconsistent + n_tied
    legend_elems = [
        Line2D([0], [0], color="#2ca25f", lw=2, label=f"utility order matches rating order ({n_consistent}/{total})"),
        Line2D([0], [0], color="#de2d26", lw=2, label=f"utility order CONTRADICTS rating order ({n_inconsistent}/{total})"),
        Line2D([0], [0], color="#999999", lw=2, label=f"tied actual rating, no order to check ({n_tied}/{total})"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=8)

    ax.set_xticks(list(RATING_TARGET.values()))
    ax.set_xticklabels([f"{v:.2f}\n({RATING_ABBREV[k]})" for k, v in RATING_TARGET.items()], fontsize=8)
    ax.set_yticks([-2, -1, 0, 1, 2]); ax.set_yticklabels(RATING_ORDER)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("U_norm -- predicted cardinal utility (fraction of this respondent's own achievable best)")
    ax.set_ylabel("Actual stated rating")
    ax.set_title(f"Wave 1: cardinal utility vs. stated rating, per pairwise comparison ({total} pairs)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("saved", path)
    print(f"Utility order matches rating order: {n_consistent}/{total} ({n_consistent/total:.1%}), "
          f"contradicts: {n_inconsistent}/{total} ({n_inconsistent/total:.1%}), "
          f"tied actual rating: {n_tied}/{total} ({n_tied/total:.1%})")


def plot_cardinal_vs_likert_by_respondent(all_results, path):
    """Same idea as plot_cardinal_vs_likert_by_pair (each pair drawn as a line from
    bundle A's (U_norm, actual rating) to bundle B's, colored green/red/gray for
    order-matches/order-contradicts/tied-rating) but faceted into one small panel PER
    RESPONDENT instead of one giant plot with all 240 pairs tangled together. With only
    ~5-7 pairs per panel instead of 240 overlapping across the whole figure, each
    person's own pattern -- and how far predicted utility sits from their stated
    rating -- is actually readable. Panels are sorted worst-MAE-first, so scanning
    top-to-bottom shows the range from "clearly badly calibrated for this person" to
    "tracks their stated ratings well."
    """
    valid = [r for r in all_results if r[3] is not None]
    valid = sorted(valid, key=lambda r: -r[3]["mean_mae"])

    ncols = 4
    nrows = (len(valid) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)

    for idx, (respid, area, variant, result) in enumerate(valid):
        ax = axes[idx // ncols][idx % ncols]
        for target in RATING_TARGET.values():
            ax.axvline(target, color="gray", linewidth=0.5, linestyle=":", zorder=0)
        for num in RATING_TO_NUM.values():
            ax.axhline(num, color="gray", linewidth=0.5, linestyle=":", zorder=0)

        by_pair = {}
        for p in result["points"]:
            by_pair.setdefault(p["code"], {})[p["side"]] = p
        n_consistent = n_inconsistent = n_tied = 0
        for code, sides in by_pair.items():
            if "A" not in sides or "B" not in sides:
                continue
            pa, pb = sides["A"], sides["B"]
            u_a, u_b = pa["u_norm"], pb["u_norm"]
            r_a, r_b = RATING_TO_NUM[pa["raw_rating"]], RATING_TO_NUM[pb["raw_rating"]]
            du, dr = u_a - u_b, r_a - r_b
            if dr == 0:
                color, n_tied = "#999999", n_tied + 1
            elif (du > 0) == (dr > 0):
                color, n_consistent = "#2ca25f", n_consistent + 1
            else:
                color, n_inconsistent = "#de2d26", n_inconsistent + 1
            ax.plot([u_a, u_b], [r_a, r_b], color=color, alpha=0.7, linewidth=1.3, zorder=1)
            ax.scatter([u_a, u_b], [r_a, r_b], color=color, s=22, alpha=0.9, zorder=2)

        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-2.4, 2.4)
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_yticklabels(["VU", "SU", "N", "SH", "VH"], fontsize=7)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", ".25", ".5", ".75", "1"], fontsize=7)
        n_pairs = n_consistent + n_inconsistent + n_tied
        ax.set_title(
            f"{respid} ({variant})\nMAE={result['mean_mae']:.2f}, order OK {n_consistent}/{n_pairs}",
            fontsize=7.5,
        )
        ax.tick_params(axis="both", labelsize=7)

    for idx in range(len(valid), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#2ca25f", lw=2, label="utility order matches rating order"),
        Line2D([0], [0], color="#de2d26", lw=2, label="utility order contradicts rating order"),
        Line2D([0], [0], color="#999999", lw=2, label="tied actual rating"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        "Wave 1: cardinal utility (x) vs. stated rating (y), per respondent -- sorted worst MAE first",
        fontsize=12, y=1.015,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("saved", path)


if __name__ == "__main__":
    v1, sim = load_all()
    cache = load_cache(CACHE_PATH)
    registry = PriorityRegistry(path=PRIORITY_REGISTRY_PATH, threshold=PRIORITY_SIMILARITY_THRESHOLD)

    respondents = [(r, "V1") for r in v1] + [(r, "SimProjects") for r in sim]
    all_results = []
    for r, label in respondents:
        print(f"Running {r['response_id']} ({label})...")
        run = run_one(r, cache=cache, priority_registry=registry)
        save_cache(cache, CACHE_PATH)
        registry.save()

        points, achievable_max = collect_bundle_points(run)
        result = evaluate_respondent(points)
        all_results.append((r["response_id"], r["area_selfreport"], label, result))
        if result:
            rho_str = f"{result['spearman_r']:.2f}" if result["spearman_r"] is not None else "n/a"
            print(f"  n={result['n']}, achievable_max={achievable_max:.2f}, Spearman r={rho_str}, "
                  f"MAE={result['mean_mae']:.3f}, exact={result['exact_rate']:.0%}, "
                  f"within-1={result['within_one_rate']:.0%}")
        else:
            print("  fewer than 2 rated bundles, skipped")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(RESULTS_DIR, f"wave1_cardinal_bundle_validation_{ts}.md")
    with open(md_path, "w") as f:
        f.write(render_report(all_results))
    print("wrote", md_path)

    png_path = os.path.join(RESULTS_DIR, f"wave1_cardinal_bundle_validation_{ts}.png")
    plot_results(all_results, png_path)

    scatter_path = os.path.join(RESULTS_DIR, f"wave1_cardinal_vs_likert_{ts}.png")
    plot_cardinal_vs_likert(all_results, scatter_path)

    pair_scatter_path = os.path.join(RESULTS_DIR, f"wave1_cardinal_vs_likert_by_pair_{ts}.png")
    plot_cardinal_vs_likert_by_pair(all_results, pair_scatter_path)

    by_respondent_path = os.path.join(RESULTS_DIR, f"wave1_cardinal_vs_likert_by_respondent_{ts}.png")
    plot_cardinal_vs_likert_by_respondent(all_results, by_respondent_path)
