"""
Visualizes how WIDE the feasible omega range actually is, per respondent, for the
joint-feasibility tier they land in (a/b/c -- tier d has no feasible region at all, so
those respondents are excluded here; see run_wave1_real_validation.py's report for the
tier breakdown itself).

The concern this answers: when we say a respondent's choices are "jointly feasible" at
some tier, does that mean there's essentially exactly ONE weighting that works (a razor's
edge -- any tiny perturbation would break feasibility, which is a bit suspicious/fragile),
or does almost the ENTIRE allowed range still work (the data barely constrains anything,
so "feasible" is nearly vacuous), or is it something genuinely in between (the data
provides real, but not total, information about each priority's weight)?

Uses joint_feasibility_tiers' per-priority omega_ranges (each priority's achievable
[min, max] interval within the winning tier's feasible region -- from two extra LP
solves per priority, already computed once inside run_one) -- but compares each
priority's achieved width to what was ALLOWED for that tier, since tier (a) bounds are
each priority's own Likert-tier box (width <= 0.25) while tiers (b)/(c) use the full
[0, 1] (width <= 1.0) -- raw widths aren't comparable across tiers, but width-as-a-
fraction-of-allowed is:
    relative_width = achieved_width / allowed_width   (0% = a single point, 100% = the
                      entire allowed range, i.e. genuinely unconstrained)

Two views:
  1. Histogram of each respondent's own MEDIAN relative width across their priorities
     -- the quick, global answer to "in general, how wide are these ranges."
  2. A detailed per-respondent strip/box plot (one row per respondent, sorted by median
     relative width, colored by tier) showing the full distribution across that
     person's own priorities -- some respondents might have most priorities tightly
     pinned and a couple totally free, which the median alone would hide.

Reuses results/llm_model_cache_wave1.json -- free, no new LLM calls, as long as that
cache is already populated (run run_wave1_real_validation.py or run_wave1_full_analysis.py
first if not).

Run locally:
    python3 plot_joint_feasibility_ranges.py
"""
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from format_wave1_results import load_all
from priority_dedup import PriorityRegistry
from run_wave0_real_validation import (
    RESULTS_DIR, run_one, load_cache,
    PRIORITY_REGISTRY_PATH, PRIORITY_SIMILARITY_THRESHOLD,
)
from run_wave1_real_validation import CACHE_PATH
from validation_v1_testing import LIKERT_TO_OMEGA

TIER_COLORS = {"a": "#4C72B0", "b": "#DD8452", "c": "#55A868"}
TIER_LABELS = {
    "a": "(a) tight Likert bounds",
    "b": "(b) ordinal-only",
    "c": "(c) unconstrained",
}


def relative_widths_for_respondent(run):
    """[(priority, relative_width, tier)] for one respondent -- None if they're tier
    (d) (no feasible region to measure)."""
    jf = run["joint_feasibility"]
    tier = jf["tier"]
    if tier not in ("a", "b", "c") or not jf["omega_ranges"]:
        return None
    importance = run["result"]["importance"]
    out = []
    for priority, (lo, hi) in jf["omega_ranges"].items():
        width = hi - lo
        if tier == "a":
            box_lo, box_hi = LIKERT_TO_OMEGA.get(importance[priority], (0.5, 0.75))
            allowed = box_hi - box_lo
        else:
            allowed = 1.0
        out.append((priority, width / allowed if allowed > 0 else 0.0, tier))
    return out


def collect_all(respondents):
    cache = load_cache(CACHE_PATH)
    registry = PriorityRegistry(path=PRIORITY_REGISTRY_PATH, threshold=PRIORITY_SIMILARITY_THRESHOLD)
    results = []
    for r, label in respondents:
        run = run_one(r, cache=cache, priority_registry=registry)
        widths = relative_widths_for_respondent(run)
        results.append((r["response_id"], r["area_selfreport"], label, widths))
    return results


def plot(results, path):
    with_widths = [(respid, area, variant, w) for respid, area, variant, w in results if w]
    n_excluded = len(results) - len(with_widths)

    medians = [
        (respid, variant, w, float(np.median([rw for _, rw, _ in w])))
        for respid, area, variant, w in with_widths
    ]
    medians.sort(key=lambda m: m[3])

    fig = plt.figure(figsize=(13, 4 + 0.28 * len(medians)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], wspace=0.35)

    # --- Panel 1: histogram of per-respondent median relative width ---
    ax1 = fig.add_subplot(gs[0])
    all_medians = [m[3] for m in medians]
    ax1.hist(all_medians, bins=np.arange(0, 1.05, 0.05), color="#4C72B0", edgecolor="white")
    ax1.axvline(np.mean(all_medians), color="#C44E52", linewidth=2,
                label=f"mean of medians = {np.mean(all_medians):.0%}")
    ax1.set_xlabel("Respondent's own median relative width\n(0% = essentially one omega, 100% = fully unconstrained)")
    ax1.set_ylabel("Number of respondents")
    ax1.set_title(f"How wide, in general?\n({len(medians)} respondents at tier a/b/c"
                  + (f", {n_excluded} at tier d excluded" if n_excluded else "") + ")")
    ax1.set_xlim(-0.02, 1.02)
    ax1.legend(fontsize=8, loc="upper right")

    # --- Panel 2: per-respondent detail, sorted by median, colored by tier ---
    ax2 = fig.add_subplot(gs[1])
    rng = np.random.default_rng(0)
    for row, (respid, variant, w, med) in enumerate(medians):
        tier = w[0][2]
        color = TIER_COLORS[tier]
        rel_widths = [rw for _, rw, _ in w]
        lo, hi = min(rel_widths), max(rel_widths)
        ax2.plot([lo, hi], [row, row], color=color, alpha=0.35, linewidth=3, zorder=1)
        ys = row + rng.uniform(-0.18, 0.18, size=len(rel_widths))
        ax2.scatter(rel_widths, ys, color=color, s=14, alpha=0.85, zorder=2)
        ax2.scatter([med], [row], color="black", marker="|", s=90, zorder=3)

    ax2.set_yticks(range(len(medians)))
    ax2.set_yticklabels([f"{respid} ({variant})" for respid, variant, _, _ in medians], fontsize=6)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_xlabel("Relative width per priority (0% = single point, 100% = fully unconstrained)\n"
                   "dots = individual priorities, black tick = this respondent's median, bar = min-max span")
    ax2.set_title("Per-respondent detail, sorted by median width")

    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker="o", color="w", markerfacecolor=TIER_COLORS[t],
                            markersize=6, label=TIER_LABELS[t]) for t in ("a", "b", "c")]
    legend_elems.append(Line2D([0], [0], marker="|", color="black", markersize=10, linewidth=0, label="median"))
    ax2.legend(handles=legend_elems, fontsize=7, loc="lower right")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("saved", path)


if __name__ == "__main__":
    v1, sim = load_all()
    respondents = [(r, "V1") for r in v1] + [(r, "SimProjects") for r in sim]
    results = collect_all(respondents)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"joint_feasibility_ranges_{ts}.png")
    plot(results, path)
