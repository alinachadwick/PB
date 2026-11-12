"""
For each real Wave-0 respondent and each of their bundle-pair comparisons:

1. Plots the model's CARDINAL Delta range [Delta_min, Delta_max] (not the discrete
   strong/weak/conflicted buckets) against which side they actually chose.
2. Also plots the respondent's own reported reaction to each bundle (the Like/Neutral/
   Dislike ratings for Set A and Set B) as a "person's cardinal score" on the same axis,
   so you can see whether the model's range and the respondent's own stated intensity
   agree, not just whether the discrete choice falls inside the range.
3. Reports each respondent's JOINT feasibility tier (validation_v1_testing.py's
   joint_feasibility_tiers, computed once inside run_one and reused here): does there
   exist a single omega weighting -- one number per priority -- that reproduces ALL of
   that respondent's choices simultaneously? This is stricter than checking each pair
   independently: the existing per-pair Delta range already asks "is SOME weighting in
   this priority's box consistent with this ONE choice"; this asks "is there ONE
   weighting that works for EVERY choice at once." A respondent can pass every per-pair
   check individually while having no single weighting that explains all of them
   together -- a real GARP-style rationalizability test for the linear-additive cardinal
   model. Reported as the STRICTEST of four nested tiers that holds: (a) within each
   priority's own Likert-tier box, (b) within [0,1] but respecting stated importance
   order, (c) within [0,1] with no ordering requirement, or (d) infeasible even then.

Run locally:
    python3 plot_cardinal_margins.py

Requires: ANTHROPIC_API_KEY in .env for whatever isn't already cached (reuses
results/llm_model_cache.json -- same cache run_wave0_real_validation.py writes to and
reads from, so running that script first, or just running this one twice, makes the
second run free), plus `matplotlib` and `scipy` (pip install matplotlib scipy
if you don't already have them).

Output: prints a per-respondent feasibility summary to stdout, and saves
results/cardinal_margins_<timestamp>.png with one panel per respondent.
"""
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from format_wave0_results import load_all
from run_wave0_real_validation import run_one, CACHE_PATH, load_cache, save_cache
from validation_v1_testing import LIKERT_TO_OMEGA, priority_contributions

RESULTS_DIR = "results"

# Model's own "strong" cutoff (matches the STRONG_THRESHOLD used in predict_pair /
# outcome_labels_for_interval in validation_v1_testing.py) -- drawn as reference lines,
# not used in any computation here. Rescaled from 2.0 when u_j moved from a 0-3 to a
# 0-1 scale.
STRONG_THRESHOLD = 0.67

# Finer-grained than the -1/0/1 mapping baked into validation_answers elsewhere -- we
# want real spread for the "person's own cardinal score," not a 3-bucket compression.
FIVE_LEVEL_RATING = {
    "Very Unhappy": -2,
    "Somewhat Unhappy": -1,
    "Neutral": 0,
    "Somewhat Happy": 1,
    "Very Happy": 2,
}


def person_cardinal_score(raw_rating_a, raw_rating_b):
    """Respondent's own reaction to bundle A minus their reaction to bundle B, on the
    same sign convention as Delta (positive favors A, negative favors B). Independent of
    -- and a useful cross-check against -- their explicit chosen-side answer."""
    a = FIVE_LEVEL_RATING.get(raw_rating_a, 0)
    b = FIVE_LEVEL_RATING.get(raw_rating_b, 0)
    return a - b


def pair_delta_range(bundle_a, bundle_b, utility_functions, importance):
    """Re-derives [Delta_min, Delta_max] the same way compute_delta_range does (via
    priority_contributions, so the per-priority c_j formula lives in exactly one place),
    keeping the per-priority c_j values around too since the bar-chart annotations use
    them elsewhere."""
    omega_ranges = {p: LIKERT_TO_OMEGA.get(level, (0.5, 0.75)) for p, level in importance.items()}
    per_priority_c = priority_contributions(bundle_a, bundle_b, utility_functions, omega_ranges)
    Delta_min = Delta_max = 0.0
    for priority, c_j in per_priority_c.items():
        lo, hi = omega_ranges[priority]
        if c_j > 0:
            Delta_max += hi * c_j
            Delta_min += lo * c_j
        elif c_j < 0:
            Delta_max += lo * c_j
            Delta_min += hi * c_j
    return Delta_min, Delta_max, per_priority_c


def build_respondent_data(run):
    r = run["result"]
    utility_functions = r["utility_functions"]
    importance = r["importance"]

    pair_rows = []
    for i, (code, (bundle_a, bundle_b), truth, raw_bp) in enumerate(
        zip(run["pair_codes"], run["bundle_pairs"], run["validation_answers"], run["raw_bundle_pairs"])
    ):
        dmin, dmax, c_map = pair_delta_range(bundle_a, bundle_b, utility_functions, importance)
        person_score = person_cardinal_score(raw_bp["rating_a"], raw_bp["rating_b"])
        chosen = "A" if truth["chosen"] == 0 else "B"
        # Same definition of "consistent" used everywhere else in this project: the
        # chosen side has to actually fall within [Delta_min, Delta_max] -- i.e. the
        # range has to reach past 0 on the chosen side, not just lean that way.
        is_consistent = (dmax > 0) if chosen == "A" else (dmin < 0)
        pair_rows.append({
            "code": code,
            "delta_min": dmin,
            "delta_max": dmax,
            "chosen": chosen,
            "is_consistent": is_consistent,
            "person_score": person_score,
            "rating_a": raw_bp["rating_a"],  # raw text: "Very Happy" / "Somewhat Happy" / "Neutral" / etc.
            "rating_b": raw_bp["rating_b"],
            "certainty": truth.get("certainty"),  # 1 ("Not at all certain") .. 5 ("Extremely certain"), or None
        })
    return pair_rows


RATING_ABBREV = {
    "Very Happy": "VH", "Somewhat Happy": "SH", "Neutral": "N",
    "Somewhat Unhappy": "SU", "Very Unhappy": "VU",
}


def plot_all(all_data, path):
    n = len(all_data)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    # Extra width so the "A: ... B: ..." annotation column has room without clipping.
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 3.1 * nrows), squeeze=False)

    for idx, (respid, area, variant, jf, pair_rows) in enumerate(all_data):
        ax = axes[idx // ncols][idx % ncols]
        # "MISS" suffix on the tick label itself so it survives even in black-and-white
        # printouts/screenshots, not just the red background.
        y_labels = [row["code"] + ("  ✗ MISS" if not row["is_consistent"] else "") for row in pair_rows]
        y = np.arange(len(pair_rows))
        # Text annotations live just outside the axes (x in axes-fraction, y in data
        # coords) so they read as a clean column regardless of each panel's own x-range.
        annot_transform = ax.get_yaxis_transform()

        for i, row in enumerate(pair_rows):
            color = "#1f5e8c" if row["chosen"] == "A" else "#9c701c"
            # Flag genuine misses -- the model's Delta range never reached the side the
            # respondent actually chose -- with a red band behind the whole row. This is
            # the only thing on the chart that means "correct vs. incorrect"; every other
            # color here means "which side," not "was it right."
            if not row["is_consistent"]:
                ax.axhspan(i - 0.42, i + 0.42, color="#c0392b", alpha=0.10, zorder=0)
            # Delta range as a horizontal bar -- opacity scales with the respondent's
            # OWN reported certainty in this choice (1="Not at all certain" through
            # 5="Extremely certain"), so a miss they were barely sure about reads as
            # visually less alarming than a miss they were extremely confident about.
            # Falls back to the original fixed alpha when certainty wasn't captured.
            certainty = row.get("certainty")
            bar_alpha = 0.15 + (certainty - 1) / 4 * 0.45 if certainty else 0.35
            ax.plot([row["delta_min"], row["delta_max"]], [i, i], color=color, linewidth=6, alpha=bar_alpha, solid_capstyle="butt")
            # Person's own reported cardinal score (A-rating minus B-rating) as a marker
            ax.scatter([row["person_score"]], [i], marker="D", s=45, color=color, edgecolor="black", linewidth=0.6, zorder=3)
            # Choice indicator at the extreme edge (A = right, B = left) for reference
            edge_x = max(abs(row["delta_min"]), abs(row["delta_max"]), abs(row["person_score"]), STRONG_THRESHOLD) * 1.15
            edge_x = edge_x if row["chosen"] == "A" else -edge_x
            ax.scatter([edge_x], [i], marker="o", s=30, color=color, zorder=3)

            # The two bundle ratings, spelled out individually (not just their
            # difference) -- e.g. "A: SH  B: SU". Deliberately NOT colored: blue/gold
            # already mean "chose A / chose B" everywhere else in this chart, and
            # reusing them here for "liked/disliked this bundle" would encode two
            # different things with the same two colors. The abbreviation itself
            # (VH/SH/N/SU/VU) already says the valence, so plain text is unambiguous.
            a_txt = RATING_ABBREV.get(row["rating_a"], "?")
            b_txt = RATING_ABBREV.get(row["rating_b"], "?")
            ax.text(1.03, i, "A:", transform=annot_transform, fontsize=7.5, va="center", ha="left", color="#555555")
            ax.text(1.08, i, a_txt, transform=annot_transform, fontsize=7.5, va="center", ha="left", color="#222222", fontweight="bold")
            ax.text(1.16, i, "B:", transform=annot_transform, fontsize=7.5, va="center", ha="left", color="#555555")
            ax.text(1.21, i, b_txt, transform=annot_transform, fontsize=7.5, va="center", ha="left", color="#222222", fontweight="bold")

            # Respondent's own certainty in this choice, spelled out as a number too --
            # the bar opacity above already encodes it visually, but only roughly
            # distinguishable across 5 levels, so the exact value goes here as well.
            c_txt = f"{certainty}/5" if certainty else "?"
            ax.text(1.30, i, "C:", transform=annot_transform, fontsize=7.5, va="center", ha="left", color="#555555")
            ax.text(1.35, i, c_txt, transform=annot_transform, fontsize=7.5, va="center", ha="left", color="#222222", fontweight="bold")

        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(STRONG_THRESHOLD, color="gray", linewidth=0.6, linestyle="--")
        ax.axvline(-STRONG_THRESHOLD, color="gray", linewidth=0.6, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.invert_yaxis()
        # Leave room on the right for the rating + certainty annotations added above.
        ax.set_xlim(right=ax.get_xlim()[1] + 0.46 * (ax.get_xlim()[1] - ax.get_xlim()[0]))
        if jf["tier"] == "a":
            feas_str = "joint feasibility: tier (a) — tight Likert bounds"
        elif jf["tier"] == "d":
            feas_str = f"joint feasibility: tier (d) — infeasible (culprits: {jf['culprits'] or 'needs >1 removed'})"
        else:
            feas_str = f"joint feasibility: tier ({jf['tier']})"
        ax.set_title(f"{respid} ({area}, {variant})\n{feas_str}", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(
        "Cardinal Delta ranges (bars) vs. respondent's own reaction score (diamonds) vs. choice (dots)\n"
        "blue = chose A, gold = chose B (which side, not whether it's right)  |  "
        "red band + ✗ MISS = the range never reached the side they actually chose  |  "
        "dashed lines = model's own ±strong-prediction threshold  |  "
        "A:/B: columns = their raw reaction to each bundle (VH=Very Happy, SH=Somewhat Happy, N=Neutral, "
        "SU=Somewhat Unhappy, VU=Very Unhappy)  |  "
        "bar opacity + C: column = their own certainty in this choice (1=Not at all certain .. 5=Extremely certain)  |  "
        "joint feasibility tier: (a) tight Likert bounds > (b) ordinal-only > (c) unconstrained > (d) infeasible "
        "-- strictest tier for which SOME single omega vector explains ALL of this respondent's choices at once",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("saved", path)


if __name__ == "__main__":
    v1, sim = load_all()
    all_respondents = [(r, "V1") for r in v1] + [(r, "SimProjects") for r in sim]
    # Reuses whatever run_wave0_real_validation.py has already generated/cached for
    # these respondents instead of regenerating every classification and utility
    # function from scratch -- this used to always hit the LLM fresh regardless of
    # results/llm_model_cache.json's contents.
    cache = load_cache(CACHE_PATH)

    all_data = []
    for r, label in all_respondents:
        print(f"Running {r['response_id']} ({label})...")
        run = run_one(r, cache=cache)
        save_cache(cache, CACHE_PATH)
        # run_one doesn't currently keep the raw per-pair rating text around -- stash it
        # here from the original parsed record so build_respondent_data can use it.
        run["raw_bundle_pairs"] = r["bundle_pairs"]

        pair_rows = build_respondent_data(run)
        jf = run["joint_feasibility"]  # computed once inside run_one, see joint_feasibility_tiers

        all_data.append((r["response_id"], r["area_selfreport"], label, jf, pair_rows))

        if jf["tier"] == "d":
            culprits = jf["culprits"]
            detail = f"(dropping {culprits} would fix it)" if culprits else "(no single pair removal fixes it)"
            print(f"  tier (d) INFEASIBLE {detail}")
        else:
            print(f"  tier ({jf['tier']}) — {jf['tier_label']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"cardinal_margins_{ts}.png")
    plot_all(all_data, path)

    tier_counts = {t: sum(1 for d in all_data if d[3]["tier"] == t) for t in "abcd"}
    print(
        f"\nJoint feasibility tiers across {len(all_data)} respondents: "
        + ", ".join(f"({t}) {tier_counts[t]}" for t in "abcd")
    )
