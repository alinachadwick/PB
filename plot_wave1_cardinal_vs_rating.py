"""
Regresses the model's predicted cardinal utility for each INDIVIDUAL bundle a real
Wave-1 respondent rated against their own stated Likert-cardinal reaction to that same
bundle (Very Unhappy .. Very Happy) -- using ONLY already-cached classifications/
utility functions from the existing Wave-1 validation cache. Makes ZERO new LLM calls
as long as results/llm_model_cache_wave1.json already covers every Wave-1 respondent
(it does, from the last full run_wave1_real_validation.py run) -- this script calls
run_one() the same way that script's __main__ originally did (no priority_registry),
so cache keys match exactly what's already stored.

For one bundle S, U(S) = sum_j omega_j * u_j(S) is a RANGE [U_min, U_max]: omega_j
ranges over that priority's Likert-tier box, and since every u_j(S) >= 0 by
construction (no second bundle to subtract, unlike the pairwise Delta computation),
U_min always uses each priority's LOW end and U_max always uses the HIGH end -- no
per-priority sign bookkeeping needed.

The regression line itself supplies the otherwise-missing calibration between the
model's arbitrary utility scale and the Likert scale -- it's fit from the data, not
assumed in advance (see the earlier discussion about U(bundle) having no natural
zero-point matching "Neutral": a regression sidesteps that by just learning whatever
affine map best fits, rather than requiring one to be invented up front).

Run locally:
    python3 plot_wave1_cardinal_vs_rating.py
"""
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from format_wave1_results import load_all
from run_wave0_real_validation import RESULTS_DIR, run_one, load_cache
from validation_v1_testing import LIKERT_TO_OMEGA

CACHE_PATH = os.path.join(RESULTS_DIR, "llm_model_cache_wave1.json")

FIVE_LEVEL_RATING = {
    "Very Unhappy": -2, "Somewhat Unhappy": -1, "Neutral": 0,
    "Somewhat Happy": 1, "Very Happy": 2,
}
RATING_ORDER = ["Very Unhappy", "Somewhat Unhappy", "Neutral", "Somewhat Happy", "Very Happy"]


def bundle_utility_range(bundle, utility_functions, importance):
    u_min = u_max = 0.0
    for priority, level in importance.items():
        u_j = utility_functions.get(priority)
        if u_j is None:
            continue
        lo, hi = LIKERT_TO_OMEGA.get(level, (0.5, 0.75))
        u_val = u_j(bundle)  # always >= 0 by construction
        u_min += lo * u_val
        u_max += hi * u_val
    return u_min, u_max


def collect_points(runs):
    points = []
    for run in runs:
        utility_functions = run["result"]["utility_functions"]
        importance = run["result"]["importance"]
        for code, (bundle_a, bundle_b), ans in zip(run["pair_codes"], run["bundle_pairs"], run["validation_answers"]):
            for bundle, raw_rating in [(bundle_a, ans["rating_a_raw"]), (bundle_b, ans["rating_b_raw"])]:
                if raw_rating not in FIVE_LEVEL_RATING:
                    continue
                u_min, u_max = bundle_utility_range(bundle, utility_functions, importance)
                points.append({
                    "u_min": u_min, "u_max": u_max, "u_mid": (u_min + u_max) / 2,
                    "rating": FIVE_LEVEL_RATING[raw_rating], "raw_rating": raw_rating,
                    "response_id": run["response_id"], "code": code,
                })
    return points


def main():
    v1, sim = load_all()
    cache = load_cache(CACHE_PATH)
    n_before = len(cache)

    print(f"Reconstructing cached utility functions for {len(v1) + len(sim)} Wave-1 respondents...")
    all_runs = [run_one(r, cache=cache) for r in v1 + sim]

    n_after = len(cache)
    if n_after > n_before:
        print(f"NOTE: {n_after - n_before} new cache entries were generated (some priorities weren't "
              f"already cached) -- these did require fresh LLM calls.")
    else:
        print("Confirmed: zero new cache entries -- everything came from the existing cache.")

    points = collect_points(all_runs)
    print(f"{len(points)} (bundle, rating) data points collected")

    x = np.array([p["u_mid"] for p in points])
    y = np.array([p["rating"] for p in points])
    xerr_lo = x - np.array([p["u_min"] for p in points])
    xerr_hi = np.array([p["u_max"] for p in points]) - x

    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]

    fig, ax = plt.subplots(figsize=(9, 6))
    rng = np.random.default_rng(0)
    y_jittered = y + rng.uniform(-0.08, 0.08, size=len(y))  # so same-rating points don't fully overlap
    ax.errorbar(x, y_jittered, xerr=[xerr_lo, xerr_hi], fmt="o", markersize=4, alpha=0.5,
                ecolor="gray", elinewidth=0.7, capsize=0, color="#4C72B0", label="bundle ratings")

    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="#C44E52", linewidth=2,
             label=f"fit: rating = {slope:.3f}·U + {intercept:.2f}  (r={r:.2f}, R²={r**2:.2f})")

    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(RATING_ORDER)
    ax.set_xlabel("Model-predicted bundle utility U(bundle) -- point = midpoint, bars = [U_min, U_max] range")
    ax.set_ylabel("Respondent's own reported reaction")
    ax.set_title(f"Wave 1: predicted cardinal utility vs. stated Likert reaction ({len(points)} bundle ratings)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"wave1_cardinal_vs_rating_{ts}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)
    print(f"Pearson r = {r:.3f}, R² = {r**2:.3f}, slope = {slope:.4f}, intercept = {intercept:.3f}")


if __name__ == "__main__":
    main()
