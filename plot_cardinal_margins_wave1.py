"""
Wave-1 counterpart to plot_cardinal_margins.py. Every piece of that script's logic --
Delta-range computation, the respondent's own cardinal reaction score, the joint
feasibility LP, the miss-flagging plot -- operates purely on the respondent-record
schema produced by build_respondent() (in format_wave0_results.py), which is identical
regardless of which wave the data came from. So nothing here is reimplemented; this
just points the same pipeline at Wave 1's data (already filtered to APPROVED Prolific
participants by format_wave1_results.load_all()) and saves to a separate output file.

Run locally:
    python3 plot_cardinal_margins_wave1.py

Same requirements as plot_cardinal_margins.py: ANTHROPIC_API_KEY in .env for whatever
isn't already cached (reuses results/llm_model_cache_wave1.json -- same cache
run_wave1_real_validation.py writes to and reads from), matplotlib + scipy.
"""
import os
from datetime import datetime

from format_wave1_results import load_all
from run_wave0_real_validation import run_one, load_cache, save_cache
from run_wave1_real_validation import CACHE_PATH
from plot_cardinal_margins import (
    RESULTS_DIR,
    build_respondent_data,
    joint_feasibility,
    plot_all,
)

if __name__ == "__main__":
    v1, sim = load_all()
    all_respondents = [(r, "V1") for r in v1] + [(r, "SimProjects") for r in sim]
    cache = load_cache(CACHE_PATH)

    all_data = []
    for r, label in all_respondents:
        print(f"Running {r['response_id']} ({label})...")
        run = run_one(r, cache=cache)
        save_cache(cache, CACHE_PATH)
        run["raw_bundle_pairs"] = r["bundle_pairs"]

        pair_rows, per_pair_c = build_respondent_data(run)
        feasible, culprits = joint_feasibility(run, per_pair_c)

        all_data.append((r["response_id"], r["area_selfreport"], label, feasible, culprits, pair_rows))

        print(f"  {'FEASIBLE' if feasible else 'INFEASIBLE'} "
              f"{'' if feasible else f'(dropping {culprits} would fix it)' if culprits else '(no single pair removal fixes it)'}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"cardinal_margins_wave1_{ts}.png")
    plot_all(all_data, path)

    n_feasible = sum(1 for d in all_data if d[3])
    print(f"\n{n_feasible}/{len(all_data)} respondents have a single weighting "
          f"(within their own Likert-tier bounds) consistent with ALL their choices.")
