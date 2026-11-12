"""
Runs real Wave-0 respondents through the validation pipeline ONCE per respondent and
produces BOTH outputs that used to require separate scripts:
  - the markdown validation report (run_wave0_real_validation.py's render_report)
  - the cardinal-margins bar chart (plot_cardinal_margins.py's plot_all)

Calling run_one() once per respondent and feeding the same result into both outputs
avoids the redundant second pass (and, before caching existed, the redundant LLM cost)
of running the two scripts separately.

Run locally:
    python3 run_wave0_full_analysis.py
"""
import os
from datetime import datetime

from format_wave0_results import load_all
from priority_dedup import PriorityRegistry
from run_wave0_real_validation import (
    RESULTS_DIR, CACHE_PATH, run_one, render_report, load_cache, save_cache,
    PRIORITY_REGISTRY_PATH, PRIORITY_SIMILARITY_THRESHOLD,
)
from plot_cardinal_margins import build_respondent_data, plot_all


def process(respondents, label, cache, registry):
    runs = []
    plot_rows = []
    for r in respondents:
        print(f"  {r['response_id']} ({r['area_selfreport']}, {label})...")
        run = run_one(r, cache=cache, priority_registry=registry)
        save_cache(cache, CACHE_PATH)
        registry.save()
        runs.append(run)

        # build_respondent_data expects the raw per-pair rating text, which run_one's
        # return dict doesn't carry (it's only needed for plotting, not the report).
        run["raw_bundle_pairs"] = r["bundle_pairs"]
        pair_rows = build_respondent_data(run)
        jf = run["joint_feasibility"]  # computed once inside run_one, see joint_feasibility_tiers
        plot_rows.append((r["response_id"], r["area_selfreport"], label, jf, pair_rows))

        if jf["tier"] == "d":
            culprits = jf["culprits"]
            detail = f"(dropping {culprits} would fix it)" if culprits else "(no single pair removal fixes it)"
            print(f"    tier (d) INFEASIBLE {detail}")
        else:
            print(f"    tier ({jf['tier']}) — {jf['tier_label']}")
    return runs, plot_rows


if __name__ == "__main__":
    v1, sim = load_all()
    cache = load_cache(CACHE_PATH)
    registry = PriorityRegistry(path=PRIORITY_REGISTRY_PATH, threshold=PRIORITY_SIMILARITY_THRESHOLD)

    print(f"Running {len(v1)} real V1 respondents...")
    v1_runs, v1_plot = process(v1, "V1", cache, registry)

    print(f"Running {len(sim)} real SimProjects respondents...")
    sim_runs, sim_plot = process(sim, "SimProjects", cache, registry)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    md_path = os.path.join(RESULTS_DIR, f"wave0_real_validation_{ts}.md")
    with open(md_path, "w") as f:
        f.write(render_report(v1_runs, sim_runs, cache=cache))
    print("wrote", md_path)

    png_path = os.path.join(RESULTS_DIR, f"cardinal_margins_{ts}.png")
    plot_all(v1_plot + sim_plot, png_path)
    print("wrote", png_path)

    all_plot = v1_plot + sim_plot
    tier_counts = {t: sum(1 for d in all_plot if d[3]["tier"] == t) for t in "abcd"}
    print(
        f"\nJoint feasibility tiers across {len(all_plot)} respondents: "
        + ", ".join(f"({t}) {tier_counts[t]}" for t in "abcd")
    )

    total = registry.hits + registry.misses
    if total:
        print(f"Priority-reuse: {registry.hits}/{total} priorities ({registry.hits/total:.0%}) "
              f"reused an existing utility function instead of generating fresh.")
