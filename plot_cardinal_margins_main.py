"""
Cardinal-margins plots for the MAIN run (type-based utilities from cached minimal
representations, floor 0, unclassifiable = flat; see experiment_priority_types.py).
Same chart as plot_cardinal_margins.plot_all / plot_cardinal_margins_wave1, written once for
Wave 1 and once for Wave 0. Priority registry is read, never saved. No LLM calls expected
(minimal reps for both waves were cached by plot_validation_accuracy_main.py).

Writes results/cardinal_margins_main_wave{1,0}_<ts>.png
Run: python3 plot_cardinal_margins_main.py
"""
import os
from datetime import datetime

import experiment_priority_types as ept
import experiment_minimal_representation as emp
from format_wave1_results import load_all
from format_wave0_results import load_all as load_all_wave0
from priority_dedup import PriorityRegistry
from run_wave0_real_validation import (
    RESULTS_DIR, run_one, PRIORITY_REGISTRY_PATH, PRIORITY_SIMILARITY_THRESHOLD,
)
from plot_cardinal_margins import build_respondent_data, plot_all


def collect(respondents, cache, reg):
    rows = []
    for r in respondents:
        run = run_one(r, cache=cache, priority_registry=reg)
        if run["result"] is None:
            continue
        run["raw_bundle_pairs"] = r["bundle_pairs"]
        rows.append((r["response_id"], r["area_selfreport"], r["variant"],
                     run["joint_feasibility"], build_respondent_data(run)))
    return rows


def main():
    v1, sim = load_all()
    w0_v1, w0_sim = load_all_wave0()
    cache = emp.load_cache(emp.CACHE_PATH)
    emp.v.build_utility_functions_v1 = ept.make_builder(set(ept.CLASSIFIABLE))
    reg = PriorityRegistry(path=PRIORITY_REGISTRY_PATH, threshold=PRIORITY_SIMILARITY_THRESHOLD)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for wave, group in ((1, v1 + sim), (0, w0_v1 + w0_sim)):
        data = collect(group, cache, reg)
        path = os.path.join(RESULTS_DIR, f"cardinal_margins_main_wave{wave}_{ts}.png")
        plot_all(data, path)
        print(f"Wave {wave}: {len(data)} respondents -> {path}")


if __name__ == "__main__":
    main()
