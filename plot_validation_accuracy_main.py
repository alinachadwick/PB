"""
Updated version of plot_validation_accuracy.py's figure for the MAIN run (type-based
utilities, floor 0, unclassifiable = flat): identical layout, Wave 0 and Wave 1 (light = Wave 0).

Bar height = # of the respondent's pairs the model is CONSISTENT with (the old figure's
"Consistent" column quantity: the chosen side is among the model's compatible outcomes, so
indecisive predictions count). Reads the per-respondent counts saved by an earlier run
(results/validation_accuracy_main_*.json, written by collect() below); pass --rerun to
recompute them (~80 s, no LLM calls).

Run: python3 plot_validation_accuracy_main.py [--rerun]
"""
import glob
import json
import sys

import plot_validation_accuracy as base


def collect():
    import experiment_priority_types as ept
    import experiment_minimal_representation as emp
    from format_wave1_results import load_all
    from format_wave0_results import load_all as load_all_wave0
    from priority_dedup import PriorityRegistry
    from run_wave0_real_validation import run_one, PRIORITY_REGISTRY_PATH, PRIORITY_SIMILARITY_THRESHOLD
    v1, sim = load_all()
    w0_v1, w0_sim = load_all_wave0()
    cache = emp.load_cache(emp.CACHE_PATH)
    emp.v.build_utility_functions_v1 = ept.make_builder(set(ept.CLASSIFIABLE))
    reg = PriorityRegistry(path=PRIORITY_REGISTRY_PATH, threshold=PRIORITY_SIMILARITY_THRESHOLD)  # never saved
    out = []
    # Wave 1 first so its registry/cache state matches the earlier Wave 1 runs exactly; Wave 0
    # priorities not yet in the minimal-rep cache are extracted (LLM) and added to it.
    for wave, group in ((1, v1 + sim), (0, w0_v1 + w0_sim)):
      for r in group:
        res = run_one(r, cache=cache, priority_registry=reg)["result"]
        if res is None:
            continue
        n_cons = sum(bool(pp["is_consistent"]) for pp in res["per_pair"])
        out.append({"id": r["response_id"], "survey_type": r["variant"], "wave": wave,
                    "correct": n_cons, "total": len(res["per_pair"])})
    return out


def main():
    files = sorted(glob.glob("results/validation_accuracy_main_*.json"))
    if "--rerun" in sys.argv or not files:
        rows = collect()
        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        json.dump(rows, open(f"results/validation_accuracy_main_{ts}.json", "w"), indent=1)
    else:
        rows = json.load(open(files[-1]))
        if "survey_type" not in rows[0]:  # older file layout: committed/indecisive split
            rows = [{"id": r["id"], "survey_type": r["variant"], "wave": 1,
                     "correct": r["committed"] + r["indecisive"], "total": r["total"]} for r in rows]
    for r in rows:
        r["accuracy"] = r["correct"] / r["total"]
    base.main(respondents=rows,
              title="Respondent accuracy, V1 vs. SimProjects -- Wave 0 + Wave 1 combined, type-based model",
              out_prefix="validation_accuracy_main")


if __name__ == "__main__":
    main()
