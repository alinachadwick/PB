"""
Wave-1 counterpart to run_wave0_real_validation.py. Runs real Wave-1 respondents
(APPROVED Prolific participants only, per format_wave1_results.py) through the same
validation pipeline used for Wave 0 and for simulated personas (`run_validation` /
`build_utility_functions_v1` / `predict_pair` / `predict_pair_lexicographic`).

Nothing about that pipeline is Wave-0-specific -- run_one() and render_variant_section()
in run_wave0_real_validation.py operate purely on the respondent-record schema produced
by build_respondent() (in format_wave0_results.py), which is identical across waves. So
this file reuses both directly rather than reimplementing them; the only wave-specific
piece is the report's framing text and the output filename.

Run locally:
    python3 run_wave1_real_validation.py

Same requirements as run_wave0_real_validation.py: ANTHROPIC_API_KEY in .env, and
SimProjects needs its 3 missing demographic exports in data/ before it'll return any
respondents (see format_wave1_results.py's warnings).
"""
import os
from datetime import datetime

from format_wave1_results import load_all
from priority_dedup import PriorityRegistry
from run_wave0_real_validation import (
    RESULTS_DIR, run_one, render_variant_section, load_cache, save_cache,
    PRIORITY_REGISTRY_PATH, PRIORITY_SIMILARITY_THRESHOLD,
)

# Separate from Wave 0's cache file -- keeps the two waves' cached models distinguishable
# on disk even though response IDs shouldn't collide across waves in practice. The
# priority-reuse REGISTRY (as opposed to this generation cache) is intentionally
# shared with Wave 0 via PRIORITY_REGISTRY_PATH -- see run_wave0_real_validation.py.
CACHE_PATH = os.path.join(RESULTS_DIR, "llm_model_cache_wave1.json")


def render_report(v1_runs, sim_runs, cache=None):
    out = []
    out.append("# Wave 1 Real-Respondent Validation\n")
    out.append(
        "Runs the real Wave-1 respondents (both survey variants, filtered to APPROVED "
        "Prolific participants only) through the same validation pipeline used for "
        "Wave 0 and for simulated personas (`run_validation` / `build_utility_functions_v1` "
        "/ `predict_pair` / `predict_pair_lexicographic`). Ground truth — bundle choice, "
        "certainty, cited priorities, importance ratings — comes directly from each "
        "respondent's real answers. Each priority's utility function is generated directly "
        "from its free text plus a positive/negative sign (no explicit LLM step maps a "
        "priority to particular projects -- that step was tried and deliberately dropped; "
        "see run_ablation_no_classification.py for the comparison that informed the "
        "decision). Generated utility functions are cached "
        f"(`{CACHE_PATH}`), so re-running this script against the same respondents "
        "reuses the same model instead of regenerating it. Priority reuse ALSO happens "
        "across DIFFERENT respondents (and across Wave 0 too): a priority "
        "sufficiently similar (same project pool, same positive/negative sign, "
        "token-overlap above threshold) to one already seen reuses that earlier "
        "utility function rather than generating a new one -- see "
        "`priority_dedup.PriorityRegistry`. A per-priority note in this report flags "
        "when that happened.\n"
    )
    render_variant_section(out, "Wave1_V1", v1_runs, cache=cache)
    render_variant_section(out, "Wave1_SimProjects", sim_runs, cache=cache)
    return "\n".join(out)


if __name__ == "__main__":
    v1, sim = load_all()
    cache = load_cache(CACHE_PATH)
    # Shared with run_wave0_real_validation.py -- see PRIORITY_REGISTRY_PATH there.
    registry = PriorityRegistry(path=PRIORITY_REGISTRY_PATH, threshold=PRIORITY_SIMILARITY_THRESHOLD)

    print(f"Running {len(v1)} real Wave-1 V1 respondents through the validation pipeline...")
    v1_runs = []
    for r in v1:
        print(f"  {r['response_id']} ({r['area_selfreport']})...")
        v1_runs.append(run_one(r, cache=cache, priority_registry=registry))
        save_cache(cache, CACHE_PATH)
        registry.save()

    print(f"Running {len(sim)} real Wave-1 SimProjects respondents through the validation pipeline...")
    sim_runs = []
    for r in sim:
        print(f"  {r['response_id']} ({r['area_selfreport']})...")
        sim_runs.append(run_one(r, cache=cache, priority_registry=registry))
        save_cache(cache, CACHE_PATH)
        registry.save()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"wave1_real_validation_{ts}.md")
    with open(path, "w") as f:
        f.write(render_report(v1_runs, sim_runs, cache=cache))
    print("wrote", path)
    total = registry.hits + registry.misses
    if total:
        print(f"Priority-reuse: {registry.hits}/{total} priorities ({registry.hits/total:.0%}) "
              f"reused an existing classification/utility function instead of generating fresh.")
