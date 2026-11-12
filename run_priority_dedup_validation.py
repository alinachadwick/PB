"""
Runs every real respondent (Wave 0 + Wave 1, both survey variants) through the
validation pipeline with the priority-reuse cache turned ON: before generating a fresh
utility function for a priority (classification is off by default pipeline-wide -- see
run_wave0_real_validation.py), checks whether a sufficiently similar
priority has already been processed in the SAME project pool (V1::urban, V1::rural,
SimProjects::urban, SimProjects::rural -- the same four pools regardless of wave, since
both waves use the literal same PROJECTS_* dicts from validation_v1_testing.py) and
reuses that entry instead of paying for a new LLM call. See priority_dedup.py for the
similarity check itself, and cluster_priorities.py for an offline, whole-corpus
visualization of the same underlying idea (how much duplication is really in the data).

run_wave0_real_validation.py and run_wave1_real_validation.py now default to
priority-reuse too (scoped to their own single wave); this script's distinct value is
running BOTH waves through ONE shared generation cache (results/llm_model_cache_dedup.json,
separate from results/llm_model_cache.json / llm_model_cache_wave1.json so it never
collides with or degrades those scripts' own per-wave runs), which is what actually lets
a match across waves save a real LLM call rather than just being registered as "seen"
in the shared priority-similarity registry those scripts also read/write. Treat this
run's consistency numbers as measuring an approximation traded off against LLM cost (a
reused utility function was generated for a different, merely SIMILAR respondent's
wording and persona context), not as a drop-in replacement for the per-respondent runs
-- compare its aggregate consistency rates against wave0_real_validation_*.md /
wave1_real_validation_*.md to see what the tradeoff costs.

Run locally (requires ANTHROPIC_API_KEY in .env):
    python3 run_priority_dedup_validation.py
"""
import os
from datetime import datetime

from priority_dedup import PriorityRegistry
from run_wave0_real_validation import (
    RESULTS_DIR, run_one, load_cache, save_cache,
    PRIORITY_REGISTRY_PATH, PRIORITY_SIMILARITY_THRESHOLD,
)

CACHE_PATH = os.path.join(RESULTS_DIR, "llm_model_cache_dedup.json")
# Shared with run_wave0_real_validation.py / run_wave1_real_validation.py, which now
# also default to priority-reuse -- one growing index across all three scripts rather
# than a separate one here that would just diverge from theirs.
REGISTRY_PATH = PRIORITY_REGISTRY_PATH
SIMILARITY_THRESHOLD = PRIORITY_SIMILARITY_THRESHOLD


def fmt_pct(x):
    return f"{x:.0%}" if x is not None else "n/a"


def load_all_respondents():
    """Every real respondent across both waves and both variants, each tagged with
    which wave it came from. Loaders are tried independently -- if one wave's raw data
    isn't present in this checkout, the other still runs."""
    all_r = []
    try:
        from format_wave0_results import load_all as load_wave0
        v1, sim = load_wave0()
        all_r += [("Wave0", r) for r in v1 + sim]
    except Exception as e:
        print(f"  (skipping Wave 0: {e})")
    try:
        from format_wave1_results import load_all as load_wave1
        v1, sim = load_wave1()
        all_r += [("Wave1", r) for r in v1 + sim]
    except Exception as e:
        print(f"  (skipping Wave 1: {e})")
    return all_r


def render_report(rows, registry):
    out = []
    out.append("# Priority-Reuse (Dedup) Validation\n")
    total_lookups = registry.hits + registry.misses
    hit_rate = registry.hits / total_lookups if total_lookups else 0.0
    out.append(
        f"{len(rows)} respondents processed (Wave 0 + Wave 1, both variants, shared per-pool "
        f"registry, similarity threshold {SIMILARITY_THRESHOLD}). Of {total_lookups} elicited-"
        f"priority lookups, **{registry.hits} ({hit_rate:.0%}) reused an existing, "
        f"sufficiently similar priority's utility function** instead of "
        f"generating a fresh one; **{registry.misses}** were novel and generated fresh.\n"
    )
    for pool_id in sorted(registry.pools.keys()):
        out.append(f"- `{pool_id}`: {len(registry.pools[pool_id])} distinct canonical priorities registered")
    out.append("")

    c_all = [run["result"]["consistency_rate"] for _, run in rows if run["result"]]
    l_all = [run["result"]["lexicographic_consistency_rate"] for _, run in rows if run["result"]]
    if c_all:
        out.append(f"- Mean cardinal consistency (dedup pipeline): {fmt_pct(sum(c_all)/len(c_all))}")
    if l_all:
        out.append(f"- Mean lexicographic consistency (dedup pipeline): {fmt_pct(sum(l_all)/len(l_all))}")
    out.append(
        "\nCompare these two numbers against the aggregate lines in your most recent "
        "wave0_real_validation_*.md / wave1_real_validation_*.md reports (same "
        "respondents, per-respondent generation instead of shared/reused) to see "
        "what accuracy this trades away for the reduced LLM cost.\n"
    )

    out.append("| Wave | Respondent | Pool | Area | Cardinal | Lex |")
    out.append("|---|---|---|---|---|---|")
    for wave_label, run in rows:
        if not run["result"]:
            continue
        out.append(
            f"| {wave_label} | {run['response_id']} | {run.get('pool_id', '?')} | {run['area']} | "
            f"{fmt_pct(run['result']['consistency_rate'])} | "
            f"{fmt_pct(run['result']['lexicographic_consistency_rate'])} |"
        )
    return "\n".join(out)


if __name__ == "__main__":
    respondents = load_all_respondents()
    cache = load_cache(CACHE_PATH)
    registry = PriorityRegistry(path=REGISTRY_PATH, threshold=SIMILARITY_THRESHOLD)

    print(f"Running {len(respondents)} real respondents through the priority-reuse pipeline...")
    rows = []
    for wave_label, r in respondents:
        print(f"  {wave_label}/{r['response_id']} ({r['variant']}, {r['area_selfreport']})...")
        run = run_one(r, cache=cache, priority_registry=registry)
        save_cache(cache, CACHE_PATH)  # incremental, so an interrupted run doesn't lose earlier work
        registry.save(REGISTRY_PATH)
        rows.append((wave_label, run))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"priority_dedup_validation_{ts}.md")
    with open(path, "w") as f:
        f.write(render_report(rows, registry))
    print("wrote", path)
    total = registry.hits + registry.misses
    print(f"Registry cache hit rate: {registry.hits}/{total} ({registry.hits/total:.0%})" if total else "no lookups")
