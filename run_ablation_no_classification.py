"""
Ablation: does the LLM-inferred per-project priority classification actually help?

`build_utility_functions_v1` has two branches:
  - "Classified": per-project good/bad/depends breakdown from
    `generate_priority_classifications`, one extra LLM call per priority.
  - "Legacy": a bare positive/negative sign per priority (inferred here purely from
    which open-ended survey question it came from -- bundle_good/mix_good -> positive,
    bundle_bad -> negative), no per-project information at all. Real respondents never
    have the old priority_projects free-select list (that survey question was
    removed), so for them this really is sign-only.

This script runs the same real Wave-1 respondents through both branches and reports
cardinal/lexicographic consistency side by side, so the answer to "does classification
help" comes from data instead of guesswork.

RESULT / STATUS: this comparison is what informed the team's decision to drop the
classification step entirely -- run_wave0_real_validation.py and
run_wave1_real_validation.py now default to skip_classification=True (the "Legacy"
column below), not the "Classified" one. This script is kept as the historical record
of that comparison and to support re-running it if the decision is ever revisited; it
does not need to be run as part of the normal reporting workflow.

Cost note: the "classified" column reuses results/llm_model_cache_wave1.json, so if
that's already populated from a prior run_wave1_real_validation.py run, it costs
nothing. The "no-classification" column is a genuinely fresh set of utility-function
LLM calls (one per elicited priority per respondent, ~same volume as the classified
run's utility-function calls -- it just skips the classification calls) written to a
separate cache file so it never collides with or overwrites the classified cache.

Run locally (requires ANTHROPIC_API_KEY in .env):
    python3 run_ablation_no_classification.py
"""
import os
from datetime import datetime

from format_wave1_results import load_all
from run_wave0_real_validation import RESULTS_DIR, run_one, load_cache, save_cache

CLASSIFIED_CACHE_PATH = os.path.join(RESULTS_DIR, "llm_model_cache_wave1.json")
NOCLASS_CACHE_PATH = os.path.join(RESULTS_DIR, "llm_model_cache_wave1_noclass.json")


def run_both(respondents, label, classified_cache, noclass_cache):
    rows = []
    for r in respondents:
        print(f"  {r['response_id']} ({r['area_selfreport']}, {label})...")
        classified = run_one(r, cache=classified_cache, skip_classification=False)
        save_cache(classified_cache, CLASSIFIED_CACHE_PATH)
        noclass = run_one(r, cache=noclass_cache, skip_classification=True)
        save_cache(noclass_cache, NOCLASS_CACHE_PATH)
        rows.append((r, classified, noclass))
    return rows


def fmt_pct(x):
    return f"{x:.0%}" if x is not None else "n/a"


def render_report(v1_rows, sim_rows):
    out = []
    out.append("# Ablation: LLM Project<>Priority Classification\n")
    out.append(
        "Compares the normal pipeline (per-project good/bad/depends classification via "
        "`generate_priority_classifications`, feeding `build_utility_functions_v1`'s "
        "\"Classified\" branch) against an ablated version that skips classification "
        "entirely and generates every utility function through the \"Legacy\" branch — "
        "a bare positive/negative sign per priority (inferred from which open-ended "
        "question it came from), with no per-project information at all. Same real "
        "Wave-1 respondents, same bundle pairs, same ground truth in both columns; only "
        "the utility-function generation input differs.\n"
    )

    for label, rows in [("Wave1_V1", v1_rows), ("Wave1_SimProjects", sim_rows)]:
        c_cardinal = [c["result"]["consistency_rate"] for _, c, _ in rows if c["result"]]
        c_lex = [c["result"]["lexicographic_consistency_rate"] for _, c, _ in rows if c["result"]]
        n_cardinal = [n["result"]["consistency_rate"] for _, _, n in rows if n["result"]]
        n_lex = [n["result"]["lexicographic_consistency_rate"] for _, _, n in rows if n["result"]]

        out.append(f"## {label} — aggregate\n")
        out.append(f"- Respondents: {len(rows)}")
        if c_cardinal and n_cardinal:
            out.append(
                f"- Mean cardinal consistency — classified: {fmt_pct(sum(c_cardinal)/len(c_cardinal))}"
                f"  vs. no-classification: {fmt_pct(sum(n_cardinal)/len(n_cardinal))}"
            )
        if c_lex and n_lex:
            out.append(
                f"- Mean lexicographic consistency — classified: {fmt_pct(sum(c_lex)/len(c_lex))}"
                f"  vs. no-classification: {fmt_pct(sum(n_lex)/len(n_lex))}"
            )
        out.append("")
        out.append(
            "| Respondent | Area | Cardinal (classified) | Cardinal (no-class) | Δ | "
            "Lex (classified) | Lex (no-class) | Δ |"
        )
        out.append("|---|---|---|---|---|---|---|---|")
        for r, c, n in rows:
            if not c["result"] or not n["result"]:
                continue
            cc = c["result"]["consistency_rate"]
            nc = n["result"]["consistency_rate"]
            cl = c["result"]["lexicographic_consistency_rate"]
            nl = n["result"]["lexicographic_consistency_rate"]
            out.append(
                f"| {r['response_id']} | {r['area_selfreport']} | {fmt_pct(cc)} | {fmt_pct(nc)} | "
                f"{(nc-cc)*100:+.0f}pp | {fmt_pct(cl)} | {fmt_pct(nl)} | {(nl-cl)*100:+.0f}pp |"
            )
        out.append("")

    return "\n".join(out)


if __name__ == "__main__":
    v1, sim = load_all()
    classified_cache = load_cache(CLASSIFIED_CACHE_PATH)
    noclass_cache = load_cache(NOCLASS_CACHE_PATH)

    print(f"Running {len(v1)} Wave-1 V1 respondents (classified vs. no-classification)...")
    v1_rows = run_both(v1, "V1", classified_cache, noclass_cache)

    print(f"Running {len(sim)} Wave-1 SimProjects respondents (classified vs. no-classification)...")
    sim_rows = run_both(sim, "SimProjects", classified_cache, noclass_cache)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"ablation_no_classification_{ts}.md")
    with open(path, "w") as f:
        f.write(render_report(v1_rows, sim_rows))
    print("wrote", path)
