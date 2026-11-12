"""
Answers exactly one question: for each real Wave-0 respondent and each of their bundle-pair
comparisons, do the two sides activate disjoint, overlapping, identical, or empty sets of
that person's IMPORTANT priorities (top Likert tier)? See analyze_wave0_activation.py's
module docstring for the "active" / "important" definitions this reuses.
"""
import os
from datetime import datetime

from format_wave0_results import load_all
from run_wave0_real_validation import run_one
from analyze_wave0_activation import analyze_respondent, RESULTS_DIR


def render_report(v1_analyses, sim_analyses):
    out = []
    out.append("# Wave 0 — Per-Pair Priority Activation (disjoint / overlapping / identical / empty)\n")
    out.append(
        "A priority is **active** in a bundle if its utility function scores that bundle "
        "above 0. **\"Important\" priorities** = a respondent's top Likert tier only "
        "(`\"I'd want this even if it meant giving up other priorities\"`). For each pair, "
        "comparing which important priorities are active on side A vs. side B gives:\n"
        "- **empty** — neither side activates any important priority\n"
        "- **disjoint** — both sides activate something, but no priority is active on both\n"
        "- **overlapping** — some priorities are active on both sides, some only on one\n"
        "- **identical** — the exact same set of important priorities is active on both sides\n"
    )

    for variant, analyses in [("Survey1_V1", v1_analyses), ("Survey1_V1_SimProjects", sim_analyses)]:
        out.append(f"## {variant}\n")
        for a in analyses:
            out.append(f"### {a['response_id']} — {a['area']}")
            out.append(f"Important priorities ({a['n_important']}): {a['important_priorities']}\n")
            out.append("| Pair | Active on A | Active on B | Relationship |")
            out.append("|---|---|---|---|")
            for pp in a["per_pair"]:
                out.append(
                    f"| {pp['code']} | {sorted(pp['active_a_important']) or '—'} | "
                    f"{sorted(pp['active_b_important']) or '—'} | **{pp['relationship_important']}** |"
                )
            out.append("")

    return "\n".join(out)


if __name__ == "__main__":
    v1, sim = load_all()

    print(f"Running {len(v1)} real V1 respondents...")
    v1_runs = [run_one(r) for r in v1]

    print(f"Running {len(sim)} real SimProjects respondents...")
    sim_runs = [run_one(r) for r in sim]

    v1_analyses = [analyze_respondent(run) for run in v1_runs]
    sim_analyses = [analyze_respondent(run) for run in sim_runs]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"wave0_pair_activation_{ts}.md")
    with open(path, "w") as f:
        f.write(render_report(v1_analyses, sim_analyses))
    print("wrote", path)
