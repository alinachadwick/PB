"""
Answers the "priority activation" batch of planning-doc questions for the real Wave-0
respondents, using the SAME utility functions/classifications built by
run_wave0_real_validation.py's pipeline (re-run here to get access to the raw u_j
functions, which run_validation doesn't persist anywhere on its own).

Definitions used throughout:
- A priority p is ACTIVE in a bundle X if u_p(X) > 0 (the model's own calibration
  anchor: 0.0 means "the bundle does essentially nothing to satisfy this criterion").
- "Important" priorities = a respondent's top Likert tier (LIKERT_MOST, "I'd want this
  even if it meant giving up other priorities") -- the literal reading of "important
  priorities" in the planning doc. All-priority activation is also computed and reported
  as supplementary context, but the headline numbers use this restriction.
- For a given (respondent, pair), comparing active-set(A) vs active-set(B) over the
  important-priority set gives one of: empty (neither side activates anything),
  disjoint (nonempty, no overlap), overlapping (share some but not all), identical
  (exactly the same priorities active on both sides).
"""
import json
from collections import Counter, defaultdict

from format_wave0_results import load_all
from run_wave0_real_validation import run_one
from validation_v1_testing import LIKERT_MOST, normalize, compute_lexicographic_evidence

RESULTS_DIR = "results"
EPS = 1e-9


def important_priorities(importance):
    return [p for p, level in importance.items() if level == LIKERT_MOST]


def activation_relationship(active_a, active_b):
    if not active_a and not active_b:
        return "empty"
    if active_a == active_b:
        return "identical"
    if active_a & active_b:
        return "overlapping"
    return "disjoint"


def compute_pair_activation(bundle_a, bundle_b, utility_functions, priority_set):
    active_a, active_b = set(), set()
    values = {}
    for p in priority_set:
        u_j = utility_functions.get(p)
        if u_j is None:
            continue
        u_a, u_b = u_j(bundle_a), u_j(bundle_b)
        values[p] = (u_a, u_b)
        if u_a > EPS:
            active_a.add(p)
        if u_b > EPS:
            active_b.add(p)
    return active_a, active_b, values


def analyze_respondent(run):
    r = run["result"]
    utility_functions = r["utility_functions"]
    importance = r["importance"]
    important = important_priorities(importance)
    all_priorities = list(importance.keys())

    per_pair = []
    ever_active_important = set()
    ever_cited_norm = set()

    for code, (bundle_a, bundle_b), truth in zip(
        run["pair_codes"], run["bundle_pairs"], run["validation_answers"]
    ):
        active_a_imp, active_b_imp, _ = compute_pair_activation(bundle_a, bundle_b, utility_functions, important)
        active_a_all, active_b_all, _ = compute_pair_activation(bundle_a, bundle_b, utility_functions, all_priorities)
        relationship_important = activation_relationship(active_a_imp, active_b_imp)
        relationship_all = activation_relationship(active_a_all, active_b_all)

        ever_active_important |= (active_a_imp | active_b_imp)
        for c in truth.get("cited_priorities", []):
            if not c.lower().startswith("other:"):
                ever_cited_norm.add(normalize(c))

        per_pair.append({
            "code": code,
            "active_a_important": active_a_imp,
            "active_b_important": active_b_imp,
            "relationship_important": relationship_important,
            "active_a_all": active_a_all,
            "active_b_all": active_b_all,
            "relationship_all": relationship_all,
            "n_important_activated": len(active_a_imp | active_b_imp),
        })

    never_activated_important = set(important) - ever_active_important
    never_cited_important = {p for p in important if normalize(p) not in ever_cited_norm}

    return {
        "response_id": run["response_id"],
        "area": run["area"],
        "important_priorities": important,
        "n_important": len(important),
        "n_all": len(all_priorities),
        "per_pair": per_pair,
        "never_activated_important": never_activated_important,
        "never_cited_important": never_cited_important,
    }


def cross_pair_diversity(analysis):
    """Which important priorities recur across multiple pairs vs. appear in only one?"""
    counts = Counter()
    for pp in analysis["per_pair"]:
        for p in (pp["active_a_important"] | pp["active_b_important"]):
            counts[p] += 1
    n_pairs = len(analysis["per_pair"])
    recurring = {p: c for p, c in counts.items() if c > 1}
    one_off = {p: c for p, c in counts.items() if c == 1}
    return counts, recurring, one_off, n_pairs


def render_report(v1_analyses, sim_analyses, v1_runs, sim_runs):
    out = []
    out.append("# Wave 0 — Priority Activation Analysis\n")
    out.append(
        "A priority is **active** in a bundle if its (LLM-built) utility function scores "
        "that bundle above 0 -- the model's own calibration anchor for \"does essentially "
        "nothing to satisfy this criterion.\" **\"Important\" priorities** = a respondent's "
        "top Likert tier only (`\"I'd want this even if it meant giving up other "
        "priorities\"`). Headline numbers below use that restriction; all-priority "
        "activation is noted where it changes the picture.\n"
    )

    all_analyses = {"Survey1_V1": v1_analyses, "Survey1_V1_SimProjects": sim_analyses}

    # --- Q1/Q2: per-pair relationship + per-person cross-pair diversity ---
    out.append("## 1–2. Per-pair activation relationship, and whether different pairs activate different priorities\n")
    for variant, analyses in all_analyses.items():
        out.append(f"### {variant}\n")
        rel_counts = Counter()
        for a in analyses:
            out.append(f"**{a['response_id']}** ({a['area']}) — {a['n_important']} important priorities: {a['important_priorities']}\n")
            out.append("| Pair | Relationship (important) | Active on A | Active on B | Relationship (all priorities) |")
            out.append("|---|---|---|---|---|")
            for pp in a["per_pair"]:
                rel_counts[pp["relationship_important"]] += 1
                out.append(
                    f"| {pp['code']} | {pp['relationship_important']} | "
                    f"{sorted(pp['active_a_important']) or '—'} | {sorted(pp['active_b_important']) or '—'} | "
                    f"{pp['relationship_all']} |"
                )
            counts, recurring, one_off, n_pairs = cross_pair_diversity(a)
            out.append("")
            if counts:
                out.append(
                    f"- Across {n_pairs} pairs: {len(recurring)} important priorit{'y' if len(recurring)==1 else 'ies'} "
                    f"recur in more than one pair ({recurring}); {len(one_off)} activate in only one pair ({list(one_off.keys())})."
                )
            else:
                out.append("- No important priority was ever active on either side of any pair for this respondent.")
            out.append("")
        out.append(f"**{variant} relationship totals across all (respondent, pair) cells:** {dict(rel_counts)}\n")

    # --- Q3: never activated / never cited ---
    out.append("## 3. Important priorities never activated, and never cited, across all of a person's comparisons\n")
    for variant, analyses in all_analyses.items():
        out.append(f"### {variant}\n")
        for a in analyses:
            out.append(f"**{a['response_id']}** ({a['area']})")
            out.append(f"- Never activated on either side of any pair: {sorted(a['never_activated_important']) or 'none — all were activated somewhere'}")
            out.append(f"- Never cited as mattering in any comparison: {sorted(a['never_cited_important']) or 'none — all were cited somewhere'}")
            both = a["never_activated_important"] & a["never_cited_important"]
            if both:
                out.append(f"- **Both never-activated AND never-cited** (never showed up anywhere in the comparisons at all): {sorted(both)}")
            out.append("")

    # --- Q4: are less-useful pairs less useful for the same or different people ---
    out.append("## 4. Are the less-useful pairs less useful for the same or different respondents?\n")
    out.append(
        "\"Less useful\" = a pair whose important-priority relationship is `empty` or `identical` "
        "for that respondent (neither side is distinguished on anything the respondent said matters most).\n"
    )
    for variant, analyses in all_analyses.items():
        out.append(f"### {variant}\n")
        by_pair = defaultdict(list)
        for a in analyses:
            for pp in a["per_pair"]:
                by_pair[pp["code"]].append((a["response_id"], pp["relationship_important"]))
        out.append("| Pair | Uninformative for | Informative for |")
        out.append("|---|---|---|")
        for code, entries in sorted(by_pair.items()):
            uninformative = [rid for rid, rel in entries if rel in ("empty", "identical")]
            informative = [rid for rid, rel in entries if rel in ("disjoint", "overlapping")]
            out.append(f"| {code} | {uninformative or '—'} | {informative or '—'} |")
        out.append("")

    # --- Q5: internal inconsistency / uncertainty by pair ---
    out.append("## 5. Which pairs generate a lot of internal inconsistency/uncertainty, and why\n")
    for variant, analyses_with_runs in [("Survey1_V1", v1_runs), ("Survey1_V1_SimProjects", sim_runs)]:
        out.append(f"### {variant}\n")
        by_pair_uncertainty = defaultdict(list)
        by_pair_inconsistent = defaultdict(list)
        for run in analyses_with_runs:
            r = run["result"]
            for code, pred, truth in zip(run["pair_codes"], r["per_pair"], run["validation_answers"]):
                by_pair_uncertainty[code].append(pred["model_uncertainty"])
                if not pred["is_consistent"] or not pred["lex_consistent"]:
                    by_pair_inconsistent[code].append(run["response_id"])
        out.append("| Pair | Mean model uncertainty | Respondents with an inconsistent prediction |")
        out.append("|---|---|---|")
        for code in sorted(by_pair_uncertainty):
            vals = by_pair_uncertainty[code]
            mean_u = sum(vals) / len(vals)
            flagged = by_pair_inconsistent.get(code, [])
            out.append(f"| {code} | {mean_u:.2f} | {flagged or '—'} |")
        out.append("")
        # Dig into WHY for the worst pair per variant
        if by_pair_inconsistent:
            worst_code = max(by_pair_inconsistent, key=lambda c: len(by_pair_inconsistent[c]))
            if len(by_pair_inconsistent[worst_code]) >= 2:
                out.append(f"**Digging into `{worst_code}`** (inconsistent for {len(by_pair_inconsistent[worst_code])} respondents):\n")
                for run in analyses_with_runs:
                    if run["response_id"] not in by_pair_inconsistent[worst_code]:
                        continue
                    idx = run["pair_codes"].index(worst_code)
                    bundle_a, bundle_b = run["bundle_pairs"][idx]
                    r = run["result"]
                    ev = compute_lexicographic_evidence(bundle_a, bundle_b, r["utility_functions"], r["importance"])
                    ev_summary = [(e["priority"], e["tier"], "A" if e["direction"] > 0 else "B") for e in ev]
                    out.append(f"- `{run['response_id']}`: evidence = {ev_summary}")
                out.append("")

    return "\n".join(out)


if __name__ == "__main__":
    v1, sim = load_all()

    print(f"Running {len(v1)} real V1 respondents...")
    v1_runs = []
    for r in v1:
        print(f"  {r['response_id']} ({r['area_selfreport']})...")
        v1_runs.append(run_one(r))

    print(f"Running {len(sim)} real SimProjects respondents...")
    sim_runs = []
    for r in sim:
        print(f"  {r['response_id']} ({r['area_selfreport']})...")
        sim_runs.append(run_one(r))

    v1_analyses = [analyze_respondent(run) for run in v1_runs]
    sim_analyses = [analyze_respondent(run) for run in sim_runs]

    import os
    from datetime import datetime
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"wave0_activation_analysis_{ts}.md")
    with open(path, "w") as f:
        f.write(render_report(v1_analyses, sim_analyses, v1_runs, sim_runs))
    print("wrote", path)
