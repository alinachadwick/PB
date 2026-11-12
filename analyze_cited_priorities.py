"""
Which priorities do people actually use when answering the pairwise comparisons, and does
that include the inseparable ones?

Motivated by: dropping inseparable priorities from the prediction barely changes accuracy.
Either (1) respondents never cite inseparable priorities on the comparison questions,
(2) they cite them but give them low importance, or (3) they cite them but the choice isn't
actually driven by them (e.g. a lexicographic respondent decides on something else).

"Inseparable" = utility function not additive over projects (least-squares R^2 < 0.95 on
every size-4 bundle of the pool), same definition as experiment_separable_only.py. Flat
(identically 0) priorities are reported separately.

For every (respondent, pair, cited priority) and every (respondent, pair) this reports:
  H1  Are they cited?  Share of citations that are inseparable vs. the share of the
      respondent's priority list that is inseparable; and OPPORTUNITY: how often an
      inseparable priority even distinguishes the two bundles (c_j != 0), and how often it
      is cited when it does.
  H2  Are they low weight?  Likert importance tier of inseparable vs. separable priorities,
      over all elicited priorities and over cited ones.
  H3  Are they used?  When an inseparable priority is cited and distinguishes the pair, does it
      point toward the bundle chosen (vs. separable cited priorities)? Does the choice follow
      the separable-only prediction anyway? Would a lexicographic rule be decided by an
      inseparable priority, and does the choice follow it?

Uses the cached utility functions (no LLM calls). Writes results/cited_priorities_<ts>.md/.json.

Run locally:
    python3 analyze_cited_priorities.py
"""
import itertools
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

import validation_v1_testing as v
from format_wave1_results import load_all
from priority_dedup import PriorityRegistry
from run_wave0_real_validation import (
    RESULTS_DIR, run_one, load_cache, PRIORITY_REGISTRY_PATH, PRIORITY_SIMILARITY_THRESHOLD, POOLS,
    resolve_cited_key,
)
from run_wave1_real_validation import CACHE_PATH

SEPARABLE_R2 = 0.95
TIER_NAME = {4: "most", 3: "trade-away", 2: "nice-to-have", 1: "not important"}


def additive_r2(u, projects):
    names = list(projects)
    bundles = list(itertools.combinations(names, v.BUNDLE_SIZE))
    X = np.array([[1.0 if n in b else 0.0 for n in names] for b in bundles])
    U = np.array([u(list(b)) for b in bundles])
    ss_tot = ((U - U.mean()) ** 2).sum()
    if ss_tot < 1e-12:
        return 1.0
    A = np.hstack([X, np.ones((len(U), 1))])
    w, *_ = np.linalg.lstsq(A, U, rcond=None)
    return 1.0 - ((U - A @ w) ** 2).sum() / ss_tot


def kind_of(u, projects):
    if u is None:
        return "missing"
    if getattr(u, "is_flat", False):
        return "flat"
    return "separable" if additive_r2(u, projects) >= SEPARABLE_R2 else "inseparable"


def frac(n, d):
    return f"{n}/{d} ({n / d:.0%})" if d else "n/a"


def analyze_run(run, label):
    """Returns (priority_rows, citation_rows, pair_rows) for one respondent."""
    result = run["result"]
    if not result:
        return [], [], []
    area = run["area"].lower()
    projects = POOLS[label]["urban" if area == "urban" else "rural"][0]
    ufs = result["utility_functions"]
    importance = result["importance"]
    kinds = {p: kind_of(u, projects) for p, u in ufs.items()}
    tier = {p: v.LIKERT_ORDER.get(importance.get(p), 0) for p in ufs}
    rid = run["response_id"]

    prio_rows = [{"respondent": rid, "variant": label, "priority": p, "kind": kinds[p], "tier": tier[p]}
                 for p in ufs]

    cites, pairs = [], []
    for code, (bA, bB), truth, matches in zip(run["pair_codes"], run["bundle_pairs"],
                                              run["validation_answers"], run["cited_vs_model"]):
        chosen = truth["chosen"]  # 0 = A, 1 = B
        c = {p: u(bA) - u(bB) for p, u in ufs.items() if u is not None}
        # --- citations
        cited_keys = set()
        for m in matches:
            key = resolve_cited_key(m["cited"])
            cited_keys.add(key)
            cites.append({"respondent": rid, "variant": label, "pair": code, "priority": key,
                          "kind": kinds.get(key, "unmatched"), "tier": tier.get(key, 0),
                          "status": m["status"], "chosen": chosen})
        # --- pair level: what differentiates, who is cited, what decides
        diff = {p: cj for p, cj in c.items() if abs(cj) > 1e-9 and kinds[p] in ("separable", "inseparable")}
        mid = {p: 0.0 for p in ufs}
        for p in ufs:  # importance midpoints are enough for the sign of Delta
            lo, hi = v.LIKERT_TO_OMEGA.get(importance.get(p), (0.5, 0.75))
            mid[p] = (lo + hi) / 2

        def sign_choice(keys):
            d = sum(mid[p] * diff[p] for p in keys if p in diff)
            if abs(d) < 1e-9:
                return None
            return 0 if d > 0 else 1

        all_pred = sign_choice(list(diff))
        sep_pred = sign_choice([p for p in diff if kinds[p] == "separable"])
        top = max((tier[p] for p in diff), default=0)
        top_set = [p for p in diff if tier[p] == top]
        votes = sum(1 if diff[p] > 0 else -1 for p in top_set)
        lex_pred = None if votes == 0 else (0 if votes > 0 else 1)
        pairs.append({
            "respondent": rid, "variant": label, "pair": code, "chosen": chosen,
            "n_diff_sep": sum(kinds[p] == "separable" for p in diff),
            "n_diff_insep": sum(kinds[p] == "inseparable" for p in diff),
            "n_cited": len(cited_keys),
            "cited_insep": sum(kinds.get(k) == "inseparable" for k in cited_keys),
            "cited_sep": sum(kinds.get(k) == "separable" for k in cited_keys),
            "insep_differs_and_cited": sum(kinds.get(k) == "inseparable" and k in diff for k in cited_keys),
            "all_pred": all_pred, "sep_pred": sep_pred, "lex_pred": lex_pred,
            "top_tier_has_insep": any(kinds[p] == "inseparable" for p in top_set),
            "top_tier_all_insep": bool(top_set) and all(kinds[p] == "inseparable" for p in top_set),
        })
    return prio_rows, cites, pairs


def summarize(prio, cites, pairs, title):
    out = [f"## {title}\n"]
    if not pairs:
        return out + ["(no data)\n"]
    kinds = defaultdict(int)
    for r in prio:
        kinds[r["kind"]] += 1
    n_p = len(prio)
    out.append(f"{len({r['respondent'] for r in prio})} respondents, {n_p} elicited priorities "
               f"({kinds['separable']} separable, {kinds['inseparable']} inseparable, {kinds['flat']} flat), "
               f"{len(pairs)} pairwise choices, {len(cites)} citations.\n")

    # ---- H1
    out.append("### H1: are inseparable priorities cited?\n")
    cit = [c for c in cites if c["kind"] in ("separable", "inseparable")]
    ci = sum(c["kind"] == "inseparable" for c in cit)
    base = [r for r in prio if r["kind"] in ("separable", "inseparable")]
    bi = sum(r["kind"] == "inseparable" for r in base)
    out.append(f"- Inseparable share of the priority list (non-flat): {frac(bi, len(base))}")
    out.append(f"- Inseparable share of citations (non-flat): {frac(ci, len(cit))}")
    n_pairs = len(pairs)
    out.append(f"- Pairs where >= 1 inseparable priority is cited: {frac(sum(p['cited_insep'] > 0 for p in pairs), n_pairs)}")
    out.append(f"- Respondents who ever cite an inseparable priority: "
               f"{frac(len({p['respondent'] for p in pairs if p['cited_insep'] > 0}), len({p['respondent'] for p in pairs}))}")
    opp = [p for p in pairs if p["n_diff_insep"] > 0]
    out.append(f"- **Opportunity**: pairs where an inseparable priority distinguishes the two bundles at all: "
               f"{frac(len(opp), n_pairs)}")
    out.append(f"- Of those, pairs where an inseparable priority that distinguishes them is cited: "
               f"{frac(sum(p['insep_differs_and_cited'] > 0 for p in opp), len(opp))}")
    opp_s = [p for p in pairs if p["n_diff_sep"] > 0]
    out.append(f"- (For comparison, separable priorities: distinguish the bundles in {frac(len(opp_s), n_pairs)} of pairs; "
               f"cited among those in "
               f"{frac(sum(p['cited_sep'] > 0 for p in opp_s), len(opp_s))})\n")

    # ---- H2
    out.append("### H2: are inseparable priorities lower weight?\n")
    out.append("| Priority set | Kind | n | Mean Likert tier (4=most, 1=not important) | % rated 'most' |")
    out.append("|---|---|---|---|---|")
    for name, rows in (("all elicited", prio), ("cited", cites)):
        for k in ("separable", "inseparable"):
            ts = [r["tier"] for r in rows if r["kind"] == k]
            if ts:
                out.append(f"| {name} | {k} | {len(ts)} | {np.mean(ts):.2f} | {sum(t == 4 for t in ts) / len(ts):.0%} |")
    out.append("")

    # ---- H3
    out.append("### H3: when cited, are they actually used?\n")
    out.append("Status of cited priorities, from the model's per-pair math "
               "(agrees = points toward the bundle chosen; irrelevant = doesn't distinguish the bundles):\n")
    out.append("| Kind | cited | agrees | disagrees | irrelevant | agree rate when distinguishing |")
    out.append("|---|---|---|---|---|---|")
    for k in ("separable", "inseparable"):
        rs = [c for c in cites if c["kind"] == k]
        a = sum(c["status"] == "agrees" for c in rs)
        d = sum(c["status"] == "disagrees" for c in rs)
        irr = sum(c["status"] == "irrelevant" for c in rs)
        out.append(f"| {k} | {len(rs)} | {a} | {d} | {irr} | {frac(a, a + d)} |")
    out.append("")

    ci_pairs = [p for p in pairs if p["insep_differs_and_cited"] > 0]
    def acc(rows, key):
        ok = sum(p[key] == p["chosen"] for p in rows if p[key] is not None)
        n = sum(p[key] is not None for p in rows)
        return frac(ok, n)
    out.append("Does the choice follow the model's prediction? (sign of weighted sum of c_j, importance midpoints as weights; "
               "pairs with no decision excluded)\n")
    out.append("| Pairs | all priorities | separable only | lexicographic (top-tier differentiating priorities) |")
    out.append("|---|---|---|---|")
    out.append(f"| all pairs ({n_pairs}) | {acc(pairs, 'all_pred')} | {acc(pairs, 'sep_pred')} | {acc(pairs, 'lex_pred')} |")
    out.append(f"| pairs citing a distinguishing inseparable priority ({len(ci_pairs)}) | {acc(ci_pairs, 'all_pred')} | "
               f"{acc(ci_pairs, 'sep_pred')} | {acc(ci_pairs, 'lex_pred')} |")
    ti = [p for p in pairs if p["top_tier_has_insep"]]
    ta = [p for p in pairs if p["top_tier_all_insep"]]
    out.append(f"| pairs where top-tier differentiating priorities include an inseparable one ({len(ti)}) | "
               f"{acc(ti, 'all_pred')} | {acc(ti, 'sep_pred')} | {acc(ti, 'lex_pred')} |")
    out.append(f"| ... and they are ALL inseparable ({len(ta)}) | {acc(ta, 'all_pred')} | {acc(ta, 'sep_pred')} | {acc(ta, 'lex_pred')} |")
    both = [p for p in pairs if p["all_pred"] is not None and p["sep_pred"] is not None]
    fix = sum(p["all_pred"] == p["chosen"] and p["sep_pred"] != p["chosen"] for p in both)
    hurt = sum(p["all_pred"] != p["chosen"] and p["sep_pred"] == p["chosen"] for p in both)
    out.append(f"\nAdding the inseparable priorities back to the separable-only prediction: fixes {fix} pairs, "
               f"breaks {hurt} pairs, out of {len(both)} where both models decide.\n")
    return out


def main():
    cache = load_cache(CACHE_PATH)
    registry = PriorityRegistry(path=PRIORITY_REGISTRY_PATH, threshold=PRIORITY_SIMILARITY_THRESHOLD)
    v1, sim = load_all()
    groups = {"V1": [], "SimProjects": []}
    for label, rs in (("V1", v1), ("SimProjects", sim)):
        for r in rs:
            print(f"  {label} {r['response_id']}...")
            run = run_one(r, cache=cache, priority_registry=registry)
            run["response_id"], run["area"] = r["response_id"], r["area_selfreport"]
            groups[label].append((run, label))

    data = {}
    md = ["# Which priorities do people use in the pairwise comparisons?\n",
          "Separable = utility function additive over projects (R² >= 0.95); inseparable = anything else. "
          "Computed from cached utility functions; no LLM calls. See the header of "
          "`analyze_cited_priorities.py` for definitions.\n"]
    for title, labels in (("All respondents", ["V1", "SimProjects"]), ("V1 only", ["V1"]), ("SimProjects only", ["SimProjects"])):
        prio, cites, pairs = [], [], []
        for lab in labels:
            for run, l in groups[lab]:
                p, c, q = analyze_run(run, l)
                prio += p; cites += c; pairs += q
        data[title] = {"priorities": prio, "citations": cites, "pairs": pairs}
        md += summarize(prio, cites, pairs, title)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(RESULTS_DIR, f"cited_priorities_{ts}")
    with open(base + ".md", "w") as f:
        f.write("\n".join(md))
    with open(base + ".json", "w") as f:
        json.dump(data, f, indent=1, default=str)
    print(f"wrote {base}.md / .json")


if __name__ == "__main__":
    main()
