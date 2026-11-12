"""
For every pairwise comparison, which of the respondent's priorities SHOULD have been
activated according to the model, and does that match which ones they checked off?

For respondent i, pair (A, B), priority j (all of the respondent's elicited priorities):
    omega_j      = Likert-tier midpoint weight, renormalized to sum to 1 over the respondent's
                   non-flat priorities (likert_to_omega_ranges with utility functions)
    gap_j        = u_j(A) - u_j(B)
    activation_j = omega_j * |gap_j|          (0 => the priority doesn't distinguish the bundles)
    share_j      = activation_j / sum_k activation_k
Priorities are ranked by activation within each comparison and annotated with: separable /
inseparable / flat (additive R^2 >= 0.95 on the pool's size-4 bundles => separable), Likert
tier, which bundle it favors, whether the respondent CITED it for that pair, and whether it
points toward the bundle they chose.

Aggregate diagnostics (overall, by variant, and by separable vs inseparable):
  - cited rate by activation-share bin and by activation rank
  - AUC: P(a cited priority has higher activation than an uncited one) per comparison
  - precision@k (k = # cited): share of the k most-activated priorities that were cited
  - cited-but-zero-activation (the model says it couldn't have mattered) and
    highly-activated-but-not-cited (the model says it should have mattered)

Uses cached utility functions (no LLM calls). Writes results/activation_vs_citations_<ts>.md
(summary + per-comparison tables) and .json (everything).

Run locally:
    python3 analyze_activation_vs_citations.py
"""
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

import validation_v1_testing as v
from analyze_cited_priorities import kind_of, frac
from format_wave1_results import load_all
from priority_dedup import PriorityRegistry
from run_wave0_real_validation import (
    RESULTS_DIR, run_one, load_cache, PRIORITY_REGISTRY_PATH, PRIORITY_SIMILARITY_THRESHOLD, POOLS,
    resolve_cited_key,
)
from run_wave1_real_validation import CACHE_PATH

TIER_NAME = {4: "most", 3: "trade-away", 2: "nice", 1: "not imp."}
SHARE_BINS = [(0.0, 0.0, "0 (no gap)"), (1e-12, 0.05, "(0, 5%]"), (0.05, 0.15, "(5, 15%]"),
              (0.15, 0.30, "(15, 30%]"), (0.30, 1.01, ">30%")]


def comparison_rows(run, label):
    result = run["result"]
    if not result:
        return []
    area = run["area"].lower()
    projects = POOLS[label]["urban" if area == "urban" else "rural"][0]
    ufs, importance = result["utility_functions"], result["importance"]
    omega = {p: (lo + hi) / 2 for p, (lo, hi) in v.likert_to_omega_ranges(importance, ufs).items()}
    kinds = {p: kind_of(u, projects) for p, u in ufs.items()}
    out = []
    for code, (bA, bB), truth in zip(run["pair_codes"], run["bundle_pairs"], run["validation_answers"]):
        cited = {resolve_cited_key(c) for c in truth["cited_priorities"]}
        chosen = "A" if truth["chosen"] == 0 else "B"
        rows = []
        for p, u in ufs.items():
            uA, uB = u(bA), u(bB)
            w = omega.get(p, 0.0)
            act = w * abs(uA - uB)
            rows.append({"priority": p, "kind": kinds[p], "tier": v.LIKERT_ORDER.get(importance.get(p), 0),
                         "omega": w, "u_A": uA, "u_B": uB, "gap": uA - uB, "activation": act,
                         "favors": "A" if uA > uB else ("B" if uB > uA else "-"),
                         "cited": p in cited})
        total = sum(r["activation"] for r in rows)
        for r in rows:
            r["share"] = r["activation"] / total if total > 0 else 0.0
            r["agrees_with_choice"] = (r["favors"] == chosen) if r["favors"] != "-" else None
        rows.sort(key=lambda r: -r["activation"])
        for i, r in enumerate(rows, 1):
            r["rank"] = i
        out.append({"respondent": run["response_id"], "variant": label, "pair": code, "chosen": chosen,
                    "n_cited": len(cited), "unmatched_citations": sorted(cited - set(ufs)),
                    "total_activation": total, "rows": rows})
    return out


def auc(rows):
    pos = [r["activation"] for r in rows if r["cited"]]
    neg = [r["activation"] for r in rows if not r["cited"]]
    if not pos or not neg:
        return None
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def precision_at_k(rows):
    k = sum(r["cited"] for r in rows)
    if k == 0:
        return None
    top = [r for r in rows[:k] if r["activation"] > 0]
    return sum(r["cited"] for r in top) / k


def summarize(comps, title):
    md = [f"## {title}\n"]
    if not comps:
        return md + ["(no data)\n"]
    allrows = [dict(r, comp=i) for i, c in enumerate(comps) for r in c["rows"]]
    md.append(f"{len(comps)} comparisons, {len({c['respondent'] for c in comps})} respondents, "
              f"{len(allrows)} (comparison, priority) cells, {sum(r['cited'] for r in allrows)} cited.\n")

    # --- cited rate by activation-share bin
    md.append("### Cited rate by how much the model says the priority should be activated\n")
    md.append("Activation share = this priority's share of the comparison's total weighted utility gap.\n")
    md.append("| Activation share | all: cited | separable: cited | inseparable: cited |")
    md.append("|---|---|---|---|")
    for lo, hi, name in SHARE_BINS:
        def in_bin(r):
            return (r["share"] == 0.0) if hi == 0.0 else (lo <= r["share"] <= hi and r["share"] > 0)
        cells = []
        for kinds in (("separable", "inseparable"), ("separable",), ("inseparable",)):
            rs = [r for r in allrows if r["kind"] in kinds and in_bin(r)]
            cells.append(frac(sum(r["cited"] for r in rs), len(rs)))
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    md.append("")

    # --- by rank
    md.append("### Cited rate by the model's activation rank within the comparison\n")
    md.append("| Rank | cited | of which separable | of which inseparable |")
    md.append("|---|---|---|---|")
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 5), (6, 99)):
        rs = [r for r in allrows if lo <= r["rank"] <= hi]
        name = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 99 else f"{lo}+")
        sep = [r for r in rs if r["kind"] == "separable"]
        ins = [r for r in rs if r["kind"] == "inseparable"]
        md.append(f"| {name} | {frac(sum(r['cited'] for r in rs), len(rs))} | "
                  f"{frac(sum(r['cited'] for r in sep), len(sep))} | {frac(sum(r['cited'] for r in ins), len(ins))} |")
    md.append("")

    # --- per-comparison agreement metrics
    aucs = [a for a in (auc(c["rows"]) for c in comps) if a is not None]
    pks = [p for p in (precision_at_k(c["rows"]) for c in comps) if p is not None]
    md.append("### Per-comparison agreement between model activation and citations\n")
    md.append(f"- Mean AUC (chance = 0.5): **{np.mean(aucs):.2f}** over {len(aucs)} comparisons with >= 1 cited and >= 1 uncited priority")
    md.append(f"- Mean precision@k (k = # cited; share of the model's top-k that were cited): **{np.mean(pks):.2f}**")
    cited_rows = [r for r in allrows if r["cited"]]
    zero = [r for r in cited_rows if r["activation"] == 0]
    md.append(f"- Cited but zero activation (model says it can't distinguish the bundles): {frac(len(zero), len(cited_rows))}; "
              f"separable {frac(sum(r['kind'] == 'separable' for r in zero), sum(r['kind'] == 'separable' for r in cited_rows))}, "
              f"inseparable {frac(sum(r['kind'] == 'inseparable' for r in zero), sum(r['kind'] == 'inseparable' for r in cited_rows))}")
    share_cited = [sum(r["share"] for r in c["rows"] if r["cited"]) for c in comps if c["total_activation"] > 0]
    md.append(f"- Share of a comparison's total activation sitting on cited priorities: mean {np.mean(share_cited):.0%}")
    top1 = [c["rows"][0] for c in comps if c["rows"] and c["rows"][0]["activation"] > 0]
    md.append(f"- The model's single most-activated priority was cited: {frac(sum(r['cited'] for r in top1), len(top1))}; "
              f"when it was cited, pointed toward the choice: "
              f"{frac(sum(bool(r['agrees_with_choice']) for r in top1 if r['cited']), sum(r['cited'] for r in top1))}")
    big_miss = [r for r in allrows if not r["cited"] and r["share"] >= 0.30]
    md.append(f"- Highly activated (>= 30% share) but NOT cited: {len(big_miss)} cells "
              f"({sum(r['kind'] == 'inseparable' for r in big_miss)} inseparable, "
              f"{sum(r['kind'] == 'separable' for r in big_miss)} separable)")
    md.append("")
    return md


def per_comparison_md(comps):
    md = ["# Per-comparison detail\n",
          "Priorities ordered by the model's expected activation (omega x |utility gap|). "
          "✔ = cited for this comparison; 'agrees' = the priority favors the bundle the respondent chose.\n"]
    by_resp = defaultdict(list)
    for c in comps:
        by_resp[(c["variant"], c["respondent"])].append(c)
    for (variant, rid), cs in by_resp.items():
        md.append(f"## {variant} {rid}\n")
        for c in cs:
            md.append(f"### Pair {c['pair']} — chose {c['chosen']} "
                      f"(total activation {c['total_activation']:.3f}, {c['n_cited']} cited"
                      + (f", unmatched citations: {c['unmatched_citations']}" if c["unmatched_citations"] else "") + ")\n")
            md.append("| # | Priority | Kind | Tier | ω | u(A) | u(B) | Activation | Share | Favors | Agrees | Cited |")
            md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in c["rows"]:
                ag = "" if r["agrees_with_choice"] is None else ("yes" if r["agrees_with_choice"] else "NO")
                md.append(f"| {r['rank']} | {r['priority']} | {r['kind']} | {TIER_NAME.get(r['tier'], '?')} | {r['omega']:.2f} | "
                          f"{r['u_A']:.2f} | {r['u_B']:.2f} | {r['activation']:.3f} | {r['share']:.0%} | {r['favors']} | {ag} | "
                          f"{'✔' if r['cited'] else ''} |")
            md.append("")
    return md


def main():
    cache = load_cache(CACHE_PATH)
    registry = PriorityRegistry(path=PRIORITY_REGISTRY_PATH, threshold=PRIORITY_SIMILARITY_THRESHOLD)
    v1, sim = load_all()
    comps = {"V1": [], "SimProjects": []}
    for label, rs in (("V1", v1), ("SimProjects", sim)):
        for r in rs:
            print(f"  {label} {r['response_id']}...")
            run = run_one(r, cache=cache, priority_registry=registry)
            run["response_id"], run["area"] = r["response_id"], r["area_selfreport"]
            comps[label] += comparison_rows(run, label)

    md = ["# Model activation vs. checked-off priorities, per comparison\n",
          "For each pairwise comparison, every priority the respondent has is ranked by "
          "ω_j × |u_j(A) − u_j(B)| (how much the model says it should matter for that comparison) and compared "
          "to the priorities they checked off. Separable = additive utility function (R² >= 0.95). "
          "No LLM calls; see the header of `analyze_activation_vs_citations.py`.\n"]
    allc = comps["V1"] + comps["SimProjects"]
    md += summarize(allc, "All respondents")
    md += summarize(comps["V1"], "V1 only")
    md += summarize(comps["SimProjects"], "SimProjects only")
    md += per_comparison_md(allc)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(RESULTS_DIR, f"activation_vs_citations_{ts}")
    with open(base + ".md", "w") as f:
        f.write("\n".join(md))
    with open(base + ".json", "w") as f:
        json.dump(allc, f, indent=1)
    print(f"wrote {base}.md / .json")


if __name__ == "__main__":
    main()
