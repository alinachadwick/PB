"""
Classify every Wave 1 priority into a TYPE using only its cached minimal representation
(z, E, s) -- no LLM calls. Taxonomy (paper, sec:priority-types):

  single_project   |S_z| = 1 (z counts one project)
  project_type     z = |B ∩ S_z|, |S_z| >= 2, and E is exactly the set of bundles at the
                   min or max achievable value of z
  type_count       z = |B ∩ S_z|, |S_z| >= 2, E anything else
  distributive     z = number of distinct labels (a function of counts across >= 2
                   values of a feature)
  unclassifiable   E empty, z constant, E = everything, or minimal rep failed to parse

"Achievable" = values of z the pool can actually produce over all size-BUNDLE_SIZE bundles;
E = [lo, hi] is clipped to that range (same as experiment_minimal_representation.py).
Role (peak/valley) and floor fields do not affect the type.

Reads  results/minimal_rep_cache__v1.json, latest results/priority_buckets_*.json (old
       bucketing, for comparison + outer buckets).
Writes results/priority_types_minimal_rep_<ts>.md and .json

Run: python3 classify_priorities_minimal_rep.py
"""
import glob
import itertools
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

from validation_v1_testing import (
    PROJECTS_URBAN, PROJECTS_RURAL, PROJECTS_SIM_URBAN, PROJECTS_SIM_RURAL, BUNDLE_SIZE,
)

RESULTS_DIR = "results"
REP_CACHE = os.path.join(RESULTS_DIR, "minimal_rep_cache__v1.json")
POOLS = {
    "V1::rural": ("Wave 1 V1 Rural", PROJECTS_RURAL),
    "V1::urban": ("Wave 1 V1 Urban", PROJECTS_URBAN),
    "SimProjects::rural": ("Wave 1 SimProjects Rural", PROJECTS_SIM_RURAL),
    "SimProjects::urban": ("Wave 1 SimProjects Urban", PROJECTS_SIM_URBAN),
}
TYPES = ["single_project", "project_type", "type_count", "distributive", "unclassifiable"]


def z_of(rep, bundle):
    z = rep["z"]
    if z["kind"] == "count":
        s = set(z["projects"])
        return sum(1 for p in bundle if p in s)
    return len({z["labels"].get(p, p) for p in bundle})


def classify(rep, projects):
    """Return (type, reason, detail dict)."""
    if rep.get("failed"):
        return "unclassifiable", "minimal representation failed to parse", {}
    z = rep["z"]
    zs = [z_of(rep, b) for b in itertools.combinations(list(projects), BUNDLE_SIZE)]
    zmin, zmax = min(zs), max(zs)
    lo, hi = sorted((rep["lo"], rep["hi"]))
    clo, chi = max(lo, zmin), min(hi, zmax)
    detail = {"zmin": zmin, "zmax": zmax, "E": [lo, hi], "E_clipped": [clo, chi]}

    if z["kind"] == "count":
        S = [p for p in z["projects"] if p in projects]
        detail["S_z"] = S
        if not S:
            return "unclassifiable", "z is constant (S_z is empty)", detail
        if zmin == zmax:
            return "unclassifiable", "z is constant over the pool", detail
        if clo > chi:
            return "unclassifiable", "E is empty (no achievable z in E)", detail
        if clo == zmin and chi == zmax:
            return "unclassifiable", "E covers every achievable z", detail
        if len(S) == 1:
            return "single_project", "|S_z| = 1", detail
        if clo == chi and clo in (zmin, zmax):
            side = "max" if clo == zmax else "min"
            return "project_type", f"E = {{{side} z}} = {{{clo}}}", detail
        return "type_count", f"E = [{clo},{chi}] is not exactly the min or max of z", detail

    labels = {p: z["labels"].get(p, p) for p in projects}
    detail["n_labels"] = len(set(labels.values()))
    detail["labels"] = labels
    if zmin == zmax:
        return "unclassifiable", "z is constant (labels collapse to one value)", detail
    if clo > chi:
        return "unclassifiable", "E is empty (no achievable z in E)", detail
    if clo == zmin and chi == zmax:
        return "unclassifiable", "E covers every achievable z", detail
    return "distributive", f"z = #distinct labels ({detail['n_labels']} labels)", detail


def load_old():
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "priority_buckets_*.json")))
    if not paths:
        return {}, None
    rows = json.load(open(paths[-1]))
    return {(r["pool_id"], r["priority"]): r for r in rows}, paths[-1]


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    reps = json.load(open(REP_CACHE))
    old, old_path = load_old()
    rows = []
    for key, rep in reps.items():
        pool_id = "::".join(key.split("::")[1:3])
        priority = rep["priority"]
        label, projects = POOLS[pool_id]
        typ, reason, detail = classify(rep, projects)
        o = old.get((pool_id, priority), {})
        rows.append({
            "priority": priority, "pool_id": pool_id, "pool": label, "type": typ, "reason": reason,
            "z_kind": rep["z"]["kind"], "role": rep["role"], "detail": detail,
            "old_sub_bucket": o.get("sub_bucket"), "outer_bucket": o.get("outer_bucket"),
        })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json.dump(rows, open(os.path.join(RESULTS_DIR, f"priority_types_minimal_rep_{ts}.json"), "w"), indent=1)

    n = len(rows)
    tc = Counter(r["type"] for r in rows)
    L = [f"# Priority types from minimal representations", "",
         f"{n} (pool, priority) entries classified from `{REP_CACHE}` (no LLM calls). "
         f"Old bucketing for comparison: `{old_path}`.", "",
         "## Overall", "",
         md_table(["Type", "n", "%"], [(t, tc[t], f"{100*tc[t]/n:.0f}%") for t in TYPES]), "",
         "## By pool", ""]
    pools = list(POOLS)
    L.append(md_table(["Type"] + [POOLS[p][0] for p in pools],
                      [[t] + [sum(1 for r in rows if r["type"] == t and r["pool_id"] == p) for p in pools]
                       for t in TYPES]))
    L += ["", "Pool totals: " + ", ".join(f"{POOLS[p][0]} = {sum(1 for r in rows if r['pool_id']==p)}" for p in pools), ""]

    outers = sorted({r["outer_bucket"] or "(none)" for r in rows})
    L += ["## By outer bucket (hand-made, from wave1buckets.pdf)", "",
          md_table(["Outer bucket"] + TYPES,
                   [[ob] + [sum(1 for r in rows if (r["outer_bucket"] or "(none)") == ob and r["type"] == t)
                            for t in TYPES] for ob in outers]), ""]

    L += ["## Peak vs valley by type", "",
          md_table(["Type", "peak", "valley"],
                   [[t, sum(1 for r in rows if r["type"] == t and r["role"] == "peak"),
                     sum(1 for r in rows if r["type"] == t and r["role"] == "valley")] for t in TYPES]), ""]

    old_types = sorted({r["old_sub_bucket"] or "(none)" for r in rows})
    L += ["## Old (behavior-based) sub-bucket vs new (minimal-rep) type", "",
          "Rows = old sub_bucket, columns = new type.", "",
          md_table(["Old \\ New"] + TYPES + ["total"],
                   [[ot] + [sum(1 for r in rows if (r["old_sub_bucket"] or "(none)") == ot and r["type"] == t)
                            for t in TYPES] + [sum(1 for r in rows if (r["old_sub_bucket"] or "(none)") == ot)]
                    for ot in old_types]), ""]

    # Unclassifiable reasons
    uc = [r for r in rows if r["type"] == "unclassifiable"]
    L += ["## Unclassifiable: reasons", "",
          md_table(["Reason", "n"], Counter(r["reason"] for r in uc).most_common()), ""]
    L += ["### All unclassifiable priorities", ""]
    for r in uc:
        L.append(f"- [{r['pool']}] _{r['reason']}_ (old: {r['old_sub_bucket']}): {r['priority']}")
    L.append("")

    # Disagreements worth looking at
    agree = {("single_project", "single_project"), ("project_type", "project_type"),
             ("type_count", "type_count"), ("distributive", "distributive")}
    dis = [r for r in rows if (r["old_sub_bucket"], r["type"]) not in agree]
    L += [f"## Examples per type (up to 8 each)", ""]
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    for t in TYPES[:-1]:
        L += [f"### {t} ({len(by[t])})", ""]
        for r in by[t][:8]:
            d = r["detail"]
            extra = f"S_z={d.get('S_z')}" if "S_z" in d else f"{d.get('n_labels')} labels"
            L.append(f"- [{r['pool']}] {r['role']}, E={d['E']} of z∈[{d['zmin']},{d['zmax']}], {extra}: {r['priority']}")
        L.append("")

    out = os.path.join(RESULTS_DIR, f"priority_types_minimal_rep_{ts}.md")
    open(out, "w").write("\n".join(L))
    print("\n".join(L[:60]))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
