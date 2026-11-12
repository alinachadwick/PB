"""
Rerun the Wave-1 validation with utility functions derived BY PRIORITY TYPE from the cached
minimal representations (no LLM calls, no re-extraction). Types come from
classify_priorities_minimal_rep.classify.

Type -> utility rule (floor = 0 everywhere):
  single_project   u = 1 if (z in E) == (role == peak) else 0, where z = 1[project in B]
  project_type     linear in z = |B ∩ S_z|: 1 at E (an extreme of z), drawn down linearly to 0 at
                   the farthest achievable z (valley mirrored)
  type_count       plateau on E, linear drawdown to 0 on each side (valley mirrored)
  distributive     same tent over z = #distinct labels
  unclassifiable   FLAT (contributes nothing)
The last three share one distance-based rule (experiment_minimal_representation.utility_from_rep,
variant 2_floor0); only z and E differ by type. The script checks the single-project rule against
that generic rule numerically.

Runs
  by_type        all types as above (the main result)
  only:<type>    only priorities of that type are active, everything else flat
  drop:<type>    everything except that type is active
Metrics are those of experiment_minimal_representation (pairwise consistency, accuracy when not
abstaining, abstention), same 40 respondents. Writes results/priority_types_validation_<ts>.md

Run: python3 experiment_priority_types.py
"""
import itertools
import json
import os
from datetime import datetime

import experiment_minimal_representation as emp
import validation_v1_testing as v
from classify_priorities_minimal_rep import classify, TYPES
from format_wave1_results import load_all

RESULTS_DIR = emp.RESULTS_DIR
CLASSIFIABLE = [t for t in TYPES if t != "unclassifiable"]


def utility_by_type(rep, projects):
    """Returns (utility, type)."""
    typ, _, _ = classify(rep, projects)
    if typ == "unclassifiable":
        return emp._flat(), typ
    if typ == "single_project":
        p = [q for q in rep["z"]["projects"] if q in projects][0]
        lo, hi = sorted((rep["lo"], rep["hi"]))
        in_e = lambda z: lo <= z <= hi
        peak = rep["role"] == "peak"
        raw = lambda bundle: 1.0 if in_e(1 if p in bundle else 0) == peak else 0.0
        return v._normalize_utility(raw, projects), typ
    return emp.utility_from_rep(rep, projects, "2_floor0"), typ


def make_builder(active):
    """active: set of types whose priorities keep their utility; all others are flat."""
    def patched(free_text_priority, projects, sign="positive", good_projects=None,
                bad_projects=None, depends_projects=None, cache=None, cache_key=None):
        key = cache_key or f"{'|'.join(projects)}::{sign}::{v.normalize(free_text_priority)}"
        rep = emp.get_rep(key, free_text_priority, sign, projects)
        u, typ = utility_by_type(rep, projects)
        if typ not in active:
            u = emp._flat()
        return u
    return patched


def check_single_project_rule():
    """Single-project rule must agree with the generic floor-0 tent."""
    reps = json.load(open(emp.rep_cache_path()))
    n = bad = 0
    pools = {"V1::rural": v.PROJECTS_RURAL, "V1::urban": v.PROJECTS_URBAN,
             "SimProjects::rural": v.PROJECTS_SIM_RURAL, "SimProjects::urban": v.PROJECTS_SIM_URBAN}
    for key, rep in reps.items():
        projects = pools["::".join(key.split("::")[1:3])]
        u, typ = utility_by_type(rep, projects)
        if typ != "single_project":
            continue
        g = emp.utility_from_rep(rep, projects, "2_floor0")
        n += 1
        if any(abs(u(b) - g(b)) > 1e-9 for b in itertools.combinations(list(projects), v.BUNDLE_SIZE)):
            bad += 1
    return n, bad


def _safe_joint_feasibility(*a, **k):
    # Diagnostic only (plots); with few active priorities some respondents have none, and the
    # LP crashes on an empty weight vector. Skip it there; it never affects the metrics.
    try:
        return _orig_jf(*a, **k)
    except ValueError:
        return None


_orig_jf = emp.run_one.__globals__["joint_feasibility_tiers"]
emp.run_one.__globals__["joint_feasibility_tiers"] = _safe_joint_feasibility


def main():
    n_sp, n_bad = check_single_project_rule()
    print(f"single-project rule vs generic tent: {n_sp} priorities, {n_bad} disagree")

    v1, sim = load_all()
    respondents = [(r, "V1") for r in v1] + [(r, "SimProjects") for r in sim]
    base_cache = emp.load_cache(emp.CACHE_PATH)

    runs = [("by_type (all types)", set(CLASSIFIABLE))]
    runs += [(f"only:{t}", {t}) for t in CLASSIFIABLE]
    runs += [(f"drop:{t}", set(CLASSIFIABLE) - {t}) for t in CLASSIFIABLE]

    summaries = {}
    orig = emp.make_patched_builder
    try:
        for name, active in runs:
            print(f"\n===== {name} =====", flush=True)
            emp.make_patched_builder = lambda variant, flag_floor=0.5, a=active: make_builder(a)
            pair_rows, bundle_rows, _ = emp.run_variant("by_type", 0.5, respondents, base_cache)
            summaries[name] = emp.summarize(pair_rows, bundle_rows)
            print(json.dumps(summaries[name], indent=1), flush=True)
    finally:
        emp.make_patched_builder = orig

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cols = [("pair_consistency", "Pairwise consistency"),
            ("pair_robust_acc", "Accuracy when not abstaining (pooled)"),
            ("pair_abstain", "Abstain"), ("bundle_spearman", "Bundle Spearman")]
    L = ["# Validation with type-based utility functions (floor 0, unclassifiable = flat)", "",
         "Wave-1 real respondents (V1 + SimProjects). Per-respondent means, except pooled accuracy.", "",
         f"Check: single-project rule agrees with the generic floor-0 tent on "
         f"{n_sp - n_bad}/{n_sp} single-project priorities.", "",
         "| Run | n | " + " | ".join(c[1] for c in cols) + " | decided / total pairs |",
         "|---|---|" + "---|" * (len(cols) + 1)]
    for name, s in summaries.items():
        L.append(f"| {name} | {s['n_resp']} | " + " | ".join(f"{s[k]:.3f}" for k, _ in cols)
                 + f" | {s['pair_pooled_decided']}/{s['pair_pooled_pairs']} |")
    L += ["", "`only:T` = only type-T priorities active (rest flat); `drop:T` = all but type T active."]
    path = os.path.join(RESULTS_DIR, f"priority_types_validation_{ts}.md")
    open(path, "w").write("\n".join(L))
    print("\n".join(L), "\nwrote", path)


if __name__ == "__main__":
    main()
