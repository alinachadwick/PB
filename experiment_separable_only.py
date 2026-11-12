"""
Experiment: how much does accuracy drop if predictions use ONLY project-level (separable)
priorities?

A priority is "project-level" if its generated utility function is additive over projects
(u(B) ~ sum of per-project values; least-squares R^2 >= SEPARABLE_R2 on every size-4 bundle
of its pool) -- i.e. single-project and project-type priorities. Everything else (type-count,
distributive, substitutes/complements, unstructured) is treated as inseparable and removed:
its utility function is replaced by a constant 0 flagged is_flat, so it gets no weight in the
pairwise/cardinal models (omega is renormalized over what's left) and contributes no evidence
to the lexicographic model.

Reuses the real pipelines unchanged (run_wave1_real_validation.py and
run_wave1_cardinal_bundle_validation.py, run via runpy), only patching safe_llm_python. Uses
the existing utility-function cache, so no LLM calls. Report files these scripts write are moved
into results/experiment_separable_only/ so they don't mix with the baseline reports.

Run locally:
    python3 experiment_separable_only.py
"""
import itertools
import os
import runpy
import shutil
from collections import Counter

import numpy as np

import validation_v1_testing as v

SEPARABLE_R2 = 0.95
RESULTS_DIR = "results"
OUT_DIR = os.path.join(RESULTS_DIR, "experiment_separable_only")
STATS = Counter()

_orig = v.safe_llm_python


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


def separable_only(func_body, projects):
    u = _orig(func_body, projects)
    if getattr(u, "is_flat", False):
        STATS["flat"] += 1
        return u
    if additive_r2(u, projects) >= SEPARABLE_R2:
        STATS["kept (project-level)"] += 1
        return u
    STATS["removed (inseparable)"] += 1
    zero = lambda bundle: 0.0
    zero.is_flat = True
    return zero


def main():
    before = set(os.listdir(RESULTS_DIR))
    v.safe_llm_python = separable_only
    for script in ("run_wave1_real_validation", "run_wave1_cardinal_bundle_validation"):
        print(f"\n===== {script} (separable priorities only) =====")
        runpy.run_module(script, run_name="__main__")
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in set(os.listdir(RESULTS_DIR)) - before:
        if f.startswith(("wave1_real_validation_", "wave1_cardinal_")):
            shutil.move(os.path.join(RESULTS_DIR, f), os.path.join(OUT_DIR, f))
    print("\nPriority function calls by category (counts calls, so repeats across respondents are included):")
    for k, n in STATS.items():
        print(f"  {k}: {n}")
    print(f"Reports moved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
