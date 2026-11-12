"""
Experiment: LLM -> minimal representation -> DETERMINISTIC utility function, vs. the fully
LLM-derived utility functions the pipeline uses today.

Pipeline under test (see "Minimal representations of priority-level utility" in the paper):
    L_j --LLM--> R_j = (z_j, E_j, s_j) [+ optional floor info] --deterministic--> u_j

The LLM is asked ONCE per priority for a small JSON record (cached in
results/minimal_rep_cache.json, so variants 2-5 differ only in which fields the deterministic
code reads, never in LLM sampling noise):
  z       what matters about a bundle. Closed vocabulary, both reduce to a scalar:
            count    -> number of projects in the bundle from a named project set
            distinct -> number of distinct labels in the bundle (LLM labels each project)
  E       plateau [lo, hi], an inclusive range of z values (0..BUNDLE_SIZE)
  s       "peak" (E is best) or "valley" (E is worst)
  floor   has_floor flag, floor (one number), floor_below / floor_above (one per side)

Deterministic map (no LLM): u = 1 on the plateau E; on each side, linear drawdown from 1 to
that side's floor value at the farthest z the pool can produce (floor = 0 => down to zero).
A valley is the mirror image: u = floor at E, rising linearly to 1 at the farthest z.
Every u then goes through validation_v1_testing._normalize_utility, same as the LLM functions.

Variants
  1_llm_full        today's fully LLM-written utility functions (cached; no new LLM calls)
  2_floor0          minimal rep, drawdown to 0 on both sides
  3_floor_flag      + yes/no "meaningful floor" flag -> fixed floor value FLAG_FLOOR if yes
  4_floor_cardinal  + one cardinal floor value (symmetric)
  5_floor_asym      + separate cardinal floor below / above the plateau

Everything downstream (omega ranges, pairwise model, cardinal bundle-rating model) is the
existing pipeline, untouched. Patches validation_v1_testing.build_utility_functions_v1 only.

Run locally:
    python3 experiment_minimal_representation.py            # all variants
    python3 experiment_minimal_representation.py --extract-only
"""
import argparse
import itertools
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime

import validation_v1_testing as v
from format_wave1_results import load_all
from priority_dedup import PriorityRegistry
from run_wave0_real_validation import (
    RESULTS_DIR, run_one, load_cache, PRIORITY_REGISTRY_PATH, PRIORITY_SIMILARITY_THRESHOLD,
)
from run_wave1_real_validation import CACHE_PATH
import run_wave1_cardinal_bundle_validation as card
from plot_cardinal_margins import build_respondent_data, plot_all

# Each extraction-prompt version gets its own cache file (minimal_rep_cache__<name>.json), so
# trying a new prompt never overwrites an old one's outputs. The default prompt is "v1"; add
# more with --prompt-file (a text file using the same {priority}/{sign}/{projects_text}/
# {bundle_size} placeholders) and select via --prompt-name. The JSON schema _parse_rep expects
# is the contract any prompt must satisfy.
PROMPT_NAME = "v1"
PROMPT_TEXT = None  # None -> built-in default in rep_prompt()


def rep_cache_path():
    return os.path.join(RESULTS_DIR, f"minimal_rep_cache__{PROMPT_NAME}.json")


FLAG_FLOORS = (0.25, 0.5)  # variant 3 is run at each of these; no LLM involved
VARIANTS = ["1_llm_full", "2_floor0", "3_floor_flag", "4_floor_cardinal", "5_floor_asym"]
STATS = Counter()


# ---------------------------------------------------------------------------
# Step 1 (LLM): natural language -> minimal representation
# ---------------------------------------------------------------------------

def rep_prompt(priority, sign, projects):
    projects_text = "\n".join(f'  - "{n}": {d}' for n, d in projects.items())
    if PROMPT_TEXT is not None:
        return PROMPT_TEXT.format(priority=priority, sign=sign.upper(),
                                  projects_text=projects_text, bundle_size=v.BUNDLE_SIZE)
    return f"""A resident of a town is choosing a bundle of {v.BUNDLE_SIZE} projects (out of the list below) in a participatory budgeting survey. They described a priority in their own words. Do NOT write a utility function. Extract only a MINIMAL REPRESENTATION of the priority: what aspect of the bundle matters, and which values of it are best (or worst).

Available projects:
{projects_text}

Priority ({sign.upper()} criterion -- positive = something they want, negative = something they want to avoid): "{priority}"

Return ONLY a JSON object with these fields:

"z": what aspect of a bundle matters. Choose exactly one kind:
   {{"kind": "count", "projects": [<project names>]}}
       z(bundle) = how many projects in the bundle are from this set. Use this for liking/disliking specific projects or a type of project ("I want at least three X", "no more than one Y", "I like Z").
   {{"kind": "distinct", "labels": {{<every project name>: <short label>}}}}
       z(bundle) = number of distinct labels among the bundle's projects. Use this ONLY for diversity / balance / spread-across-groups priorities ("a mix of things", "not all aimed at one group"), labelling every project along the dimension the person cares about.
   If the priority has no real connection to any project, use {{"kind": "count", "projects": []}}.
"lo", "hi": integers in 0..{v.BUNDLE_SIZE}; the inclusive range of z values that form the extremal region E.
"role": "peak" if E is the best region of z for this person, "valley" if E is the worst region. (A "wouldn't like it if X" priority is usually a valley whose E is the bad values of z; "I want at least 3" is a peak with lo=3, hi={v.BUNDLE_SIZE}.)
"has_floor": true if moving AWAY from E does NOT drive satisfaction all the way to the worst case -- i.e. partial progress still counts for something meaningful (e.g. "I'd like at least 3" -- having 2 is still decent). false if satisfaction really collapses to nothing away from E (e.g. "I want exactly 2", "must include X").
"floor": a number in [0, 1] -- how much satisfaction remains at the farthest z from E, as a fraction of satisfaction at E (0 = none, 1 = no loss at all). Use 0 if has_floor is false.
"floor_below": same quantity, but only for z values BELOW E (fewer than lo). Use 0 if not applicable.
"floor_above": same quantity, but only for z values ABOVE E (more than hi). Use 0 if not applicable.

Return ONLY the JSON object, no explanation."""


def _parse_rep(text, projects):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON")
    rep = json.loads(m.group(0))
    z = rep["z"]
    if z["kind"] == "count":
        z = {"kind": "count", "projects": [p for p in z.get("projects", []) if p in projects]}
    elif z["kind"] == "distinct":
        z = {"kind": "distinct", "labels": {p: str(z["labels"].get(p, p)) for p in projects}}
    else:
        raise ValueError("bad z kind")
    clamp = lambda x, a, b: max(a, min(b, x))
    return {
        "z": z,
        "lo": int(clamp(rep["lo"], 0, v.BUNDLE_SIZE)),
        "hi": int(clamp(rep["hi"], 0, v.BUNDLE_SIZE)),
        "role": "valley" if rep.get("role") == "valley" else "peak",
        "has_floor": bool(rep.get("has_floor", False)),
        "floor": float(clamp(float(rep.get("floor", 0) or 0), 0, 1)),
        "floor_below": float(clamp(float(rep.get("floor_below", 0) or 0), 0, 1)),
        "floor_above": float(clamp(float(rep.get("floor_above", 0) or 0), 0, 1)),
    }


_rep_cache = None


def get_rep(key, priority, sign, projects):
    global _rep_cache
    if _rep_cache is None:
        _rep_cache = json.load(open(rep_cache_path())) if os.path.exists(rep_cache_path()) else {}
    if key in _rep_cache:
        return _rep_cache[key]
    rep = None
    for _ in range(2):
        try:
            rep = _parse_rep(v.call_llm(rep_prompt(priority, sign, projects), max_tokens=1500), projects)
            break
        except Exception as e:
            print(f"  [Warning] minimal-rep parse failed for {priority!r}: {e}")
    if rep is None:  # flat: no discriminating content
        rep = {"z": {"kind": "count", "projects": []}, "lo": 0, "hi": 0, "role": "peak",
               "has_floor": False, "floor": 0.0, "floor_below": 0.0, "floor_above": 0.0, "failed": True}
    rep["priority"] = priority
    _rep_cache[key] = rep
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(_rep_cache, open(rep_cache_path(), "w"), indent=1)
    return rep


# ---------------------------------------------------------------------------
# Step 2 (deterministic): minimal representation -> utility function
# ---------------------------------------------------------------------------

def z_of(rep, bundle):
    z = rep["z"]
    if z["kind"] == "count":
        s = set(z["projects"])
        return sum(1 for p in bundle if p in s)
    return len({z["labels"].get(p, p) for p in bundle})


def utility_from_rep(rep, projects, variant, flag_floor=0.5):
    zs = [z_of(rep, b) for b in itertools.combinations(list(projects), v.BUNDLE_SIZE)]
    zmin, zmax = min(zs), max(zs)
    lo, hi = sorted((rep["lo"], rep["hi"]))
    lo, hi = max(lo, zmin), min(hi, zmax)  # clip E to what the pool can actually produce
    if zmin == zmax or lo > hi:
        return _flat()

    if variant == "2_floor0":
        f_below = f_above = 0.0
    elif variant == "3_floor_flag":
        f_below = f_above = flag_floor if rep["has_floor"] else 0.0
    elif variant == "4_floor_cardinal":
        f_below = f_above = rep["floor"] if rep["has_floor"] else 0.0
    elif variant == "5_floor_asym":
        f_below = rep["floor_below"] if rep["has_floor"] else 0.0
        f_above = rep["floor_above"] if rep["has_floor"] else 0.0
    else:
        raise ValueError(variant)

    def tent(z, f_b, f_a):
        if lo <= z <= hi:
            return 1.0
        if z < lo:
            return f_b + (1 - f_b) * (z - zmin) / (lo - zmin)
        return f_a + (1 - f_a) * (zmax - z) / (zmax - hi)

    if rep["role"] == "peak":
        raw = lambda bundle: tent(z_of(rep, bundle), f_below, f_above)
    else:
        # Valley: u = f at E, 1 at the farthest z. A single depth f is used for both sides.
        f_e = (f_below + f_above) / 2
        raw = lambda bundle: 1 - (1 - f_e) * tent(z_of(rep, bundle), 0.0, 0.0)
    return v._normalize_utility(raw, projects)


def _flat():
    zero = lambda bundle: 0.0
    zero.is_flat = True
    return zero


# ---------------------------------------------------------------------------
# Running the existing validation pipeline with the patched utility builder
# ---------------------------------------------------------------------------

_orig_build = v.build_utility_functions_v1


def make_patched_builder(variant, flag_floor=0.5):
    def patched(free_text_priority, projects, sign="positive", good_projects=None,
                bad_projects=None, depends_projects=None, cache=None, cache_key=None):
        # cache_key is the registry-canonical key, so a priority reused across respondents
        # reuses the same minimal representation -- exactly mirroring the baseline's reuse.
        key = cache_key or f"{'|'.join(projects)}::{sign}::{v.normalize(free_text_priority)}"
        rep = get_rep(key, free_text_priority, sign, projects)
        u = utility_from_rep(rep, projects, variant, flag_floor)
        STATS[f"{variant}:{'flat' if getattr(u, 'is_flat', False) else 'nonflat'}"] += 1
        return u
    return patched


def run_variant(variant, flag_floor, respondents, base_cache):
    v.build_utility_functions_v1 = _orig_build if variant == "1_llm_full" \
        else make_patched_builder(variant, flag_floor)
    registry = PriorityRegistry(path=PRIORITY_REGISTRY_PATH, threshold=PRIORITY_SIMILARITY_THRESHOLD)
    pair_rows, bundle_rows, plot_data = [], [], []
    for r, label in respondents:
        run = run_one(r, cache=base_cache, priority_registry=registry)  # never saved
        res = run["result"]
        if res is None:
            continue
        run["raw_bundle_pairs"] = r["bundle_pairs"]
        plot_data.append((r["response_id"], r["area_selfreport"], label,
                          run["joint_feasibility"], build_respondent_data(run)))
        # Abstain = the model's interval for Delta = U(A) - U(B) contains 0 (it can plausibly
        # predict either option). Accuracy is over the remaining pairs, where the whole
        # interval sits on one side of 0 and so names a side.
        decided = [(pp, t) for pp, t in zip(res["per_pair"], run["validation_answers"])
                   if not (pp["Delta_min"] <= 1e-9 and pp["Delta_max"] >= -1e-9)]
        n_pairs = len(res["per_pair"])
        pair_rows.append({
            "consistency": res["consistency_rate"],
            "n_pairs": n_pairs, "n_decided": len(decided),
            "n_correct": sum((pp["Delta_min"] > 0) == (t["chosen"] == 0) for pp, t in decided),
            "abstain": 1 - len(decided) / n_pairs if n_pairs else None,
            "lex": res["lexicographic_consistency_rate"],
        })
        points, _ = card.collect_bundle_points(run)
        ev = card.evaluate_respondent(points)
        if ev:
            bundle_rows.append(ev)
    return pair_rows, bundle_rows, plot_data


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(pair_rows, bundle_rows):
    return {
        "n_resp": len(pair_rows),
        "pair_consistency": mean(r["consistency"] for r in pair_rows),
        "pair_pooled_decided": sum(r["n_decided"] for r in pair_rows),
        "pair_pooled_pairs": sum(r["n_pairs"] for r in pair_rows),
        "pair_robust_acc": (sum(r["n_correct"] for r in pair_rows)
                            / max(1, sum(r["n_decided"] for r in pair_rows))),
        "pair_abstain": mean(r["abstain"] for r in pair_rows),
        "pair_lex_consistency": mean(r["lex"] for r in pair_rows),
        "bundle_spearman": mean(r["spearman_r"] for r in bundle_rows),
        "bundle_mae": mean(r["mean_mae"] for r in bundle_rows),
        "bundle_exact": mean(r["exact_rate"] for r in bundle_rows),
        "bundle_within1": mean(r["within_one_rate"] for r in bundle_rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-only", action="store_true",
                    help="only build/populate the minimal-representation cache")
    ap.add_argument("--prompt-name", default="v1", help="names the extraction prompt + its cache file")
    ap.add_argument("--prompt-file", help="text file with a custom extraction prompt (see PROMPT_NAME note)")
    ap.add_argument("--plots", action="store_true", help="save a cardinal-margins plot per variant")
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    args = ap.parse_args()
    global PROMPT_NAME, PROMPT_TEXT
    PROMPT_NAME = args.prompt_name
    if args.prompt_file:
        PROMPT_TEXT = open(args.prompt_file).read()

    v1, sim = load_all()
    respondents = [(r, "V1") for r in v1] + [(r, "SimProjects") for r in sim]
    base_cache = load_cache(CACHE_PATH)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries = {}
    for variant in args.variants:
        for ff in (FLAG_FLOORS if variant == "3_floor_flag" else (None,)):
            name = variant if ff is None else f"{variant}@{ff}"
            print(f"\n===== {name} =====", flush=True)
            pair_rows, bundle_rows, plot_data = run_variant(variant, ff or 0.5, respondents, base_cache)
            if args.plots:  # same chart as plot_cardinal_margins_wave1, one per variant
                plot_all(plot_data, os.path.join(
                    RESULTS_DIR, f"minimal_rep_margins_{PROMPT_NAME}_{name.replace('@', '_')}_{run_ts}.png"))
            summaries[name] = summarize(pair_rows, bundle_rows)
            print(json.dumps(summaries[name], indent=1), flush=True)
            if args.extract_only:
                break
        if args.extract_only and variant != "1_llm_full":
            break

    reps = list((json.load(open(rep_cache_path())) if os.path.exists(rep_cache_path()) else {}).values())
    rep_stats = {
        "n_priorities": len(reps),
        "z=count": sum(r["z"]["kind"] == "count" and r["z"]["projects"] != [] for r in reps),
        "z=distinct": sum(r["z"]["kind"] == "distinct" for r in reps),
        "z=empty(flat)": sum(r["z"]["kind"] == "count" and r["z"]["projects"] == [] for r in reps),
        "valley": sum(r["role"] == "valley" for r in reps),
        "has_floor": sum(r["has_floor"] for r in reps),
        "parse_failed": sum(bool(r.get("failed")) for r in reps),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"minimal_rep_experiment_{PROMPT_NAME}_{ts}.md")
    cols = [("pair_consistency", "Pairwise consistency"), ("pair_robust_acc", "Accuracy when not abstaining (pooled)"),
            ("pair_abstain", "Abstain (interval contains 0)")]
    lines = ["# Minimal representation -> deterministic utility vs. fully-LLM utility\n",
             "Wave-1 real respondents (V1 + SimProjects). Means are per-respondent, then averaged.\n",
             "| Variant | n | " + " | ".join(c[1] for c in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['n_resp']} | " + " | ".join(f"{s[k]:.3f}" for k, _ in cols) + " |")
    lines += ["", "## Minimal-representation statistics (over cached priorities)", "```",
              json.dumps(rep_stats, indent=1), "```"]
    open(path, "w").write("\n".join(lines))
    print("\nwrote", path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
