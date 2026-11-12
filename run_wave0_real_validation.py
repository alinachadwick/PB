"""
Run real Wave-0 Survey1_V1 respondents through the SAME validation pipeline used for
simulated personas (run_validation / build_utility_functions_v1 / predict_pair /
predict_pair_lexicographic), per team direction: reuse the existing pipeline rather than
building a new one, and replace the survey's now-removed priority<>project classification
question with an LLM-inferred guess (generate_priority_classifications), same as the
simulated-persona pipeline already does.

Ground truth (bundle choice, certainty, cited priorities, importance ratings) comes
directly from the real respondent's own answers -- nothing about the OUTCOME side is
LLM-generated, only the priority<>project classification INPUT that the model needs and
the survey no longer collects.

Scoped to Survey1_V1 respondents only (project pools now match the current QSF). SimProjects
uses a different, larger project taxonomy that isn't wired into validation_v1_testing.py yet.
"""
import io
import json
import os
from datetime import datetime

from format_wave0_results import load_all
from priority_dedup import PriorityRegistry
from validation_v1_testing import (
    PROJECTS_URBAN, PROJECTS_RURAL,
    BUNDLE_PAIRS_URBAN, BUNDLE_PAIRS_RURAL,
    PROJECTS_SIM_URBAN, PROJECTS_SIM_RURAL,
    BUNDLE_PAIRS_SIM_URBAN, BUNDLE_PAIRS_SIM_RURAL,
    CRITERIA_SIGN_BY_QUESTION,
    select_top_priorities, generate_priority_classifications,
    run_validation, normalize,
    LIKERT_MOST, LIKERT_TRADE_AWAY, LIKERT_NICE, LIKERT_NOT,
    priority_contributions, likert_to_omega_ranges, joint_feasibility_tiers,
)

RESULTS_DIR = "results"

# LLM-generated classifications and utility-function source, cached to disk keyed by
# respondent + priority so re-running this script against the same respondents doesn't
# re-hit the LLM (cost) and doesn't silently change the model out from under you between
# runs (LLM generation isn't deterministic -- without this, every re-run gets different
# utility functions for the same person, which makes results impossible to compare
# across runs). Simulated personas don't use this: they're regenerated fresh each run by
# design, and only real respondents have a stable ID worth caching against.
CACHE_PATH = os.path.join(RESULTS_DIR, "llm_model_cache.json")

# Shared across run_wave0_real_validation.py, run_wave1_real_validation.py, and
# run_priority_dedup_validation.py -- one growing index of "priorities we've already
# seen," so reuse benefits accumulate across scripts/runs over time rather than each
# script maintaining its own blind spot. Lightweight bookkeeping only (text + tokens +
# sign), safe to share even though each script still writes its generated
# classifications/utility functions to its own separate CACHE_PATH above.
PRIORITY_REGISTRY_PATH = os.path.join(RESULTS_DIR, "priority_registry.json")
PRIORITY_SIMILARITY_THRESHOLD = 0.5


def load_cache(path=CACHE_PATH):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(cache, path=CACHE_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=1)

CRITERION_SIGN_BY_TAG = {
    "open-good": "positive",
    "Q204": "positive",
    "mix-good": "positive",
    "open-bad": "negative",
}

# Coarser than CRITERION_SIGN_BY_TAG (which collapses mix-good into "positive" for the
# purposes of the utility-function sign) -- this preserves the three-way distinction
# between the survey's separate elicitation questions, for reporting: "open-good"/"Q204"
# ("what would you fund") and "mix-good" ("how should the overall mix look") are both
# positive, but describe genuinely different kinds of priorities (an individual
# thing vs. a property of the combination), and "open-bad" ("what would make a bad
# bundle") is the negative counterpart.
CRITERION_CATEGORY_BY_TAG = {
    "open-good": "Good",
    "Q204": "Good",
    "mix-good": "Mix",
    "open-bad": "Bad",
}

CERTAINTY_TO_INT = {
    "Not at all certain": 1,
    "Somewhat certain": 2,
    "Moderately certain": 3,
    "Very certain": 4,
    "Extremely certain": 5,
}

RATING_TO_INT = {
    "Very Unhappy": -1, "Somewhat Unhappy": -1,
    "Neutral": 0,
    "Somewhat Happy": 1, "Very Happy": 1,
}

# Same 5-level abbreviations used in plot_cardinal_margins.py's RATING_ABBREV --
# duplicated here (rather than imported) since that module imports run_one from this
# one, and a reverse import would be circular.
RATING_ABBREV = {
    "Very Happy": "VH", "Somewhat Happy": "SH", "Neutral": "N",
    "Somewhat Unhappy": "SU", "Very Unhappy": "VU",
}

# Expected pair-code order for each pool, matching BUNDLE_PAIRS_*'s definition order in
# validation_v1_testing.py (both pulled from the same QSF choices, same order).
EXPECTED_ORDER_RURAL = ["PN.vs.EE", "EN.vs.PS", "EP.vs.EN", "EN.vs.EE", "ES.vs.EP"]
EXPECTED_ORDER_URBAN = ["SH.vs.SS", "HN.vs.ES", "SE.VS.ES", "EH.vs.SN", "HS.vs.PE"]
EXPECTED_ORDER_SIM_RURAL = EXPECTED_ORDER_RURAL + ["Q221", "Q220"]
EXPECTED_ORDER_SIM_URBAN = EXPECTED_ORDER_URBAN + ["Q213", "Q212"]


def stub_persona_context(r):
    """Substitute for the simulated pipeline's LLM-authored persona_description/profile:
    grounds the LLM in this respondent's own words instead of a fictional character, for
    the two pipeline calls (tie-break ranking, priority classification) that expect one."""
    lines = "\n".join(f"- {c['text']}" for c in r["criteria"])
    desc = (
        f"A real participatory-budgeting survey respondent (self-reported area: "
        f"{r['area_selfreport']}) who listed these priorities, in their own words:\n{lines}"
    )
    profile = "(no separate profile -- use the priorities above directly)"
    return desc, profile


def compute_criteria_signs(r):
    return {normalize(c["text"]): CRITERION_SIGN_BY_TAG.get(c["question"], "positive") for c in r["criteria"]}


def resolve_cited_key(cited_text):
    """Map a raw cited_priorities entry (either the elicited-priority text verbatim, or
    an "Other: <free text>" validation citation) to the same normalize()'d key used
    everywhere else (importance, utility_functions, priority_contributions) to refer to
    that priority -- matching how run_validation folds "Other: ..." citations in via
    extract_other_criteria (keyed by normalize() of the text AFTER the "Other:" prefix)."""
    if cited_text.lower().startswith("other:"):
        return normalize(cited_text.split(":", 1)[1].strip())
    return normalize(cited_text)


def match_cited_priorities(cited_priorities, contributions, chosen_side):
    """
    For each priority the respondent said drove THIS specific choice, check whether it
    actually had any bearing on the model's own math for this pair, and if so, whether
    it pointed the same way as the choice they made. c_j > 0 favors A, c_j < 0 favors B
    (see priority_contributions); a priority with c_j == 0 didn't distinguish the two
    bundles at all, so it could not have driven this particular choice even if it's
    important to the respondent in general.
    """
    matches = []
    for cited in cited_priorities:
        key = resolve_cited_key(cited)
        c_j = contributions.get(key)
        if c_j is None:
            status = "unmatched"
        elif c_j == 0:
            status = "irrelevant"
        else:
            predicted_side = "A" if c_j > 0 else "B"
            status = "agrees" if predicted_side == chosen_side else "disagrees"
        matches.append({"cited": cited, "c_j": c_j, "status": status})
    return matches


def build_survey_info(r, projects, cache=None, skip_classification=True, cache_key_prefix=None,
                       priority_key_fn=None):
    """
    skip_classification: defaults to True as of the team decision to drop the explicit
    LLM priority<->project mapping step entirely. When True, never calls the LLM
    per-project classification step (generate_priority_classifications) and leaves
    priority_classifications empty, which forces build_utility_functions_v1's "Legacy"
    branch (sign-only) for every priority in run_validation's _build_uj -- utility
    functions are generated from nothing but the priority's free text plus a bare
    positive/negative sign (real respondents never have a priority_projects free-select
    list; that survey question was removed independently, earlier). Pass
    skip_classification=False to restore the old (per-project-classified) behavior --
    still fully supported and used by run_ablation_no_classification.py for comparison.

    priority_key_fn: optional, forwarded to generate_priority_classifications -- see
    that function's docstring. Used by run_one's priority-reuse mode to route a
    priority's classification cache lookup onto a previously-seen, similar priority.
    Irrelevant when skip_classification=True (the default), since no classification
    call happens in that case.
    """
    importance_ratings = {normalize(c["text"]): c["importance"] for c in r["criteria"]}
    criteria_signs = compute_criteria_signs(r)

    persona_description, profile = stub_persona_context(r)
    # Kept for reporting context (which 3 priorities are non-negotiable) but no longer
    # used to scope classification -- every elicited priority is classified now.
    top_priorities = select_top_priorities(persona_description, profile, importance_ratings, n=3)

    if skip_classification:
        priority_classifications = {}
    else:
        all_priorities = [c["text"] for c in r["criteria"]]
        priority_classifications = generate_priority_classifications(
            persona_description, profile, all_priorities, projects,
            cache=cache, cache_key_prefix=cache_key_prefix or f"{r['variant']}::{r['response_id']}",
            priority_key_fn=priority_key_fn,
        )

    survey_info = {
        "importance": importance_ratings,
        "priority_classifications": priority_classifications,
        "criteria_signs": criteria_signs,
        "idc": [],
    }
    return survey_info, top_priorities, priority_classifications


def build_validation_answers(r, expected_order):
    codes = [bp["code"] for bp in r["bundle_pairs"]]
    if codes != expected_order:
        raise ValueError(f"{r['response_id']}: pair order {codes} != expected {expected_order}")
    answers = []
    for bp in r["bundle_pairs"]:
        chosen = 0 if bp["chosen_side"] == "A" else 1
        certainty = CERTAINTY_TO_INT.get(bp["certainty"])
        answers.append({
            "chosen": chosen,
            "certainty": certainty,
            "cited_priorities": bp["cited_priorities"],
            "bundle_a_rating": RATING_TO_INT.get(bp["rating_a"]),
            "bundle_b_rating": RATING_TO_INT.get(bp["rating_b"]),
            "rating_a_raw": bp["rating_a"],
            "rating_b_raw": bp["rating_b"],
        })
    return answers


POOLS = {
    "V1": {
        "urban": (PROJECTS_URBAN, BUNDLE_PAIRS_URBAN, EXPECTED_ORDER_URBAN),
        "rural": (PROJECTS_RURAL, BUNDLE_PAIRS_RURAL, EXPECTED_ORDER_RURAL),
    },
    "SimProjects": {
        "urban": (PROJECTS_SIM_URBAN, BUNDLE_PAIRS_SIM_URBAN, EXPECTED_ORDER_SIM_URBAN),
        "rural": (PROJECTS_SIM_RURAL, BUNDLE_PAIRS_SIM_RURAL, EXPECTED_ORDER_SIM_RURAL),
    },
}


def run_one(r, cache=None, skip_classification=True, priority_registry=None):
    """
    skip_classification: see build_survey_info -- defaults to True (the LLM
    priority<->project classification step is no longer part of the standard
    pipeline). Still uses a distinct cache-key namespace ("...::noclass") whenever
    True, so today's default-mode utility functions never collide with (or overwrite)
    any old classified cache entries left over from before this default flipped.

    priority_registry: optional priority_dedup.PriorityRegistry. When given, switches
    to a cache namespace SHARED across respondents within the same project pool
    (pool_id, e.g. "V1::urban") instead of the default per-respondent namespace, and
    routes each priority's utility-function cache lookup through
    registry.resolve(pool_id, priority_text) -- so a priority sufficiently similar to
    one already seen (from ANY respondent in that pool, including across waves, since
    both waves use the literal same PROJECTS_* dicts per pool) reuses that entry
    instead of paying for a fresh LLM call. This is an approximation traded off against
    LLM cost: the reused utility function was generated for someone else's (similar)
    wording, not this respondent's own. Off by default; independent of
    skip_classification, which is now on by default regardless.
    """
    area = r["area_selfreport"].lower()
    pool_key = "urban" if area == "urban" else "rural"
    projects, bundle_pairs, expected_order = POOLS[r["variant"]][pool_key]
    pool_id = f"{r['variant']}::{pool_key}"

    demographics = {"age": "25-34", "demographic_area": area}
    priority_canonical_keys = {}
    if priority_registry is not None:
        base_prefix = f"pool::{pool_id}"
        criteria_signs = compute_criteria_signs(r)
        # Resolve each of this respondent's DISTINCT priorities against the registry
        # exactly ONCE up front, not once per cache "kind" (classification vs. utility
        # function each need their own cache key downstream). Resolving lazily inside
        # priority_key_fn would call registry.resolve() twice per priority -- once
        # while building the classify cache key, again for the utility cache key -- and
        # since both calls tokenize the same underlying text, the second call would
        # always find the first call's own just-registered entry as a "hit" against
        # itself. That's pure noise, not real cross-respondent reuse, and it would
        # silently inflate the hit-rate stats reported below.
        for c in r["criteria"]:
            norm_text = normalize(c["text"])
            if norm_text not in priority_canonical_keys:
                sign = criteria_signs.get(norm_text, "positive")
                priority_canonical_keys[norm_text] = priority_registry.resolve(pool_id, c["text"], sign=sign)

        def priority_key_fn(p, _map=priority_canonical_keys):
            # Pure lookup, no further registry.resolve() calls -- see note above. Falls
            # back to p unchanged for anything not in the map (e.g. a criterion
            # surfaced later via an "Other:" validation citation, which isn't part of
            # r["criteria"]) -- exactly as if no registry were in use for that one.
            return _map.get(normalize(p), p)
    else:
        base_prefix = f"{r['variant']}::{r['response_id']}"
        priority_key_fn = None
    cache_key_prefix = f"{base_prefix}::noclass" if skip_classification else base_prefix
    survey_info, top_priorities, priority_classifications = build_survey_info(
        r, projects, cache=cache, skip_classification=skip_classification,
        cache_key_prefix=cache_key_prefix, priority_key_fn=priority_key_fn,
    )
    validation_answers = build_validation_answers(r, expected_order)

    # run_validation auto-dumps a full per-priority breakdown (u(A), u(B), c_j, and each
    # priority's contribution to Delta) for every pair the CARDINAL model got wrong --
    # captured here instead of letting it print to stdout, so it ends up in the report.
    debug_buffer = io.StringIO()
    result = run_validation(
        demographics, "v1", survey_info, bundle_pairs, validation_answers,
        projects_override=projects, cache=cache, cache_key_prefix=cache_key_prefix,
        file=debug_buffer, priority_key_fn=priority_key_fn,
    )
    # For each pair, check whether the priorities the respondent SAYS drove that choice
    # actually had any bearing on the model's own math for it (nonzero contribution to
    # Delta), and if so, whether the direction agrees with the choice they made. Reuses
    # the exact per-priority c_j the cardinal model itself computes -- no new LLM calls.
    omega_ranges = likert_to_omega_ranges(result["importance"]) if result else {}
    cited_vs_model = []
    joint_feasibility = None
    if result:
        for (bA, bB), truth in zip(bundle_pairs, validation_answers):
            contributions = priority_contributions(bA, bB, result["utility_functions"], omega_ranges)
            chosen_side = "A" if truth["chosen"] == 0 else "B"
            cited_vs_model.append(match_cited_priorities(truth["cited_priorities"], contributions, chosen_side))
        joint_feasibility = joint_feasibility_tiers(
            bundle_pairs, expected_order, validation_answers,
            result["utility_functions"], result["importance"],
        )

    return {
        "response_id": r["response_id"],
        "area": r["area_selfreport"],
        "criteria": r["criteria"],  # full elicited list, original text + importance
        "demographics": r.get("demographics") or {},  # Prolific demographics; empty for Wave 0 (no export to join)
        "n_criteria": len(r["criteria"]),
        "top_priorities": top_priorities,
        "priority_classifications": priority_classifications,
        "importance_ratings": survey_info["importance"],
        "validation_answers": validation_answers,
        "debug_output": debug_buffer.getvalue(),
        "bundle_pairs": bundle_pairs,
        "pair_codes": expected_order,
        "result": result,
        "cited_vs_model": cited_vs_model,
        "joint_feasibility": joint_feasibility,
        "cache_key_prefix": cache_key_prefix,
        "pool_id": pool_id,
        "priority_canonical_keys": priority_canonical_keys,
        "skip_classification": skip_classification,
    }


def format_classification(classification):
    """Compact, readable summary of one priority's per-project classification --
    groups by category and drops "neither" entries (almost always the majority and
    not informative), instead of dumping the full project-by-project JSON."""
    if classification is None:
        return "_no classification found (lookup mismatch -- check normalize() keys)_"
    if classification.get("not_about_projects", True):
        return "_not about any particular project_"
    proj_map = classification.get("projects", {})
    by_cat = {"good": [], "bad": [], "depends": []}
    for proj, cat in proj_map.items():
        if cat in by_cat:
            by_cat[cat].append(proj)
    parts = []
    for cat, label in [("good", "supports"), ("bad", "opposes"), ("depends", "depends on mix")]:
        if by_cat[cat]:
            parts.append(f"**{label}:** {', '.join(sorted(by_cat[cat]))}")
    return "; ".join(parts) if parts else "_no project stood out as good/bad/depends_"


def render_variant_section(out, variant_label, runs, cache=None):
    cache = cache or {}
    n = len(runs)
    consistency = [r["result"]["consistency_rate"] for r in runs if r["result"]]
    lex_consistency = [r["result"]["lexicographic_consistency_rate"] for r in runs if r["result"]]
    out.append(f"## {variant_label} — aggregate\n")
    out.append(f"- Respondents: {n}")
    if consistency:
        out.append(f"- Mean cardinal consistency rate: {sum(consistency)/len(consistency):.0%}")
    if lex_consistency:
        out.append(f"- Mean lexicographic consistency rate: {sum(lex_consistency)/len(lex_consistency):.0%}")
    all_matches = [m for run in runs for matches in run.get("cited_vs_model", []) for m in matches]
    if all_matches:
        n_cited = len(all_matches)
        n_used = sum(1 for m in all_matches if m["status"] in ("agrees", "disagrees"))
        n_agree = sum(1 for m in all_matches if m["status"] == "agrees")
        n_disagree = sum(1 for m in all_matches if m["status"] == "disagrees")
        n_irrelevant = sum(1 for m in all_matches if m["status"] == "irrelevant")
        n_unmatched = sum(1 for m in all_matches if m["status"] == "unmatched")
        out.append(
            f"- Cited-priority relevance: {n_used}/{n_cited} citations ({n_used/n_cited:.0%}) "
            f"actually influenced the model's prediction for that pair (nonzero contribution to "
            f"Delta); {n_irrelevant} were irrelevant to that specific pair, {n_unmatched} didn't "
            f"resolve to a tracked priority. Of the {n_used} that were used, {n_agree} pointed "
            f"toward the bundle the respondent actually chose and {n_disagree} pointed the "
            f"other way."
        )
    tiers = [run["joint_feasibility"]["tier"] for run in runs if run.get("joint_feasibility")]
    if tiers:
        tier_counts = {t: tiers.count(t) for t in "abcd"}
        out.append(
            "- Joint feasibility (strictest tier reached per respondent — see per-respondent detail "
            "below for what each tier means): "
            + ", ".join(f"({t}) {tier_counts[t]}" for t in "abcd")
        )
    out.append("")

    for run in runs:
        r = run["result"]
        out.append(f"## {run['response_id']} — {run['area']}\n")
        demo = run.get("demographics") or {}
        if demo:
            parts = []
            if demo.get("location"):
                parts.append(f"{demo['location']}" + (f", {demo['country_of_residence']}"
                             if demo.get("country_of_residence") and demo["country_of_residence"] != "United States" else ""))
            if demo.get("age"):
                parts.append(f"age {demo['age']}")
            if demo.get("sex"):
                parts.append(demo["sex"])
            if demo.get("political_affiliation"):
                parts.append(demo["political_affiliation"])
            if demo.get("education"):
                parts.append(demo["education"])
            if demo.get("employment_status"):
                parts.append(demo["employment_status"])
            if demo.get("ethnicity"):
                parts.append(demo["ethnicity"])
            out.append(f"- Demographics (Prolific): {' · '.join(parts)}\n")
        out.append(f"- Elicited criteria: {run['n_criteria']} (all listed below, not just the top-3)")
        out.append(f"- Top-3 (non-negotiable) priorities: {run['top_priorities']}")
        out.append(f"- Cardinal consistency rate: {r['consistency_rate']:.0%}")
        out.append(f"- Lexicographic consistency rate: {r['lexicographic_consistency_rate']:.0%}")
        if r["robust_accuracy"] is not None:
            out.append(f"- Strong-prediction accuracy: {r['robust_accuracy']:.0%} ({r['n_robust']}/{r['n_total']})")
        else:
            out.append("- No strong (robust) predictions made")
        if r["calibration"] is not None:
            out.append(f"- Calibration (Pearson r, uncertainty vs. certainty): {r['calibration']:.2f}")
        jf = run.get("joint_feasibility")
        if jf:
            out.append(f"- Joint feasibility: **tier {jf['tier']}** — {jf['tier_label']}")
        out.append("")

        if jf:
            if jf["tier"] == "d":
                out.append(
                    "<details><summary><b>Joint feasibility detail</b> — no omega vector, even fully "
                    "unconstrained, explains all of this respondent's choices at once</summary>\n"
                )
                if jf["culprits"]:
                    out.append(f"- Dropping any ONE of these pairs would restore feasibility: {jf['culprits']}")
                else:
                    out.append("- No single pair's removal restores feasibility (the violation needs 2+ pairs).")
                out.append("</details>\n")
            else:
                importance = r["importance"]
                omega_ranges = jf["omega_ranges"]
                widths = [hi - lo for lo, hi in omega_ranges.values()]
                mean_width = sum(widths) / len(widths) if widths else 0.0
                out.append(
                    "<details><summary><b>Joint feasibility detail</b> — achievable omega range per "
                    f"priority (mean width {mean_width:.2f}; narrower = this respondent's choices pin "
                    "that weight down more tightly)</summary>\n"
                )
                out.append("| Priority | Stated importance | Achieved range | Width |")
                out.append("|---|---|---|---|")
                for p in jf["priorities"]:
                    lo, hi = omega_ranges[p]
                    out.append(f"| {p} | {importance.get(p, '?')} | [{lo:.2f}, {hi:.2f}] | {hi - lo:.2f} |")
                out.append("\n</details>\n")

        out.append("| Pair | Chosen | Cardinal model | Consistent | A rated | B rated | Cited used |")
        out.append("|---|---|---|---|---|---|---|")
        cited_vs_model = run.get("cited_vs_model") or [[] for _ in run["pair_codes"]]
        for code, pred, truth, matches in zip(
            run["pair_codes"], r["per_pair"], run["validation_answers"], cited_vs_model
        ):
            chosen = "A" if truth["chosen"] == 0 else "B"
            compat = pred["compatible_outcomes"]
            model_str = compat[0] if len(compat) == 1 else "[" + ", ".join(compat) + "]"
            a_rated = RATING_ABBREV.get(truth.get("rating_a_raw"), "?")
            b_rated = RATING_ABBREV.get(truth.get("rating_b_raw"), "?")
            if matches:
                n_used = sum(1 for m in matches if m["status"] in ("agrees", "disagrees"))
                n_agree = sum(1 for m in matches if m["status"] == "agrees")
                cited_str = f"{n_used}/{len(matches)} used ({n_agree} agree)"
            else:
                cited_str = "–"
            out.append(f"| {code} | {chosen} | {model_str} | {pred['is_consistent']} | "
                       f"{a_rated} | {b_rated} | {cited_str} |")
        out.append("")

        cited_lines = []
        for code, matches in zip(run["pair_codes"], cited_vs_model):
            for m in matches:
                symbol = {"agrees": "✓", "disagrees": "✗", "irrelevant": "·", "unmatched": "?"}[m["status"]]
                c_j_str = f"{m['c_j']:+.2f}" if m["c_j"] is not None else "n/a"
                cited_lines.append(f"{code}: {symbol} \"{m['cited']}\" (c_j={c_j_str}, {m['status']})")
        if cited_lines:
            out.append(
                "<details><summary><b>Cited priorities vs. model-relevant priorities</b> "
                "(✓ = this priority's own utility function pointed toward the bundle the "
                "respondent chose; ✗ = it pointed the other way; · = the two bundles didn't "
                "differ on it at all, so it couldn't have driven this choice; "
                "? = didn't resolve to a tracked priority)</summary>\n"
            )
            out.append("```text")
            out.extend(cited_lines)
            out.append("```")
            out.append("</details>\n")

        debug_text = (run.get("debug_output") or "").strip()
        if debug_text:
            # debug_pair labels pairs positionally ("Pair 1", "Pair 2", ...) in the same
            # order as pair_codes -- swap in the actual code so it's not just a number.
            for i, code in enumerate(run["pair_codes"]):
                debug_text = debug_text.replace(f"Pair {i + 1} —", f"Pair {i + 1} ({code}) —")
            out.append(
                "<details><summary><b>Why the wrong cardinal predictions were wrong</b> "
                "(per-priority u(A), u(B), and each priority's contribution to Delta, for "
                "every pair the cardinal model missed above — this is the "
                "<code>Consistent</code> column specifically, not <code>Lex consistent</code>)"
                "</summary>\n"
            )
            out.append("```text")
            out.append(debug_text)
            out.append("```")
            out.append("</details>\n")

        skip_classification = run.get("skip_classification", True)
        header = ("**All elicited priorities, importance ratings, and generated utility function:**\n"
                  if skip_classification else
                  "**All elicited priorities, importance ratings, LLM classification, and generated "
                  "utility function:**\n")
        out.append(header)
        prefix = run["cache_key_prefix"]
        canonical_keys = run.get("priority_canonical_keys") or {}
        for c in run["criteria"]:
            text, importance = c["text"], c["importance"]
            norm_text = normalize(text)
            classification = run["priority_classifications"].get(norm_text)
            # Which of the three elicitation questions this priority came from --
            # "Good" (an individual thing they'd fund), "Bad" (something that would
            # make a bundle bad), or "Mix" (a property of the overall combination) --
            # independent of the (now-disabled-by-default) per-project
            # supports/opposes/depends classification.
            category = CRITERION_CATEGORY_BY_TAG.get(c["question"], "Good")
            # When priority-reuse is active, the actual cache key for this priority may
            # be a DIFFERENT (earlier, similar) priority's text -- normalize(text) alone
            # would silently miss the lookup for anything that got reused. Cache keys
            # are always normalize()'d (see run_validation's _uj_cache_key), so match
            # that here too rather than using the registry's raw, originally-cased text.
            key_text = normalize(canonical_keys.get(norm_text, norm_text))
            out.append(f"<details><summary><b>{importance}</b> [{category}] — {text}</summary>\n")
            # The classification step is off by default now (see skip_classification) --
            # priority_classifications is always empty in that mode, so a "Classification:"
            # line would always print the same non-informative placeholder for every
            # single priority. Only show it when classification was actually run (i.e.
            # skip_classification=False was explicitly requested, e.g. for the ablation).
            if not skip_classification:
                out.append(f"- Classification: {format_classification(classification)}")
            if key_text != norm_text:
                out.append(f"- _Reused a previously-generated utility function for a similar priority: "
                           f"\"{canonical_keys.get(norm_text, norm_text)}\"_")
            source = cache.get(f"{prefix}::utility::{key_text}")
            if source:
                out.append("\n```python")
                out.append(source)
                out.append("```")
            else:
                out.append("- _utility function source not found in cache_")
            out.append("\n</details>\n")


def render_report(v1_runs, sim_runs, cache=None):
    out = []
    out.append("# Wave 0 Real-Respondent Validation\n")
    out.append(
        "Runs the real Wave-0 respondents (both survey variants) through the same "
        "validation pipeline used for simulated personas (`run_validation` / "
        "`build_utility_functions_v1` / `predict_pair` / `predict_pair_lexicographic`). "
        "Ground truth — bundle choice, certainty, cited priorities, importance ratings — "
        "comes directly from each respondent's real answers. Each priority's utility "
        "function is generated directly from its free text plus a positive/negative sign "
        "(no explicit LLM step maps a priority to particular projects -- that step was "
        "tried and deliberately dropped; see run_ablation_no_classification.py for the "
        "comparison that informed the decision). Generated utility functions are cached "
        "(`results/llm_model_cache.json`), so re-running this script against the same "
        "respondents reuses the same model instead of regenerating it. Priority reuse ALSO "
        "happens across DIFFERENT respondents: a priority sufficiently similar (same "
        "project pool, same positive/negative sign, token-overlap above threshold) to one "
        "already seen reuses that earlier utility function rather than generating a new "
        "one -- see `priority_dedup.PriorityRegistry`. A per-priority note in this report "
        "flags when that happened.\n"
    )
    render_variant_section(out, "Survey1_V1", v1_runs, cache=cache)
    render_variant_section(out, "Survey1_V1_SimProjects", sim_runs, cache=cache)
    return "\n".join(out)


if __name__ == "__main__":
    v1, sim = load_all()
    cache = load_cache()
    # Priority-reuse is on by default for this script now: before generating a fresh
    # classification/utility function, check whether a sufficiently similar priority
    # (same project pool, same positive/negative sign, token-overlap above threshold)
    # has already been seen -- by this run, an earlier run of this script, or a run of
    # run_wave1_real_validation.py / run_priority_dedup_validation.py, since they all
    # share PRIORITY_REGISTRY_PATH -- and reuse its entry instead of paying for a new
    # LLM call. Deliberately conservative (see priority_dedup.PriorityRegistry): when
    # it's not confident two priorities are the same, it generates fresh rather than
    # guessing.
    registry = PriorityRegistry(path=PRIORITY_REGISTRY_PATH, threshold=PRIORITY_SIMILARITY_THRESHOLD)

    print(f"Running {len(v1)} real V1 respondents through the validation pipeline...")
    v1_runs = []
    for r in v1:
        print(f"  {r['response_id']} ({r['area_selfreport']})...")
        v1_runs.append(run_one(r, cache=cache, priority_registry=registry))
        save_cache(cache)  # incremental, so an interrupted run doesn't lose earlier work
        registry.save()

    print(f"Running {len(sim)} real SimProjects respondents through the validation pipeline...")
    sim_runs = []
    for r in sim:
        print(f"  {r['response_id']} ({r['area_selfreport']})...")
        sim_runs.append(run_one(r, cache=cache, priority_registry=registry))
        save_cache(cache)
        registry.save()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"wave0_real_validation_{ts}.md")
    with open(path, "w") as f:
        f.write(render_report(v1_runs, sim_runs, cache=cache))
    print("wrote", path)
    total = registry.hits + registry.misses
    if total:
        print(f"Priority-reuse: {registry.hits}/{total} priorities ({registry.hits/total:.0%}) "
              f"reused an existing classification/utility function instead of generating fresh.")
