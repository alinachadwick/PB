"""
Runs a roster of 20 diverse simulated personas through the full v1 survey
pipeline (elicitation -> bundle-pair validation -> importance rating -> top-3
classification -> prediction) and writes a single Markdown report meant to be
shared with the team: a roster table, pooled/group-level consistency stats,
and a per-persona breakdown.

Reuses every pipeline function from validation_v1_testing.py unmodified —
this file only adds the persona roster and the reporting layer.

Usage:
    python3 run_persona_batch_20.py            # full 20-persona batch
    python3 run_persona_batch_20.py --n 5       # trial run on the first 5 personas
"""

import argparse
from datetime import datetime

import pandas as pd

from validation_v1_testing import (
    CRITERIA_QUESTIONS_URBAN,
    CRITERIA_SIGN_BY_QUESTION,
    PROJECTS_ALL,
    PROJECTS_RURAL,
    PROJECTS_URBAN,
    BUNDLE_PAIRS_RURAL,
    BUNDLE_PAIRS_URBAN,
    generate_persona_profile,
    generate_criteria,
    generate_validation_answers,
    generate_likert_ratings,
    select_top_priorities,
    generate_priority_classifications,
    run_validation,
    normalize,
)
from research_stats import (
    describe_priority_counts,
    test_predictive_validity,
    compare_cardinal_vs_lex,
)

# ---------------------------------------------------------------------------
# Roster — the 9 personas already in TEST_PERSONAS, plus 11 new ones chosen to
# fill gaps: political affiliation (only conservative/vague-left/progressive
# were covered before), education level (not tracked at all before), and
# additional disability angles beyond mobility (hearing, chronic illness).
# ---------------------------------------------------------------------------

PB_NEVER = "I've never heard of it"
PB_HEARD = "I've heard of it but haven't participated"
PB_DONE = "I've participated in it"

PERSONAS_20 = [
    {
        "description": (
            "You are a woman in her mid-50s who owns a small hardware store in an urban neighborhood. "
            "You vote Republican and are skeptical of government spending on projects you see as "
            "'nice to have' rather than essential. You believe the city should focus on core services — "
            "safety, infrastructure, keeping things running — before funding anything extra. You're "
            "friendly in person but you don't think city money should be spent loosely, and you get "
            "frustrated when a budget feels like it's trying to please everyone instead of being responsible."
        ),
        "area": "urban", "education": "Some college, no degree",
        "political_affiliation": "Republican", "pb_experience": PB_HEARD,
    },
    {
        "description": (
            "You are a man in his 40s living in a small rural town. You use a wheelchair after a "
            "workplace injury several years ago, and you're acutely aware of which public spaces are "
            "actually accessible to you and which aren't — you think about curb cuts, ramps, and "
            "accessible restrooms before deciding whether it's worth going somewhere. You care about "
            "seniors and other disabled residents too, since you see how the same barriers affect them. "
            "You want the town to treat accessibility and health services as basic infrastructure, "
            "not an afterthought."
        ),
        "area": "rural", "education": "Bachelor's degree",
        "political_affiliation": "Independent", "pb_experience": PB_HEARD,
    },
    {
        "description": (
            "You are a man in his early 30s who runs a small family farm outside of town. You're short "
            "on time most days and don't waste words when you don't have to, but you know exactly what "
            "you think about how the town should spend money. You care most about keeping young people "
            "from leaving the area — that means jobs and things for kids to do. You also think road and "
            "park maintenance matter more than people realize, since you drive them every day for work. "
            "You vote based on the person, not the party."
        ),
        "area": "rural", "education": "High school graduate (includes GED or equivalent credential)",
        "political_affiliation": "Independent", "pb_experience": PB_NEVER,
    },
    {
        "description": (
            "You are an egalitarian man in his 50s from Chicago, IL. "
            "You believe city resources should prioritize the most vulnerable residents — "
            "unhoused people, low-income families. "
            "You are skeptical of environmental projects when basic needs aren't being met."
        ),
        "area": "urban", "education": "Bachelor's degree",
        "political_affiliation": "Democrat", "pb_experience": PB_DONE,
    },
    {
        "description": (
            "You are an environmentalist woman in her 30s from rural Mississippi. "
            "You grew up surrounded by nature and believe sustainability is the most urgent issue. "
            "You distrust short-term fixes without addressing root causes."
        ),
        "area": "rural", "education": "Bachelor's degree",
        "political_affiliation": "Independent", "pb_experience": PB_HEARD,
    },
    {
        "description": (
            "You are a woman in her 40s who lives in the city. You vote Democrat most of the time "
            "but you're not super political. You have two dogs and you like going to the farmers market "
            "on weekends. You work in HR at a mid-size company. You care about the environment "
            "but also think people need help too. You're not sure what you think about homeless people — "
            "you feel bad for them but also the situation feels complicated."
        ),
        "area": "urban", "education": "Bachelor's degree",
        "political_affiliation": "Democrat", "pb_experience": PB_NEVER,
    },
    {
        "description": (
            "You are a man in his 60s who moved from a city to a small rural town ten years ago. "
            "You like hiking and being outdoors. You have kids who are now grown up and left. "
            "You think education is important but you also think kids today have too many opportunities "
            "handed to them. You vote Republican sometimes. You're worried about jobs leaving the area. "
            "You garden a lot and think local food is important."
        ),
        "area": "rural", "education": "Bachelor's degree",
        "political_affiliation": "Independent", "pb_experience": PB_NEVER,
    },
    {
        "description": (
            "You are a 28-year-old who just moved to the city for work. You rent an apartment. "
            "You don't have kids but your sister does and you're close with her family. "
            "You care about climate change but you also have a car because public transit is bad. "
            "You're generally pretty busy and haven't thought much about local politics. "
            "You want the city to feel nicer but you're not sure what that means exactly."
        ),
        "area": "urban", "education": "Bachelor's degree",
        "political_affiliation": "No political affiliation", "pb_experience": PB_NEVER,
    },
    {
        "description": (
            "You are a woman in your early 50s living in a rural community. "
            "English is your second language and you sometimes find forms confusing. "
            "You have a large family and most of your concern is about your kids and grandkids. "
            "You also help out at your church's food pantry on weekends. "
            "You don't really follow the news much but you know your town needs more jobs."
        ),
        "area": "rural", "education": "Some high school, no degree",
        "political_affiliation": "No political affiliation", "pb_experience": PB_NEVER,
    },
    # --- new personas ---
    {
        "description": (
            "You are a woman in her 40s who works as a nurse at a local urgent care clinic. You register "
            "as an Independent and don't feel like either major party really gets healthcare or "
            "public-health infrastructure right. You're pragmatic — you've seen firsthand what happens "
            "when basic services are underfunded, but you're also skeptical of programs that sound good "
            "on paper without a clear plan behind them. You care about things that keep people healthy "
            "and safe day to day, and you get frustrated by spending that feels more symbolic than functional."
        ),
        "area": "suburban", "education": "Associate's degree",
        "political_affiliation": "Independent", "pb_experience": PB_HEARD,
    },
    {
        "description": (
            "You are a man in his mid-20s who drives for a rideshare app and picks up other gig work "
            "when you can. You don't identify with any political party — you haven't thought much about "
            "local politics before, it all feels pretty removed from your actual life. Money is tight and "
            "you're focused on practical things: is it safe to be out at night, are the streets in decent "
            "shape, is there anywhere nearby to get cheap food. You don't have strong abstract opinions — "
            "you just know what would actually make your day-to-day easier."
        ),
        "area": "urban", "education": "Some college, no degree",
        "political_affiliation": "No political affiliation", "pb_experience": PB_NEVER,
    },
    {
        "description": (
            "You are a woman in her 30s who works for a local nonprofit and attends city council meetings "
            "regularly. You're a committed Democrat and think a lot about structural inequities — who gets "
            "left out of decisions, and which spending patterns quietly reinforce advantages some "
            "neighborhoods already have. You're not interested in symbolic gestures; you want to see "
            "concrete investment in the communities that need it most. You have detailed, well-reasoned "
            "opinions about how participatory budgeting should work."
        ),
        "area": "urban", "education": "Graduate or professional school degree (MA, PhD, JD, MD, ...)",
        "political_affiliation": "Democrat", "pb_experience": PB_DONE,
    },
    {
        "description": (
            "You are a man in his 50s who owns and rents out a few small properties around town. You vote "
            "Republican and lean libertarian — you think government should do the basics well and "
            "otherwise stay out of the way. You're already frustrated by property taxes and you're "
            "instinctively skeptical of any new program, even ones that sound nice, because you've seen "
            "budgets balloon before. You'd rather the town fund fewer things and do them well than spread "
            "money across a lot of initiatives."
        ),
        "area": "suburban", "education": "Bachelor's degree",
        "political_affiliation": "Republican", "pb_experience": PB_HEARD,
    },
    {
        "description": (
            "You are a woman in her early 30s raising two kids on your own while working full-time at a "
            "retail job. Money and time are both tight — you're short on time and don't have patience for "
            "long, drawn-out surveys, but you know exactly what you think. You care most about anything "
            "that makes life easier or safer for your kids — daycare, safe places for them to be after "
            "school, basic affordability. You'd rather just say plainly what you need than write an essay "
            "about it. You vote Democrat when you vote at all."
        ),
        "area": "urban", "education": "High school graduate (includes GED or equivalent credential)",
        "political_affiliation": "Democrat", "pb_experience": PB_NEVER,
    },
    {
        "description": (
            "You are a woman in her 60s, a retired teacher who is deaf and communicates primarily through "
            "ASL and written text. You register as an Independent. Accessibility is central to how you "
            "think about any public spending — not just physical accessibility, but whether services and "
            "information are actually usable by people who communicate differently. You're articulate and "
            "detail-oriented, and you tend to notice when a project's description quietly assumes everyone "
            "can hear an announcement or use a phone line."
        ),
        "area": "suburban", "education": "Graduate or professional school degree (MA, PhD, JD, MD, ...)",
        "political_affiliation": "Independent", "pb_experience": PB_HEARD,
    },
    {
        "description": (
            "You are a man in his 40s, a military veteran who now works as an electrician and is active "
            "in his local union. You vote Republican. You care about infrastructure and jobs that pay well "
            "and last — the kind of work that actually builds something. You're skeptical of programs you "
            "see as handouts rather than investment, but you feel strongly that emergency services and "
            "support for veterans and first responders shouldn't be an afterthought. You're blunt and "
            "don't sugarcoat your opinions."
        ),
        "area": "rural", "education": "Associate's degree",
        "political_affiliation": "Republican", "pb_experience": PB_NEVER,
    },
    {
        "description": (
            "You are a woman in her 40s who immigrated to the US about 15 years ago and now runs a small "
            "restaurant. You vote Democrat. You care a lot about the town investing in the community — "
            "safety, opportunities for people who are just getting started the way you once were, and "
            "support for other small businesses like yours. You remember what it was like to feel "
            "unfamiliar with how local government worked, and you want the process to be something people "
            "like you can actually understand and take part in."
        ),
        "area": "suburban", "education": "Some college, no degree",
        "political_affiliation": "Democrat", "pb_experience": PB_NEVER,
    },
    {
        "description": (
            "You are a man in his late 60s, retired, whose kids have grown up and moved away. You vote "
            "Republican. You're on a fixed income and you notice every time property taxes creep up. "
            "You'd rather see money go toward maintaining what the town already has — roads, basic "
            "services — than toward new programs, however well-intentioned. You grew up in this town and "
            "have strong opinions about what it needs, formed over decades of watching budgets come and go."
        ),
        "area": "rural", "education": "High school graduate (includes GED or equivalent credential)",
        "political_affiliation": "Republican", "pb_experience": PB_HEARD,
    },
    {
        "description": (
            "You are a woman in her 30s who lives with a chronic illness that leaves you with limited "
            "energy on any given day. You vote Democrat and consider yourself a disability advocate, "
            "though your focus is different from mobility-specific issues — you think about healthcare "
            "access, flexibility, and services that don't assume everyone can show up in person or "
            "navigate long processes. Because your energy is limited, you tend to focus your engagement on "
            "a few things you care about most rather than spreading yourself across every issue."
        ),
        "area": "urban", "education": "Bachelor's degree",
        "political_affiliation": "Democrat", "pb_experience": PB_DONE,
    },
    {
        "description": (
            "You are a man in his 50s who leads a small congregation in a rural town. You register as an "
            "Independent — your faith shapes your politics more than either party does. You spend a lot of "
            "time with people in your community who are struggling, whether that's food insecurity, "
            "addiction, or just not having enough support, and that shapes what you think the town should "
            "prioritize. You tend to frame your reasoning in terms of caring for the most vulnerable people "
            "in the community, and you give thoughtful, warm, fairly detailed answers."
        ),
        "area": "rural", "education": "Graduate or professional school degree (MA, PhD, JD, MD, ...)",
        "political_affiliation": "Independent", "pb_experience": PB_HEARD,
    },
]

assert len(PERSONAS_20) == 20, f"expected 20 personas, got {len(PERSONAS_20)}"


# ---------------------------------------------------------------------------
# Pipeline — same call sequence as run_persona_tests, just returning
# structured data per persona instead of printing to a results file.
# ---------------------------------------------------------------------------

def run_one_persona(persona, index):
    area = persona["area"]
    if area in ("urban", "suburban"):
        validation_projects, bundle_pairs = PROJECTS_URBAN, BUNDLE_PAIRS_URBAN
    else:
        validation_projects, bundle_pairs = PROJECTS_RURAL, BUNDLE_PAIRS_RURAL

    profile = generate_persona_profile(persona["description"])

    criteria_by_question = generate_criteria(
        persona["description"], profile, CRITERIA_QUESTIONS_URBAN, PROJECTS_ALL
    )
    elicited_criteria = []
    for cl in criteria_by_question.values():
        elicited_criteria.extend(cl)
    seen = set()
    elicited_criteria = [c for c in elicited_criteria if not (c in seen or seen.add(c))]

    validation_answers = generate_validation_answers(
        persona["description"], profile, elicited_criteria, bundle_pairs, validation_projects
    )

    importance_ratings = generate_likert_ratings(persona["description"], profile, criteria_by_question)
    top_priorities = select_top_priorities(persona["description"], profile, importance_ratings, n=3)
    priority_classifications = generate_priority_classifications(
        persona["description"], profile, top_priorities, validation_projects
    )

    criteria_signs = {}
    for q_label, criteria_list in criteria_by_question.items():
        sign = CRITERIA_SIGN_BY_QUESTION.get(q_label, "positive")
        for c in criteria_list:
            criteria_signs[normalize(c)] = sign

    survey_info = {
        "importance": importance_ratings,
        "priority_classifications": priority_classifications,
        "criteria_signs": criteria_signs,
        "idc": [],
    }

    # run_validation's Under-18 skip check needs an "age" key; none of these
    # personas are meant to trigger it, so this is just a compatibility value —
    # age isn't tracked as a real field here per the v10 survey update.
    demographics = {"age": "18-30", "demographic_area": area}

    result = run_validation(demographics, "v1", survey_info, bundle_pairs, validation_answers)

    return {
        "index": index,
        "persona": persona,
        "profile": profile,
        "criteria_by_question": criteria_by_question,
        "n_priorities": len(elicited_criteria),
        "importance_ratings": importance_ratings,
        "top_priorities": top_priorities,
        "priority_classifications": priority_classifications,
        "validation_answers": validation_answers,
        "bundle_pairs": bundle_pairs,
        "result": result,
    }


def run_batch():
    all_results = []
    for i, persona in enumerate(PERSONAS_20, start=1):
        print(f"[{i}/{len(PERSONAS_20)}] Running: {persona['description'][:70]}...")
        all_results.append(run_one_persona(persona, i))
    return all_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_dataframe(all_results):
    rows = []
    for r in all_results:
        res = r["result"]
        n_total = res["n_total"]
        rows.append({
            "respondent_id": r["index"],
            "label": r["persona"]["description"],
            "group_area": r["persona"]["area"],
            "group_political": r["persona"]["political_affiliation"],
            "group_education": r["persona"]["education"],
            "group_pb_experience": r["persona"]["pb_experience"],
            "n_priorities": r["n_priorities"],
            "cardinal_n_correct": round(res["consistency_rate"] * n_total),
            "cardinal_n_total": n_total,
            "lex_n_correct": round(res["lexicographic_consistency_rate"] * n_total),
            "lex_n_total": n_total,
            "strong_pred_accuracy": res["robust_accuracy"],
            "n_robust": res["n_robust"],
        })
    return pd.DataFrame(rows)


def render_report(all_results, df, path):
    lines = []
    out = lines.append

    out(f"# 20-Persona Survey Batch — Consistency Report")
    out("")
    out(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · {len(all_results)} simulated personas "
        f"× 5 bundle-pair comparisons each = {5 * len(all_results)} total comparisons.")
    out("")
    out("Each persona independently goes through the full v1 pipeline: elicits its own "
        "priorities, makes 5 bundle-pair choices, rates importance, and gets scored by both "
        "the cardinal and lexicographic prediction models against its own choices.")
    out("")

    out("## Roster")
    out("")
    for r in all_results:
        p = r["persona"]
        out(f"**{r['index']}.** {p['description']}")
        out("")
        out(f"*Area: {p['area']} · Education: {p['education']} · "
            f"Political affiliation: {p['political_affiliation']} · PB experience: {p['pb_experience']}*")
        out("")

    out("## Bundle comparisons reference")
    out("")
    out("Every rural persona sees the same 5 rural pairs; every urban/suburban persona sees "
        "the same 5 urban pairs. Per-persona tables below refer back to these by number.")
    out("")
    for area_label, pairs in [("Rural", BUNDLE_PAIRS_RURAL), ("Urban / suburban", BUNDLE_PAIRS_URBAN)]:
        out(f"**{area_label}**")
        out("")
        for i, (bA, bB) in enumerate(pairs, start=1):
            out(f"- Pair {i}: **A** = {', '.join(bA)} **vs. B** = {', '.join(bB)}")
        out("")

    out("## Aggregate results")
    out("")
    desc = describe_priority_counts(df)
    out(f"- **Priorities elicited per persona:** mean {desc['mean']:.1f}, median {desc['median']:.0f}, "
        f"range [{desc['min']}, {desc['max']}]")
    for label, ncol, tcol in [("Cardinal", "cardinal_n_correct", "cardinal_n_total"),
                               ("Lexicographic", "lex_n_correct", "lex_n_total")]:
        stats = test_predictive_validity(df, ncol, tcol, label)
        p_str = (f"p={stats['wilcoxon_p_vs_chance']:.4f}"
                 if stats["wilcoxon_p_vs_chance"] == stats["wilcoxon_p_vs_chance"] else stats["note"])
        out(f"- **{label} consistency:** {stats['mean_consistency']:.0%} "
            f"(95% CI [{stats['ci95'][0]:.0%}, {stats['ci95'][1]:.0%}]) · vs. chance: {p_str}")
    cmp = compare_cardinal_vs_lex(df)
    cmp_detail = cmp["note"] or f"p={cmp['wilcoxon_p']:.4f}"
    out(f"- **Cardinal vs. lexicographic:** mean difference {cmp['mean_diff']:+.0%} ({cmp_detail})")
    n_strong = df["n_robust"].sum()
    n_total_pairs = df["cardinal_n_total"].sum()
    out(f"- **Strong (confident) predictions:** {n_strong}/{n_total_pairs} comparisons "
        f"({n_strong / n_total_pairs:.0%}) produced a strong cardinal prediction at all.")
    out("")

    out("## Per-persona detail")
    out("")
    for r in all_results:
        p = r["persona"]
        res = r["result"]
        out(f"### Persona {r['index']} — {p['area']}, {p['political_affiliation']}")
        out("")
        out(p["description"])
        out("")
        out(f"- Area: {p['area']} · Education: {p['education']} · "
            f"Political affiliation: {p['political_affiliation']} · PB experience: {p['pb_experience']}")
        out(f"- Priorities elicited: {r['n_priorities']} · Top 3 (non-negotiable tier): "
            f"{', '.join(r['top_priorities']) if r['top_priorities'] else '(none reached that tier)'}")
        out(f"- **Consistency: {res['consistency_rate']:.0%}** · "
            f"**Lexicographic: {res['lexicographic_consistency_rate']:.0%}**"
            + (f" · Strong-pred accuracy: {res['robust_accuracy']:.0%} "
               f"({res['n_robust']}/{res['n_total']} strong predictions)"
               if res["robust_accuracy"] is not None else " · No strong predictions made"))
        out("")

        out("**Elicited criteria & importance ratings**")
        out("")
        for q_label, criteria_list in r["criteria_by_question"].items():
            out(f"*{q_label}*")
            out("")
            if not criteria_list:
                out("- (none)")
            for c in criteria_list:
                rating = r["importance_ratings"].get(normalize(c), "?")
                out(f"- {c} — *{rating}*")
            out("")

        out("**Top-3 priority classifications** (per-project answers, only asked for the "
            "non-negotiable-tier priorities)")
        out("")
        if not r["top_priorities"]:
            out("(No priority reached the non-negotiable tier, so this step was skipped.)")
            out("")
        for tp in r["top_priorities"]:
            classification = r["priority_classifications"].get(normalize(tp), {})
            out(f"- **{tp}**")
            if classification.get("not_about_projects"):
                out("  - Does not relate to particular projects.")
            else:
                for proj, cat in classification.get("projects", {}).items():
                    cat_label = {
                        "good": "Supports funding this project",
                        "bad": "Opposes funding this project",
                        "depends": "Could support, depending on other projects funded",
                        "neither": "Is irrelevant to this project",
                    }.get(cat, cat)
                    out(f"  - {proj}: {cat_label}")
        out("")

        out("**Bundle-pair validation answers**")
        out("")
        out("| Pair | Bundle A rating | Bundle B rating | Chosen | Cardinal | Lex | Consistent | Certainty | Cited priorities |")
        out("|------|------------------|------------------|--------|----------|-----|------------|-----------|-------------------|")
        rating_label = {1: "Like", 0: "Neutral", -1: "Dislike", None: "—"}
        for i, (pred, truth) in enumerate(
            zip(res["per_pair"], r["validation_answers"]), start=1
        ):
            chosen_label = "A" if truth["chosen"] == 0 else "B"
            compatible = pred["compatible_outcomes"]
            cardinal_str = compatible[0] if len(compatible) == 1 else "/".join(compatible)
            consistent_mark = "✅" if pred["is_consistent"] else "❌"
            certainty = truth.get("certainty")
            certainty_str = str(certainty) if certainty is not None else "—"
            cited = "; ".join(truth.get("cited_priorities", [])) or "—"
            a_rating = rating_label.get(truth.get("bundle_a_rating"), truth.get("bundle_a_rating"))
            b_rating = rating_label.get(truth.get("bundle_b_rating"), truth.get("bundle_b_rating"))
            out(f"| {i} | {a_rating} | {b_rating} | {chosen_label} | {cardinal_str} | {pred['lex_outcome']} | "
                f"{consistent_mark} | {certainty_str} | {cited} |")
        out("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                         help="Only run the first N personas (for a quick trial before the full 20).")
    args = parser.parse_args()

    if args.n is not None:
        PERSONAS_20 = PERSONAS_20[:args.n]

    results = run_batch()
    df = build_dataframe(results)
    suffix = f"_trial{args.n}" if args.n is not None else ""
    report_path = f"results/persona_batch_20_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}.md"
    render_report(results, df, report_path)
    print(f"\nReport written to {report_path}")
