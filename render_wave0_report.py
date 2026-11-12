"""
Render the parsed real Wave-0 pilot responses (from format_wave0_results.py) into a
human-readable Markdown report for the team, plus the data-quality findings that came
out of parsing them.
"""
from collections import Counter
from format_wave0_results import load_all, RESULTS_DIR

LIKERT_SHORT = {
    "I'd want this even if it meant giving up other priorities": "Most important",
    "Important, but I'd trade it away for more important ones": "Trade away",
    "Nice to have": "Nice to have",
    "This isn't really important to me": "Not important",
}


def fmt_bundle(lines):
    return "; ".join(lines)


def area_label(r):
    return r["area_selfreport"] or "?"


def duration_fmt(secs):
    if secs is None:
        return "?"
    m, s = divmod(secs, 60)
    return f"{m}m{s:02d}s"


def render_respondent(out, r, i):
    out.append(f"### {r['variant']} respondent {i} — {area_label(r)}, {r['political_affiliation'] or 'affiliation not given'}\n")
    out.append(f"*Response ID: `{r['response_id']}` · Prolific PID: `{r['prolific_pid'] or '?'}` "
                f"· started {r['start']} · duration {duration_fmt(r['duration_s'])} "
                f"· progress {r['progress']}% · finished: {r['finished']}*\n")
    out.append(f"- PB experience: {r['heard'] or '?'}")
    out.append(f"- Video comprehension check: {r['video_check'] or '?'}")
    out.append(f"- Budget-constraint attention check: {r['attention_check'] or '?'}")
    out.append("")

    out.append(f"**Elicited priorities & importance ratings** ({r['n_criteria_elicited']} elicited, "
                f"{r['n_importance_rated']} rated)\n")
    if r["criteria"]:
        out.append("| # | Priority | Importance |")
        out.append("|---|---|---|")
        for j, c in enumerate(r["criteria"], 1):
            level = LIKERT_SHORT.get(c["importance"], c["importance"])
            text = c["text"].replace("|", "/").replace("\n", " ")
            out.append(f"| {j} | {text} | {level} |")
        out.append("")
    else:
        out.append("*(none captured — respondent left before this section)*\n")
    if r["extra_criteria_no_rating"]:
        out.append(f"*{len(r['extra_criteria_no_rating'])} additional elicited item(s) had no matching importance rating "
                    f"(export ran out of importance_N slots or the respondent dropped off before rating them).*\n")
    if r["extra_importance_no_criterion"]:
        out.append(f"*{len(r['extra_importance_no_criterion'])} importance rating(s) have no matching elicited-criterion text "
                    f"— likely a priority typed via the survey's \"Add another priority\" box, which isn't captured "
                    f"in the export (see piping finding below).*\n")

    if r["bundle_pairs"]:
        out.append("**Bundle-pair comparisons**\n")
        out.append("| Pair | Chosen | Certainty | Reaction to A | Reaction to B | Cited priorities |")
        out.append("|---|---|---|---|---|---|")
        for bp in r["bundle_pairs"]:
            chosen = f"**{bp['chosen_side']}**" if bp["chosen_side"] else "?"
            cited = "; ".join(bp["cited_priorities"]) if bp["cited_priorities"] else "—"
            cited = cited.replace("|", "/")
            if bp["cited_unresolved"]:
                cited += f" *(+{len(bp['cited_unresolved'])} unresolved)*"
            out.append(f"| {bp['code']} | {chosen} | {bp['certainty'] or '?'} | {bp['rating_a'] or '?'} | {bp['rating_b'] or '?'} | {cited} |")
        out.append("")
        for bp in r["bundle_pairs"]:
            out.append(f"- `{bp['code']}` — **A:** {fmt_bundle(bp['bundle_a'])}  ·  **B:** {fmt_bundle(bp['bundle_b'])}")
        out.append("")
    else:
        out.append("*(bundle-comparison section not reached)*\n")

    if r["feedback_rating"] or r["feedback_text"]:
        out.append(f"**Feedback:** rated the survey \"{r['feedback_rating'] or '?'}\" at letting them express their views."
                    + (f" Comment: \"{r['feedback_text']}\"" if r["feedback_text"] else ""))
        out.append("")
    out.append("---\n")


def aggregate_section(out, label, rows):
    out.append(f"### {label} — aggregate ({len(rows)} respondents)\n")
    if not rows:
        out.append("*No qualifying respondents.*\n")
        return
    finished = sum(1 for r in rows if r["finished"] == "True")
    durs = [r["duration_s"] for r in rows if r["duration_s"] and r["finished"] == "True"]
    out.append(f"- Finished: {finished}/{len(rows)}")
    if durs:
        out.append(f"- Median completion time (finished only): {duration_fmt(sorted(durs)[len(durs)//2])} "
                    f"(range {duration_fmt(min(durs))}–{duration_fmt(max(durs))})")
    areas = Counter(area_label(r) for r in rows)
    out.append(f"- Area self-report: {dict(areas)}")
    parties = Counter(r["political_affiliation"] or "(not reached)" for r in rows)
    out.append(f"- Political affiliation: {dict(parties)}")
    heard = Counter(r["heard"] or "(not reached)" for r in rows)
    out.append(f"- PB familiarity: {dict(heard)}")
    att_ok = sum(1 for r in rows if r["attention_check"] == "They can fund any set of four projects")
    out.append(f"- Attention check answered correctly (\"any set of four projects\"): {att_ok}/{len(rows)}")
    n_pairs = sum(len(r["bundle_pairs"]) for r in rows)
    certs = Counter(bp["certainty"] for r in rows for bp in r["bundle_pairs"] if bp["certainty"])
    out.append(f"- Bundle-pair comparisons answered: {n_pairs} across {len(rows)} respondents")
    out.append(f"- Certainty distribution: {dict(certs)}")
    fb = Counter(r["feedback_rating"] for r in rows if r["feedback_rating"])
    out.append(f"- \"How well did the survey let you express your views\": {dict(fb)}")
    out.append("")


def build_report():
    v1, sim = load_all()
    out = []
    out.append("# Wave 0 Pilot Results — Real Respondents\n")
    out.append("Filtered to genuine Prolific responses (`Status = IP Address`, valid `PROLIFIC_PID`) "
                "started after 2026-08-31 15:00. Two additional \"IP Address\" rows were excluded as internal "
                "test clicks — same lab IP address (`18.29.117.182`), an unsubstituted `{{%PROLIFIC_PID%}}` "
                "placeholder, and abandoned within 30 seconds at <10% progress.\n")
    out.append(f"- **Survey1_V1**: {len(v1)} qualifying respondents")
    out.append(f"- **Survey1_V1_SimProjects**: {len(sim)} qualifying respondents\n")
    out.append("A 6th V1 \"IP Address\" row (`FS_7OUdwZ4nYjrEThC`, real Prolific PID, real external IP, started "
                "post-3PM) bailed at 20% progress before the bundle-comparison section and is excluded here per "
                "team direction — it's not counted toward the 5. The 2 remaining `FS_...` partials (94%, 94% "
                "progress) are kept: they have valid, distinct Prolific PIDs and reached the bundle-comparison "
                "section, so they read as genuine participants who didn't finish rather than test noise.\n")

    out.append("## Data-quality notes on the export\n")
    out.append(
        "- **\"Which priorities mattered in this choice\" is recoverable, just not from the raw export text.** "
        "Q93 and its siblings show raw placeholder labels (`Click to write Choice N`) in the CSV, but respondents "
        "themselves saw their own real priority text — it's relabeled live in the browser by JS that reads the same "
        "ordered priority list used for the importance-rating matrix. The mapping from `Choice N` back to that list "
        "is fixed and deterministic (verified identical across all 10 V1 tags, all 4 SimProjects-only tags, and both "
        "instruments), so it's been reconstructed here rather than reported as missing. \"Other:\" free-text answers "
        "are included directly.\n"
    )
    out.append(
        "- **Priority-classification questions (Q160/157/158, gates Q196/197/198) and the drag-and-drop ranking "
        "(Q161) don't appear anywhere in this data** — per direction from the team, these were intentionally removed "
        "from the current instrument, not a bug. The embedded-data fields that used to feed them "
        "(`ALL_PRIORITIES`, `priority_1`–`7`, `top_priority_1/2/3`, `rank_candidates`, `needs_rank`) are correspondingly "
        "blank/frozen-at-default for every respondent, which is expected now rather than something to fix. "
        "**Practical upshot: none of the statistical methods that assumed per-project priority classification "
        "(the lexicographic model as previously spec'd, and likely Bradley-Terry/rationalizability as scoped) can "
        "run against this input as-is — flagged separately for a methods discussion.**\n"
    )

    out.append("## Aggregate summary\n")
    aggregate_section(out, "Survey1_V1", v1)
    aggregate_section(out, "Survey1_V1_SimProjects", sim)

    out.append("## Per-respondent detail — Survey1_V1\n")
    for i, r in enumerate(v1, 1):
        render_respondent(out, r, i)

    out.append("## Per-respondent detail — Survey1_V1_SimProjects\n")
    for i, r in enumerate(sim, 1):
        render_respondent(out, r, i)

    return "\n".join(out)


if __name__ == "__main__":
    import os
    from datetime import datetime
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = build_report()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"wave0_pilot_report_{ts}.md")
    with open(path, "w") as f:
        f.write(report)
    print("wrote", path)
