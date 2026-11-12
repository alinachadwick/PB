"""
Accuracy breakdown for the MAIN run (type-based utilities, floor 0, unclassifiable = flat;
see experiment_priority_types.py), over all Wave-1 pairwise comparisons.

Each pair is placed in exactly one participant case, from their own 5-level happiness ratings
of the two bundles (Very Unhappy .. Very Happy):
  consistent    rating of the chosen bundle > rating of the other (ordinal and cardinal agree)
  indifferent   the two ratings are equal
  inconsistent  rating of the chosen bundle < rating of the other (rating contradicts choice)
  (no rating)   a rating is missing -- reported separately, excluded from the percentages' cases
and, within each, a model outcome from the range Delta = [Dmin, Dmax] = U(A) - U(B):
  correct      entire range on the side the participant chose
  incorrect    entire range on the other side
  indecisive   range contains 0

Writes results/accuracy_breakdown_<ts>.{md,json,png} and a LaTeX table to the .md.
No LLM calls. Run: python3 analyze_accuracy_breakdown.py
"""
import json
import os
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiment_priority_types as ept
import experiment_minimal_representation as emp
from format_wave1_results import load_all
from priority_dedup import PriorityRegistry
from run_wave0_real_validation import run_one, PRIORITY_REGISTRY_PATH, PRIORITY_SIMILARITY_THRESHOLD

LEVEL = {"Very Unhappy": 0, "Somewhat Unhappy": 1, "Neutral": 2, "Somewhat Happy": 3, "Very Happy": 4}
CASES = [("consistent", "Consistent (rating agrees with choice)"),
         ("indifferent", "Indifferent (equal ratings)"),
         ("inconsistent", "Inconsistent (rating contradicts choice)")]
OUTCOMES = ["correct", "incorrect", "indecisive"]


def collect():
    v1, sim = load_all()
    base = emp.load_cache(emp.CACHE_PATH)
    emp.v.build_utility_functions_v1 = ept.make_builder(set(ept.CLASSIFIABLE))
    reg = PriorityRegistry(path=PRIORITY_REGISTRY_PATH, threshold=PRIORITY_SIMILARITY_THRESHOLD)  # never saved
    rows = []
    for r in v1 + sim:
        run = run_one(r, cache=base, priority_registry=reg)
        res = run["result"]
        if res is None:
            continue
        for k, (pp, ans) in enumerate(zip(res["per_pair"], run["validation_answers"])):
            ra, rb = LEVEL.get(ans["rating_a_raw"]), LEVEL.get(ans["rating_b_raw"])
            chosen = ans["chosen"]  # 0 = A, 1 = B
            if ra is None or rb is None:
                case = "no_rating"
            else:
                diff = (ra - rb) * (1 if chosen == 0 else -1)  # chosen minus other
                case = "consistent" if diff > 0 else "indifferent" if diff == 0 else "inconsistent"
            dmin, dmax = pp["Delta_min"], pp["Delta_max"]
            if dmin <= 1e-9 and dmax >= -1e-9:
                outcome = "indecisive"
            else:
                outcome = "correct" if (dmin > 0) == (chosen == 0) else "incorrect"
            rows.append({"variant": r["variant"], "response_id": r["response_id"], "pair": k,
                         "case": case, "outcome": outcome, "chosen": "AB"[chosen],
                         "rating_a": ans["rating_a_raw"], "rating_b": ans["rating_b_raw"],
                         "dmin": dmin, "dmax": dmax})
    return rows


def pct(n, d):
    return 100 * n / d if d else 0.0


def make_figure(rows, path):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    # green/red (not the blue/orange used for survey versions); the red is lighter than the
    # green so they also differ in lightness, and every segment carries its % label
    colors = {"correct": "#1a8a4a", "incorrect": "#e2574c", "indecisive": "#bdbcb6"}
    ink = {"correct": "white", "incorrect": "white", "indecisive": "#0b0b0b"}
    n_all = len(rows)
    fig, ax = plt.subplots(figsize=(7.5, 2.9))
    for i, (key, label) in enumerate(reversed(CASES)):
        sub = [r for r in rows if r["case"] == key]
        n = len(sub)
        left = 0
        for o in OUTCOMES:
            w = pct(sum(r["outcome"] == o for r in sub), n)
            if w > 0:
                ax.barh(i, w, left=left, color=colors[o], edgecolor="white", linewidth=2, height=0.62,
                        label=o.capitalize() if i == 0 else None)
                if w >= 6:
                    ax.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center", color=ink[o], fontsize=9)
            left += w
        ax.text(101.5, i, f"n = {n} ({pct(n, n_all):.0f}% of pairs)", va="center", fontsize=9, color="#52514e")
    ax.set_yticks(range(len(CASES)))
    ax.set_yticklabels([lbl for _, lbl in reversed(CASES)])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the participant's pairwise comparisons in this case (%)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(title="Model prediction", ncol=3, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), fontsize=9, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    rows = collect()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(emp.RESULTS_DIR, f"accuracy_breakdown_{ts}")
    json.dump(rows, open(base + ".json", "w"), indent=1)
    make_figure([r for r in rows if r["case"] != "no_rating"], base + ".png")

    n_all = len(rows)
    c = Counter((r["case"], r["outcome"]) for r in rows)
    cases = [k for k, _ in CASES] + (["no_rating"] if any(r["case"] == "no_rating" for r in rows) else [])
    L = [f"# Accuracy breakdown, main run ({n_all} pairs, 40 respondents)", "",
         "Counts, then % of ALL pairs (rows sum to the case share). Model outcome is relative to the "
         "participant's stated choice.", "",
         "| Participant case | correct | incorrect | indecisive | total |", "|---|---|---|---|---|"]
    for k in cases:
        t = sum(c[(k, o)] for o in OUTCOMES)
        L.append(f"| {k} | " + " | ".join(f"{c[(k, o)]} ({pct(c[(k, o)], n_all):.2f}%)" for o in OUTCOMES)
                 + f" | {t} ({pct(t, n_all):.2f}%) |")
    tot = {o: sum(c[(k, o)] for k in cases) for o in OUTCOMES}
    L.append("| total | " + " | ".join(f"{tot[o]} ({pct(tot[o], n_all):.2f}%)" for o in OUTCOMES) + f" | {n_all} (100%) |")

    L += ["", "## Within each case (row %, share of that case)", "",
          "| Participant case | n | correct | incorrect | indecisive | correct / decided |", "|---|---|---|---|---|---|"]
    for k in cases:
        t = sum(c[(k, o)] for o in OUTCOMES)
        dec = c[(k, "correct")] + c[(k, "incorrect")]
        L.append(f"| {k} | {t} | " + " | ".join(f"{pct(c[(k, o)], t):.1f}%" for o in OUTCOMES)
                 + f" | {pct(c[(k, 'correct')], dec):.1f}% |")

    insp = Counter(r["response_id"] for r in rows if r["case"] == "inconsistent")
    L += ["", f"Inconsistent pairs come from {len(insp)} respondents: "
          + ", ".join(f"{k[-6:]}: {v}" for k, v in insp.most_common()), ""]

    L += ["## Decisively incorrect pairs", ""]
    for r in rows:
        if r["outcome"] == "incorrect":
            L.append(f"- {r['variant']} …{r['response_id'][-6:]} pair {r['pair']}: chose {r['chosen']}, "
                     f"A={r['rating_a']}, B={r['rating_b']}, Δ=[{r['dmin']:.2f}, {r['dmax']:.2f}], case={r['case']}")

    # LaTeX table in the same shape as tab:accuracy_breakdown
    pc = lambda k, o: f"{pct(c[(k, o)], n_all):.2f}\\%"
    names = {"consistent": "Consistent", "indifferent": "Indifferent", "inconsistent": "Inconsistent",
             "no_rating": "No rating"}
    tex = ["\\begin{table}[t]", "\\centering", "\\begin{tabular}{lccc|c}",
           "    & \\textbf{Decisively correct} & \\textbf{Decisively incorrect} & \\textbf{Indecisive} & \\textbf{Total} \\\\ \\hline"]
    for k in cases:
        t = sum(c[(k, o)] for o in OUTCOMES)
        tex.append(f"    \\textbf{{{names[k]}}} & " + " & ".join(pc(k, o) for o in OUTCOMES)
                   + f" & {pct(t, n_all):.2f}\\% \\\\")
    tex[-1] = tex[-1] + " \\hline"
    tex.append("    \\textbf{Total} & " + " & ".join(f"{pct(tot[o], n_all):.2f}\\%" for o in OUTCOMES) + " & 100\\% \\\\")
    tex += ["\\end{tabular}", "\\caption{Model predictions by participant case.}",
            "\\label{tab:accuracy_breakdown}", "\\end{table}"]
    L += ["", "## LaTeX", "", "```latex"] + tex + ["```"]

    open(base + ".md", "w").write("\n".join(L))
    print("\n".join(L), "\nwrote", base + ".md/.json/.png")


if __name__ == "__main__":
    main()
