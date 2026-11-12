"""
Quick statistical read on a results_*.txt file: given each persona only answers
5 pairwise comparisons, what can we actually claim, per-persona vs. pooled?

Usage:
    python3 analyze_results.py [path/to/results_*.txt]   # defaults to the newest file in results/
"""

import glob
import math
import os
import re
import sys

RESULTS_DIR = "results"
N_PAIRS_PER_PERSONA = 5  # fixed by BUNDLE_PAIRS_* in validation_v1_testing.py


def latest_results_file():
    files = glob.glob(os.path.join(RESULTS_DIR, "results_*.txt"))
    if not files:
        raise SystemExit(f"No results_*.txt files found in {RESULTS_DIR}/")
    return max(files, key=os.path.getmtime)


def wilson_ci(successes, n, z=1.96):
    """95% Wilson score interval — behaves sanely at small n, unlike the normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def binom_two_sided_p(k, n, p=0.5):
    """Exact two-sided binomial test p-value against a chance baseline, no scipy needed."""
    def pmf(i):
        return math.comb(n, i) * p**i * (1 - p) ** (n - i)
    observed = pmf(k)
    eps = 1e-9
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= observed * (1 + eps)))


def parse_personas(path):
    """
    Splits a results_*.txt file into PERSONA blocks and pulls out each block's
    already-computed summary stats. Blocks missing a 'Consistency rate:' line
    are incomplete (the run was still in progress when we read the file) and
    are skipped rather than guessed at.
    """
    text = open(path).read()
    blocks = re.split(r"\n(?=PERSONA: )", text)
    personas = []
    skipped_incomplete = 0

    for block in blocks:
        if not block.startswith("PERSONA:"):
            continue

        n_pairs = len(re.findall(r"^\s*Pair \d+:", block, re.MULTILINE))
        cons_m = re.search(r"Consistency rate:\s+(\d+)%", block)
        lex_m = re.search(r"Lexicographic consistency rate:\s+(\d+)%", block)
        strong_m = re.search(r"Strong pred accuracy:\s+(\d+)%\s+\((\d+)/(\d+) strong predictions\)", block)

        if cons_m is None or lex_m is None:
            skipped_incomplete += 1
            continue

        n_total = n_pairs or N_PAIRS_PER_PERSONA
        cardinal_rate = int(cons_m.group(1)) / 100
        lex_rate = int(lex_m.group(1)) / 100

        if strong_m:
            strong_rate = int(strong_m.group(1)) / 100
            n_strong = int(strong_m.group(2))
        else:
            strong_rate, n_strong = None, 0

        personas.append({
            "label": block.split("\n", 1)[0][len("PERSONA: "):][:70],
            "n_total": n_total,
            "cardinal_correct": round(cardinal_rate * n_total),
            "lex_correct": round(lex_rate * n_total),
            "n_strong": n_strong,
            "strong_correct": round(strong_rate * n_strong) if strong_rate is not None else 0,
        })

    return personas, skipped_incomplete


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else latest_results_file()
    print(f"Analyzing: {path}\n")

    personas, skipped = parse_personas(path)
    if skipped:
        print(f"(Skipped {skipped} persona block(s) that were incomplete — run still in progress "
              f"when this file was read.)\n")
    if not personas:
        raise SystemExit("No complete persona blocks found.")

    n_personas = len(personas)
    n_pairs = sum(p["n_total"] for p in personas)
    print(f"Complete personas: {n_personas}  |  Total pairwise comparisons: {n_pairs} "
          f"({N_PAIRS_PER_PERSONA} per persona)\n")

    print("=" * 78)
    print("PER-PERSONA (n=5 each) — this is what a single respondent's results look like")
    print("=" * 78)
    for p in personas:
        lo, hi = wilson_ci(p["cardinal_correct"], p["n_total"])
        print(f"  {p['label']}...")
        print(f"    Cardinal:  {p['cardinal_correct']}/{p['n_total']}  "
              f"95% CI [{lo:.0%}, {hi:.0%}]  (width: {hi-lo:.0%})")
    print()

    print("=" * 78)
    print("POOLED ACROSS ALL COMPLETE PERSONAS")
    print("=" * 78)

    for name, key in [("Cardinal consistency", "cardinal_correct"), ("Lexicographic consistency", "lex_correct")]:
        correct = sum(p[key] for p in personas)
        lo, hi = wilson_ci(correct, n_pairs)
        pval = binom_two_sided_p(correct, n_pairs, p=0.5)
        sig = "not distinguishable from chance" if pval > 0.05 else "significantly above/below chance"
        print(f"{name}: {correct}/{n_pairs} = {correct/n_pairs:.0%}  "
              f"95% CI [{lo:.0%}, {hi:.0%}]  |  vs. 50% chance: p={pval:.3f} ({sig})")

    n_strong_total = sum(p["n_strong"] for p in personas)
    strong_correct_total = sum(p["strong_correct"] for p in personas)
    print()
    if n_strong_total:
        lo, hi = wilson_ci(strong_correct_total, n_strong_total)
        pval = binom_two_sided_p(strong_correct_total, n_strong_total, p=0.5)
        sig = "not distinguishable from chance" if pval > 0.05 else "significantly above/below chance"
        print(f"Strong-prediction accuracy (the metric that actually tests discriminative power,\n"
              f"since 'conflicted'/'indifferent' predictions count as automatically consistent\n"
              f"with either choice and inflate the two rates above):")
        print(f"  {strong_correct_total}/{n_strong_total} = {strong_correct_total/n_strong_total:.0%}  "
              f"95% CI [{lo:.0%}, {hi:.0%}]  |  vs. 50% chance: p={pval:.3f} ({sig})")
        print(f"  (Only {n_strong_total}/{n_pairs} = {n_strong_total/n_pairs:.0%} of all comparisons "
              f"produced a strong prediction at all.)")
    else:
        print("No strong predictions anywhere in this file — no discriminative-accuracy claim is possible.")

    print()
    print("=" * 78)
    print("WHAT IT WOULD TAKE: sample size needed to pin down accuracy to +-10 points (95% CI)")
    print("=" * 78)
    target_p, target_halfwidth = 0.70, 0.10
    for n_target in [5, 10, 20, 30, 50, 100, 200, 500]:
        k = round(target_p * n_target)
        lo, hi = wilson_ci(k, n_target)
        halfwidth = (hi - lo) / 2
        marker = "  <- roughly there" if halfwidth <= target_halfwidth else ""
        print(f"  n={n_target:>4} comparisons: CI half-width ~ {halfwidth:.0%}{marker}")
    print(f"\n  At {N_PAIRS_PER_PERSONA} pairs/persona, that's ~{math.ceil(100/N_PAIRS_PER_PERSONA)} "
          f"respondents needed just to get in the neighborhood of +-10 points on the POOLED rate —\n"
          f"  and that still says nothing about any individual respondent, whose own 5-pair result\n"
          f"  will always carry a wide interval no matter how many other respondents you add.")


if __name__ == "__main__":
    main()
