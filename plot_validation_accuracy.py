"""
Plots per-respondent pairwise cardinal-consistency accuracy (correct / total pairs,
from each respondent's "Consistent" column) for the most recent Wave 0 and Wave 1
real-respondent validation reports, pooled into one chart: V1 respondents on the
left half, SimProjects respondents on the right half. Within each half, Wave 0 and
Wave 1 respondents are pooled and sorted together by accuracy; wave is distinguished
by bar opacity (solid = Wave 1, lighter = Wave 0) since hue already encodes survey
type.

Reads directly from the latest results/wave{0,1}_real_validation_*.md report for
each wave (no LLM calls, no cache -- just parses the "Consistent" column of each
respondent's pairs table) so it stays cheap to re-run after any new validation run.

Run locally:
    python3 plot_validation_accuracy.py
"""
import glob
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

RESULTS_DIR = "results"

# hue carries survey type (identity); alpha carries wave (a secondary dimension)
COLOR_V1 = "#2a78d6"
COLOR_SP = "#eb6834"
ALPHA_WAVE0 = 0.5
ALPHA_WAVE1 = 1.0

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#dddddd"
BASELINE = "#999999"
SURFACE = "#ffffff"

V1_PAIRS = 5
SP_PAIRS = 7


def latest_report(wave):
    pattern = os.path.join(RESULTS_DIR, f"wave{wave}_real_validation_*.md")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match {pattern}")
    return matches[-1]


def parse_report(path, wave):
    """Extracts per-respondent (survey_type, correct, total) from a real_validation
    report's per-respondent pairs tables, using each table's own 'Consistent' column."""
    text = open(path).read()
    sections = re.split(r"^## ", text, flags=re.M)[1:]
    survey_type = None
    respondents = []
    for part in sections:
        header_line, body = part.split("\n", 1)
        header_line = header_line.strip()
        if "aggregate" in header_line:
            name = header_line.split("—")[0].strip()
            survey_type = "SimProjects" if "SimProjects" in name else "V1"
            continue
        m = re.match(r"([^\s—]+)\s*—\s*(\S+)", header_line)
        if not m:
            continue
        resp_id, region = m.group(1), m.group(2)
        table_match = re.search(r"\|\s*Pair\s*\|.*\n\|[-\s|]+\n((?:\|.*\n?)+)", body)
        header_match = re.search(r"\|\s*Pair\s*\|.*", body)
        if not table_match or not header_match:
            continue
        cols = [c.strip() for c in header_match.group(0).strip().strip("|").split("|")]
        if "Consistent" not in cols:
            continue
        consistent_idx = cols.index("Consistent")
        correct = total = 0
        for row in table_match.group(1).strip().split("\n"):
            if not row.strip().startswith("|"):
                break
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if len(cells) <= consistent_idx or cells[consistent_idx] not in ("True", "False"):
                continue
            total += 1
            correct += cells[consistent_idx] == "True"
        if total == 0:
            continue
        respondents.append({
            "id": resp_id, "region": region, "wave": wave, "survey_type": survey_type,
            "correct": correct, "total": total, "accuracy": correct / total,
        })
    return respondents


def draw_half(ax, respondents, color, x0, bar_w, gap):
    """Draws one half (all respondents for one survey type, both waves pooled and
    sorted descending by accuracy). Returns (x_center, mean_accuracy, x_end)."""
    ordered = sorted(respondents, key=lambda r: r["accuracy"], reverse=True)
    xs = x0 + np.arange(len(ordered)) * (bar_w + gap)
    for x, r in zip(xs, ordered):
        alpha = ALPHA_WAVE1 if r["wave"] == 1 else ALPHA_WAVE0
        ax.bar(x, r["accuracy"], width=bar_w, color=color, alpha=alpha,
               edgecolor="none", zorder=3)
    mean_acc = np.mean([r["accuracy"] for r in ordered]) if ordered else 0
    x_end = xs[-1] + bar_w if len(xs) else x0
    x_center = (x0 + x_end) / 2
    return x_center, mean_acc, x0, x_end


def main(respondents=None, title="Respondent accuracy, V1 vs. SimProjects -- Wave 0 + Wave 1 combined",
         out_prefix="validation_accuracy"):
    """respondents: optional list of {"survey_type","wave","correct","total","accuracy"} dicts
    (e.g. from a fresh model run) in place of the latest wave 0/1 markdown reports."""
    if respondents is None:
        reports = {0: latest_report(0), 1: latest_report(1)}
        respondents = []
        for wave, path in reports.items():
            respondents += parse_report(path, wave)
    waves_present = {r["wave"] for r in respondents}

    v1 = [r for r in respondents if r["survey_type"] == "V1"]
    sp = [r for r in respondents if r["survey_type"] == "SimProjects"]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    bar_w, gap, group_gap = 0.8, 0.35, 3.0
    v1_center, v1_mean, v1_x0, v1_x1 = draw_half(ax, v1, COLOR_V1, 0, bar_w, gap)
    sp_x0 = v1_x1 + group_gap
    sp_center, sp_mean, sp_x0, sp_x1 = draw_half(ax, sp, COLOR_SP, sp_x0, bar_w, gap)

    for y in (0, 0.25, 0.5, 0.75, 1.0):
        ax.axhline(y, color=GRIDLINE, linewidth=1, zorder=1)
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=2)

    divider_x = (v1_x1 + sp_x0) / 2
    ax.axvline(divider_x, color=BASELINE, linewidth=1, linestyle=(0, (3, 3)), zorder=2)

    ax.hlines(v1_mean, v1_x0, v1_x1, color=COLOR_V1, linewidth=1.5,
              linestyle=(0, (4, 3)), zorder=4)
    ax.hlines(sp_mean, sp_x0, sp_x1, color=COLOR_SP, linewidth=1.5,
              linestyle=(0, (4, 3)), zorder=4)

    ax.text(v1_center, -0.09, f"V1  (n={len(v1)})", ha="center", va="top",
            fontsize=11, fontweight="bold", color=TEXT_SECONDARY)
    ax.text(sp_center, -0.09, f"SimProjects  (n={len(sp)})", ha="center", va="top",
            fontsize=11, fontweight="bold", color=TEXT_SECONDARY)

    ax.set_ylim(0, 1.08)
    ax.set_xlim(v1_x0 - 1, sp_x1 + 1)

    # left axis: # correct out of 5 (V1's pair count); right axis: # correct out of 7
    # (SimProjects' pair count). Both index the same 0-1 accuracy scale, just with
    # each half's own denominator, matching how each half's bars were actually scored.
    ax.set_yticks([k / V1_PAIRS for k in range(V1_PAIRS + 1)])
    ax.set_yticklabels([str(k) for k in range(V1_PAIRS + 1)], color=COLOR_V1, fontsize=9)
    ax.set_ylabel(f"# correct (of {V1_PAIRS}) -- V1", color=COLOR_V1, fontsize=10)

    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks([k / SP_PAIRS for k in range(SP_PAIRS + 1)])
    ax_right.set_yticklabels([str(k) for k in range(SP_PAIRS + 1)], color=COLOR_SP, fontsize=9)
    ax_right.set_ylabel(f"# correct (of {SP_PAIRS}) -- SimProjects", color=COLOR_SP, fontsize=10)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["bottom"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.tick_params(length=0)

    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_V1, alpha=ALPHA_WAVE1, label="V1 -- Wave 1"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_V1, alpha=ALPHA_WAVE0, label="V1 -- Wave 0"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_SP, alpha=ALPHA_WAVE1, label="SimProjects -- Wave 1"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_SP, alpha=ALPHA_WAVE0, label="SimProjects -- Wave 0"),
    ]
    if 0 not in waves_present:
        legend_handles = [h for h in legend_handles if "Wave 0" not in h.get_label()]
    ax.set_title(
        title,
        fontsize=14, fontweight="bold", color=TEXT_PRIMARY, pad=42, loc="left",
    )

    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.1),
              ncol=len(legend_handles), frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
              handlelength=1.2, handleheight=1.2, columnspacing=1.4)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{out_prefix}_{ts}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", facecolor=SURFACE)
    print(f"Wrote {path}")
    print(f"  V1 (n={len(v1)}): mean {v1_mean:.0%}")
    print(f"  SimProjects (n={len(sp)}): mean {sp_mean:.0%}")


if __name__ == "__main__":
    main()
