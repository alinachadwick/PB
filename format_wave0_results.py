"""
Parse and format the real Wave-0 pilot responses (Survey1_V1 and Survey1_V1_SimProjects)
into a human-readable report, and surface data-quality findings.

Unlike run_persona_batch_20.py, this does NOT run the LLM utility-function / prediction
pipeline against real respondents -- the priority-classification piping (Q160/157/158,
Q196/197/198) never fires for real respondents (see PIPING_BUG_NOTE below), so there is
no reliable per-project classification to model against. This script formats what the
export actually captured and calls out what didn't come through.
"""
import csv
import json
import re
from datetime import datetime

RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# CSV sources
# ---------------------------------------------------------------------------
V1_FILES = [
    "data/Survey1_V1_September 1, 2026_13.29.csv",
    "data/Survey1_V1_September 1, 2026_13.30.csv",
]
SIM_FILES = [
    "data/Survey1_V1_SimProjects_September 1, 2026_13.28.csv",
    "data/Survey1_V1_SimProjects_September 1, 2026_13.28 2.csv",
]

# Internal test clicks: same lab IP (18.29.117.182), literal unsubstituted
# {{%PROLIFIC_PID%}} placeholder, abandoned within 30s at <10% progress.
# FS_7OUdwZ4nYjrEThC: real participant but bailed at 20% progress before the
# bundle-comparison section -- excluded per team direction (not counted toward Wave 0's 5).
EXCLUDE_RESPONSE_IDS = {"FS_3fT2KXEEtkaTN97", "FS_71pz9JkvlHzEyeN", "FS_7OUdwZ4nYjrEThC"}

# "Which priorities mattered in this choice" (Q93 and its siblings) is a checkbox question
# whose option labels get relabeled client-side, in the browser, to the respondent's own
# elicited-priority text -- via JS that reads Q75_selected_priorities (the same ordered list
# used to populate the importance-rating matrix) and walks the rendered DOM in ChoiceOrder,
# skipping the "Other:" row. The respondent sees their real priorities and answers correctly;
# what gets EXPORTED is the choice's static underlying label ("Click to write Choice N"), not
# the relabeled text. Because ChoiceOrder and the label text are fixed and identical across
# every one of these questions in both instruments (verified against both QSFs), "Choice N"
# maps deterministically back to position idx in the same elicitation-order priority list
# used for importance_N: idx = N-1 for N in 1..7, idx = N-2 for N in 9..20 (choice value 8
# is the relocated "Other:" row, not a real "Choice 8").
def _build_choice_label_to_idx():
    out = {f"Click to write Choice {n}": n - 1 for n in range(1, 8)}
    out.update({f"Click to write Choice {n}": n - 2 for n in range(9, 21)})
    return out


CHOICE_LABEL_TO_IDX = _build_choice_label_to_idx()

CUTOFF = "2026-08-31 15:00:00"

# ---------------------------------------------------------------------------
# Bundle-pair reference data (pulled directly from the QSFs: Survey1_V1 (20).qsf
# and Survey1_V1_SimProjects (8).qsf), keyed by the pair's DataExportTag.
# "a"/"b" are choice 1 / choice 2 as defined in the QSF (fixed, not randomized
# per-respondent as far as we can tell -- what varies is which the respondent picked).
# ---------------------------------------------------------------------------

V1_PAIRS = {
    "PN.vs.EE": {"area": "town", "q": 9,  "a": ["Slower, Safer Roads", "Security Cameras", "Addiction Support Programs", "Food Pantry"],
                 "b": ["Public Nature Preserve", "EV Chargers", "Free Summer Camp", "Food Pantry"]},
    "EN.vs.PS": {"area": "town", "q": 13, "a": ["Addiction Support Programs", "Free Hygiene Products", "Public Nature Preserve", "EV Chargers"],
                 "b": ["Slower, Safer Roads", "Free Summer Camp", "Food Pantry", "EV Chargers"]},
    "EP.vs.EN": {"area": "town", "q": 17, "a": ["Free Summer Camp", "Addiction Support Programs", "Free Hygiene Products", "Public Nature Preserve"],
                 "b": ["Security Cameras", "EV Chargers", "Slower, Safer Roads", "Public Nature Preserve"]},
    "EN.vs.EE": {"area": "town", "q": 21, "a": ["Free Hygiene Products", "Free Summer Camp", "Addiction Support Programs", "Food Pantry"],
                 "b": ["EV Chargers", "Slower, Safer Roads", "Addiction Support Programs", "Food Pantry"]},
    "ES.vs.EP": {"area": "town", "q": 25, "a": ["Free Summer Camp", "Food Pantry", "Public Nature Preserve", "EV Chargers"],
                 "b": ["Security Cameras", "Slower, Safer Roads", "Public Nature Preserve", "EV Chargers"]},
    "SH.vs.SS": {"area": "city", "q": 9,  "a": ["Parking -> Protected Bike Lane", "Security Cameras", "Addiction Support Programs", "Food Pantry"],
                 "b": ["Free Summer Camp", "Park Cleanup and Renovation", "EV Chargers", "Food Pantry"]},
    "HN.vs.ES": {"area": "city", "q": 13, "a": ["Addiction Support Programs", "Park Cleanup and Renovation", "Free Hygiene Products", "EV Chargers"],
                 "b": ["Food Pantry", "Parking -> Protected Bike Lane", "Free Summer Camp", "EV Chargers"]},
    "SE.VS.ES": {"area": "city", "q": 17, "a": ["Free Hygiene Products", "Free Summer Camp", "Addiction Support Programs", "Park Cleanup and Renovation"],
                 "b": ["Security Cameras", "EV Chargers", "Parking -> Protected Bike Lane", "Park Cleanup and Renovation"]},
    "EH.vs.SN": {"area": "city", "q": 21, "a": ["Free Hygiene Products", "Free Summer Camp", "Food Pantry", "Addiction Support Programs"],
                 "b": ["EV Chargers", "Parking -> Protected Bike Lane", "Food Pantry", "Addiction Support Programs"]},
    "HS.vs.PE": {"area": "city", "q": 25, "a": ["Food Pantry", "Free Summer Camp", "Park Cleanup and Renovation", "EV Chargers"],
                 "b": ["Security Cameras", "Parking -> Protected Bike Lane", "Park Cleanup and Renovation", "EV Chargers"]},
}

SIM_PAIRS = {
    "PN.vs.EE": {"area": "town", "q": 9,  "a": ["Public Nature Preserve", "Solar Panel Canopy", "Planting Trees", "Free Summer Camp"],
                 "b": ["Food Pantry", "Late Bus for After-School Programs", "Classroom Resources", "Free Summer Camp"]},
    "EN.vs.PS": {"area": "town", "q": 13, "a": ["Veterans Memorial Gathering Space", "Community Space Grants", "Public Nature Preserve", "Summer Movie Drive-in"],
                 "b": ["Slower, Safer Roads", "Classroom Resources", "Emergency Utility Fund", "Summer Movie Drive-in"]},
    "EP.vs.EN": {"area": "town", "q": 17, "a": ["Classroom Resources", "Late Bus for After-School Programs", "Public Nature Preserve", "Free Summer Camp"],
                 "b": ["Planting Trees", "Slower, Safer Roads", "Solar Panel Canopy", "Free Summer Camp"]},
    "EN.vs.EE": {"area": "town", "q": 21, "a": ["Public Nature Preserve", "Planting Trees", "Slower, Safer Roads", "Emergency Utility Fund"],
                 "b": ["Food Pantry", "Free Hygiene Products", "Community Space Grants", "Emergency Utility Fund"]},
    "ES.vs.EP": {"area": "town", "q": 25, "a": ["Late Bus for After-School Programs", "Food Pantry", "Emergency Utility Fund", "Free Hygiene Products"],
                 "b": ["Planting Trees", "Free Summer Camp", "Veterans Memorial Gathering Space", "Free Hygiene Products"]},
    "Q221":     {"area": "town", "q": 29, "a": ["Public Nature Preserve", "Slower, Safer Roads", "Planting Trees", "Solar Panel Canopy"],
                 "b": ["Summer Movie Drive-in", "Free Hygiene Products", "Classroom Resources", "Solar Panel Canopy"]},
    "Q220":     {"area": "town", "q": 33, "a": ["Veterans Memorial Gathering Space", "Summer Movie Drive-in", "Community Space Grants", "Free Hygiene Products"],
                 "b": ["Emergency Utility Fund", "Food Pantry", "Community Space Grants", "Free Hygiene Products"]},
    "SH.vs.SS": {"area": "city", "q": 9,  "a": ["Park Cleanup and Renovation", "Solar Panel Canopy", "Planting Trees", "Free Summer Camp"],
                 "b": ["Food Pantry", "Free Bus Passes for Students", "Classroom Resources", "Free Summer Camp"]},
    "HN.vs.ES": {"area": "city", "q": 13, "a": ["Veterans Memorial Gathering Space", "Community Space Grants", "Park Cleanup and Renovation", "Summer Movie Nights in the Park"],
                 "b": ["Parking -> Protected Bike Lane", "Classroom Resources", "Emergency Utility Fund", "Summer Movie Nights in the Park"]},
    "SE.VS.ES": {"area": "city", "q": 17, "a": ["Classroom Resources", "Free Bus Passes for Students", "Park Cleanup and Renovation", "Free Summer Camp"],
                 "b": ["Planting Trees", "Parking -> Protected Bike Lane", "Solar Panel Canopy", "Free Summer Camp"]},
    "EH.vs.SN": {"area": "city", "q": 21, "a": ["Park Cleanup and Renovation", "Planting Trees", "Parking -> Protected Bike Lane", "Emergency Utility Fund"],
                 "b": ["Food Pantry", "Free Hygiene Products", "Community Space Grants", "Emergency Utility Fund"]},
    "HS.vs.PE": {"area": "city", "q": 25, "a": ["Free Bus Passes for Students", "Food Pantry", "Emergency Utility Fund", "Free Hygiene Products"],
                 "b": ["Planting Trees", "Free Summer Camp", "Veterans Memorial Gathering Space", "Free Hygiene Products"]},
    "Q213":     {"area": "city", "q": 29, "a": ["Park Cleanup and Renovation", "Parking -> Protected Bike Lane", "Planting Trees", "Solar Panel Canopy"],
                 "b": ["Summer Movie Nights in the Park", "Free Hygiene Products", "Classroom Resources", "Solar Panel Canopy"]},
    "Q212":     {"area": "city", "q": 33, "a": ["Veterans Memorial Gathering Space", "Summer Movie Nights in the Park", "Community Space Grants", "Free Hygiene Products"],
                 "b": ["Emergency Utility Fund", "Food Pantry", "Community Space Grants", "Free Hygiene Products"]},
}

# pair_code -> (choice_col, certainty_col, cited_col, cited_text_col, rating_a_col, rating_b_col)
V1_PAIR_COLS = {
    "PN.vs.EE": ("PN.vs.EE", "Q94",  "Q93",  "Q93_11_TEXT",  "Q194_1", "Q194_2"),
    "EN.vs.PS": ("EN.vs.PS", "Q147", "Q119", "Q119_11_TEXT", "Q193_1", "Q193_2"),
    "EP.vs.EN": ("EP.vs.EN", "Q148", "Q120", "Q120_11_TEXT", "Q192_1", "Q192_2"),
    "EN.vs.EE": ("EN.vs.EE", "Q149", "Q127", "Q127_11_TEXT", "Q191_1", "Q191_2"),
    "ES.vs.EP": ("ES.vs.EP", "Q150", "Q130", "Q130_11_TEXT", "Q190_1", "Q190_2"),
    "SH.vs.SS": ("SH.vs.SS", "Q151", "Q121", "Q121_11_TEXT", "Q189_1", "Q189_2"),
    "HN.vs.ES": ("HN.vs.ES", "Q152", "Q122", "Q122_11_TEXT", "Q188_1", "Q188_2"),
    "SE.VS.ES": ("SE.VS.ES", "Q153", "Q123", "Q123_11_TEXT", "Q187_1", "Q187_2"),
    "EH.vs.SN": ("EH.vs.SN", "Q154", "Q136", "Q136_11_TEXT", "Q186_1", "Q186_2"),
    "HS.vs.PE": ("HS.vs.PE", "Q155", "Q139", "Q139_11_TEXT", "Q185_1", "Q185_2"),
}

SIM_PAIR_COLS = dict(V1_PAIR_COLS)
SIM_PAIR_COLS.update({
    "Q221": ("Q221", "Q218", "Q216", "Q216_11_TEXT", "Q214_1", "Q214_2"),
    "Q220": ("Q220", "Q219", "Q217", "Q217_11_TEXT", "Q215_1", "Q215_2"),
    "Q213": ("Q213", "Q210", "Q208", "Q208_11_TEXT", "Q206_1", "Q206_2"),
    "Q212": ("Q212", "Q211", "Q209", "Q209_11_TEXT", "Q207_1", "Q207_2"),
})

CRIT_PREFIXES = ["open-good", "Q204", "mix-good", "open-bad"]
CRIT_ORDER_KEY = {"open-good": 0, "Q204": 0, "mix-good": 1, "open-bad": 2}


def clean_html(html):
    if not html:
        return []
    html = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    html = html.replace("&nbsp;", " ").replace("&rarr;", "->").replace("&amp;", "&")
    html = re.sub(r"<br\s*/?>", "\n", html)
    html = re.sub(r"<[^>]+>", "", html)
    html = html.replace("​", "")
    lines = [re.sub(r"\s+", " ", l).strip() for l in html.split("\n") if l.strip()]
    return lines


def strip_emoji_and_ws(s):
    s = re.sub(r"[\U0001F000-\U0001FFFF☀-➿️]", "", s)
    return re.sub(r"\s+", " ", s).strip().rstrip(".")


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    fields = rows[0]
    data = rows[3:]
    idx = {name: i for i, name in enumerate(fields)}
    return fields, idx, data


def qualifying_rows(files):
    out = []
    for f in files:
        fields, idx, data = load_csv(f)
        for r in data:
            rid = r[idx["ResponseId"]]
            status = r[idx["Status"]]
            start = r[idx["StartDate"]]
            if status == "IP Address" and start >= CUTOFF and rid not in EXCLUDE_RESPONSE_IDS:
                d = {fields[i]: r[i] for i in range(len(fields))}
                d["_source_file"] = f
                out.append(d)
    return out


def get(d, col):
    return (d.get(col) or "").strip()


def extract_criteria(d):
    """Ordered list of (question_tag, text) matching the survey's elicitation order:
    open-good/Q204 (why fund) -> mix-good (project mix) -> open-bad (why not fund)."""
    items = []
    for prefix in CRIT_PREFIXES:
        for k, v in d.items():
            if k.startswith(prefix + "_") and v.strip():
                try:
                    n = int(k.split("_")[-1])
                except ValueError:
                    continue
                items.append((CRIT_ORDER_KEY[prefix], n, prefix, v.strip()))
    items.sort(key=lambda x: (x[0], x[1]))
    return [(p, t) for _, _, p, t in items]


def extract_importance(d):
    out = []
    i = 1
    while True:
        col = f"importance_{i}"
        if col not in d:
            break
        v = get(d, col)
        if v:
            out.append(v)
        i += 1
    return out


def match_choice(chosen_lines, pair_def):
    chosen_norm = {strip_emoji_and_ws(l).lower() for l in chosen_lines}

    def score(bundle):
        norm = [strip_emoji_and_ws(p).lower() for p in bundle if p != "?"]
        hits = sum(1 for p in norm if any(p in c or c in p for c in chosen_norm))
        return hits

    sa, sb = score(pair_def["a"]), score(pair_def["b"])
    if sa == 0 and sb == 0:
        return None
    return "A" if sa >= sb else "B"


def resolve_cited_priorities(cited_raw, criteria_texts):
    """Map raw 'Click to write Choice N' labels back to elicited-priority text via
    CHOICE_LABEL_TO_IDX. Returns (resolved_list, unresolved_labels)."""
    if not cited_raw:
        return [], []
    resolved, unresolved = [], []
    for label in [l.strip() for l in cited_raw.split(",") if l.strip()]:
        idx = CHOICE_LABEL_TO_IDX.get(label)
        if idx is not None and idx < len(criteria_texts):
            resolved.append(criteria_texts[idx])
        else:
            unresolved.append(label)
    return resolved, unresolved


def find_pair_records(d, pair_cols, pair_defs, criteria_texts):
    records = []
    for code, (choice_col, cert_col, cited_col, cited_text_col, ra_col, rb_col) in pair_cols.items():
        raw = get(d, choice_col)
        if not raw:
            continue
        chosen_lines = clean_html(raw)
        pair_def = pair_defs[code]
        chosen_side = match_choice(chosen_lines, pair_def)
        cited = get(d, cited_col)
        cited_other = get(d, cited_text_col)
        resolved, unresolved = resolve_cited_priorities(cited, criteria_texts)
        if cited_other:
            resolved.append(f"Other: {cited_other}")
        records.append({
            "code": code,
            "area": pair_def["area"],
            "q": pair_def["q"],
            "bundle_a": pair_def["a"],
            "bundle_b": pair_def["b"],
            "chosen_side": chosen_side,
            "chosen_lines": chosen_lines,
            "certainty": get(d, cert_col),
            "cited_raw": cited,
            "cited_priorities": resolved,
            "cited_unresolved": unresolved,
            "rating_a": get(d, ra_col),
            "rating_b": get(d, rb_col),
        })
    records.sort(key=lambda r: (r["area"], r["q"]))
    return records


def build_respondent(d, variant):
    pair_cols = V1_PAIR_COLS if variant == "V1" else SIM_PAIR_COLS
    pair_defs = V1_PAIRS if variant == "V1" else SIM_PAIRS

    criteria = extract_criteria(d)
    criteria_texts = [t for _, t in criteria]
    importance = extract_importance(d)
    n = min(len(criteria), len(importance))
    rated = [{"question": criteria[i][0], "text": criteria[i][1], "importance": importance[i]} for i in range(n)]
    extra_criteria = criteria[n:]
    extra_importance = importance[n:]

    return {
        "variant": variant,
        "response_id": d["ResponseId"],
        "source_file": d["_source_file"],
        "start": d["StartDate"],
        "end": d["EndDate"],
        "duration_s": int(d["Duration (in seconds)"]) if d.get("Duration (in seconds)") else None,
        "progress": d.get("Progress"),
        "finished": d.get("Finished"),
        "prolific_pid": get(d, "PROLIFIC_PID"),
        "heard": get(d, "heard"),
        "video_check": get(d, "Q79"),
        "area_selfreport": get(d, "urban"),
        "attention_check": get(d, "attention") or get(d, "Q203"),
        "political_affiliation": get(d, "zip") or get(d, "zip_7_TEXT"),
        "criteria": rated,
        "extra_criteria_no_rating": extra_criteria,
        "extra_importance_no_criterion": extra_importance,
        "n_criteria_elicited": len(criteria),
        "n_importance_rated": len(importance),
        "bundle_pairs": find_pair_records(d, pair_cols, pair_defs, criteria_texts),
        "feedback_rating": get(d, "Q72") or get(d, "Q205"),
        "feedback_text": get(d, "Q73"),
    }


def load_all():
    v1 = [build_respondent(d, "V1") for d in qualifying_rows(V1_FILES)]
    sim = [build_respondent(d, "SimProjects") for d in qualifying_rows(SIM_FILES)]
    return v1, sim


if __name__ == "__main__":
    v1, sim = load_all()
    print(f"V1: {len(v1)} qualifying respondents")
    print(f"SimProjects: {len(sim)} qualifying respondents")
