"""
Wave-1 counterpart to format_wave0_results.py -- same parsing logic (Qualtrics 3-row
header, bundle-pair/project reference data, cited-priorities resolution via
CHOICE_LABEL_TO_IDX, etc.), reused directly from that module rather than duplicated,
since Wave 1 is the same survey instrument in the same export format.

The one thing that's genuinely different from Wave 0: Wave 0 didn't have a clean
authoritative "is this a real, approved participant" signal, so it used ad hoc
heuristics (a hardcoded time cutoff, hand-picked exclusion of specific internal-test
response IDs). Wave 1 has Prolific's own demographic-export files, each with a `Status`
column -- per instruction, only respondents whose Prolific submission Status is
APPROVED should be reported here. That's a strictly better and more authoritative
filter than Wave 0's heuristics, and it has a useful side effect: internal test clicks
(which use the literal unsubstituted `{{%PROLIFIC_PID%}}` placeholder, or no PID at
all) can never match an approved Prolific participant ID, so they're excluded for free
without needing any special-cased ID list.

Deliberately built without inspecting data/wave1v1.csv or data/wave1-simprojects.csv --
only the demographic-export files (also in data/) were read, to determine the Status
schema and the join key (Prolific "Participant id" == the survey's PROLIFIC_PID column).
If Wave 1's export column layout has changed from Wave 0's in some way not yet known,
the parsing calls below (imported from format_wave0_results) are the first place to
check.
"""
import csv
import glob
import re

from format_wave0_results import (
    RESULTS_DIR,
    build_respondent,
    load_csv,
)

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
V1_FILE = "data/wave1v1.csv"
SIM_FILE = "data/wave1-simprojects.csv"

# Every Prolific study that fed either survey needs its own demographic export for the
# APPROVED filter to see those respondents at all -- V1 and SimProjects were recruited
# under entirely different sets of study IDs (confirmed by checking STUDY_ID values
# directly), so don't assume a fixed count or a 1:1 file-to-variant mapping. Globbed
# rather than hardcoded so dropping in a newly-exported file (e.g. after adding a study)
# just works without editing this file again. Membership is still checked as a union
# across all of them against each survey's own PROLIFIC_PID column, since a participant
# only ever appears in one study.
DEMOGRAPHIC_EXPORT_GLOB = "data/prolific_demographic_export_*.csv"


def load_approved_participant_ids():
    approved = set()
    for path in sorted(glob.glob(DEMOGRAPHIC_EXPORT_GLOB)):
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if (row.get("Status") or "").strip().upper() == "APPROVED":
                    pid = (row.get("Participant id") or "").strip()
                    if pid:
                        approved.add(pid)
    return approved


# Curated subset of Prolific's demographic-export columns worth surfacing in reports --
# friendly key -> exact CSV column name.
DEMOGRAPHIC_FIELDS = {
    "location": "Current u.s state of residence",
    "age": "Age",
    "sex": "Sex",
    "political_affiliation": "U.s. political affiliation",
    "socioeconomic_status": "Socioeconomic status",
    "education": "Highest education level completed",
    "ethnicity": "Ethnicity simplified",
    "employment_status": "Employment status",
    "student_status": "Student status",
    "country_of_residence": "Country of residence",
}


def load_demographics_by_pid():
    """{participant_id: {friendly_key: value, ...}} across every demographic export in
    data/. Indexes every row regardless of Status -- harmless, since only APPROVED
    participants end up surviving qualifying_rows()'s filter anyway."""
    by_pid = {}
    for path in sorted(glob.glob(DEMOGRAPHIC_EXPORT_GLOB)):
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                pid = (row.get("Participant id") or "").strip()
                if not pid:
                    continue
                by_pid[pid] = {
                    key: (row.get(col) or "").strip()
                    for key, col in DEMOGRAPHIC_FIELDS.items()
                }
    return by_pid


def known_study_ids():
    """Study IDs we actually have a demographic export for, parsed from the export
    filenames themselves (that's where the study ID lives -- these files don't carry
    a Study ID column of their own)."""
    ids = set()
    for path in glob.glob(DEMOGRAPHIC_EXPORT_GLOB):
        m = re.search(r"prolific_demographic_export_([0-9a-f]+)\.csv$", path)
        if m:
            ids.add(m.group(1))
    return ids


def warn_uncovered_studies(path):
    """Flags study IDs that have real (non-placeholder) respondents in this survey file
    but no corresponding demographic export in data/ -- those respondents can never be
    counted as APPROVED no matter their actual status, since we have no Status for them
    at all. This is exactly the SimProjects situation from Sept 2026: two exports were
    provided but covered only the studies feeding wave1v1.csv, leaving
    wave1-simprojects.csv's three studies with zero visibility."""
    fields, idx, data = load_csv(path)
    if "STUDY_ID" not in idx or "PROLIFIC_PID" not in idx:
        return
    known = known_study_ids()
    counts = {}
    for r in data:
        pid = r[idx["PROLIFIC_PID"]].strip()
        sid = r[idx["STUDY_ID"]].strip()
        if pid and pid != "{{%PROLIFIC_PID%}}" and sid and sid != "{{%STUDY_ID%}}":
            counts[sid] = counts.get(sid, 0) + 1
    missing = {sid: n for sid, n in counts.items() if sid not in known}
    if missing:
        print(
            f"WARNING: {path} has real respondents under stud{'y' if len(missing)==1 else 'ies'} "
            f"with no matching data/prolific_demographic_export_*.csv -- these can never "
            f"be counted as APPROVED regardless of their actual status: {missing} "
            f"(counts = rows under that study ID). Export that study's demographic data "
            f"from Prolific and drop it into data/ to fix."
        )


def qualifying_rows(path, approved_ids):
    warn_uncovered_studies(path)
    fields, idx, data = load_csv(path)
    if "PROLIFIC_PID" not in idx:
        raise KeyError(
            f"{path}: no PROLIFIC_PID column found -- export format may have changed "
            f"from Wave 0's; check the header row before trusting this filter."
        )
    out = []
    for r in data:
        pid = r[idx["PROLIFIC_PID"]].strip()
        if pid and pid in approved_ids:
            d = {fields[i]: r[i] for i in range(len(fields))}
            d["_source_file"] = path
            out.append(d)
    return out


def _build_with_demographics(rows, variant, demographics_by_pid):
    out = []
    for d in rows:
        resp = build_respondent(d, variant)
        resp["demographics"] = demographics_by_pid.get(resp["prolific_pid"], {})
        out.append(resp)
    return out


def load_all():
    approved = load_approved_participant_ids()
    demographics_by_pid = load_demographics_by_pid()
    v1 = _build_with_demographics(qualifying_rows(V1_FILE, approved), "V1", demographics_by_pid)
    sim = _build_with_demographics(qualifying_rows(SIM_FILE, approved), "SimProjects", demographics_by_pid)
    return v1, sim


if __name__ == "__main__":
    approved = load_approved_participant_ids()
    print(f"{len(approved)} APPROVED participant IDs across both demographic exports")
    v1, sim = load_all()
    print(f"V1: {len(v1)} APPROVED respondents")
    print(f"SimProjects: {len(sim)} APPROVED respondents")
