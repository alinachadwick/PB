
import json
import os
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import requests
import numpy as np
from scipy.optimize import linprog
from dotenv import load_dotenv
from validation_v2_helpers import *

load_dotenv()

# ---------------------------------------------------------------------------
# Projects & Priorities
# ---------------------------------------------------------------------------

# Real project pools from the currently-deployed Qualtrics survey
# (Survey1_V1 (20).qsf, pulled Sept 2026 -- Q182/Q183 choice lists). Every
# respondent sees BOTH pools during elicitation; only the validation-phase
# bundle comparisons are filtered to the respondent's own pool, gated on a
# single condition in the Survey Flow: QID6 ("urban/suburban/rural") ==
# "Urban" selected vs not-selected. Suburban and Rural are bucketed together
# (PROJECTS_RURAL / "town" pairs) and are distinct from Urban
# (PROJECTS_URBAN / "city" pairs) -- confirmed both in the QSF's branch logic
# and empirically in every real Wave-0 respondent (see run_validation's area
# check below). 6 of the 8 projects are shared between the two pools; only
# the last 2 differ.
PROJECTS_RURAL = {
    "Food Pantry": "Stock and staff a food pantry that provides free groceries to food-insecure households.",
    "Free Hygiene Products": "Provide free menstrual products, diapers, baby wipes, and other hygiene supplies in public buildings for three years.",
    "Free Summer Camp": "Fund a free one-week summer camp for all K-12 students.",
    "Addiction Support Programs": "Fund free counseling and support programs for residents facing substance abuse and addiction.",
    "EV Chargers": "Install public EV charging stations in municipal parking areas.",
    "Security Cameras": "Install security cameras in several high-traffic public areas to help deter and investigate crime.",
    "Public Nature Preserve": "Set aside and maintain public land as a protected nature preserve open for hiking and wildlife.",
    "Slower, Safer Roads": "Add traffic-calming measures -- speed bumps, narrower lanes, crosswalk improvements -- to slow traffic on residential streets.",
}

PROJECTS_URBAN = {
    "Food Pantry": "Stock and staff a food pantry that provides free groceries to food-insecure households.",
    "Free Hygiene Products": "Provide free menstrual products, diapers, baby wipes, and other hygiene supplies in public buildings for three years.",
    "Free Summer Camp": "Fund a free one-week summer camp for all K-12 students.",
    "Addiction Support Programs": "Fund free counseling and support programs for residents facing substance abuse and addiction.",
    "EV Chargers": "Install public EV charging stations in municipal parking areas.",
    "Security Cameras": "Install security cameras in several high-traffic public areas to help deter and investigate crime.",
    "Park Cleanup and Renovation": "Clean up and renovate a public park by upgrading playground equipment and installing benches.",
    "Protected Bike Lane": "Convert street parking on a busy road into a protected bike lane.",
}

# Elicitation shows respondents BOTH pools combined, regardless of area.
PROJECTS_ALL = {**PROJECTS_RURAL, **PROJECTS_URBAN}

PRIORITIES_URBAN = {
    "group": [
        "Women, parents and caregivers",
        "Families with babies or young children",
        "Unhoused and low-income residents",
        "Pedestrians",
        "Seniors and People with Disabilities",
        "Users of parks and public spaces",
    ],
    "impact": [
        "Making sure everyone's basic needs are met",
        "Public health and safety",
        "Environment and sustainability",
        "Public space and community amenities",
        "Animal welfare / pet support",
        "Sanitation and cleanliness",
    ],
}

PRIORITIES_RURAL = {
    "group": [
        "Children and Students",
        "Low-Income Residents",
        "Seniors",
        "Local workers and Job Seekers",
        "Users of Parks and Public Spaces",
    ],
    "impact": [
        "Education and pre-professional development",
        "Environment and sustainability",
        "Making sure everyone's basic needs are met",
        "Health and wellness",
    ],
}

PROJECT_PRIORITY_SCORES_RURAL = {
    "Classroom Resources": {
        "Children and Students": 3, "Low-Income Residents": 1, "Seniors": 0,
        "Local workers and Job Seekers": 0, "Users of Parks and Public Spaces": 0,
        "Education and pre-professional development": 3, "Environment and sustainability": 0,
        "Making sure everyone's basic needs are met": 1, "Health and wellness": 0,
    },
    "Pollination of Green Space": {
        "Children and Students": 0, "Low-Income Residents": 0, "Seniors": 0,
        "Local workers and Job Seekers": 0, "Users of Parks and Public Spaces": 3,
        "Education and pre-professional development": 0, "Environment and sustainability": 3,
        "Making sure everyone's basic needs are met": 0, "Health and wellness": 1,
    },
    "Food Fridges": {
        "Children and Students": 1, "Low-Income Residents": 3, "Seniors": 1,
        "Local workers and Job Seekers": 0, "Users of Parks and Public Spaces": 0,
        "Education and pre-professional development": 0, "Environment and sustainability": 0,
        "Making sure everyone's basic needs are met": 3, "Health and wellness": 2,
    },
    "Food Bank": {
        "Children and Students": 1, "Low-Income Residents": 3, "Seniors": 2,
        "Local workers and Job Seekers": 0, "Users of Parks and Public Spaces": 0,
        "Education and pre-professional development": 1, "Environment and sustainability": 0,
        "Making sure everyone's basic needs are met": 3, "Health and wellness": 2,
    },
    "Manufacturing Internships": {
        "Children and Students": 2, "Low-Income Residents": 2, "Seniors": 0,
        "Local workers and Job Seekers": 3, "Users of Parks and Public Spaces": 0,
        "Education and pre-professional development": 3, "Environment and sustainability": 0,
        "Making sure everyone's basic needs are met": 1, "Health and wellness": 0,
    },
    "Farmers Market": {
        "Children and Students": 1, "Low-Income Residents": 1, "Seniors": 3,
        "Local workers and Job Seekers": 1, "Users of Parks and Public Spaces": 2,
        "Education and pre-professional development": 0, "Environment and sustainability": 1,
        "Making sure everyone's basic needs are met": 2, "Health and wellness": 2,
    },
    "Finance Seminars and Homeownership Classes": {
        "Children and Students": 3, "Low-Income Residents": 2, "Seniors": 0,
        "Local workers and Job Seekers": 2, "Users of Parks and Public Spaces": 0,
        "Education and pre-professional development": 3, "Environment and sustainability": 0,
        "Making sure everyone's basic needs are met": 1, "Health and wellness": 0,
    },
    "Rewild Pollinator Meadows": {
        "Children and Students": 0, "Low-Income Residents": 0, "Seniors": 0,
        "Local workers and Job Seekers": 0, "Users of Parks and Public Spaces": 3,
        "Education and pre-professional development": 0, "Environment and sustainability": 3,
        "Making sure everyone's basic needs are met": 0, "Health and wellness": 1,
    },
}

PROJECT_PRIORITY_SCORES_URBAN = {
    "Free Hygiene Products": {
        "Women, parents and caregivers": 3, "Families with babies or young children": 3,
        "Unhoused and low-income residents": 2, "Pedestrians": 0,
        "Seniors and People with Disabilities": 1, "Users of parks and public spaces": 0,
        "Making sure everyone's basic needs are met": 3, "Public health and safety": 2,
        "Environment and sustainability": 0, "Public space and community amenities": 0,
        "Animal welfare / pet support": 0, "Sanitation and cleanliness": 1,
    },
    "Supplies for Unhoused Residents": {
        "Women, parents and caregivers": 1, "Families with babies or young children": 0,
        "Unhoused and low-income residents": 3, "Pedestrians": 0,
        "Seniors and People with Disabilities": 1, "Users of parks and public spaces": 0,
        "Making sure everyone's basic needs are met": 3, "Public health and safety": 1,
        "Environment and sustainability": 0, "Public space and community amenities": 0,
        "Animal welfare / pet support": 0, "Sanitation and cleanliness": 1,
    },
    "More Space for Street Trees": {
        "Women, parents and caregivers": 0, "Families with babies or young children": 1,
        "Unhoused and low-income residents": 0, "Pedestrians": 2,
        "Seniors and People with Disabilities": 1, "Users of parks and public spaces": 3,
        "Making sure everyone's basic needs are met": 0, "Public health and safety": 1,
        "Environment and sustainability": 3, "Public space and community amenities": 2,
        "Animal welfare / pet support": 0, "Sanitation and cleanliness": 0,
    },
    "Safer Crosswalks": {
        "Women, parents and caregivers": 1, "Families with babies or young children": 2,
        "Unhoused and low-income residents": 1, "Pedestrians": 3,
        "Seniors and People with Disabilities": 3, "Users of parks and public spaces": 1,
        "Making sure everyone's basic needs are met": 0, "Public health and safety": 3,
        "Environment and sustainability": 0, "Public space and community amenities": 1,
        "Animal welfare / pet support": 0, "Sanitation and cleanliness": 0,
    },
    "Rat Control Smart Traps": {
        "Women, parents and caregivers": 1, "Families with babies or young children": 1,
        "Unhoused and low-income residents": 1, "Pedestrians": 1,
        "Seniors and People with Disabilities": 1, "Users of parks and public spaces": 1,
        "Making sure everyone's basic needs are met": 0, "Public health and safety": 2,
        "Environment and sustainability": 1, "Public space and community amenities": 0,
        "Animal welfare / pet support": 0, "Sanitation and cleanliness": 3,
    },
    "Better Trash and Recycling Bins": {
        "Women, parents and caregivers": 0, "Families with babies or young children": 0,
        "Unhoused and low-income residents": 0, "Pedestrians": 1,
        "Seniors and People with Disabilities": 1, "Users of parks and public spaces": 1,
        "Making sure everyone's basic needs are met": 0, "Public health and safety": 1,
        "Environment and sustainability": 2, "Public space and community amenities": 1,
        "Animal welfare / pet support": 0, "Sanitation and cleanliness": 3,
    },
    "Low-Cost Veterinary Care": {
        "Women, parents and caregivers": 1, "Families with babies or young children": 1,
        "Unhoused and low-income residents": 2, "Pedestrians": 0,
        "Seniors and People with Disabilities": 3, "Users of parks and public spaces": 0,
        "Making sure everyone's basic needs are met": 1, "Public health and safety": 2,
        "Environment and sustainability": 0, "Public space and community amenities": 0,
        "Animal welfare / pet support": 3, "Sanitation and cleanliness": 0,
    },
    "Public EV Chargers": {
        "Women, parents and caregivers": 0, "Families with babies or young children": 0,
        "Unhoused and low-income residents": 0, "Pedestrians": 0,
        "Seniors and People with Disabilities": 0, "Users of parks and public spaces": 1,
        "Making sure everyone's basic needs are met": 0, "Public health and safety": 0,
        "Environment and sustainability": 3, "Public space and community amenities": 2,
        "Animal welfare / pet support": 0, "Sanitation and cleanliness": 0,
    },
}

# Exact wording from the deployed survey's importance question (QID69, tag
# "importance"). Tier 1 is explicitly framed as non-negotiable ("even if it
# meant giving up other priorities") — a lexicographic/must-have framing, not
# just "very important" — which is why it gets its own H tier below rather
# than being blended with tier 2.
LIKERT_MOST = "I'd want this even if it meant giving up other priorities"
LIKERT_TRADE_AWAY = "Important, but I'd trade it away for more important ones"
LIKERT_NICE = "Nice to have"
LIKERT_NOT = "This isn't really important to me"

LIKERT_TO_OMEGA = {
    LIKERT_MOST:       (0.75, 1.0),
    LIKERT_TRADE_AWAY: (0.5, 0.75),
    LIKERT_NICE:       (0.25, 0.5),
    LIKERT_NOT:        (0.0, 0.25),
}

LIKERT_ORDER = {
    LIKERT_MOST: 4,
    LIKERT_TRADE_AWAY: 3,
    LIKERT_NICE: 2,
    LIKERT_NOT: 1,
}

# 4 Likert levels compressed into the 3 argument-strength tiers (H/M/L) used by
# the lexicographic comparison: H = {tier 1 alone}, M = {tiers 2, 3}, L = {tier 4}.
LIKERT_TIER = {
    LIKERT_MOST: "H",
    LIKERT_TRADE_AWAY: "M",
    LIKERT_NICE: "M",
    LIKERT_NOT: "L",
}

UTILITY_FUNC_EXAMPLES = [
    {
        "priority": "Food Pantry",
        "sign": "positive",
        "explanation": "The priority names the project directly, so having that project fully satisfies it -- 1.0 when it's in the bundle, 0.0 when it isn't.",
        "function": """
def u_j(bundle):
    # The priority IS the project: a clean, direct match. No other project has any
    # stated connection, so nothing else affects this.
    return 1.0 if "Food Pantry" in bundle else 0.0
""",
    },
    {
        "priority": "investing in repairs to make the roads safer, since car accidents are a leading cause of death",
        "sign": "positive",
        "explanation": "One project (Slower, Safer Roads) fully and directly addresses this, and no other project is connected -- don't invent one -- so having it fully satisfies the priority.",
        "function": """
def u_j(bundle):
    # Slower, Safer Roads is the only project with a real, stated connection to
    # road safety. Clean direct match -- no bonus tacked on for unrelated projects.
    return 1.0 if "Slower, Safer Roads" in bundle else 0.0
""",
    },
    {
        "priority": "I would combine the food pantry and hygiene products, one location would make sense",
        "sign": "positive",
        "explanation": "The priority explicitly describes wanting BOTH projects together -- a complement stated in the text itself, not one we're inferring.",
        "function": """
def u_j(bundle):
    bundle_set = set(bundle)
    has_pantry = "Food Pantry" in bundle_set
    has_hygiene = "Free Hygiene Products" in bundle_set
    if has_pantry and has_hygiene:
        return 1.0  # exactly what the priority describes
    if has_pantry or has_hygiene:
        return 0.3  # only half of the stated combination
    return 0.0
""",
    },
    {
        "priority": "I care about the environment",
        "sign": "positive",
        "explanation": "Several available projects are environmental, but for a broad priority like this they're substitutes, not complements -- having all of them isn't three times as good as having one.",
        "function": """
def u_j(bundle):
    environmental = {"Planting Trees", "Solar Panel Canopy", "Public Nature Preserve"}
    count = len(environmental & set(bundle))
    if count == 0:
        return 0.0
    # Diminishing returns: the first environmental project matters most.
    return 0.5 + 0.25 * (count - 1)
""",
    },
    {
        "priority": "make sure kids are taken care of",
        "sign": "positive",
        "explanation": "A few different projects serve kids in different ways. Each adds something, but a vague priority like this shouldn't reward stacking all of them as if they were independent must-haves.",
        "function": """
def u_j(bundle):
    kid_focused = {"Free Summer Camp", "Classroom Resources", "Late Bus for After-School Programs"}
    count = len(kid_focused & set(bundle))
    if count == 0:
        return 0.0
    return 0.4 + 0.3 * (count - 1)
""",
    },
    {
        "priority": "I don't want more surveillance in this town",
        "sign": "negative",
        "explanation": "Something the person wants to AVOID. Security Cameras is the only project that creates the unwanted property, so avoiding it fully satisfies the priority.",
        "function": """
def u_j(bundle):
    # NEGATIVE criterion: reward bundles that avoid Security Cameras.
    return 0.0 if "Security Cameras" in bundle else 1.0
""",
    },
    {
        "priority": "the set of projects shouldn't ignore basic needs entirely",
        "sign": "negative",
        "explanation": "Something the person wants to AVOID (a bundle with zero basic-needs projects). Reward bundles that include at least one.",
        "function": """
def u_j(bundle):
    # NEGATIVE criterion rewritten as positive: reward AVOIDING the bad property
    # (a bundle with no basic-needs project at all).
    basic_needs = {"Food Pantry", "Free Hygiene Products", "Emergency Utility Fund"}
    count = len(basic_needs & set(bundle))
    return min(count * 0.5, 1.0)
""",
    },
    {
        "priority": "make sure the four projects benefit different groups, not just one",
        "sign": "positive",
        "explanation": "This is about the MIX of projects, not any single one -- reward breadth of who benefits, grounded in what each project's own description says it serves.",
        "function": """
def u_j(bundle):
    tags = {
        "Food Pantry": {"low_income", "families"},
        "Free Hygiene Products": {"low_income", "families"},
        "Free Summer Camp": {"kids", "families"},
        "Classroom Resources": {"kids"},
        "Addiction Support Programs": {"vulnerable_adults"},
        "Security Cameras": {"general_public"},
        "Slower, Safer Roads": {"general_public", "pedestrians"},
    }
    covered = set()
    for p in bundle:
        covered |= tags.get(p, set())
    return min(len(covered) * 0.25, 1.0)
""",
    },
    {
        "priority": "n/a",
        "sign": "positive",
        "explanation": "The priority text has no substance at all, so there is nothing to build a function from. Returning 0.0 for every bundle is correct -- ONLY for text like this, not for priorities that say something real but only loosely connect to the projects.",
        "function": """
def u_j(bundle):
    return 0.0
""",
    },
    {
        "priority": "too much money went to one type of project instead of a mix",
        "sign": "negative",
        "explanation": "Something the person wants to AVOID (over-concentration in one category). Reward bundles that spread across categories.",
        "function": """
def u_j(bundle):
    categories = {
        "Food Pantry": "needs", "Free Hygiene Products": "needs", "Emergency Utility Fund": "needs",
        "Free Summer Camp": "youth", "Classroom Resources": "youth",
        "Planting Trees": "environment", "Solar Panel Canopy": "environment",
        "Security Cameras": "safety", "Slower, Safer Roads": "safety",
    }
    present = [categories[p] for p in bundle if p in categories]
    if not present:
        return 0.3
    distinct = len(set(present))
    dominant_share = max(present.count(c) for c in set(present)) / len(present)
    # Reward category diversity, penalize one category dominating the bundle.
    return max(0.0, min(1.0, distinct * 0.3 - dominant_share * 0.2))
""",
    },
]

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def call_llm(prompt, retries=3, backoff=5, max_tokens=1024):
    """Calls Claude claude-sonnet-4-6 and returns the text response."""
    import time
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")
    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    for attempt in range(retries):
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                json=payload, headers=headers,
                timeout=120,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            # ConnectionError covers transient network blips (e.g. "Connection reset by
            # peer") that aren't a Timeout but are just as clearly worth retrying rather
            # than crashing a long batch run over a one-off network hiccup.
            wait = backoff * (2 ** attempt)
            print(f"  Connection error, retrying in {wait}s...")
            time.sleep(wait)
            continue
        if response.status_code in (529, 503):
            wait = backoff * (2 ** attempt)
            print(f"  {response.status_code} from Claude, retrying in {wait}s...")
            time.sleep(wait)
            continue
        if not response.ok:
            print(f"  HTTP {response.status_code} error body: {response.text[:500]}")
        response.raise_for_status()
        return response.json()["content"][0]["text"]
    raise RuntimeError(f"Claude API unavailable after {retries} retries")


def pearson_coeff(xs, ys):
    n = len(xs)
    if n < 10:
        return None
    mx, my = sum(xs)/n, sum(ys)/n
    num   = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    denom = (sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys)) ** 0.5
    return num/denom if denom > 1e-9 else None

def normalize(s):
    return " ".join(s.lower().split())

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# Step 1: parse survey → {priority: likert_string}
def parse_survey(version, survey_info, priorities):
    importance = dict(survey_info.get("importance", {}))
    return importance

# Step 2: likert → omega ranges
def likert_to_omega_ranges(importance_dict, utility_functions=None):
    """{priority: (lo, hi)} weight interval from the Likert tier.

    When utility_functions is given: priorities whose u_j is flat (identically 0 -- see
    _normalize_utility) are dropped, as they're functionally irrelevant, and the remaining
    intervals are scaled by one common factor so their MIDPOINTS sum to 1 (||omega||_1 = 1,
    which with each u_j in [0, 1] puts U(B) = sum_j omega_j u_j in [0, 1]). Scaling lo and hi
    by the same factor keeps each interval's relative width."""
    ranges = {
        priority: LIKERT_TO_OMEGA.get(level, (0.5, 0.75))
        for priority, level in importance_dict.items()
    }
    if utility_functions is None:
        return ranges
    ranges = {p: r for p, r in ranges.items()
              if not getattr(utility_functions.get(p), "is_flat", False)}
    total_mid = sum((lo + hi) / 2 for lo, hi in ranges.values())
    if total_mid <= 0:
        return ranges
    return {p: (lo / total_mid, hi / total_mid) for p, (lo, hi) in ranges.items()}

# Step 3: LLM-derived utility functions

# Punctuation the model sometimes emits that reads fine in prose/comments but is an
# invalid token outside of a string/comment in Python 3 (e.g. an em dash used in place
# of a minus sign) -- replaced with the plain-ASCII equivalent as cheap defense-in-depth
# before compiling. Safe to apply unconditionally: none of our project names contain
# these characters, so this can't change which project a string-equality check matches.
_UNICODE_PUNCTUATION_FIXES = {
    "—": "-", "–": "-",             # em dash, en dash
    "‘": "'", "’": "'",             # curly single quotes
    "“": '"', "”": '"',             # curly double quotes
    "…": "...",                          # ellipsis
}


def extract_code_block(response):
    """
    Extracts the FIRST fenced code block (```python ... ``` or ``` ... ```) from an
    LLM response, applying the Unicode-punctuation fixes above.

    Handles two real failure modes seen in practice with naive prefix/suffix stripping
    (response.strip().removeprefix("```python")...removesuffix("```")):
      1. Leading/trailing prose around a single code block despite being told to
         return ONLY the function.
      2. The model second-guessing itself mid-response and starting a SECOND draft
         after the first complete block (observed once in ~370 real generations,
         complete with "Wait, I used `import math`... let me rewrite" narration) --
         sometimes cut off mid-draft by max_tokens. Naive suffix-stripping keeps
         everything between the first and (new) last fence, corrupting an otherwise
         perfectly good first function with trailing prose. Taking only the FIRST
         fenced block sidesteps this entirely.

    Falls back to the old strip-prefix/suffix behavior if no fenced block is found at
    all (e.g. the model returned bare code with no markdown fence).
    """
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", response, re.DOTALL)
    code = match.group(1).strip() if match else (
        response.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()
    )
    for bad, good in _UNICODE_PUNCTUATION_FIXES.items():
        code = code.replace(bad, good)
    return code


def _compiles(func_body):
    try:
        compile(func_body, "<llm>", "exec")
        return True
    except SyntaxError:
        return False


def safe_llm_python(func_body, projects):
    import signal
    def _timeout(signum, frame): raise TimeoutError()
    # Flat, bundle-size-independent placeholder: sum(0.5 for _ in bundle) would give
    # bigger bundles a bigger score for every priority that fails to parse, silently
    # biasing every Delta computation involving it toward preferring larger bundles
    # for reasons that have nothing to do with the actual priority.
    fallback = lambda bundle: 0.0
    fallback.is_flat = True
    try:
        local_ns = {}
        exec(compile(func_body, "<llm>", "exec"), {}, local_ns)
        u_j_raw = local_ns.get("u_j")
        if u_j_raw is None:
            raise ValueError("no u_j defined")
    except SyntaxError as e:
        print(f"  [Warning] bad Python from LLM: {e}. Using flat fallback.")
        return fallback
    def u_j(bundle):
        try:
            signal.signal(signal.SIGALRM, _timeout)
            signal.alarm(1)
            result = u_j_raw(bundle)
            signal.alarm(0)
            return float(result)
        except Exception as e:
            signal.alarm(0)
            return 0.0
    try:
        u_j([])
    except Exception:
        return fallback
    return _normalize_utility(u_j, projects)


BUNDLE_SIZE = 4


def _normalize_utility(u_j, projects):
    """Post-processing applied to every generated u_j, so these properties hold by
    construction rather than depending on the prompt:
      - a flat function (same value on every bundle) becomes exactly 0.0 -- a priority with
        no discriminating content is functionally irrelevant, and downstream code drops it;
      - otherwise u_j is divided by its own maximum over every size-BUNDLE_SIZE bundle of the
        pool, so its best bundle scores exactly 1.0 (0.0 still means "does nothing"). How much
        a person cares about a priority lives in omega_j, not in the scale of u_j;
      - values are clamped to [0, 1]."""
    import itertools
    names = list(projects)
    vals = [u_j(list(b)) for b in itertools.combinations(names, BUNDLE_SIZE)]
    hi, lo = max(vals), min(vals)
    if hi - lo < 1e-9 or hi <= 0:
        zero = lambda bundle: 0.0
        zero.is_flat = True
        return zero

    def normalized(bundle):
        return min(1.0, max(0.0, u_j(bundle) / hi))
    normalized.is_flat = False
    return normalized


def build_utility_functions_v1(free_text_priority, projects, sign="positive",
                                good_projects=None, bad_projects=None, depends_projects=None,
                                cache=None, cache_key=None):
    """
    Builds u_j for one priority.

    Two modes:
    - Classified (good_projects/bad_projects/depends_projects given): a per-project
      good/bad/neither/depends rating for every project, fed straight into the prompt,
      since a single priority can legitimately have both good and bad projects under
      it. Off by the standard pipeline's default (see run_wave0_real_validation.py);
      still supported for run_ablation_no_classification.py's comparison.
    - Legacy (no classification given, the default path now): a bare positive/negative
      sign plus the priority's own free text -- no per-project information at all.
      There used to also be an optional free-select project list here
      (priority_projects), but no caller has ever populated it for years now (that
      survey question was removed independently of the classification question), so
      it was dead weight in the prompt and has been removed.

    cache/cache_key: optional. When both are given and cache_key is already present in cache,
    skips the LLM call entirely and rebuilds u_j from the cached Python source (instant, free,
    and deterministic across re-runs). Otherwise generates fresh and -- if cache/cache_key were
    given -- stores the raw source under cache_key before returning, so the caller can persist
    it. Real respondents have a stable identity to key on across re-runs; simulated personas are
    regenerated fresh each run and simply don't pass these, which leaves that behavior unchanged.
    """
    if cache is not None and cache_key is not None and cache_key in cache:
        return safe_llm_python(cache[cache_key], projects)

    examples_text = ""
    for ex in UTILITY_FUNC_EXAMPLES:
        sign_label = f"[{ex['sign'].upper()} criterion]"
        examples_text += (
            f'---\n{sign_label} Priority: "{ex["priority"]}"\n'
            f'What the person said: {ex["explanation"]}\n'
            f'Python function:\n```python{ex["function"]}```\n\n'
        )

    projects_text = "\n".join(f'  - "{name}": {desc}' for name, desc in projects.items())

    rules_text = """    Rules:
    1. Function must be named u_j, one argument only.
    2. Capture non-separability: diminishing returns, substitutes, complements, must-haves.
    3. Standard Python only, no imports.
    4. Calibrate the scale consistently, anchored like this:
       - 0.0 means the bundle does essentially nothing to satisfy this criterion
         (for a negative criterion, 0.0 means the bundle fully has the bad property).
       - 1.0 is the best a bundle could possibly do on this criterion -- a maximally
         clear, textbook-perfect example. Do not shrink a function's scale just because the
         priority seems minor or only loosely tied to the projects: how much a person cares
         about a priority is handled separately, so here just describe which bundles satisfy
         it more or less, using the full 0-1 range.
       - Use the full range in between for partial or ambiguous matches. Do not treat this
         as a near-binary indicator (e.g. "0 or 1") unless the criterion is genuinely binary.
    5. Do NOT use a simple weighted sum.
    6. Return 0.0 for every bundle ONLY if the priority text has no substance at all (e.g.
       "n/a", "none", a bare fragment with no content). If the priority says anything real,
       even if it is only loosely related to the projects, build the function that best
       reflects what it says -- do not return a flat value just because the connection is
       indirect."""

    if good_projects or bad_projects or depends_projects:
        good_projects = good_projects or []
        bad_projects = bad_projects or []
        depends_projects = depends_projects or []

        parts = []
        if good_projects:
            parts.append(
                "Projects the respondent says RAISE utility for this priority (GOOD):\n"
                + "\n".join(f'  - "{p}"' for p in good_projects)
            )
        if bad_projects:
            parts.append(
                "Projects the respondent says LOWER utility for this priority (BAD):\n"
                + "\n".join(f'  - "{p}"' for p in bad_projects)
            )
        if depends_projects:
            parts.append(
                "Projects whose effect on this priority DEPENDS on what else is in the bundle "
                "(model this as genuine non-separability — e.g. diminishing returns, a "
                "complement/substitute relationship, or a threshold effect involving these "
                "specific projects):\n"
                + "\n".join(f'  - "{p}"' for p in depends_projects)
            )
        classification_text = "\n\n".join(parts)

        prompt = f"""You are translating a participatory budgeting criterion into a Python utility function.

    Function signature: def u_j(bundle: list[str]) -> float
    - bundle is a list of project name strings
    - return a float between 0.0 and 1.0

    The respondent classified each available project's relationship to this priority
    individually, rather than labeling the whole priority as simply positive or negative.
    Some projects raise utility, some lower it, some have a context-dependent effect, and any
    project not mentioned below was judged to have no real effect on this priority at all.
    Build u_j directly from these per-project classifications — do not assume the whole
    priority is uniformly positive or negative.

{rules_text}

    Available projects:
    {projects_text}

    {classification_text}

    Projects not listed above were judged to have no real effect on this priority — do not
    build the function around them unless there's a strong reason to model an interaction with
    one of the classified projects above.

    Examples (showing both positive and negative criteria, for style/structure only — this
    criterion's own polarity comes from the classifications above, not from these examples):
    {examples_text}

    Now write u_j for:
    Priority: "{free_text_priority}"

    Return ONLY the Python function, no explanation, no markdown."""

    else:
        if sign == "positive":
            sign_instruction = (
                "This is a POSITIVE criterion — something the person WANTS. "
                "u_j should return HIGHER values for bundles that better satisfy this criterion."
            )
        else:
            sign_instruction = (
                "This is a NEGATIVE criterion — something the person wants to avoid. "
                "Your utility function should penalize bundles containing whatever creates "
                "the unwanted property. Return HIGHER values for bundles that avoid the bad "
                "property and LOWER values for bundles that contain it."
            )

        prompt = f"""You are translating a participatory budgeting criterion into a Python utility function.

    Function signature: def u_j(bundle: list[str]) -> float
    - bundle is a list of project name strings
    - return a float between 0.0 and 1.0

    {sign_instruction}

{rules_text}

    Available projects:
    {projects_text}

    Infer from the wording of the priority and the project descriptions which projects (if
    any) are actually connected to it -- central matches versus merely peripheral ones.

    Before deciding, briefly check EVERY project above individually rather than stopping at
    the first one that seems related -- a priority about a general concept (breadth of
    benefit, safety, environmental impact, etc.) may plausibly connect to more than one
    project. If you need to compare how broadly different projects reach people, ground the
    comparison in concrete facts stated in their descriptions (who is it open to, is
    membership/ownership of something required, etc.) rather than a general impression of
    which one "feels" more broad or important -- different kinds of targeting (e.g. by income
    vs. by ownership of something) aren't automatically comparable without thinking through
    who's actually included.

    Examples (showing both positive and negative criteria):
    {examples_text}

    Now write u_j for:
    [{sign.upper()} criterion] Priority: "{free_text_priority}"

    Return ONLY the Python function, no explanation, no markdown."""

    response = call_llm(prompt)
    func_body = extract_code_block(response)
    if not _compiles(func_body):
        # Rare (~1/370 in practice): the model second-guesses itself mid-response and
        # the extracted block doesn't parse. One retry with a fresh generation is cheap
        # and recovers a real function instead of settling for the flat placeholder.
        print(f"  [Warning] LLM response for {free_text_priority!r} didn't parse -- retrying once.")
        response = call_llm(prompt)
        func_body = extract_code_block(response)
    if cache is not None and cache_key is not None:
        cache[cache_key] = func_body
    return safe_llm_python(func_body, projects)


def infer_criterion_sign(free_text_priority):
    """
    Classifies a freeform criterion (e.g. surfaced via an 'Other:' validation
    citation) as a POSITIVE criterion (something wanted) or a NEGATIVE
    criterion (something to avoid), since build_utility_functions_v1 needs
    this to prompt the LLM correctly.
    """
    prompt = f"""A participatory-budgeting respondent gave this as a reason for preferring one bundle of projects over another:

"{free_text_priority}"

Is this describing a property the respondent WANTS a bundle to have (positive), or a property the respondent wants a bundle to AVOID (negative)?

Respond with ONLY one word: "positive" or "negative"."""
    response = call_llm(prompt).strip().lower()
    return "negative" if "negative" in response else "positive"


# ---------------------------------------------------------------------------
# Outcome labels and Delta computation
# ---------------------------------------------------------------------------

PREDICT_STRONG_A    = "strong predict A"
PREDICT_WEAK_A      = "weak predict A"
PREDICT_CONFLICTED  = "conflicted (A or B)"
PREDICT_WEAK_B      = "weak predict B"
PREDICT_STRONG_B    = "strong predict B"
PREDICT_INDIFFERENT = "indifferent"

CONSISTENT_WITH = {
    0: {PREDICT_STRONG_A, PREDICT_WEAK_A, PREDICT_CONFLICTED, PREDICT_INDIFFERENT},
    1: {PREDICT_STRONG_B, PREDICT_WEAK_B, PREDICT_CONFLICTED, PREDICT_INDIFFERENT},
}

def outcome_labels_for_interval(Delta_min, Delta_max, r_max, tau_r=0.0, STRONG_THRESHOLD=0.67):
    if r_max < tau_r:
        return [PREDICT_INDIFFERENT]
    T = STRONG_THRESHOLD
    labels = []
    if Delta_min < -T:
        labels.append(PREDICT_STRONG_B)
    if Delta_min < 0 and Delta_max > -T:
        labels.append(PREDICT_WEAK_B)
    if Delta_min < 0 and Delta_max > 0:
        labels.append(PREDICT_CONFLICTED)
    if Delta_min < T and Delta_max > 0:
        labels.append(PREDICT_WEAK_A)
    if Delta_max > T:
        labels.append(PREDICT_STRONG_A)
    seen = set()
    return [l for l in labels if not (l in seen or seen.add(l))]


def compute_delta_range(bundle_S, bundle_Sp, utility_functions, omega_ranges):
    Delta_min = Delta_max = r_min = r_max = 0.0
    for priority, (lo, hi) in omega_ranges.items():
        u_j = utility_functions.get(priority)
        if u_j is None:
            continue
        u_S  = u_j(bundle_S)
        u_Sp = u_j(bundle_Sp)
        c_j = max(0.0, u_S - u_Sp) - max(0.0, u_Sp - u_S)
        if c_j > 0:
            Delta_max += hi * c_j; Delta_min += lo * c_j
        elif c_j < 0:
            Delta_max += lo * c_j; Delta_min += hi * c_j
        intensity = max(0.0, u_S - u_Sp) + max(0.0, u_Sp - u_S)
        r_min += lo * intensity; r_max += hi * intensity
    return Delta_min, Delta_max, r_min, r_max

def compute_lexicographic_evidence(
    bundle_A,
    bundle_B,
    utility_functions,
    importance_ratings
):
    """
    Returns criterion-level ordinal evidence.

    Each criterion contributes:
      +1 if it favors A
      -1 if it favors B
       0 if tied

    Importance level determines lexicographic priority.
    """

    evidence = []

    for priority, rating in importance_ratings.items():
        u_j = utility_functions.get(priority)

        if u_j is None:
            continue

        u_A = u_j(bundle_A)
        u_B = u_j(bundle_B)

        if u_A > u_B:
            direction = 1
        elif u_B > u_A:
            direction = -1
        else:
            direction = 0

        if direction != 0:
            evidence.append({
                "priority": priority,
                "importance": rating,
                "level": LIKERT_ORDER.get(rating, 3),
                "tier": LIKERT_TIER.get(rating, "M"),
                "direction": direction,
                "u_A": u_A,
                "u_B": u_B,
            })

    return evidence

def priority_contributions(bundle_A, bundle_B, utility_functions, omega_ranges):
    """
    Per-priority contribution c_j to Delta for one bundle pair -- the same c_j computed
    inside compute_delta_range/debug_pair, but returned per-priority instead of only
    folded into the aggregate Delta_min/Delta_max. c_j > 0 means that priority's utility
    function favors bundle_A, c_j < 0 favors bundle_B, c_j == 0 means the priority is
    indifferent between the two bundles (and so could not have driven a choice between
    them, regardless of how important the respondent says it is in general).

    Used to check whether a priority the respondent says they used for a given choice
    actually had any bearing on the model's own math for that specific pair.
    """
    contributions = {}
    for priority, u_j in utility_functions.items():
        if priority not in omega_ranges or u_j is None:
            continue
        u_A, u_B = u_j(bundle_A), u_j(bundle_B)
        contributions[priority] = max(0.0, u_A - u_B) - max(0.0, u_B - u_A)
    return contributions


def build_per_pair_contributions(bundle_pairs, pair_codes, validation_answers, utility_functions, omega_ranges):
    """(code, c_map, chosen) per pair -- the input joint_feasibility_tiers needs. c_map is
    that pair's full per-priority contribution dict (priority_contributions); chosen is
    0 (A) or 1 (B) from the respondent's actual answer."""
    per_pair = []
    for code, (bundle_A, bundle_B), truth in zip(pair_codes, bundle_pairs, validation_answers):
        c_map = priority_contributions(bundle_A, bundle_B, utility_functions, omega_ranges)
        per_pair.append((code, c_map, truth["chosen"]))
    return per_pair


def _pair_constraint_rows(per_pair_c, priorities, indices, eps):
    """LP rows encoding: for a pair the respondent chose A, sum(c_i*omega_i) >= eps
    (chose B: <= -eps) -- rearranged into linprog's A_ub @ x <= b_ub form."""
    A_ub, b_ub = [], []
    for i in indices:
        _, c_map, chosen = per_pair_c[i]
        row = [c_map.get(p, 0.0) for p in priorities]
        if chosen == 0:
            A_ub.append([-x for x in row]); b_ub.append(-eps)
        else:
            A_ub.append(row); b_ub.append(-eps)
    return A_ub, b_ub


def _ordinal_constraint_rows(priorities, importance):
    """omega_i >= omega_j whenever priority i's stated Likert tier is strictly higher
    than priority j's (ties impose no constraint between them), as omega_j - omega_i <= 0."""
    n = len(priorities)
    levels = [LIKERT_ORDER.get(importance[p], 3) for p in priorities]
    A_ub, b_ub = [], []
    for i in range(n):
        for j in range(n):
            if levels[i] > levels[j]:
                row = [0.0] * n
                row[j] = 1.0
                row[i] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)
    return A_ub, b_ub


def _lp_solve(n, bounds, A_ub, b_ub):
    return linprog(c=np.zeros(n), A_ub=A_ub or None, b_ub=b_ub or None, bounds=bounds, method="highs")


def _achievable_intervals(priorities, bounds, A_ub, b_ub):
    """Per-priority achievable interval within a feasible constraint set -- the "interval
    hull" of the feasible polytope, projected onto each axis via 2 small LP solves
    (minimize, then maximize omega_j). Narrow => this respondent's choices (plus, for
    tiers a/b, their stated importance) tightly pin down that weight; wide => barely
    constrained. Doesn't capture correlation between priorities (true polytope shape),
    only each dimension's own extent."""
    n = len(priorities)
    intervals = {}
    for j, p in enumerate(priorities):
        obj = np.zeros(n); obj[j] = 1.0
        lo_res = linprog(c=obj, A_ub=A_ub or None, b_ub=b_ub or None, bounds=bounds, method="highs")
        hi_res = linprog(c=-obj, A_ub=A_ub or None, b_ub=b_ub or None, bounds=bounds, method="highs")
        lo = lo_res.x[j] if lo_res.success else bounds[j][0]
        hi = -hi_res.fun if hi_res.success else bounds[j][1]
        intervals[p] = (lo, hi)
    return intervals


JOINT_FEASIBILITY_TIER_LABELS = {
    "a": "(a) tight — a weighting inside every priority's own Likert-tier box explains all choices",
    "b": "(b) ordinal — the exact Likert boxes don't work, but a weighting exists that still "
         "respects every stated importance ranking",
    "c": "(c) unconstrained — only a weighting that ignores stated importance entirely explains "
         "all choices",
    "d": "(d) infeasible — no weighting whatsoever (even ignoring stated importance) explains all "
         "choices; a genuine rationality violation",
}


def joint_feasibility_tiers(bundle_pairs, pair_codes, validation_answers, utility_functions, importance, eps=1e-3):
    """
    Four-tier joint-feasibility hierarchy, tightest to loosest, reporting the STRICTEST
    tier that holds -- does there exist a single omega vector (one weight per priority)
    consistent with EVERY one of this respondent's choices at once:

      (a) omega_i constrained to this priority's own Likert-tier box (LIKERT_TO_OMEGA) --
          exactly what the cardinal model itself uses for prediction.
      (b) omega_i in [0, 1], dropping the exact numeric bounds but keeping the
          constraint that a priority rated MORE important than another must have
          weight >= that other priority's (ties impose no constraint between them).
      (c) omega_i in [0, 1], no ordering requirement at all -- does ANY weighting
          explain every choice, ignoring stated importance completely.
      (d) not even (c) has a solution -- genuinely infeasible, independent of how
          Likert levels get mapped to numbers or of stated importance altogether.

    (a)'s feasible region is a subset of (b)'s: the four Likert-tier boxes are
    contiguous and non-overlapping in importance order (MOST=[.75,1] > TRADE_AWAY=
    [.5,.75] > NICE=[.25,.5] > NOT=[0,.25]), so any omega satisfying its own tier's box
    already satisfies the ordinal constraint against every lower-tier priority. (b)'s
    region is in turn a subset of (c)'s (identical [0,1] box, ordinal constraint just
    dropped). So checking a -> b -> c in order and stopping at the first success finds
    the unique strictest tier that holds.

    Returns:
      tier: "a" | "b" | "c" | "d"
      tier_label: human-readable description of that tier (JOINT_FEASIBILITY_TIER_LABELS)
      culprits: leave-one-out diagnostic, only computed for tier "d" against the
                LOOSEST (c) constraint set -- pairs whose removal would restore
                feasibility even without any assumption about relative importance
      omega_ranges: {priority: (min, max)} -- the ACHIEVABLE interval for each
                    priority's weight within the winning tier's own feasible region
                    (how tightly stated importance + all these choices jointly pin down
                    that weight). None for tier "d", where no such region exists.
      omega_point: {priority: weight} -- ONE actual witness omega found while solving
                   the winning tier's feasibility LP (an arbitrary vertex of the
                   feasible region, not a "canonical center" -- linprog with a zero
                   objective just returns whatever feasible point it finds first).
                   Genuinely consistent with ALL of this respondent's pairwise choices,
                   so usable downstream to predict utility for bundles they weren't
                   directly compared on. None for tier "d".
      priorities: the priority list, in the order used for omega_ranges/omega_point
    """
    omega_ranges_box = likert_to_omega_ranges(importance, utility_functions)
    priorities = list(omega_ranges_box.keys())
    n = len(priorities)
    per_pair_c = build_per_pair_contributions(
        bundle_pairs, pair_codes, validation_answers, utility_functions, omega_ranges_box
    )
    all_idx = range(len(per_pair_c))
    pair_A_ub, pair_b_ub = _pair_constraint_rows(per_pair_c, priorities, all_idx, eps)

    tight_bounds = [omega_ranges_box[p] for p in priorities]
    loose_bounds = [(0.0, 1.0)] * n
    ordinal_A_ub, ordinal_b_ub = _ordinal_constraint_rows(priorities, importance)

    res_a = _lp_solve(n, tight_bounds, pair_A_ub, pair_b_ub)
    res_b = _lp_solve(n, loose_bounds, ordinal_A_ub + pair_A_ub, ordinal_b_ub + pair_b_ub)
    res_c = _lp_solve(n, loose_bounds, pair_A_ub, pair_b_ub)

    if res_a.success:
        tier, bounds, A_ub, b_ub, witness = "a", tight_bounds, pair_A_ub, pair_b_ub, res_a.x
    elif res_b.success:
        tier, bounds, A_ub, b_ub, witness = "b", loose_bounds, ordinal_A_ub + pair_A_ub, ordinal_b_ub + pair_b_ub, res_b.x
    elif res_c.success:
        tier, bounds, A_ub, b_ub, witness = "c", loose_bounds, pair_A_ub, pair_b_ub, res_c.x
    else:
        tier = "d"
        bounds = A_ub = b_ub = witness = None

    if tier == "d":
        culprits = []
        for skip in all_idx:
            idxs = [i for i in all_idx if i != skip]
            A_ub2, b_ub2 = _pair_constraint_rows(per_pair_c, priorities, idxs, eps)
            if _lp_solve(n, loose_bounds, A_ub2, b_ub2).success:
                culprits.append(per_pair_c[skip][0])
        return {
            "tier": tier, "tier_label": JOINT_FEASIBILITY_TIER_LABELS[tier],
            "culprits": culprits, "omega_ranges": None, "omega_point": None, "priorities": priorities,
        }

    omega_ranges = _achievable_intervals(priorities, bounds, A_ub, b_ub)
    return {
        "tier": tier, "tier_label": JOINT_FEASIBILITY_TIER_LABELS[tier],
        "culprits": None, "omega_ranges": omega_ranges,
        "omega_point": dict(zip(priorities, witness)), "priorities": priorities,
    }


def debug_pair(bundle_A, bundle_B, utility_functions, omega_ranges, pair_label="", file=None):
    """
    Writes a full per-priority breakdown of how Delta is computed for a bundle pair.
    Pass `file` (an open file object) to redirect output there instead of stdout.
    """
    out = lambda *a, **k: print(*a, file=file, **k)
    out(f"\n{'='*70}")
    if pair_label:
        out(f"DEBUG: {pair_label}")
    out(f"  Bundle A: {bundle_A}")
    out(f"  Bundle B: {bundle_B}")
    out(f"{'─'*70}")
    out(f"  {'Priority':<45} {'ω range':<14} {'u(A)':>6} {'u(B)':>6} {'c_j':>7} {'[Δmin,Δmax] contrib':>22}")
    out(f"{'─'*70}")

    Delta_min = Delta_max = 0.0
    rows = []

    for priority, (lo, hi) in omega_ranges.items():
        u_j = utility_functions.get(priority)
        if u_j is None:
            rows.append((priority, lo, hi, None, None, None, 0, 0))
            continue
        u_A = u_j(bundle_A)
        u_B = u_j(bundle_B)
        c_j = max(0.0, u_A - u_B) - max(0.0, u_B - u_A)
        if c_j > 0:
            contrib_max = hi * c_j
            contrib_min = lo * c_j
        elif c_j < 0:
            contrib_max = lo * c_j
            contrib_min = hi * c_j
        else:
            contrib_max = contrib_min = 0.0
        Delta_max += contrib_max
        Delta_min += contrib_min
        rows.append((priority, lo, hi, u_A, u_B, c_j, contrib_min, contrib_max))

    rows.sort(key=lambda r: abs(r[7]) if r[7] is not None else 0, reverse=True)

    for priority, lo, hi, u_A, u_B, c_j, contrib_min, contrib_max in rows:
        if u_A is None:
            out(f"  {priority:<45} [{lo:.1f},{hi:.1f}]      (no utility function)")
            continue
        direction = "→A" if c_j > 0 else ("→B" if c_j < 0 else "  ")
        out(f"  {priority:<45} [{lo:.1f},{hi:.1f}]  "
              f"{u_A:>6.2f} {u_B:>6.2f} {c_j:>+7.2f} "
              f"[{contrib_min:>+6.2f},{contrib_max:>+6.2f}] {direction}")

    out(f"{'─'*70}")
    out(f"  {'TOTAL':<73} [{Delta_min:>+6.2f},{Delta_max:>+6.2f}]")
    out(f"\n  Delta range: [{Delta_min:.3f}, {Delta_max:.3f}]")
    compatible = outcome_labels_for_interval(Delta_min, Delta_max, r_max=999)
    out(f"  Compatible outcomes: {compatible}")
    out(f"{'='*70}\n")


def predict_pair(bundle_S, bundle_Sp, utility_functions, omega_ranges,
                 tau_r=0.0, tau_kappa=0.0):
    Delta_min, Delta_max, r_min, r_max = compute_delta_range(
        bundle_S, bundle_Sp, utility_functions, omega_ranges
    )

    if r_max < tau_r:
        outcome = PREDICT_INDIFFERENT
    elif Delta_min > tau_kappa * max(r_min, 1e-9):
        STRONG_THRESHOLD = 0.67  # ALINA TODO: tune -- rescaled from 2.0 when u_j moved from a 0-3 to a 0-1 scale
        outcome = PREDICT_STRONG_A if Delta_min > STRONG_THRESHOLD else PREDICT_WEAK_A
    elif Delta_max < -tau_kappa * max(r_min, 1e-9):
        STRONG_THRESHOLD = 0.67
        outcome = PREDICT_STRONG_B if abs(Delta_max) > STRONG_THRESHOLD else PREDICT_WEAK_B
    else:
        outcome = PREDICT_CONFLICTED

    delta_width = Delta_max - Delta_min
    if delta_width < 1e-9:
        uncertainty = 0.0
    elif Delta_min < 0 < Delta_max:
        uncertainty = min(abs(Delta_min), Delta_max) / delta_width
    else:
        uncertainty = 0.0

    return {
        "outcome": outcome,
        "compatible_outcomes": outcome_labels_for_interval(Delta_min, Delta_max, r_max, tau_r),
        "Delta_min": Delta_min, "Delta_max": Delta_max,
        "r_min": r_min, "r_max": r_max,
        "uncertainty": uncertainty,
    }


def evaluate(predictions, validation_answers):
    results = []
    consistent_total = 0
    lex_consistent_total = 0
    robust_correct   = 0
    robust_total     = 0
    uncertainty_scores = []
    certainty_scores   = []

    for pred, truth in zip(predictions, validation_answers):
        outcome = pred["outcome"]
        chosen  = truth["chosen"]

        is_consistent = any(
            label in CONSISTENT_WITH.get(chosen, set())
            for label in pred["compatible_outcomes"]
        )
        consistent_total += int(is_consistent)
        lex_outcome = pred["lexicographic_outcome"]

        lex_consistent = (
            (truth["chosen"] == 0 and lex_outcome in [
                PREDICT_STRONG_A,
                PREDICT_WEAK_A,
            ])
            or
            (truth["chosen"] == 1 and lex_outcome in [
                PREDICT_STRONG_B,
                PREDICT_WEAK_B,
            ])
            or
            lex_outcome == PREDICT_CONFLICTED
        )

        lex_consistent_total += int(lex_consistent)

        is_robust = outcome in (PREDICT_STRONG_A, PREDICT_STRONG_B)
        if is_robust:
            robust_total += 1
            predicted_chosen = 0 if outcome == PREDICT_STRONG_A else 1
            robust_correct += int(predicted_chosen == chosen)

        certainty = truth.get("certainty")
        if certainty is not None:
            uncertainty_scores.append(pred["uncertainty"])
            certainty_scores.append((5 - certainty) / 4.0)

        results.append({
            "outcome":              outcome,
            "compatible_outcomes":  pred["compatible_outcomes"],
            "is_consistent":        is_consistent,
            "model_uncertainty":    pred["uncertainty"],
            "Delta_min":            pred["Delta_min"],
            "Delta_max":            pred["Delta_max"],
            "r_min":                pred["r_min"],
            "r_max":                pred["r_max"],
            "cited_priorities":     truth.get("cited_priorities", []),
            "lex_outcome": lex_outcome,
            "lex_consistent": lex_consistent,
        })

    n = len(predictions)
    calibration = pearson_coeff(uncertainty_scores, certainty_scores) if certainty_scores else None

    return {
        "consistency_rate": consistent_total / n if n else None,
        "robust_accuracy":  robust_correct / robust_total if robust_total else None,
        "abstain_rate":     1 - robust_total / n if n else None,
        "calibration":      calibration,
        "n_robust":         robust_total,
        "n_total":          n,
        "per_pair":         results,
        "lexicographic_consistency_rate": lex_consistent_total / n if n else None,
    }

def predict_pair_lexicographic(
    bundle_A,
    bundle_B,
    utility_functions,
    importance_ratings
):
    """
    Ordinally robust lexicographic comparison: cancel same-tier arguments,
    then require the cumulative H, H+M, H+M+L residual counts to all favor
    one side (with at least one strict) before declaring a winner.
    """
    evidence = compute_lexicographic_evidence(
        bundle_A,
        bundle_B,
        utility_functions,
        importance_ratings
    )

    if not evidence:
        return {
            "outcome": PREDICT_INDIFFERENT,
            "evidence": []
        }

    d = {"H": 0, "M": 0, "L": 0}
    for e in evidence:
        d[e["tier"]] += e["direction"]

    cumulative = [d["H"], d["H"] + d["M"], d["H"] + d["M"] + d["L"]]

    a_defeats_b = all(c >= 0 for c in cumulative) and any(c > 0 for c in cumulative)
    b_defeats_a = all(c <= 0 for c in cumulative) and any(c < 0 for c in cumulative)

    if a_defeats_b:
        outcome = PREDICT_STRONG_A
    elif b_defeats_a:
        outcome = PREDICT_STRONG_B
    else:
        outcome = PREDICT_CONFLICTED

    return {
        "outcome": outcome,
        "evidence": evidence
    }


def extract_other_criteria(validation_answers):
    """
    Extract new criteria introduced during validation.

    Returns:
        list of new criterion strings
    """
    others = []

    for answer in validation_answers:
        for criterion in answer.get("cited_priorities", []):
            if criterion.lower().startswith("other:"):
                new = criterion.split(":", 1)[1].strip()
                if new:
                    others.append(new)

    # Deduplicate while preserving order
    seen = set()
    return [
        c for c in others
        if not (normalize(c) in seen or seen.add(normalize(c)))
    ]

def run_validation(demographics, version, survey_info, bundle_pairs,
                   validation_answers, tau_r=0.0, tau_kappa=0.0, file=None,
                   projects_override=None, cache=None, cache_key_prefix=None,
                   priority_key_fn=None):
    """
    projects_override: pass a {name: description} dict to use instead of deriving the
    project pool from demographics["demographic_area"] -- needed for instruments other
    than the base Survey1_V1 (e.g. SimProjects' larger project taxonomy), which use a
    different pool than PROJECTS_URBAN/PROJECTS_RURAL. priorities/project_scores are
    only consulted by the (unused-here) v2 branch, so they're left as V1's regardless.

    cache/cache_key_prefix: optional, forwarded to build_utility_functions_v1 for every
    priority's utility function -- see that function's docstring for the caching
    contract. Pass a prefix stable and unique per respondent to skip regenerating
    utility functions (an LLM call each) on re-runs.

    priority_key_fn: optional. When given, transforms a priority's normalized text
    before it's used to build the utility-function cache key, letting a caller route a
    priority onto a previously-cached, sufficiently-similar priority's utility function
    instead of generating a fresh one (see priority_dedup.PriorityRegistry).
    """
    if demographics["age"] == "Under 18":
        print("Respondent under 18, skipping.")
        return None
    area = demographics["demographic_area"]
    # Survey Flow gates the bundle-pair pool on a single condition: QID6 == "Urban"
    # selected vs not-selected -- Suburban buckets with Rural, not with Urban.
    if area == "urban":
        projects, priorities = PROJECTS_URBAN, PRIORITIES_URBAN
        project_scores = PROJECT_PRIORITY_SCORES_URBAN
    elif area in ("rural", "suburban"):
        projects, priorities = PROJECTS_RURAL, PRIORITIES_RURAL
        project_scores = PROJECT_PRIORITY_SCORES_RURAL
    else:
        print("Unknown area."); return None
    if projects_override is not None:
        projects = projects_override
    if version not in ("v1", "v2"):
        print("Unknown version."); return None

    importance   = parse_survey(version, survey_info, priorities)
    omega_ranges = likert_to_omega_ranges(importance)

    if version == "v2":
        utility_functions = build_utility_functions_v2(project_scores)
    else:
        criteria_signs = survey_info.get("criteria_signs", {})
        # Per-project good/bad/neither/depends classification, only present for the
        # respondent's top-3 priorities (the live survey doesn't ask this for the rest).
        priority_classifications = survey_info.get("priority_classifications", {})

        def _uj_cache_key(p):
            if not cache_key_prefix:
                return None
            key_text = priority_key_fn(p) if priority_key_fn else p
            # normalize() here (matching generate_priority_classifications's _key)
            # matters once priority_key_fn is in play: p arrives already normalized,
            # but priority_key_fn may return a registry entry's ORIGINAL, differently-
            # cased canonical text (e.g. "Food Pantry" vs. the "food pantry" p came in
            # as) -- without re-normalizing, that casing mismatch would silently split
            # what should be one cache key into two.
            return f"{cache_key_prefix}::utility::{normalize(key_text)}"

        def _build_uj(p):
            classification = priority_classifications.get(p)
            if classification and not classification.get("not_about_projects", False):
                proj_map = classification.get("projects", {})
                good = [proj for proj, cat in proj_map.items() if cat == "good"]
                bad = [proj for proj, cat in proj_map.items() if cat == "bad"]
                depends = [proj for proj, cat in proj_map.items() if cat == "depends"]
                return build_utility_functions_v1(
                    p, projects,
                    good_projects=good, bad_projects=bad, depends_projects=depends,
                    cache=cache, cache_key=_uj_cache_key(p),
                )
            return build_utility_functions_v1(
                p, projects, sign=criteria_signs.get(p, "positive"),
                cache=cache, cache_key=_uj_cache_key(p),
            )

        utility_functions = {p: _build_uj(p) for p in importance.keys()}

        # Fold in criteria the persona surfaced via "Other: ..." validation
        # citations that weren't part of the original elicited criteria set.
        # Added with a low default importance and re-scored across ALL
        # bundle pairs (not just the one where each criterion was cited). These never
        # went through the top-3 classification step, so they always use the legacy
        # sign-based path.
        known = set(importance.keys())
        new_criteria = [
            c for c in extract_other_criteria(validation_answers)
            if normalize(c) not in known
        ]
        for c in new_criteria:
            key = normalize(c)
            importance[key] = LIKERT_NICE
            sign = infer_criterion_sign(c)
            utility_functions[key] = build_utility_functions_v1(
                c, projects, sign=sign, cache=cache, cache_key=_uj_cache_key(key)
            )
        omega_ranges = likert_to_omega_ranges(importance, utility_functions)

    predictions = []

    for bS, bSp in bundle_pairs:
        cardinal_pred = predict_pair(
            bS,
            bSp,
            utility_functions,
            omega_ranges,
            tau_r,
            tau_kappa
        )

        lex_pred = predict_pair_lexicographic(
            bS,
            bSp,
            utility_functions,
            importance
        )

        cardinal_pred["lexicographic_outcome"] = lex_pred["outcome"]
        cardinal_pred["lexicographic_evidence"] = lex_pred["evidence"]

        predictions.append(cardinal_pred)
    result = evaluate(predictions, validation_answers)

    # Auto-debug inconsistent pairs
    for i, (pred, truth, (bA, bB)) in enumerate(
        zip(result["per_pair"], validation_answers, bundle_pairs)
    ):
        if not pred["is_consistent"]:
            debug_pair(
                bA, bB, utility_functions, omega_ranges,
                pair_label=f"Pair {i+1} — persona chose {'A' if truth['chosen']==0 else 'B'}, model: {pred['compatible_outcomes']}",
                file=file,
            )

    # Exposed for downstream diagnostics (e.g. per-priority activation analysis)
    # that need to evaluate u_j on arbitrary bundles beyond what evaluate()'s
    # per-pair summary captures.
    result["utility_functions"] = utility_functions
    result["importance"] = importance
    return result

# ---------------------------------------------------------------------------
# Testing — persona based
# ---------------------------------------------------------------------------

# Exact wording from the deployed survey (QID65/QID67/QID66). Note there is no
# live "mix_bad" question — it exists as a defined question (tag "mix-bad") but
# sits in the survey's "Trash / Unused Questions" block, which isn't reachable
# from the live Survey Flow, so it's omitted here.
CRITERIA_QUESTIONS_URBAN = {
    "bundle_good": (
        "What priorities did you care about when choosing sets of projects? A priority "
        "could be a group or cause you want to support, a particular project you want "
        "funded, or how you want the funded projects to combine or balance different "
        "goals. Please list one priority per box, one sentence each. Leave any extra "
        "boxes blank. List up to 6 distinct priorities (one per line)."
    ),
    "mix_good": (
        "You may also care about the overall mix of projects. For example, you might "
        "want the projects to be diverse or balanced in some way, to include at least "
        "some number of projects of a certain type, or to include a certain pair of "
        "projects because they work particularly well together. Are there any "
        "priorities you want to add about the mix of projects? List up to 4 criteria."
    ),
    "bundle_bad": (
        "Imagine a set of projects that you would be unhappy to see funded. What would "
        "make a set of projects a bad choice? For example, a bad set could include a "
        "type of project you want to avoid, omit a project you care about, or combine "
        "projects in a way that bothers you. List each reason in its own box, one "
        "sentence each. List up to 6 criteria."
    ),
}
CRITERIA_QUESTIONS_RURAL = CRITERIA_QUESTIONS_URBAN


CRITERIA_SIGN_BY_QUESTION = {
    "bundle_good": "positive",
    "bundle_bad":  "negative",
    "mix_good":    "positive",
}

# Real 5 comparisons per pool, pulled directly from the current QSF's bundle-pair
# question choices (Survey1_V1 (20).qsf: QID92/95/98/126/129 = "town"/rural pool,
# QID101/104/107/135/138 = "city"/urban pool -- see the routing note above
# PROJECTS_RURAL for why "town" maps to rural and "city" maps to urban).
BUNDLE_PAIRS_RURAL = [
    (  # PN.vs.EE
        ["Slower, Safer Roads", "Security Cameras", "Addiction Support Programs", "Food Pantry"],
        ["Public Nature Preserve", "EV Chargers", "Free Summer Camp", "Food Pantry"],
    ),
    (  # EN.vs.PS
        ["Addiction Support Programs", "Free Hygiene Products", "Public Nature Preserve", "EV Chargers"],
        ["Slower, Safer Roads", "Free Summer Camp", "Food Pantry", "EV Chargers"],
    ),
    (  # EP.vs.EN
        ["Free Summer Camp", "Addiction Support Programs", "Free Hygiene Products", "Public Nature Preserve"],
        ["Security Cameras", "EV Chargers", "Slower, Safer Roads", "Public Nature Preserve"],
    ),
    (  # EN.vs.EE
        ["Free Hygiene Products", "Free Summer Camp", "Addiction Support Programs", "Food Pantry"],
        ["EV Chargers", "Slower, Safer Roads", "Addiction Support Programs", "Food Pantry"],
    ),
    (  # ES.vs.EP
        ["Free Summer Camp", "Food Pantry", "Public Nature Preserve", "EV Chargers"],
        ["Security Cameras", "Slower, Safer Roads", "Public Nature Preserve", "EV Chargers"],
    ),
]

BUNDLE_PAIRS_URBAN = [
    (  # SH.vs.SS
        ["Protected Bike Lane", "Security Cameras", "Addiction Support Programs", "Food Pantry"],
        ["Free Summer Camp", "Park Cleanup and Renovation", "EV Chargers", "Food Pantry"],
    ),
    (  # HN.vs.ES
        ["Addiction Support Programs", "Park Cleanup and Renovation", "Free Hygiene Products", "EV Chargers"],
        ["Food Pantry", "Protected Bike Lane", "Free Summer Camp", "EV Chargers"],
    ),
    (  # SE.vs.ES
        ["Free Hygiene Products", "Free Summer Camp", "Addiction Support Programs", "Park Cleanup and Renovation"],
        ["Security Cameras", "EV Chargers", "Protected Bike Lane", "Park Cleanup and Renovation"],
    ),
    (  # EH.vs.SN
        ["Free Hygiene Products", "Free Summer Camp", "Food Pantry", "Addiction Support Programs"],
        ["EV Chargers", "Protected Bike Lane", "Food Pantry", "Addiction Support Programs"],
    ),
    (  # HS.vs.PE
        ["Food Pantry", "Free Summer Camp", "Park Cleanup and Renovation", "EV Chargers"],
        ["Security Cameras", "Protected Bike Lane", "Park Cleanup and Renovation", "EV Chargers"],
    ),
]

# ---------------------------------------------------------------------------
# SimProjects variant -- a separate instrument (Survey1_V1_SimProjects (8).qsf)
# testing a larger, ~13-project-per-pool menu. Same area-routing rule as V1
# (QID6 == "Urban" selected vs not-selected; confirmed in this QSF's Survey Flow
# too). 6 of the 13 projects per pool overlap with the V1 taxonomy; descriptions
# for those are reused verbatim. Each pool has 7 bundle-pair comparisons (vs. 5
# for V1), pulled directly from the QSF's bundle-pair question choices.
# ---------------------------------------------------------------------------

PROJECTS_SIM_RURAL = {
    "Food Pantry": "Stock and staff a food pantry that provides free groceries to food-insecure households.",
    "Free Hygiene Products": "Provide free menstrual products, diapers, baby wipes, and other hygiene supplies in public buildings for three years.",
    "Free Summer Camp": "Fund a free one-week summer camp for all K-12 students.",
    "Classroom Resources": "Invest in classroom supplies and learning resources to support K–12 teachers.",
    "Public Nature Preserve": "Set aside and maintain public land as a protected nature preserve open for hiking and wildlife.",
    "Slower, Safer Roads": "Add traffic-calming measures -- speed bumps, narrower lanes, crosswalk improvements -- to slow traffic on residential streets.",
    "Solar Panel Canopy": "Install solar panel canopies over a public parking lot to generate clean energy and provide shade.",
    "Planting Trees": "Plant trees along streets and in public spaces to add shade and greenery.",
    "Emergency Utility Fund": "Create an emergency fund to help low-income households cover unexpected utility bills.",
    "Community Space Grants": "Give small grants to community groups to host events and activities in public spaces.",
    "Veterans Memorial Gathering Space": "Build a memorial gathering space to honor local veterans.",
    "Summer Movie Drive-in": "Host free outdoor movie drive-in nights over the summer.",
    "Late Bus for After-School Programs": "Fund a late bus route so students can stay for after-school programs and still get home.",
}

PROJECTS_SIM_URBAN = {
    "Food Pantry": "Stock and staff a food pantry that provides free groceries to food-insecure households.",
    "Free Hygiene Products": "Provide free menstrual products, diapers, baby wipes, and other hygiene supplies in public buildings for three years.",
    "Free Summer Camp": "Fund a free one-week summer camp for all K-12 students.",
    "Classroom Resources": "Invest in classroom supplies and learning resources to support K–12 teachers.",
    "Park Cleanup and Renovation": "Clean up and renovate a public park by upgrading playground equipment and installing benches.",
    "Protected Bike Lane": "Convert street parking on a busy road into a protected bike lane.",
    "Solar Panel Canopy": "Install solar panel canopies over a public parking lot to generate clean energy and provide shade.",
    "Planting Trees": "Plant trees along streets and in public spaces to add shade and greenery.",
    "Emergency Utility Fund": "Create an emergency fund to help low-income households cover unexpected utility bills.",
    "Community Space Grants": "Give small grants to community groups to host events and activities in public spaces.",
    "Veterans Memorial Gathering Space": "Build a memorial gathering space to honor local veterans.",
    "Summer Movie Nights in the Park": "Host free outdoor movie nights in the park over the summer.",
    "Free Bus Passes for Students": "Provide free bus passes so students can get to and from school.",
}

BUNDLE_PAIRS_SIM_RURAL = [
    (  # PN.vs.EE
        ["Public Nature Preserve", "Solar Panel Canopy", "Planting Trees", "Free Summer Camp"],
        ["Food Pantry", "Late Bus for After-School Programs", "Classroom Resources", "Free Summer Camp"],
    ),
    (  # EN.vs.PS
        ["Veterans Memorial Gathering Space", "Community Space Grants", "Public Nature Preserve", "Summer Movie Drive-in"],
        ["Slower, Safer Roads", "Classroom Resources", "Emergency Utility Fund", "Summer Movie Drive-in"],
    ),
    (  # EP.vs.EN
        ["Classroom Resources", "Late Bus for After-School Programs", "Public Nature Preserve", "Free Summer Camp"],
        ["Planting Trees", "Slower, Safer Roads", "Solar Panel Canopy", "Free Summer Camp"],
    ),
    (  # EN.vs.EE
        ["Public Nature Preserve", "Planting Trees", "Slower, Safer Roads", "Emergency Utility Fund"],
        ["Food Pantry", "Free Hygiene Products", "Community Space Grants", "Emergency Utility Fund"],
    ),
    (  # ES.vs.EP
        ["Late Bus for After-School Programs", "Food Pantry", "Emergency Utility Fund", "Free Hygiene Products"],
        ["Planting Trees", "Free Summer Camp", "Veterans Memorial Gathering Space", "Free Hygiene Products"],
    ),
    (  # Q221
        ["Public Nature Preserve", "Slower, Safer Roads", "Planting Trees", "Solar Panel Canopy"],
        ["Summer Movie Drive-in", "Free Hygiene Products", "Classroom Resources", "Solar Panel Canopy"],
    ),
    (  # Q220
        ["Veterans Memorial Gathering Space", "Summer Movie Drive-in", "Community Space Grants", "Free Hygiene Products"],
        ["Emergency Utility Fund", "Food Pantry", "Community Space Grants", "Free Hygiene Products"],
    ),
]

BUNDLE_PAIRS_SIM_URBAN = [
    (  # SH.vs.SS
        ["Park Cleanup and Renovation", "Solar Panel Canopy", "Planting Trees", "Free Summer Camp"],
        ["Food Pantry", "Free Bus Passes for Students", "Classroom Resources", "Free Summer Camp"],
    ),
    (  # HN.vs.ES
        ["Veterans Memorial Gathering Space", "Community Space Grants", "Park Cleanup and Renovation", "Summer Movie Nights in the Park"],
        ["Protected Bike Lane", "Classroom Resources", "Emergency Utility Fund", "Summer Movie Nights in the Park"],
    ),
    (  # SE.vs.ES
        ["Classroom Resources", "Free Bus Passes for Students", "Park Cleanup and Renovation", "Free Summer Camp"],
        ["Planting Trees", "Protected Bike Lane", "Solar Panel Canopy", "Free Summer Camp"],
    ),
    (  # EH.vs.SN
        ["Park Cleanup and Renovation", "Planting Trees", "Protected Bike Lane", "Emergency Utility Fund"],
        ["Food Pantry", "Free Hygiene Products", "Community Space Grants", "Emergency Utility Fund"],
    ),
    (  # HS.vs.PE
        ["Free Bus Passes for Students", "Food Pantry", "Emergency Utility Fund", "Free Hygiene Products"],
        ["Planting Trees", "Free Summer Camp", "Veterans Memorial Gathering Space", "Free Hygiene Products"],
    ),
    (  # Q213
        ["Park Cleanup and Renovation", "Protected Bike Lane", "Planting Trees", "Solar Panel Canopy"],
        ["Summer Movie Nights in the Park", "Free Hygiene Products", "Classroom Resources", "Solar Panel Canopy"],
    ),
    (  # Q212
        ["Veterans Memorial Gathering Space", "Summer Movie Nights in the Park", "Community Space Grants", "Free Hygiene Products"],
        ["Emergency Utility Fund", "Food Pantry", "Community Space Grants", "Free Hygiene Products"],
    ),
]

TEST_PERSONAS = [
    (
        "You are a woman in her mid-50s who owns a small hardware store in an urban neighborhood. "
        "You vote Republican and are skeptical of government spending on projects you see as "
        "'nice to have' rather than essential. You believe the city should focus on core services — "
        "safety, infrastructure, keeping things running — before funding anything extra. You're "
        "friendly in person but you don't think city money should be spent loosely, and you get "
        "frustrated when a budget feels like it's trying to please everyone instead of being responsible.",
        "urban", "51-60",
    ),
    (
        "You are a man in his 40s living in a small rural town. You use a wheelchair after a "
        "workplace injury several years ago, and you're acutely aware of which public spaces are "
        "actually accessible to you and which aren't — you think about curb cuts, ramps, and "
        "accessible restrooms before deciding whether it's worth going somewhere. You care about "
        "seniors and other disabled residents too, since you see how the same barriers affect them. "
        "You want the town to treat accessibility and health services as basic infrastructure, "
        "not an afterthought.",
        "rural", "41-50",
    ),
    (
        "You are a man in his early 30s who runs a small family farm outside of town. You're short "
        "on time most days and don't waste words when you don't have to, but you know exactly what "
        "you think about how the town should spend money. You care most about keeping young people "
        "from leaving the area — that means jobs and things for kids to do. You also think road and "
        "park maintenance matter more than people realize, since you drive them every day for work. "
        "You vote based on the person, not the party.",
        "rural", "31-40",
    ),
    (
        "You are an egalitarian man in his 50s from Chicago, IL. "
        "You believe city resources should prioritize the most vulnerable residents — "
        "unhoused people, low-income families. "
        "You are skeptical of environmental projects when basic needs aren't being met.",
        "urban", "51-60",
    ),
    (
        "You are an environmentalist woman in her 30s from rural Mississippi. "
        "You grew up surrounded by nature and believe sustainability is the most urgent issue. "
        "You distrust short-term fixes without addressing root causes.",
        "rural", "31-40",
    ),
    (
        "You are a woman in her 40s who lives in the city. You vote Democrat most of the time "
        "but you're not super political. You have two dogs and you like going to the farmers market "
        "on weekends. You work in HR at a mid-size company. You care about the environment "
        "but also think people need help too. You're not sure what you think about homeless people — "
        "you feel bad for them but also the situation feels complicated.",
        "urban", "41-50",
    ),
    (
        "You are a man in his 60s who moved from a city to a small rural town ten years ago. "
        "You like hiking and being outdoors. You have kids who are now grown up and left. "
        "You think education is important but you also think kids today have too many opportunities "
        "handed to them. You vote Republican sometimes. You're worried about jobs leaving the area. "
        "You garden a lot and think local food is important.",
        "rural", "61-70",
    ),
    (
        "You are a 28-year-old who just moved to the city for work. You rent an apartment. "
        "You don't have kids but your sister does and you're close with her family. "
        "You care about climate change but you also have a car because public transit is bad. "
        "You're generally pretty busy and haven't thought much about local politics. "
        "You want the city to feel nicer but you're not sure what that means exactly.",
        "urban", "18-30",
    ),
    (
        "You are a woman in your early 50s living in a rural community. "
        "English is your second language and you sometimes find forms confusing. "
        "You have a large family and most of your concern is about your kids and grandkids. "
        "You also help out at your church's food pantry on weekends. "
        "You don't really follow the news much but you know your town needs more jobs.",
        "rural", "51-60",
    ),
]


def respondent_instructions(persona_description):
    desc = persona_description.lower()
    if "english is your second language" in desc or "forms confusing" in desc:
        return (
            "\n\nIMPORTANT: You are not very comfortable with written English and find "
            "forms confusing. Write short, simple answers. Make occasional grammar mistakes. "
            "Sometimes misread the question slightly. Skip parts you don't understand."
        )
    elif "not super political" in desc or "haven't thought much" in desc or "not sure what" in desc:
        return (
            "\n\nIMPORTANT: You haven't thought much about these issues before. "
            "Your answers are a bit vague and sometimes trail off. "
            "You might repeat yourself or not fully answer the question."
        )
    elif "skeptical" in desc or "vote republican" in desc:
        return (
            "\n\nIMPORTANT: You are somewhat skeptical of government spending. "
            "Keep your answers short and practical. You don't over-explain."
        )
    elif "short on time" in desc or "no patience for long" in desc:
        return (
            "\n\nIMPORTANT: You know exactly what you think, but you're busy and don't like "
            "writing more than you have to. Give short, decisive answers — 1-2 sentences per "
            "item. Your answers should be specific and confident, not vague — you just don't "
            "elaborate."
        )
    else:
        return ""


def generate_persona_profile(persona_description):
    prompt = (
        f"You are roleplaying as the following person:\n{persona_description}\n\n"
        "Write a short internal monologue (3-5 sentences) describing what matters most "
        "to you when thinking about how a city should spend its public budget. "
        "Be specific about which groups and types of impact you care about most. "
        "Write in the first person."
    )
    return call_llm(prompt)


def generate_criteria(persona_description, profile, questions, projects):
    projects_list = "\n".join(f"- {name}: {desc}" for name, desc in projects.items())
    questions_text = "\n\n".join(f"[{label}]: {q}" for label, q in questions.items())
    degradation = respondent_instructions(persona_description)

    prompt = (
        f"You are roleplaying as the following person:\n{persona_description}\n\n"
        f"Here is how you think about participatory budgeting:\n{profile}\n\n"
        "Imagine participatory budgeting is being done in your town. There are two "
        "separate groups of 8 candidate projects (listed below); in each group, exactly "
        f"four projects can be funded:\n{projects_list}\n\n"
        "Answer each question below in character. List each criterion on its own line. "
        "Be concrete — refer to projects or groups by name where relevant. "
        f"List between 0 and 6 criteria per question.\n\n{questions_text}"
        f"{degradation}\n\n"
        f"Respond with ONLY a JSON object mapping each label to a LIST of criterion strings. "
        f"Example: {{\"bundle_good\": [\"helps vulnerable residents\"], \"bundle_bad\": [\"...\"]}} "
        f"Include every label: {list(questions.keys())}"
    )
    response = call_llm(prompt)
    response = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {label: ["unclear"] for label in questions}


def generate_likert_ratings(persona_description, profile, criteria_by_question):
    """
    Matches QID69 ("importance"): asked AFTER the respondent has already made
    their 5 bundle-pair choices, not before — see run_persona_tests for the
    call ordering.

    Uses numbered keys in the LLM exchange rather than asking the model to
    echo each full criterion string back as a JSON key — LLMs don't reliably
    reproduce free text byte-for-byte across separate calls (a dropped
    trailing period or a lightly reworded phrase breaks a text-based lookup
    silently), so referencing criteria by number sidesteps that entirely.
    """
    all_criteria = []
    for cl in criteria_by_question.values():
        all_criteria.extend(cl)
    seen = set()
    unique = [c for c in all_criteria if not (c in seen or seen.add(c))]
    criteria_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(unique))
    degradation = respondent_instructions(persona_description)

    prompt = (
        f"You are roleplaying as the following person:\n{persona_description}\n\n"
        f"Here is how you think about participatory budgeting:\n{profile}\n\n"
        f"You previously listed these priorities:\n{criteria_text}\n\n"
        "In the event that your priorities can't all be satisfied at once, which are "
        "most important to you? For each priority, choose ONLY one of these exact levels:\n"
        f"\"{LIKERT_MOST}\"\n"
        f"\"{LIKERT_TRADE_AWAY}\"\n"
        f"\"{LIKERT_NICE}\"\n"
        f"\"{LIKERT_NOT}\"\n\n"
        f"Reserve \"{LIKERT_MOST}\" for the small handful of priorities that are genuinely "
        "non-negotiable for you — most people should NOT end up with most of their "
        "priorities at this level. Rate the rest relative to those top few, using the "
        "full range of levels.\n"
        f"{degradation}\n\n"
        "Respond with ONLY a JSON object mapping each priority's NUMBER (as a string, e.g. "
        "\"1\") to its importance level — use the number only, do not repeat the priority "
        "text. Include every priority number."
    )
    response = call_llm(prompt)
    response = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    result = {normalize(c): LIKERT_TRADE_AWAY for c in unique}
    try:
        ratings = json.loads(response)
        for i, c in enumerate(unique):
            level = ratings.get(str(i + 1))
            if level is not None:
                result[normalize(c)] = level
    except json.JSONDecodeError:
        pass
    return result


def rank_tied_priorities(persona_description, profile, tied_priorities):
    """
    Matches QID161: shown only when a respondent marks multiple priorities at the
    top tier ("I'd want this even if..."), asking them to drag those into order.
    Returns tied_priorities reordered from most to least important.
    """
    priorities_text = "\n".join(f"- {p}" for p in tied_priorities)
    prompt = (
        f"You are roleplaying as the following person:\n{persona_description}\n\n"
        f"Here is how you think about participatory budgeting:\n{profile}\n\n"
        "You marked all of the following priorities as ones you'd want even if it meant "
        f"giving up other priorities:\n{priorities_text}\n\n"
        "Among these, which matter most? Order them from most important to least "
        "important.\n\n"
        "Respond with ONLY a JSON list of the priority strings, most important first. "
        "Use the exact text given above."
    )
    response = call_llm(prompt)
    response = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        ranked = json.loads(response)
        def normalize(s): return " ".join(s.lower().split())
        by_norm = {normalize(p): p for p in tied_priorities}
        ordered = [by_norm[normalize(p)] for p in ranked if normalize(p) in by_norm]
        # include anything the model dropped, so nothing silently disappears
        ordered += [p for p in tied_priorities if p not in ordered]
        return ordered
    except (json.JSONDecodeError, KeyError):
        return tied_priorities


def select_top_priorities(persona_description, profile, importance_ratings, n=3):
    """
    Matches the live survey's tie-break mechanism: priorities at the top tier
    ("I'd want this even if...") are ranked by the respondent via drag-and-drop
    (QID161) when there's more than one, rather than broken randomly. If fewer
    than n priorities reach the top tier, the remaining slots are filled from the
    next tier down, in the order they were originally elicited (the survey doesn't
    show a ranking step for that shortfall case).
    """
    tiered = {}
    for priority, level in importance_ratings.items():
        tiered.setdefault(LIKERT_ORDER.get(level, 3), []).append(priority)

    ordered = []
    for tier_level in sorted(tiered.keys(), reverse=True):
        tier_priorities = tiered[tier_level]
        if len(ordered) >= n:
            break
        if tier_level == LIKERT_ORDER[LIKERT_MOST] and len(tier_priorities) > 1:
            tier_priorities = rank_tied_priorities(persona_description, profile, tier_priorities)
        ordered.extend(tier_priorities)

    return ordered[:n]


LABEL_TO_CATEGORY = {
    "supports funding this project": "good",
    "opposes funding this project": "bad",
    "could support funding this project, depending on which other projects are funded": "depends",
    "is irrelevant to this project": "neither",
}


def classify_one_priority(persona_description, profile, priority, projects, cache=None, cache_key=None):
    """
    One LLM call for one priority: matches QID196/197/198 (gate) + QID157/158/160
    (per-project matrix) — first ask explicitly whether the priority relates to
    particular projects or only to the set as a whole; if it relates to particular
    projects, rate EVERY available project on the real survey's 4-point scale.

    Called once per priority (rather than batching every priority into one prompt)
    so response size stays bounded by project-pool size regardless of how many
    priorities a respondent has -- a respondent with 12 priorities against a
    13-project pool would otherwise risk truncating a single combined JSON response.

    cache/cache_key: optional, see build_utility_functions_v1's docstring for the
    caching contract -- same idea, applied to the classification dict instead of
    Python source.
    """
    if cache is not None and cache_key is not None and cache_key in cache:
        return cache[cache_key]

    projects_text = "\n".join(f"- {name}: {desc}" for name, desc in projects.items())
    degradation = respondent_instructions(persona_description)

    prompt = f"""
        You are roleplaying as the following person:

        {persona_description}

        Here is how you think about participatory budgeting:

        {profile}

        This is one of your priorities:

        "{priority}"

        The available projects are:

        {projects_text}

        First decide whether this priority relates to particular projects, or whether
        it only relates to the set of projects as a whole rather than any particular
        project (e.g. "I want the budget to be fair" wouldn't point to specific
        projects).

        If it relates to particular projects, then for EACH available project, decide
        whether — from this priority's point of view — that project:
          "Supports funding this project"
          "Opposes funding this project"
          "Could support funding this project, depending on which other projects are funded"
          "Is irrelevant to this project"

        Guidelines:
        - Rate every project if this priority relates to particular projects. Don't skip any.
        - Think like the respondent, not like an objective expert — there are no right answers.
        - Use the project names exactly as written.
        {degradation}

        Respond with ONLY a JSON object of one of these two forms:

        {{"relates_to_particular_projects": true, "projects": {{"Project A": "Supports funding this project", "Project B": "Opposes funding this project", "Project C": "Is irrelevant to this project", "Project D": "Could support funding this project, depending on which other projects are funded"}}}}

        {{"relates_to_particular_projects": false}}

        Use the project names EXACTLY as written above, and if it relates to
        particular projects, include every project listed above.
        """

    response = call_llm(prompt)
    response = (
        response.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

    result = {"not_about_projects": True, "projects": {}}
    try:
        data = json.loads(response)
        relates = bool(data.get("relates_to_particular_projects", False))
        proj_map = {}
        if relates:
            proj_map = {
                proj: LABEL_TO_CATEGORY.get(str(label).strip().lower(), "neither")
                for proj, label in data.get("projects", {}).items()
            }
        result = {"not_about_projects": not relates, "projects": proj_map}
    except (json.JSONDecodeError, AttributeError):
        pass
    if cache is not None and cache_key is not None:
        cache[cache_key] = result
    return result


def generate_priority_classifications(persona_description, profile, priorities, projects,
                                       cache=None, cache_key_prefix=None, priority_key_fn=None):
    """
    Classifies EVERY priority passed in (not just a top-3 subset) against the
    project pool, one LLM call per priority via classify_one_priority -- this used
    to only run on a respondent's top-3 priorities (matching the real survey's now-
    removed QID196/197/198 + QID157/158/160 questions, which only ever asked about
    the top 3), leaving every other priority to fall back to a sign-only utility
    function with no explicit project association. Since that survey question is
    gone, there's no reason to keep that scoping -- every priority gets the same
    explicit per-project classification now.

    cache/cache_key_prefix: optional. When given, each priority's classification is
    cached under f"{cache_key_prefix}::classify::{normalize(priority)}" -- pass a
    prefix that's stable and unique per respondent (e.g. their response ID) to skip
    re-calling the LLM on re-runs.

    priority_key_fn: optional. When given, transforms a priority's raw text before it's
    used to build the LLM cache key -- NOT the returned dict's key, which always stays
    normalize(p), so callers keep looking results up by the respondent's own wording.
    Lets a caller route a priority onto a previously-cached, sufficiently-similar
    priority's classification instead of generating a fresh one (see
    priority_dedup.PriorityRegistry, used by run_wave0_real_validation.py's optional
    priority-reuse mode).

    Returns:
        {
            "helping seniors": {
                "not_about_projects": False,
                "projects": {"Community Center": "good", "Public EV Chargers": "neither", ...},
            },
            "fairness in general": {"not_about_projects": True, "projects": {}},
            ...
        }
    """
    def _key(p):
        if not cache_key_prefix:
            return None
        key_text = priority_key_fn(p) if priority_key_fn else p
        return f"{cache_key_prefix}::classify::{normalize(key_text)}"

    return {
        normalize(p): classify_one_priority(persona_description, profile, p, projects,
                                             cache=cache, cache_key=_key(p))
        for p in priorities
    }


def generate_validation_answers(persona_description, profile, elicited_criteria, bundle_pairs, projects):
    """
    Matches Validation.rural/urban: for each comparison, respondents pick a bundle
    (QID92-style), rate each bundle individually on a 3-point like/neutral/dislike
    scale (QID194-style), report certainty 1-5 (QID94-style), and select which of
    their OWN previously-written priorities mattered — a checklist/recognition task,
    not free recall — with an "Other:" write-in (QID93-style). This runs BEFORE
    importance is rated (see run_persona_tests), so it's grounded in the raw
    elicited criteria list, not yet-computed importance ratings.
    """
    projects_list = "\n".join(f"- {name}: {desc}" for name, desc in projects.items())
    criteria_text = "\n".join(f"- {c}" for c in elicited_criteria)
    pairs_text = "".join(
        f"\nComparison {i+1}:\n  Bundle A: {', '.join(bA)}\n  Bundle B: {', '.join(bB)}\n"
        for i, (bA, bB) in enumerate(bundle_pairs)
    )
    degradation = respondent_instructions(persona_description)

    prompt = (
        f"You are roleplaying as the following person:\n{persona_description}\n\n"
        f"Here is how you think about participatory budgeting:\n{profile}\n\n"
        f"Available projects:\n{projects_list}\n\n"
        f"Earlier, you listed these priorities:\n{criteria_text}\n\n"
        "For each comparison below:\n"
        "1. Which set of projects should the town fund, in your opinion? Choose A or B.\n"
        "2. How do you feel about EACH individual set of four projects on its own — "
        "\"dislike\" (-1), \"neutral\" (0), or \"like\" (+1)?\n"
        "3. How certain are you about your choice? Pick a level 1-5, where "
        "1 = \"Not at all certain\", 2 = \"Somewhat certain\", 3 = \"Moderately certain\", "
        "4 = \"Very certain\", 5 = \"Extremely certain\".\n"
        "4. Which of YOUR priorities listed above mattered most in choosing this option over "
        "the other one? Select 1-3 from your own list above, using the exact text. If an "
        "important priority isn't in your list, add it as 'Other: <new priority>'.\n"
        f"{pairs_text}"
        f"{degradation}\n\n"
        "Respond with ONLY a JSON list, one object per comparison in order:\n"
        "[{\"chosen\": 0, \"bundle_a_rating\": 1, \"bundle_b_rating\": -1, \"certainty\": 4, "
        "\"cited_priorities\": [\"criterion\"]}, ...]\n"
        "0 = Bundle A, 1 = Bundle B."
    )
    response = call_llm(prompt)
    response = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        answers = json.loads(response)
        for a in answers:
            a["chosen"] = int(a["chosen"])
            a["certainty"] = int(a["certainty"]) if a.get("certainty") is not None else None
        return answers
    except (json.JSONDecodeError, KeyError, ValueError):
        return [{"chosen": 0, "cited_priorities": [], "certainty": None} for _ in bundle_pairs]


def print_persona_result(persona_description, profile, criteria_by_question,
                         importance_ratings, validation_answers, result, bundle_pairs, priority_classifications,
                         demographics=None, file=None):
    out = lambda *a, **k: print(*a, file=file, **k)
    out("=" * 70)
    out(f"PERSONA: {persona_description}")
    if demographics:
        out(f"DEMOGRAPHICS: area={demographics.get('demographic_area')} age={demographics.get('age')}")
    out()
    out("PROFILE:")
    for line in profile.strip().split("\n"):
        out(f"  {line}")
    out()
    out("ELICITED CRITERIA + RATINGS:")
    n_priorities = 0
    for label, criteria_list in criteria_by_question.items():
        sign = CRITERIA_SIGN_BY_QUESTION.get(label, "positive")
        sign_marker = "+" if sign == "positive" else "-"
        out(f"  [{label}] ({sign_marker})")
        for c in criteria_list:
            def normalize(s): return " ".join(s.lower().split())
            rating = importance_ratings.get(normalize(c), "?")
            out(f"    • {c}  ->  {rating}")
            n_priorities += 1
    out(f"TOTAL ELICITED PRIORITIES: {n_priorities}")
    out()
    out(f"TOP PRIORITIES (per-project classification): {list(priority_classifications.keys())}")
    out(json.dumps(priority_classifications, indent=2))
    out()
    out("VALIDATION PREDICTIONS:")
    for i, (pred, truth, (bA, bB)) in enumerate(
        zip(result["per_pair"], validation_answers, bundle_pairs)
    ):
        chosen_label = "A" if truth["chosen"] == 0 else "B"
        compatible = pred["compatible_outcomes"]
        model_str = compatible[0] if len(compatible) == 1 else "[" + ", ".join(compatible) + "]"
        out(f"  Pair {i+1}:")
        out(f"    A: {bA}")
        out(f"    B: {bB}")
        out(f"    Persona chose: {chosen_label}  |  Cardinal model: {model_str}  |  "
              f"Consistent: {pred['is_consistent']}")
        out(f"    Delta=[{pred['Delta_min']:.2f}, {pred['Delta_max']:.2f}]  "
              f"Uncertainty: {pred['model_uncertainty']:.2f}")
        out(f"    Lexicographic model: {pred['lex_outcome']}  |  "
              f"Consistent: {pred['lex_consistent']}")
        out(f"    Cited: {truth['cited_priorities']}")
    out()
    out(f"  Consistency rate:  {result['consistency_rate']:.0%}")
    out()
    out(
        f"  Lexicographic consistency rate: "
        f"{result['lexicographic_consistency_rate']:.0%}"
    )
    if result["robust_accuracy"] is not None:
        out(f"  Strong pred accuracy: {result['robust_accuracy']:.0%} "
              f"({result['n_robust']}/{result['n_total']} strong predictions)")
    else:
        out("  No strong predictions made")
    if result["calibration"] is not None:
        out(f"  Calibration (Pearson r): {result['calibration']:.2f}")
    out()


def run_persona_tests(tau_r=0.0, tau_kappa=0.0):
    """
    Ordering here matches the live Survey Flow: elicitation (V1.start) shows BOTH
    project pools to every respondent regardless of area; only the validation-phase
    bundle comparisons (Validation.rural/urban) filter to the respondent's own
    area's pool. Importance rating (V1.end, "importance") happens AFTER the bundle
    choices, not before — so validation answers are grounded in the raw elicited
    criteria list, and the top-3 priority selection/classification only happens
    once importance ratings exist.
    """
    results_filename = f"results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    print(f"Writing results to {results_filename}")

    with open(results_filename, "w") as f:
        for persona_description, area, age in TEST_PERSONAS:
            print(f"\nRunning persona: {persona_description[:60]}...")
            if area == "urban":
                validation_projects, bundle_pairs = PROJECTS_URBAN, BUNDLE_PAIRS_URBAN
            else:
                validation_projects, bundle_pairs = PROJECTS_RURAL, BUNDLE_PAIRS_RURAL

            demographics = {"age": age, "demographic_area": area}
            profile = generate_persona_profile(persona_description)

            # Elicitation: both pools, regardless of area.
            criteria_by_question = generate_criteria(
                persona_description, profile, CRITERIA_QUESTIONS_URBAN, PROJECTS_ALL
            )
            elicited_criteria = []
            for criteria_list in criteria_by_question.values():
                elicited_criteria.extend(criteria_list)
            seen = set()
            elicited_criteria = [c for c in elicited_criteria if not (c in seen or seen.add(c))]

            # Bundle-pair validation choices, BEFORE importance is rated — matches
            # the real question order.
            validation_answers = generate_validation_answers(
                persona_description, profile, elicited_criteria, bundle_pairs, validation_projects
            )

            # Importance rating, AFTER the bundle choices.
            importance_ratings = generate_likert_ratings(persona_description, profile, criteria_by_question)
            # Kept for reporting context (which 3 priorities are non-negotiable) but no
            # longer used to scope classification -- every elicited priority is classified.
            top_priorities = select_top_priorities(persona_description, profile, importance_ratings, n=3)
            priority_classifications = generate_priority_classifications(
                persona_description, profile, elicited_criteria, validation_projects
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

            result = run_validation(
                demographics, "v1", survey_info,
                bundle_pairs, validation_answers,
                tau_r=tau_r, tau_kappa=tau_kappa,
                file=f,
            )
            if result:
                print_persona_result(
                    persona_description, profile, criteria_by_question,
                    importance_ratings, validation_answers, result, bundle_pairs, priority_classifications,
                    demographics=demographics, file=f,
                )


if __name__ == "__main__":
    run_persona_tests()