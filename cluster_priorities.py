"""
Looks across every real respondent (Wave 0 + Wave 1, both survey variants) at the raw
priority text they wrote, and groups near-duplicate/similar priorities together -- e.g.
"Food pantry because people need help" and "A food pantry is needed to help feed the
low income residents" should land in the same cluster even though the wording differs.
Answers: how much genuine duplication is there in what people write, and does it
cluster into recognizable recurring themes?

Lightweight, dependency-free (no sklearn/embeddings) TF-IDF + average-linkage
clustering over the priority text -- everything here is numpy/scipy, both already
required by plot_cardinal_margins.py.

Two outputs:
  1. results/priority_clusters_<ts>.md -- every cluster with 2+ members, listing each
     member's exact text, respondent, and which wave/pool it came from.
  2. results/priority_clusters_<ts>.png -- horizontal bar chart of the largest clusters.

This clustering is deliberately corpus-wide (pools mixed together) since the goal here
is "do people write about similar things," not project-pool-specific reuse -- that's
priority_dedup.PriorityRegistry's job (used online by run_wave0_real_validation.py's
optional priority-reuse mode), which scopes matches to a single project pool because a
cached utility function's Python source references literal project names from one
specific pool.

Positive priorities ("what you want") and negative ones ("what you'd want to avoid" --
elicited via the survey's separate bundle_bad/open-bad question) are clustered in
COMPLETELY SEPARATE spaces, never compared against each other at all. Token-overlap/
TF-IDF similarity is sign-blind -- "Security Cameras" and "I don't want more security
cameras, people want privacy" share most of their non-stopword tokens, since negation
words aren't distinctive on their own -- so without this split they'd land in the same
cluster despite being opposites. Splitting the corpus up front makes that structurally
impossible rather than just unlikely (a same-sign-only distance penalty could still be
outvoted by enough close same-sign neighbors pulling an opposite-sign item in via
average-linkage chaining).

After the TF-IDF pass, runs one LLM merge-check call PER SIGN BUCKET (so two calls
total, never mixing positive/negative) that looks at every cluster's representative
text and proposes merges the token-overlap metric missed -- typos ("pantrys" vs
"pantry"), singular/plural, or abbreviation-vs-full-name ("Slower, Safer Roads" vs
"Safe roads"). This step is skipped automatically (with a printed note) if
ANTHROPIC_API_KEY isn't set, so the script still runs for free/offline if you just want
the TF-IDF pass.

Run locally:
    python3 cluster_priorities.py              # all respondents, both variants
    python3 cluster_priorities.py SimProjects   # only SimProjects respondents
    python3 cluster_priorities.py V1            # only V1 respondents
"""
import json
import os
import re
from collections import Counter
from datetime import datetime

import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

from run_wave0_real_validation import compute_criteria_signs
from validation_v1_testing import call_llm, normalize

RESULTS_DIR = "results"

SIGN_COLOR = {"positive": "#4C72B0", "negative": "#C44E52"}
SIGN_DESCRIPTION = {
    "positive": "things the respondent WANTS funded",
    "negative": "things the respondent wants to AVOID / would be unhappy to see funded",
}

# Distance threshold for merging two priorities into the same cluster (0 = identical
# token sets, 1 = no shared tokens at all). Tuned by eyeballing the resulting clusters
# against the actual survey text -- lower to split more aggressively, raise to lump
# more loosely-related priorities together.
DISTANCE_THRESHOLD = 0.6

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if",
    "in", "is", "it", "its", "just", "like", "more", "my", "no", "not", "of", "on",
    "or", "our", "should", "so", "some", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "too", "very", "want", "was", "we", "were",
    "what", "when", "which", "who", "will", "with", "would", "you", "your", "also",
    "us", "me", "than", "about", "into", "one", "any", "all", "get", "having",
}


def tokenize(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if w and w not in STOPWORDS and len(w) > 2]


def build_tfidf_matrix(docs_tokens):
    vocab = {}
    for tokens in docs_tokens:
        for t in set(tokens):
            vocab.setdefault(t, len(vocab))
    n_docs, n_vocab = len(docs_tokens), len(vocab)
    df = np.zeros(n_vocab)
    for tokens in docs_tokens:
        for t in set(tokens):
            df[vocab[t]] += 1
    idf = np.log((n_docs + 1) / (df + 1)) + 1.0
    mat = np.zeros((n_docs, n_vocab))
    for i, tokens in enumerate(docs_tokens):
        for t, c in Counter(tokens).items():
            mat[i, vocab[t]] = c * idf[vocab[t]]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def collect_priorities(variant_filter=None):
    """Every elicited priority across both waves and both survey variants, tagged with
    where it came from. Each wave's loader is tried independently -- if one wave's raw
    data isn't present in this checkout, the other still runs.

    variant_filter: optional, e.g. "SimProjects" or "V1" -- restricts to respondents of
    that one variant. None (default) keeps both, as before.
    """
    entries = []
    sources = []

    try:
        from format_wave0_results import load_all as load_wave0
        v1, sim = load_wave0()
        sources.append(("Wave0", v1 + sim))
    except Exception as e:
        print(f"  (skipping Wave 0: {e})")

    try:
        from format_wave1_results import load_all as load_wave1
        v1, sim = load_wave1()
        sources.append(("Wave1", v1 + sim))
    except Exception as e:
        print(f"  (skipping Wave 1: {e})")

    for wave_label, respondents in sources:
        for r in respondents:
            if variant_filter is not None and r["variant"] != variant_filter:
                continue
            area = (r["area_selfreport"] or "").lower()
            pool_bucket = "urban" if area == "urban" else "rural"
            pool_id = f"{r['variant']}::{pool_bucket}"
            signs = compute_criteria_signs(r)
            for c in r["criteria"]:
                text = (c["text"] or "").strip()
                if not text:
                    continue
                entries.append({
                    "text": text,
                    "wave": wave_label,
                    "variant": r["variant"],
                    "pool_id": pool_id,
                    "response_id": r["response_id"],
                    "importance": c["importance"],
                    "sign": signs.get(normalize(text), "positive"),
                })
    return entries


def cluster_entries(entries, distance_threshold=DISTANCE_THRESHOLD):
    """Groups a list of entries (each with a "text" key) into clusters via TF-IDF +
    average-linkage. Callers are responsible for only ever passing in entries that are
    safe to compare against each other -- see the sign-splitting in main() below."""
    if len(entries) < 2:
        return [[e] for e in entries]

    docs_tokens = [tokenize(e["text"]) for e in entries]
    mat = build_tfidf_matrix(docs_tokens)

    sim_matrix = np.clip(mat @ mat.T, 0.0, 1.0)
    dist = 1.0 - sim_matrix
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # guard against float asymmetry before squareform
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    clusters = {}
    for label, entry in zip(labels, entries):
        clusters.setdefault(label, []).append(entry)
    return list(clusters.values())


def llm_merge_groups(cluster_reps, sign):
    """
    One LLM call: given the representative text for every cluster in one sign bucket,
    asks which clusters actually refer to the same real-world priority despite the
    TF-IDF pass missing it (typos, plurals, abbreviation vs. full name, etc.), and
    which are merely related-but-distinct (which should NOT be merged). Returns a list
    of merge-groups, each a list of 2+ indices into cluster_reps; empty list if nothing
    should merge, the call fails, or the response can't be parsed.
    """
    numbered = "\n".join(f"{i}: {text}" for i, text in enumerate(cluster_reps))
    prompt = f"""Below is a numbered list of short priority statements from a participatory
budgeting survey. Each one is a representative example already standing in for a
cluster of near-identical wordings -- these are all {SIGN_DESCRIPTION[sign]}.

Some of these items actually refer to the exact same specific real-world priority/topic
even though the wording differs -- e.g. typos ("pantrys" vs "pantry"), singular vs.
plural, or an abbreviation/short name vs. the full project name (e.g. "Slower, Safer
Roads" vs "Safe roads"). Group ONLY items like that together.

Do NOT group items that are merely related but are actually distinct priorities (e.g.
"Food Pantry" and "Free Hygiene Products" are both basic-needs items but are NOT the
same priority -- leave them separate). When in doubt, do not group.

Items:
{numbered}

Respond with ONLY a JSON array of merge groups, where each group is a list of 2 or more
item numbers that all refer to the same specific priority. Omit any item number that
doesn't belong in a merge group -- do not list singletons, and do not include groups of
size 1. Example response: [[3, 17], [9, 42, 108]]

If nothing should be merged, respond with: []"""

    try:
        response = call_llm(prompt, max_tokens=4096)
    except Exception as e:
        print(f"    (LLM merge check failed for {sign}: {e} -- skipping)")
        return []

    response = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    # The model occasionally ignores "respond with ONLY JSON" and prepends a sentence
    # of prose -- fall back to pulling out the first [...] block rather than failing
    # the whole pass over what's usually just one stray leading line.
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match:
        response = match.group(0)
    try:
        groups = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        print(f"    (LLM merge check for {sign} returned unparsable JSON -- skipping)")
        return []

    valid_groups = []
    for g in groups if isinstance(groups, list) else []:
        if not isinstance(g, list):
            continue
        idxs = sorted({i for i in g if isinstance(i, int) and 0 <= i < len(cluster_reps)})
        if len(idxs) >= 2:
            valid_groups.append(idxs)
    return valid_groups


def apply_merge_groups(clusters, groups):
    """Union-find over cluster indices so that if the LLM's groups overlap/chain (e.g.
    [[0,1],[1,2]]) they still merge transitively into one cluster rather than
    conflicting."""
    parent = list(range(len(clusters)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for g in groups:
        for a, b in zip(g, g[1:]):
            union(a, b)

    merged = {}
    for i, cluster in enumerate(clusters):
        merged.setdefault(find(i), []).extend(cluster)
    return list(merged.values())


def summarize_clusters_for_chart(cluster_text_lists):
    """
    One LLM call: given every chart cluster's full list of member texts, returns a
    short (3-6 word) label per cluster describing its shared theme -- e.g. "Food
    pantry" rather than truncating "Food Pantry because people need help" into
    "Food Pantry because people ne…" or showing the full sentence on the chart.
    Falls back to each cluster's shortest raw member (the old behavior) for any
    cluster whose summary the LLM call doesn't return cleanly, and for all clusters if
    the call fails outright or ANTHROPIC_API_KEY isn't set. The underlying cluster
    membership is never touched by this -- it only changes what text labels the bar.
    """
    fallback = [min(texts, key=len) for texts in cluster_text_lists]
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return fallback

    blocks = []
    for i, texts in enumerate(cluster_text_lists):
        examples = "\n".join(f"    - {t}" for t in texts[:6])
        blocks.append(f"{i}:\n{examples}")
    numbered = "\n".join(blocks)

    prompt = f"""Below are {len(cluster_text_lists)} numbered groups of near-duplicate priority
statements from a participatory budgeting survey (each group is a cluster of
differently-worded restatements of the same underlying priority; only up to 6 example
members are shown per group). For EACH group, write a short label (3-6 words, plain
text, no leading dash or number) that names the shared priority/topic -- concise enough
to use as a chart axis label, specific enough to distinguish it from the other groups.

Groups:
{numbered}

Respond with ONLY a JSON array of {len(cluster_text_lists)} strings, one label per
group, in the same order as the groups above."""

    try:
        response = call_llm(prompt, max_tokens=2048)
    except Exception as e:
        print(f"    (cluster summarization failed: {e} -- using raw text instead)")
        return fallback

    response = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match:
        response = match.group(0)
    try:
        labels = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        print("    (cluster summarization returned unparsable JSON -- using raw text instead)")
        return fallback

    if not isinstance(labels, list) or len(labels) != len(cluster_text_lists):
        print(f"    (cluster summarization returned {len(labels) if isinstance(labels, list) else 'non-list'} "
              f"labels for {len(cluster_text_lists)} clusters -- using raw text instead)")
        return fallback

    return [
        str(label).strip() if isinstance(label, str) and label.strip() else fallback[i]
        for i, label in enumerate(labels)
    ]


def main(variant_filter=None):
    entries = collect_priorities(variant_filter=variant_filter)
    n_respondents = len(set(e["response_id"] for e in entries))
    scope = f" ({variant_filter} only)" if variant_filter else ""
    print(f"{len(entries)} elicited priorities across {n_respondents} respondents{scope}")
    if len(entries) < 3:
        print("Not enough priorities to cluster.")
        return

    # Cluster positive ("what you want") and negative ("what you'd want to avoid")
    # priorities in fully separate spaces -- see module docstring for why this can't
    # just be a distance penalty within one combined clustering pass.
    pos_entries = [e for e in entries if e["sign"] != "negative"]
    neg_entries = [e for e in entries if e["sign"] == "negative"]
    print(f"  {len(pos_entries)} positive, {len(neg_entries)} negative")

    use_llm_merge = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not use_llm_merge:
        print("  ANTHROPIC_API_KEY not set -- skipping LLM merge-check pass "
              "(clusters below are TF-IDF-only; typo/plural/abbreviation near-misses won't be merged).")

    all_clusters = []  # list of (sign, cluster) across both buckets
    for sign, bucket in [("positive", pos_entries), ("negative", neg_entries)]:
        clusters = cluster_entries(bucket)
        if use_llm_merge and len(clusters) >= 2:
            reps = [min((e["text"] for e in c), key=len) for c in clusters]
            print(f"  checking {len(clusters)} {sign} clusters for LLM-detected merges...")
            groups = llm_merge_groups(reps, sign)
            if groups:
                merged_away = sum(len(g) for g in groups) - len(groups)
                print(f"    merging {len(groups)} group(s) in {sign} bucket "
                      f"({len(clusters)} -> {len(clusters) - merged_away} clusters)")
                clusters = apply_merge_groups(clusters, groups)
            else:
                print(f"    no merges found in {sign} bucket")
        for cluster in clusters:
            all_clusters.append((sign, cluster))

    all_clusters.sort(key=lambda sc: len(sc[1]), reverse=True)
    multi = [(sign, c) for sign, c in all_clusters if len(c) > 1]
    n_in_multi = sum(len(c) for _, c in multi)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Markdown report: every cluster with 2+ members ---
    md = []
    md.append(f"# Priority Clustering{scope}\n")
    md.append(
        f"{len(entries)} elicited priorities from {n_respondents} respondents{scope} "
        f"({len(pos_entries)} positive / {len(neg_entries)} negative), grouped by "
        f"TF-IDF + average-linkage clustering on the raw priority text (distance "
        f"threshold {DISTANCE_THRESHOLD}), plus one LLM merge-check pass per sign "
        f"bucket (skipped if ANTHROPIC_API_KEY wasn't set) that catches near-misses the "
        f"token-overlap metric alone can't, e.g. typos/plurals or an abbreviation vs. "
        f"the full name (\"Slower, Safer Roads\" vs. \"Safe roads\"). Positive and "
        f"negative priorities are clustered completely separately and never compared to "
        f"each other at any stage, so a \"wants X\" priority can never land in the same "
        f"cluster as an \"opposes X\" one. {len(all_clusters)} clusters total; "
        f"{len(multi)} have 2+ members.\n"
    )
    md.append(
        f"**{n_in_multi}/{len(entries)} priorities ({n_in_multi/len(entries):.0%}) belong "
        f"to a cluster shared with at least one other priority** — this is roughly the "
        f"population `priority_dedup.py`'s online reuse cache is meant to catch (that "
        f"cache additionally requires the match to be in the same project pool, so its "
        f"actual hit rate will be somewhat lower than this corpus-wide figure).\n"
    )
    for i, (sign, cluster) in enumerate(multi, 1):
        pools = Counter(e["pool_id"] for e in cluster)
        pool_str = ", ".join(f"{p} x{n}" for p, n in pools.most_common())
        md.append(f"## Cluster {i} [{sign}] — {len(cluster)} priorities ({pool_str})\n")
        for e in cluster:
            md.append(f"- [{e['wave']}/{e['response_id']}] *({e['importance']})* {e['text']}")
        md.append("")

    suffix = f"_{variant_filter.lower()}" if variant_filter else ""
    md_path = os.path.join(RESULTS_DIR, f"priority_clusters{suffix}_{ts}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print("wrote", md_path)

    # --- Bar chart: largest clusters, color-coded by sign ---
    top_n = min(25, len(multi))
    if top_n == 0:
        print("No clusters with 2+ members -- skipping chart.")
        return
    top = multi[:top_n]
    sizes = [len(c) for _, c in top]
    colors = [SIGN_COLOR[sign] for sign, _ in top]
    print(f"  summarizing {top_n} chart clusters into short labels...")
    reps = summarize_clusters_for_chart([[e["text"] for e in c] for _, c in top])

    # Left margin needs to scale with the longest label or matplotlib clips it against
    # the figure's fixed width -- tight_layout() alone doesn't reliably rescue very long
    # tick labels, and bbox_inches="tight" on savefig (below) only trims excess
    # whitespace, it doesn't add room that was never there.
    max_label_len = max((len(r) for r in reps), default=0)
    fig, ax = plt.subplots(figsize=(10 + max_label_len * 0.09, max(4, 0.35 * top_n)))
    y = np.arange(top_n)
    ax.barh(y, sizes, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(reps, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Priorities in this cluster")
    ax.set_title(f"Largest priority clusters{scope} ({len(entries)} priorities, {n_respondents} respondents)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SIGN_COLOR[s]) for s in ("positive", "negative")]
    ax.legend(handles, ["positive (wants)", "negative (avoid)"], loc="lower right", fontsize=8)
    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, f"priority_clusters{suffix}_{ts}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print("wrote", png_path)


if __name__ == "__main__":
    import sys
    main(variant_filter=sys.argv[1] if len(sys.argv) > 1 else None)
