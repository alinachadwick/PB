"""
Online near-duplicate detection for elicited priorities.

Used by run_wave0_real_validation.py's optional priority-reuse mode (run_one's
`priority_registry` param): before generating a fresh classification/utility function
for a priority via the LLM, check whether a sufficiently similar priority has already
been processed in the SAME project pool and, if so, reuse its cache entry instead of
paying for a new LLM call. Pools matter because a cached utility function's Python
source references literal project name strings from one specific project set
(urban/rural pools differ, V1 vs. SimProjects pools differ) -- a match across pools
would produce a function that references projects that don't exist in the other pool.

Uses the same token-Jaccard-style similarity idea as cluster_priorities.py's offline
TF-IDF clustering, simplified here to compare one new priority against a running list
of previously-seen ones (cheap, no need to rebuild a corpus-wide similarity matrix on
every lookup) rather than clustering a whole corpus at once.
"""
import json
import os
import re

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if",
    "in", "is", "it", "its", "just", "like", "more", "my", "no", "not", "of", "on",
    "or", "our", "should", "so", "some", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "too", "very", "want", "was", "we", "were",
    "what", "when", "which", "who", "will", "with", "would", "you", "your", "also",
    "us", "me", "than", "about", "into", "one", "any", "all", "get", "having",
}


def _stem(word):
    """Minimal, conservative suffix-stripping to catch common plurals/typos ("pantrys"
    vs "pantry", "roads" vs "road") without pulling in a full stemmer dependency.
    Deliberately narrow -- only strips endings very unlikely to change a word's core
    meaning, and skips short words entirely where stripping could produce nonsense
    (e.g. "bus" -> "bu", "gas" is left alone by the length guard)."""
    if len(word) <= 4:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("es") and not word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return {_stem(w) for w in text.split() if w and w not in STOPWORDS and len(w) > 2}


def jaccard(tokens_a, tokens_b):
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class PriorityRegistry:
    """
    Persisted (JSON) map of pool_id -> [{"canonical": raw priority text, "tokens": [..],
    "sign": "positive"|"negative"|None}]. Accumulates across separate script runs, not
    just within one process, so the reuse rate improves the more respondents you've run
    through it.

    resolve(pool_id, text, sign=None) returns the canonical text a caller should use for
    cache-key purposes: the priority's OWN text if nothing similar enough has been seen
    yet in this pool, or an earlier match's text if one exists -- routing the caller's
    cache lookup for that canonical text onto an existing classification/utility-
    function entry instead of generating a new one. Tracks `hits`/`misses` for reporting.

    sign: pass the priority's positive/negative framing (e.g. which open-ended survey
    question it came from -- "what you want" vs. "what you'd want to avoid") when known.
    A candidate match is skipped whenever both sides have a known sign and they differ.
    This matters because token-overlap similarity is sign-blind: "Security Cameras" and
    "I don't want more security cameras, people want privacy" share most of their
    non-stopword tokens (negation words like "don't"/"not" aren't distinctive tokens on
    their own), so without this guard a "wants X" priority could get silently matched
    onto an "opposes X" priority's classification/utility function -- inverting the
    result instead of approximating it.
    """

    def __init__(self, path=None, threshold=0.5):
        self.path = path
        self.threshold = threshold
        self.pools = {}
        self.hits = 0
        self.misses = 0
        if path and os.path.exists(path):
            with open(path) as f:
                self.pools = json.load(f)

    def resolve(self, pool_id, text, sign=None):
        tokens = tokenize(text)
        entries = self.pools.setdefault(pool_id, [])
        best_score, best_entry = 0.0, None
        for entry in entries:
            entry_sign = entry.get("sign")
            if sign is not None and entry_sign is not None and sign != entry_sign:
                continue
            score = jaccard(tokens, set(entry["tokens"]))
            if score > best_score:
                best_score, best_entry = score, entry
        if best_entry is not None and best_score >= self.threshold:
            self.hits += 1
            return best_entry["canonical"]
        self.misses += 1
        entries.append({"canonical": text, "tokens": sorted(tokens), "sign": sign})
        return text

    def save(self, path=None):
        path = path or self.path
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.pools, f, indent=1)
