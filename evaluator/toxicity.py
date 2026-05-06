import re
import json
import os

# ------------------------------------------------------------------ #
# Toxicity lexicon                                                     #
# Loaded from external JSON if available, otherwise falls back to     #
# the built-in default. This allows the lexicon to be maintained      #
# without touching source code.                                        #
#                                                                      #
# Expected JSON format: {"word": weight, ...}                         #
# Weights should be in (0.0, 1.0] — higher = more severe.            #
# ------------------------------------------------------------------ #

DEFAULT_TOXIC_WORDS = {
    # Violence / threats
    "kill":        0.9,
    "murder":      0.9,
    "attack":      0.8,
    "threat":      0.8,
    "harm":        0.7,
    "hurt":        0.7,
    "destroy":     0.7,
    "violence":    0.8,
    # Harassment / hostility
    "hate":        0.6,
    "harass":      0.8,
    "bully":       0.7,
    "abuse":       0.8,
    "discriminate":0.8,
    "racist":      0.9,
    "sexist":      0.9,
    # Insults / demeaning
    "stupid":      0.5,
    "idiot":       0.5,
    "moron":       0.5,
    "useless":     0.4,
    "incompetent": 0.4,
    "worthless":   0.5,
    # Workplace-specific
    "fired":       0.3,
    "terminate":   0.3,
    "lawsuit":     0.4,
    "sue":         0.4,
}

_LEXICON_PATH = os.environ.get(
    "TOXIC_LEXICON_PATH",
    "evaluator/toxic_lexicon.json"
)


def load_lexicon(path=_LEXICON_PATH):
    """
    Load the toxicity lexicon from an external JSON file.

    Falls back to DEFAULT_TOXIC_WORDS if the file does not exist
    or cannot be parsed, so the evaluator always has a usable lexicon.

    Args:
        path (str): Path to the JSON lexicon file.

    Returns:
        dict: {word: weight} mapping.
    """
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lexicon = json.load(f)
                if isinstance(lexicon, dict) and lexicon:
                    return lexicon
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Toxicity] Failed to load lexicon from '{path}': {e}. "
                  f"Using default lexicon.")
    return DEFAULT_TOXIC_WORDS


TOXIC_WORDS = load_lexicon()

# Maximum possible raw score — used for normalisation
_MAX_RAW_SCORE = sum(TOXIC_WORDS.values()) or 1.0

# ------------------------------------------------------------------ #
# Negation patterns                                                    #
# Detects common HR phrases where a toxic-looking word is used        #
# neutrally, e.g. "cancel the kill switch", "end the hate speech     #
# policy review".                                                      #
# ------------------------------------------------------------------ #

_NEGATION_PREFIXES = [
    r"cancel", r"remove", r"stop", r"end", r"prevent",
    r"avoid", r"address", r"resolve", r"discuss", r"review",
    r"policy\s+on", r"training\s+on", r"awareness\s+of"
]

_NEGATION_PATTERN = re.compile(
    r"(?:" + "|".join(_NEGATION_PREFIXES) + r")"
    r"\s+(?:\w+\s+){0,3}",
    re.IGNORECASE
)


def _is_negated(word, text):
    """
    Return True if the toxic word appears to be used in a neutral /
    negated context in the given text.

    Looks for a negation prefix within a short window before the word.

    Args:
        word (str): The toxic keyword.
        text (str): Full lowercased input text.

    Returns:
        bool
    """
    pattern = re.compile(
        r"(?:" + "|".join(_NEGATION_PREFIXES) + r")"
        r"\s+(?:\w+\s+){0,3}" + re.escape(word),
        re.IGNORECASE
    )
    return bool(pattern.search(text))


# ------------------------------------------------------------------ #
# Severity classification                                              #
# ------------------------------------------------------------------ #

def get_severity(score):
    """
    Map a normalised toxicity score to a human-readable severity tier.

    Args:
        score (float): Normalised toxicity score in [0.0, 1.0].

    Returns:
        str: "high" | "medium" | "low" | "none"
    """
    if score >= 0.8:    return "high"
    elif score >= 0.5:  return "medium"
    elif score > 0.0:   return "low"
    return "none"


# ------------------------------------------------------------------ #
# Core detection                                                       #
# ------------------------------------------------------------------ #

def is_toxic(text, threshold=0.5):
    """
    Determine whether text is toxic based on weighted keyword matching.

    Detection pipeline:
      1. Empty / whitespace-only input → safe, score 0.0
      2. Whole-word regex match against TOXIC_WORDS lexicon
      3. Negation check — neutral HR usage suppresses the match
      4. Normalised score = raw_weight_sum / max_possible_score
      5. Severity tier assigned from normalised score

    Args:
        text      (str):   Input text to evaluate.
        threshold (float): Normalised score at or above which text is
                           flagged as toxic (default: 0.5).

    Returns:
        dict: {
            "toxic"        : bool,
            "score"        : float,   # normalised to [0.0, 1.0]
            "threshold"    : float,
            "severity"     : str,     # "high" | "medium" | "low" | "none"
            "matched_terms": list,
            "reason"       : str
        }
    """
    # ---- empty / whitespace guard --------------------------------- #
    if not text or not text.strip():
        return {
            "toxic":         False,
            "score":         0.0,
            "threshold":     threshold,
            "severity":      "none",
            "matched_terms": [],
            "reason":        "Empty input"
        }

    text_lower = text.lower()
    raw_score  = 0.0
    matched_terms = []

    for word, weight in TOXIC_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text_lower):
            if not _is_negated(word, text_lower):
                raw_score += weight
                matched_terms.append(word)

    # Normalise score against the maximum possible accumulation
    score      = min(raw_score / _MAX_RAW_SCORE, 1.0)
    is_flagged = score >= threshold
    severity   = get_severity(score)

    reason = (
        f"Detected toxic terms: {', '.join(matched_terms)}"
        if matched_terms
        else "No toxic terms detected"
    )

    return {
        "toxic":         is_flagged,
        "score":         round(score, 4),
        "threshold":     threshold,
        "severity":      severity,
        "matched_terms": matched_terms,
        "reason":        reason
    }


def toxicity_score(text):
    """
    Convenience wrapper: return just the normalised toxicity score.

    Delegates to is_toxic() to avoid duplicating detection logic.

    Args:
        text (str): Input text.

    Returns:
        float: Normalised toxicity score in [0.0, 1.0].
    """
    return is_toxic(text)["score"]
