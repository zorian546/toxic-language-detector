"""Shared preprocessing and inference for training and the browser dashboard."""

from collections import Counter
from dataclasses import dataclass
import html
import json
from math import isclose
from pathlib import Path
import re
import unicodedata

from nltk import NaiveBayesClassifier, __version__ as NLTK_VERSION, ngrams
from nltk.probability import DictionaryProbDist
from nltk.tokenize import TweetTokenizer


ROOT = Path(__file__).resolve().parent
TOKENIZER = TweetTokenizer(preserve_case=True, match_phone_numbers=False)
URL = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
MENTION = re.compile(r"(?<![\w@])@[a-z0-9_]+\b", re.IGNORECASE)
BOUNDARY = re.compile(r"<url>|<user>|[.!?…]+|[\r\n\v\f\u0085\u2028\u2029]+")
WORD = re.compile(r"(?u)[^\W_]+(?:'[^\W_]+)*\Z")
APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})
FORMAT_MARKS = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
NEGATIONS = frozenset({"no", "not", "nor", "never", "neither", "cannot"})


def clean(text):
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = FORMAT_MARKS.sub("", text.translate(APOSTROPHES)).lower()

    def mask_url(match):
        url = match.group()
        suffix = url[len(url.rstrip(".,!?;:)]}")):]
        return " <url> " + suffix

    text = URL.sub(mask_url, text)
    return MENTION.sub(" <user> ", text)


def duplicate_key(text):
    """Group equivalent case, whitespace, mask and Unicode forms before splitting."""
    return " ".join(clean(text).split())


class Preprocessor:
    def __init__(self, variants):
        self.variants = variants
        forms = sorted(variants, key=lambda form: (-len(form), form))
        self.variant_pattern = (
            re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, forms)) + r")(?!\w)")
            if forms else None
        )

    def segments(self, text):
        text = clean(text)
        if self.variant_pattern:
            text = self.variant_pattern.sub(lambda match: self.variants[match[0]], text)
        for segment in BOUNDARY.split(text):
            words = []
            for token in TOKENIZER.tokenize(segment):
                if token.startswith("#") and WORD.fullmatch(token[1:]):
                    token = token[1:]
                if WORD.fullmatch(token):
                    words.append(token)
                elif token not in {",", "'", '"', "“", "”", "(", ")"} and words:
                    yield tuple(words)
                    words = []
            if words:
                yield tuple(words)

    def counts(self, text):
        return count_segments(self.segments(text))


def count_segments(segments):
    return Counter(
        " ".join(group)
        for segment in segments
        for size in (1, 2, 3)
        for group in ngrams(segment, size)
    )


def features(counts, vocabulary):
    return {term: term in counts for term in vocabulary}


def restore_naive_bayes(state):
    priors = DictionaryProbDist(state["priors"])
    probabilities = {
        (label, term): DictionaryProbDist({False: pair[0], True: pair[1]})
        for label, table in state["conditionals"].items()
        for term, pair in table.items()
    }
    return NaiveBayesClassifier(priors, probabilities)


def probability(value):
    return type(value) in (int, float) and 0 < value < 1


def validate_model(state):
    if state["version"] != 1 or state["selected"] not in {"rules", "naive_bayes"}:
        raise ValueError("Unsupported detector artifact")
    if not probability(state["threshold"]):
        raise ValueError("Invalid Naive Bayes cutoff")
    if state.get("tokenizer_version", NLTK_VERSION) != NLTK_VERSION:
        raise ValueError(f"This model requires NLTK {state['tokenizer_version']}, found {NLTK_VERSION}")
    for key in ("lexicon", "vocabulary"):
        terms = state[key]
        if not isinstance(terms, list) or any(not isinstance(term, str) or not term for term in terms):
            raise ValueError(f"Invalid detector {key}")
        if len(set(terms)) != len(terms):
            raise ValueError(f"Duplicate entries in detector {key}")
    variants = state["variants"]
    if not isinstance(variants, dict) or any(
        not form or clean(form) != form or not isinstance(term, str) or not WORD.fullmatch(term)
        for form, term in variants.items()
    ):
        raise ValueError("Invalid reviewed spelling variants")
    model = state["naive_bayes"]
    labels = {"clean", "abusive"}
    priors = model["priors"]
    tables = model["conditionals"]
    if (priors.keys() != labels or tables.keys() != labels
            or not all(map(probability, priors.values())) or not isclose(sum(priors.values()), 1)):
        raise ValueError("Invalid Naive Bayes class probabilities")
    for table in tables.values():
        if table.keys() != set(state["vocabulary"]) or any(
            not isinstance(pair, list) or len(pair) != 2
            or not all(map(probability, pair)) or not isclose(sum(pair), 1)
            for pair in table.values()
        ):
            raise ValueError("Invalid Naive Bayes feature probabilities")


@dataclass(frozen=True)
class Prediction:
    flagged: bool
    rules_flagged: bool
    nb_flagged: bool
    matches: dict
    score: float
    signals: tuple


class Detector:
    def __init__(self, state):
        validate_model(state)
        self.selected = state["selected"]
        self.lexicon = frozenset(state["lexicon"])
        self.preprocessor = Preprocessor(state["variants"])
        self.vocabulary = tuple(state["vocabulary"])
        self.model = restore_naive_bayes(state["naive_bayes"])
        self.threshold = state["threshold"]
        self.evaluation = state.get("evaluation")
        weights = state["naive_bayes"]["conditionals"]
        self.positive_signals = tuple(sorted(
            (term for term in self.vocabulary
             if weights["abusive"][term][1] > weights["clean"][term][1]),
            key=lambda term: (
                -weights["abusive"][term][1] / weights["clean"][term][1], term
            ),
        ))

    @classmethod
    def load(cls, path=ROOT / "artifacts" / "detector.json"):
        try:
            return cls(json.loads(Path(path).read_text(encoding="utf-8")))
        except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError(f"Could not load the detector artifact at {path}: {error}") from error

    def predict_counts(self, counts):
        matches = {term: counts[term] for term in sorted(self.lexicon & counts.keys())}
        score = self.model.prob_classify(features(counts, self.vocabulary)).prob("abusive")
        nb_flagged = score >= self.threshold
        rules_flagged = bool(matches)
        flagged = rules_flagged if self.selected == "rules" else nb_flagged
        signals = tuple(term for term in self.positive_signals if term in counts)[:4]
        return Prediction(flagged, rules_flagged, nb_flagged, matches, score, signals)

    def predict(self, text):
        return self.predict_counts(self.preprocessor.counts(text))
