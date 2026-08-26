"""
matching/knowledge_base.py

Loads the user-supplied knowledge-base files (STEP 17) and provides fuzzy
matching (STEP 18) of OCR'd ingredient tokens against them.

Because we don't know the exact column names in advance, we implement a
column-discovery heuristic that inspects header names and sample values
to guess which column holds:
  - the primary ingredient/nutrient name
  - alternate names (if any)

This module is defensive: if a KB file is missing or malformed, matching
simply falls back to "no match" rather than crashing the pipeline.
"""

import os

import config

try:
    from rapidfuzz import process as _rf_process, fuzz as _rf_fuzz

    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


NAME_COLUMN_HINTS = [
    "ingredient", "name", "ingredient_name", "primary_name", "item",
    "nutrient", "nutrient_name", "product", "product_name",
]
ALT_NAME_COLUMN_HINTS = [
    "alternate", "alt_name", "alt_names", "synonym", "synonyms",
    "aka", "also_known_as", "other_names",
]


def discover_columns(df):
    """
    Given a pandas DataFrame, prints available columns and returns a dict
    with best-guess column names:
        {"name_column": str or None, "alt_column": str or None}
    """
    columns = list(df.columns)
    print(f"[knowledge_base] Available columns: {columns}")

    def find_best(hints):
        cols_lower = {c: str(c).strip().lower() for c in columns}
        # exact hint match first
        for hint in hints:
            for c, cl in cols_lower.items():
                if cl == hint:
                    return c
        # substring match second
        for hint in hints:
            for c, cl in cols_lower.items():
                if hint in cl:
                    return c
        return None

    name_col = find_best(NAME_COLUMN_HINTS)
    alt_col = find_best(ALT_NAME_COLUMN_HINTS)

    # Fallback: if nothing matched, assume the first string-like column
    # is the name column.
    if name_col is None and columns:
        name_col = columns[0]

    print(f"[knowledge_base] Guessed name_column={name_col!r}, alt_column={alt_col!r}")

    return {"name_column": name_col, "alt_column": alt_col}


def _split_alt_names(value):
    if value is None:
        return []
    text = str(value)
    if not text or text.lower() == "nan":
        return []
    import re

    parts = re.split(r"[;,/|]", text)
    return [p.strip() for p in parts if p.strip()]


def _build_vocab(df, columns):
    """
    Builds a flat list of (canonical_name, searchable_name) pairs: one
    entry for the primary name and one for each alternate name, all
    pointing back to the same canonical_name.
    """
    name_col = columns.get("name_column")
    alt_col = columns.get("alt_column")

    vocab = []
    if name_col is None:
        return vocab

    for _, row in df.iterrows():
        canonical = row.get(name_col)
        if canonical is None or str(canonical).strip() == "" or str(canonical).lower() == "nan":
            continue
        canonical = str(canonical).strip()
        vocab.append((canonical, canonical))

        if alt_col is not None:
            for alt in _split_alt_names(row.get(alt_col)):
                vocab.append((canonical, alt))

    return vocab


class KnowledgeBase:
    """
    Lazily loads and caches the three knowledge-base files, exposing a
    unified fuzzy-match interface.
    """

    def __init__(
        self,
        ingredient_csv=None,
        nutrition_csv=None,
        personal_care_xlsx=None,
    ):
        self.ingredient_csv = ingredient_csv or config.INGREDIENT_KB_CSV
        self.nutrition_csv = nutrition_csv or config.NUTRITION_KB_CSV
        self.personal_care_xlsx = personal_care_xlsx or config.PERSONAL_CARE_KB_XLSX

        self._food_vocab = None
        self._personal_care_vocab = None
        self._nutrition_vocab = None

        self.load_errors = {}

    def _load_csv(self, path, label):
        import pandas as pd

        if not os.path.isfile(path):
            self.load_errors[label] = f"File not found: {path}"
            return None
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            self.load_errors[label] = f"Failed to read {path}: {e}"
            return None

    def _load_xlsx(self, path, label):
        import pandas as pd

        if not os.path.isfile(path):
            self.load_errors[label] = f"File not found: {path}"
            return None
        try:
            df = pd.read_excel(path)
            return df
        except Exception as e:
            self.load_errors[label] = f"Failed to read {path}: {e}"
            return None

    def _get_food_vocab(self):
        if self._food_vocab is not None:
            return self._food_vocab
        df = self._load_csv(self.ingredient_csv, "ingredient_csv")
        if df is None:
            self._food_vocab = []
            return self._food_vocab
        cols = discover_columns(df)
        self._food_vocab = _build_vocab(df, cols)
        print(f"[knowledge_base] Loaded {len(self._food_vocab)} food ingredient vocab entries")
        return self._food_vocab

    def _get_personal_care_vocab(self):
        if self._personal_care_vocab is not None:
            return self._personal_care_vocab
        df = self._load_xlsx(self.personal_care_xlsx, "personal_care_xlsx")
        if df is None:
            self._personal_care_vocab = []
            return self._personal_care_vocab
        cols = discover_columns(df)
        self._personal_care_vocab = _build_vocab(df, cols)
        print(
            f"[knowledge_base] Loaded {len(self._personal_care_vocab)} "
            "personal-care ingredient vocab entries"
        )
        return self._personal_care_vocab

    def _get_nutrition_vocab(self):
        if self._nutrition_vocab is not None:
            return self._nutrition_vocab
        df = self._load_csv(self.nutrition_csv, "nutrition_csv")
        if df is None:
            self._nutrition_vocab = []
            return self._nutrition_vocab
        cols = discover_columns(df)
        self._nutrition_vocab = _build_vocab(df, cols)
        print(f"[knowledge_base] Loaded {len(self._nutrition_vocab)} nutrition vocab entries")
        return self._nutrition_vocab

    def get_ingredient_names(self, domain="food"):
        """
        Returns a flat list of ingredient name strings (primary + alt),
        used e.g. by detection.ingredient_region's vocabulary fallback.
        """
        vocab = (
            self._get_personal_care_vocab()
            if domain == "personal_care"
            else self._get_food_vocab()
        )
        return [searchable for _, searchable in vocab]

    def match_ingredient(self, ocr_text, domain="food", threshold=None):
        """
        Fuzzy-matches a single OCR ingredient token against the
        appropriate knowledge base.

        Returns:
            {"ocr_text": str, "matched_name": str or None, "similarity": float}
        """
        threshold = threshold if threshold is not None else config.FUZZY_KB_MATCH_THRESHOLD

        vocab = (
            self._get_personal_care_vocab()
            if domain == "personal_care"
            else self._get_food_vocab()
        )

        if not vocab or not ocr_text or not ocr_text.strip():
            return {"ocr_text": ocr_text, "matched_name": None, "similarity": 0.0}

        query = ocr_text.strip().lower()
        searchable_names = [s.lower() for _, s in vocab]

        if _HAS_RAPIDFUZZ:
            result = _rf_process.extractOne(
                query, searchable_names, scorer=_rf_fuzz.WRatio
            )
            if result is None:
                return {"ocr_text": ocr_text, "matched_name": None, "similarity": 0.0}
            match_text, score, idx = result
            canonical = vocab[idx][0]
        else:
            # simple fallback: exact / substring match only
            canonical = None
            score = 0.0
            for canon, searchable in vocab:
                s = searchable.lower()
                if s == query:
                    canonical, score = canon, 100.0
                    break
                if s in query or query in s:
                    if len(s) > 3 and 100.0 * len(s) / max(len(query), 1) > score:
                        canonical, score = canon, 85.0

        if canonical is None or score < config.FUZZY_KB_MATCH_MIN_ACCEPT:
            return {"ocr_text": ocr_text, "matched_name": None, "similarity": round(float(score), 1)}

        if score < threshold:
            return {"ocr_text": ocr_text, "matched_name": None, "similarity": round(float(score), 1)}

        return {
            "ocr_text": ocr_text,
            "matched_name": canonical,
            "similarity": round(float(score), 1),
        }

    def match_ingredient_list(self, ocr_tokens, domain="food", threshold=None):
        return [self.match_ingredient(tok, domain=domain, threshold=threshold) for tok in ocr_tokens]
