"""
nlp/ingredient_corrector.py

An optional, future-ready NLP layer for correcting and normalizing OCR ingredient lists.
It normalizes text, removes section headers/unrelated text, splits ingredients into
individual phrases, and matches them fuzzy against the knowledge base.
Supports pluggable external correctors (like an LLM) later.
"""

import re
from detection.ocr_detector import normalize_ocr_text

class IngredientCorrector:
    """
    Standardizes and fuzzy-corrects OCR'd ingredient text lines.
    """

    def __init__(self, kb=None):
        self.kb = kb

    def clean_text(self, text):
        """
        Normalizes OCR text, removes leading headings, and cleans stray characters.
        """
        if not text:
            return ""
        
        # Lowercase and standard unicode normalization
        cleaned = normalize_ocr_text(text)
        
        # Remove common ingredient headings/anchors from the start of the text
        heading_patterns = [
            r'^ingredients\s*[:\-\.]?',
            r'^ingredient\s*[:\-\.]?',
            r'^composition\s*[:\-\.]?',
            r'^contents\s*[:\-\.]?',
            r'^made from\s*[:\-\.]?',
            r'^contains\s*[:\-\.]?'
        ]
        
        for pat in heading_patterns:
            cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE).strip()
            
        return cleaned

    def split_phrases(self, text):
        """
        Splits a merged ingredient paragraph into individual ingredient tokens.
        Handles commas and semicolons, while ignoring splits inside parentheses.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        # Split on commas/semicolons, but only if they are not inside parentheses
        # Using a regex-based split that respects parentheses
        tokens = []
        current_token = []
        paren_depth = 0

        for char in cleaned:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth = max(0, paren_depth - 1)
            
            if (char == ',' or char == ';') and paren_depth == 0:
                tokens.append("".join(current_token).strip())
                current_token = []
            else:
                current_token.append(char)

        if current_token:
            tokens.append("".join(current_token).strip())

        # Filter out empty or extremely short/junk tokens
        refined_tokens = []
        for t in tokens:
            t_clean = re.sub(r'[\.\*\:\-\s]+$', '', t).strip() # clean trailing punctuation
            t_clean = re.sub(r'^[\.\*\:\-\s]+', '', t_clean).strip() # clean leading punctuation
            if len(t_clean) >= 2:
                refined_tokens.append(t_clean)

        return refined_tokens

    def correct_and_match(self, text, domain="food"):
        """
        Performs the complete correction and matching pipeline.
        
        Args:
            text: raw text from ingredients ROI OCR
            domain: "food" or "personal_care"
            
        Returns:
            list of dicts containing:
                "ocr_text": original OCR token
                "normalized_text": normalized OCR token
                "corrected_ingredient": canonical KB name or None
                "match_confidence": matching score (0.0 to 1.0) or None
                "matched_name": canonical KB name or None
                "confidence": matching score (0.0 to 1.0) or None
        """
        tokens = self.split_phrases(text)
        results = []

        if not self.kb:
            # Fallback if KB is not provided
            return [{
                "ocr_text": tok,
                "normalized_text": normalize_ocr_text(tok),
                "corrected_ingredient": None,
                "match_confidence": None,
                "matched_name": None,
                "confidence": None
            } for tok in tokens]

        matched_kb_items = self.kb.match_ingredient_list(tokens, domain=domain)
        
        for tok, m in zip(tokens, matched_kb_items):
            sim = m["similarity"] / 100.0 if m["matched_name"] else None
            results.append({
                "ocr_text": tok,
                "normalized_text": normalize_ocr_text(tok),
                "corrected_ingredient": m["matched_name"],
                "match_confidence": sim,
                "matched_name": m["matched_name"],
                "confidence": sim
            })

        return results

    def external_llm_correct(self, text, api_client=None):
        """
        Placeholder/Interface for future generative LLM correction.
        """
        # In V2, this is not implemented/mandatory, but interface is preserved.
        return text
