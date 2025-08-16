# ============================================================
# File: text_preprocessing.py
# Author: Mohammed Munazir
# Description: Basic pre-processing utilities for text data
# ============================================================

import re
import spacy

try:
    NLP = spacy.load("en_core_web_sm")
except:
    NLP = None

EXCLUDED_PRONOUNS = {
    "i", "we", "they", "you", "he", "she", "it", "me", "us", "them",
    "him", "her", "myself", "ourselves", "yourself", "yourselves",
    "themselves", "someone", "everyone", "anyone", "nobody", "people"
}

def clean_text(text):
    """Basic text cleaner — lowercase, remove extra spaces."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text.strip().lower())

def extract_aspects_spacy(text, nlp_model=None):
    """Extract meaningful noun phrases, excluding pronouns."""
    nlp = nlp_model or NLP
    if not nlp:
        return []

    doc = nlp(text)
    aspects = []
    for chunk in doc.noun_chunks:
        cleaned = chunk.text.strip().lower()
        if (
            cleaned in EXCLUDED_PRONOUNS
            or all(token.is_stop for token in chunk)
            or len(cleaned) < 2
        ):
            continue
        aspects.append(chunk.text.strip())

    if not aspects:
        nouns = [
            token.text for token in doc
            if token.pos_ == "NOUN" and token.text.lower() not in EXCLUDED_PRONOUNS and not token.is_stop
        ]
        aspects = [nouns[0]] if nouns else [text.strip()]
    return aspects
