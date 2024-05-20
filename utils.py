import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import nltk

nltk.download("punkt")


def preprocess_lyrics(lyrics):
    # Convert to lowercase
    lyrics = lyrics.lower()
    # Remove special characters
    lyrics = re.sub(f"[{re.escape(string.punctuation)}]", "", lyrics)
    # Tokenize
    tokens = word_tokenize(lyrics)
    # Join tokens back into a single string
    return " ".join(tokens)


def preprocess_input(lyrics):
    # Preprocess the lyrics: lowercasing, removing punctuation, etc.
    lyrics = lyrics.lower()
    lyrics = "".join(c for c in lyrics if c.isalnum() or c.isspace())
    return lyrics.split()


from collections import Counter


def cosine_similarity(tokens1, tokens2):
    vec1 = Counter(tokens1)
    vec2 = Counter(tokens2)

    common = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in common])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = (sum1**0.5) * (sum2**0.5)

    if not denominator:
        return 0.0
    return float(numerator) / denominator
