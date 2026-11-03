"""
Small, fast BM25 that replaces `rank_bm25` in the retriever.

Why:
  * IDF = log(1 + (N - n + 0.5) / (n + 0.5)) is ALWAYS positive (Lucene / Elasticsearch formula),
    so a word that is in most chunks gets a tiny weight instead of a negative one.
  * Query words found in more than `max_df` of the chunks are ignored (unless nothing else is left),
    so "shall", "building", "what" cannot drown the real keywords.
  * Uses an inverted index: only chunks that contain a query word are scored (rank_bm25 loops over
    every chunk for every word, which is slow with 30,000 chunks).
  * Takes the top-k directly. No "stop at the first zero" logic.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np


class BM25Index:
    def __init__(self, texts, tokenizer, k1: float = 1.5, b: float = 0.75, max_df: float = 0.5):
        self.tokenizer, self.k1, self.b, self.max_df = tokenizer, k1, b, max_df
        self.n = len(texts)
        postings = defaultdict(lambda: ([], []))
        lengths = np.zeros(self.n, dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = tokenizer(text)
            lengths[i] = len(tokens)
            for term, count in Counter(tokens).items():
                ids, tfs = postings[term]
                ids.append(i)
                tfs.append(count)
        self.doc_len = lengths
        self.avgdl = float(lengths.mean()) or 1.0
        self.post = {t: (np.asarray(ids, dtype=np.int32), np.asarray(tfs, dtype=np.float32))
                     for t, (ids, tfs) in postings.items()}
        self.idf = {t: math.log(1.0 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
                    for t, (ids, _) in self.post.items()}

    def stats(self) -> str:
        """Print this once after building. A healthy index has a vocabulary of thousands of words."""
        return f"BM25 index: {self.n} chunks, vocabulary {len(self.post)}, avg chunk length {self.avgdl:.0f} tokens"

    def search(self, query: str, top_k: int = 50, mask=None) -> list[tuple[int, float]]:
        """Return [(chunk_index, score), ...] best first. mask: optional boolean array (True = allowed chunk)."""
        terms = [t for t in dict.fromkeys(self.tokenizer(query)) if t in self.post]
        rare = [t for t in terms if len(self.post[t][0]) / self.n <= self.max_df]
        terms = rare or terms
        scores = np.zeros(self.n, dtype=np.float32)
        for t in terms:
            ids, tf = self.post[t]
            length_norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[ids] / self.avgdl)
            scores[ids] += self.idf[t] * tf * (self.k1 + 1.0) / (tf + length_norm)
        if mask is not None:
            scores = np.where(mask, scores, 0.0)
        k = min(top_k, self.n)
        if k == 0:
            return []
        best = np.argpartition(-scores, k - 1)[:k]
        best = best[np.argsort(-scores[best])]
        return [(int(i), float(scores[i])) for i in best if scores[i] > 0]
