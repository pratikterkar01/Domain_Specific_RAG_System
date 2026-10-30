"""
Hybrid retrieval pipeline for an existing Qdrant collection.

    query -> [BM25 top-50] ----+
                               +--> RRF (k=60) --> top-N --> cross-encoder rerank --> top-5
    query -> [BGE-M3 top-50] --+                                   |
                                                                   v
                                  parent-section expansion + "best first and last" ordering
                                                                   v
                                          grounded, cited, abstain-capable prompt

Install:
    pip install qdrant-client sentence-transformers rank-bm25

Your collections are read from a local Qdrant folder (no server needed).

Usage:
    python hybrid_rag.py --path ./qdrant_db --collection my_collection_name --query "What is the minimum COP in Section 6.5.1?"

IMPORTANT: --embed-model must be the SAME model that built the collection you query
(BGE-M3 for BGE-M3 collections, multilingual-e5-base for E5 ones, and so on).
"""
from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from Utils import Config,logger
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from Model import RetrivalClass

global_Variable = Config.get_settings()
logger = logger.get_logger(__name__)
DEFAULT_PATH = "./qdrant_db"   # folder you used with QdrantClient(path=...)

## Find which DB present and initialize them
def make_client(path: str = DEFAULT_PATH) -> QdrantClient:
    logger.info("Inside make_client, Open the local Qdrant folder (embedded mode, no server). Fails if the folder does not exist.")
    if not os.path.isdir(path):
        logger.error(f"Quadrant DB not present at {path}")
        raise SystemExit(f"Qdrant folder not found: {path}\nPass the folder you used with QdrantClient(path=...) via --path.")
    try:
        client = QdrantClient(path=path)
        names = [c.name for c in client.get_collections().collections]
        logger.info(f"{len(names)} DB found")
    except Exception as e:
        # local mode allows one process at a time: close notebooks / other scripts that have the folder open
        logger.error(f"Internal system error for opening DB : {e}")
        raise SystemExit(f"Could not open {path}: {e}\nIf another notebook or script has this folder open, close it first.")
    print(f"Opened {path}; collections: {names}")
    return client


TEXT_CANDIDATES = ("text", "page_content", "content", "chunk_text", "chunk")


def flat(payload: dict | None) -> dict:
    """LangChain-style payloads nest section/source under 'metadata'; lift them to the top level."""
    payload = payload or {}
    meta = payload.get("metadata")
    return {**meta, **payload} if isinstance(meta, dict) else payload


def autodetect_fields(client: QdrantClient, collection: str):
    """Set the vector name and text field for THIS collection (your 9 DBs may differ). Returns (vector_size, payload_keys, n_points)."""
    global TEXT_FIELD, VECTOR_NAME
    info = client.get_collection(collection)
    vecs = info.config.params.vectors
    if isinstance(vecs, dict):  # named vectors
        VECTOR_NAME = next(iter(vecs))
        size = vecs[VECTOR_NAME].size
    else:
        VECTOR_NAME, size = None, vecs.size
    pts, _ = client.scroll(collection, limit=1, with_payload=True, with_vectors=False)
    keys = list(flat(pts[0].payload).keys()) if pts else []
    for c in (TEXT_FIELD, *TEXT_CANDIDATES):
        if c in keys:
            TEXT_FIELD = c
            break
    else:
        raise SystemExit(f"[{collection}] no text field found in payload keys {keys}. Set TEXT_FIELD at the top of hybrid_rag.py.")
    return size, keys, info.points_count

# --------------------------------------------------------------------------
# BM25 (built once from the chunks already stored in Qdrant)
# --------------------------------------------------------------------------
# Keeps clause numbers ("6.5.1"), units and acronyms ("COP", "kW/m2") as single tokens.
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.\-/][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def load_all_chunks(client: QdrantClient, collection: str) -> tuple[list, list[str], list[dict]]:
    """Scroll every point's payload (no vectors) so BM25 sees exactly the same chunks."""
    ids, texts, payloads = [], [], []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            ids.append(p.id)
            fp = flat(p.payload)
            texts.append(fp.get(TEXT_FIELD, ""))
            payloads.append(fp)
        if offset is None:
            break
    return ids, texts, payloads


class HybridRetriever:
    def __init__(self, client: QdrantClient, collection: str, embed_model: str, reranker_name: str = global_Variable.RERANKER_NAME,
                 embedder=None, reranker=None, query_prefix: str = ""):
        # embedder / reranker can be passed in so 9 collections can share loaded models.
        # query_prefix: e.g. "query: " for E5 models (empty for BGE-M3).
        self.client = client
        self.collection = collection
        self.query_prefix = query_prefix
        self.embedder = embedder or SentenceTransformer(embed_model)
        self.reranker = reranker or CrossEncoder(reranker_name, max_length=512)

        ids, texts, payloads = load_all_chunks(client, collection)
        if not ids:
            raise RuntimeError(f"Collection '{collection}' returned no points.")
        self.ids, self.texts, self.payloads = ids, texts, payloads
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    # ---- individual retrievers -------------------------------------------
    def _dense(self, query: str, flt: models.Filter | None) -> list[RetrivalClass.Hit]:
        vec = self.embedder.encode(self.query_prefix + query, normalize_embeddings=True).tolist()
        res = self.client.query_points(
            collection_name=self.collection,
            query=vec,
            using=VECTOR_NAME,
            limit=global_Variable.DENSE_TOP_K,
            query_filter=flt,
            with_payload=True,
        ).points
        return [
            RetrivalClass.Hit(id=str(p.id), text=flat(p.payload).get(TEXT_FIELD, ""), payload=flat(p.payload), dense_rank=r)
            for r, p in enumerate(res, start=1)
        ]

    def _bm25(self, query: str, flt_fn) -> list[RetrivalClass.Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[RetrivalClass.Hit] = []
        for i in order:
            if scores[i] <= 0:
                break
            if flt_fn and not flt_fn(self.payloads[i]):
                continue
            hits.append(RetrivalClass.Hit(id=str(self.ids[i]), text=self.texts[i], payload=self.payloads[i], bm25_rank=len(hits) + 1))
            if len(hits) == global_Variable.BM25_TOP_K:
                break
        return hits

    # ---- fusion ------------------------------------------------------------
    @staticmethod
    def rrf(dense: list[RetrivalClass.Hit], sparse: list[RetrivalClass.Hit], k: int = global_Variable.RRF_K) -> list[RetrivalClass.Hit]:
        merged: dict[str, RetrivalClass.Hit] = {}
        for hits, attr in ((dense, "dense_rank"), (sparse, "bm25_rank")):
            for h in hits:
                m = merged.setdefault(h.id, h)
                setattr(m, attr, getattr(h, attr))
                m.rrf_score += 1.0 / (k + getattr(h, attr))
        return sorted(merged.values(), key=lambda h: h.rrf_score, reverse=True)

    # ---- rerank --------------------------------------------------------------
    def rerank(self, query: str, hits: list[RetrivalClass.Hit], top_k: int = global_Variable.FINAL_TOP_K) -> list[RetrivalClass.Hit]:
        if not hits:
            return []
        scores = self.reranker.predict([(query, h.text) for h in hits], batch_size=16, show_progress_bar=False)
        for h, s in zip(hits, scores):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: h.rerank_score, reverse=True)[:top_k]

    # ---- full pipeline ---------------------------------------------------------
    def search(self, query: str, section: str | None = None, source: str | None = None, top_k: int = global_Variable.FINAL_TOP_K):
        """Optional metadata filters: section / source (matched on the payload fields above)."""
        t = RetrivalClass.Timings()
        conds = []
        if section:
            conds.append(models.FieldCondition(key=global_Variable.SECTION_FIELD, match=models.MatchValue(value=section)))
        if source:
            conds.append(models.FieldCondition(key=global_Variable.SOURCE_FIELD, match=models.MatchValue(value=source)))
        qflt = models.Filter(must=conds) if conds else None

        def py_filter(payload: dict) -> bool:
            return (not section or payload.get(global_Variable.SECTION_FIELD) == section) and (
                not source or payload.get(global_Variable.SOURCE_FIELD) == source
            )

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as ex:  # run both retrievers in parallel
            f_dense = ex.submit(self._dense, query, qflt)
            f_bm25 = ex.submit(self._bm25, query, py_filter if conds else None)
            dense, sparse = f_dense.result(), f_bm25.result()
        t.ms["retrieve_parallel"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fused = self.rrf(dense, sparse)[:global_Variable.RERANK_POOL]
        t.ms["rrf"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        top = self.rerank(query, fused, top_k)
        t.ms["rerank"] = (time.perf_counter() - t0) * 1000

        t.ms["total"] = sum(t.ms.values())
        return top, t


# --------------------------------------------------------------------------
# Context assembly: parent expansion + best chunks first and last
# --------------------------------------------------------------------------
def expand_to_parents(hits: list[RetrivalClass.Hit]) -> list[RetrivalClass.Hit]:
    """Send the parent section instead of the child chunk; drop duplicate parents."""
    seen, out = set(), []
    for h in hits:  # hits are already sorted best-first
        key = h.payload.get(global_Variable.PARENT_ID_FIELD) or h.id
        if key in seen:
            continue
        seen.add(key)
        parent_text = h.payload.get(global_Variable.PARENT_TEXT_FIELD)
        if parent_text:
            h = RetrivalClass.Hit(**{**h.__dict__, "text": parent_text})
        out.append(h)
    return out


def order_best_first_and_last(hits: list[RetrivalClass.Hit]) -> list[RetrivalClass.Hit]:
    """Counter 'lost in the middle': best at the start, 2nd best at the end, weakest in the middle."""
    if len(hits) <= 2:
        return hits
    return [hits[0]] + hits[2:] + [hits[1]]


def build_prompt(question: str, hits: list[RetrivalClass.Hit]) -> str:
    blocks = []
    for n, h in enumerate(hits, start=1):
        src = h.payload.get(global_Variable.SOURCE_FIELD, "unknown source")
        sec = h.payload.get(global_Variable.SECTION_FIELD, "")
        blocks.append(f"[{n}] ({src} {sec})\n{h.text}")
    context = "\n\n".join(blocks)
    return (
        "You are an assistant for building codes and standards.\n"
        "Answer ONLY from the numbered clauses below. After every claim, cite the clause as [n].\n"
        'If the clauses do not contain the answer, reply exactly: "Not found in the documents."\n'
        "Do not use outside knowledge. Keep numbers and units exactly as written.\n\n"
        f"CLAUSES:\n{context}\n\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True, help="one of your 9 Qdrant collections")
    ap.add_argument("--query", required=True)
    ap.add_argument("--path", default=DEFAULT_PATH, help="local Qdrant folder, e.g. ./qdrant_db")
    ap.add_argument("--embed-model", default="BAAI/bge-m3", help="must match the collection's embedding model")
    ap.add_argument("--section", default=None, help="optional metadata filter")
    ap.add_argument("--source", default=None, help="optional metadata filter")
    args = ap.parse_args()

    client = make_client(args.path)

    retriever = HybridRetriever(client, args.collection, args.embed_model)
    top, timings = retriever.search(args.query, section=args.section, source=args.source)

    print("\nTop results (after rerank):")
    for h in top:
        print(
            f"  rerank={h.rerank_score:6.3f}  rrf={h.rrf_score:.4f}  "
            f"dense_rank={h.dense_rank}  bm25_rank={h.bm25_rank}  "
            f"{h.payload.get(global_Variable.SOURCE_FIELD, '')} {h.payload.get(global_Variable.SECTION_FIELD, '')}"
        )

    context_hits = order_best_first_and_last(expand_to_parents(top))
    prompt = build_prompt(args.query, context_hits)

    print("\nLatency (ms): " + ", ".join(f"{k}={v:.0f}" for k, v in timings.ms.items()))
    print("\n----- PROMPT FOR THE LLM -----\n" + prompt)


if __name__ == "__main__":
    main()