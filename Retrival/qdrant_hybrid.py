"""
Hybrid search done INSIDE Qdrant (no manual BM25 code at query time).

How it works
  * Each chunk is stored with TWO vectors:  "dense" (your embedding)  and  "bm25" (a sparse keyword vector).
  * The sparse vector holds, for every word in the chunk, its BM25 term weight (saturated tf, length-normalised).
    The collection is created with  modifier=IDF , so Qdrant itself multiplies by the word rarity (IDF) at search
    time, using the live statistics of the collection. IDF is always positive in Qdrant.
  * One query_points() call runs dense search + sparse search and fuses them with RRF (k = 60). Metadata filters
    are applied inside both searches. Then the cross-encoder reranks the fused list.

Words are turned into numbers with a stable hash, so nothing has to be saved besides the DB. The tokenizer keeps
clause numbers ("6.5.1") and units ("kW/m2") as single words, which the stock FastEmbed BM25 would split apart.

USE 1: convert your 9 existing DBs (no re-embedding: the stored dense vectors are copied)
    python qdrant_hybrid.py convert --src ./DB --dst ./DB_hybrid
    python qdrant_hybrid.py check   --db ./DB_hybrid/BGE-M3_Embedding/StructureAwareChunking

USE 2: build a NEW DB in the right shape from the start (call this in your ingestion code)
    build_hybrid_db("./DB_new/x", "my_collection", chunks, embed_fn, dense_size=1024)
    # chunks   = [{"id": 0, "text": "...", "payload": {"source": "ECBC_2024.pdf", "page": 7, ...}}, ...]
    # embed_fn = lambda texts: model.encode(texts, normalize_embeddings=True).tolist()

USE 3: evaluate it
    python eval_retrieval.py --test questions.xlsx --db-root ./DB_hybrid --engine native
"""
from __future__ import annotations

import argparse
import os
import re
import time
import zlib
from collections import Counter

from Utils import Config
from qdrant_client import QdrantClient, models
from  Model import RetrivalClass


global_Variable = Config.get_settings()
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.\-/][A-Za-z0-9]+)*")  # keeps 6.5.1, kW/m2, ECBC-2017 as one word


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def token_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF  # stable across runs and machines


# --------------------------------------------------------------------------
# sparse (BM25) encoder
# --------------------------------------------------------------------------
def encode_doc(text: str, avgdl: float) -> models.SparseVector:
    """BM25 term weights of one chunk. Qdrant multiplies these by IDF when searching."""
    tokens = tokenize(text)
    dl = len(tokens)
    idx, val = [], []
    for term, tf in Counter(tokens).items():
        idx.append(token_id(term))
        val.append(tf * (global_Variable.K1 + 1.0) / (tf + global_Variable.K1 * (1.0 - global_Variable.B + global_Variable.B * dl / avgdl)))
    return models.SparseVector(indices=idx, values=val)


def encode_query(text: str) -> models.SparseVector | None:
    terms = list(dict.fromkeys(tokenize(text)))
    if not terms:
        return None
    return models.SparseVector(indices=[token_id(t) for t in terms], values=[1.0] * len(terms))


# --------------------------------------------------------------------------
# collection set-up and building
# --------------------------------------------------------------------------
def create_hybrid_collection(client: QdrantClient, name: str, dense_size: int,
                             distance: models.Distance = models.Distance.COSINE) -> None:
    """The configuration that must exist BEFORE the chunks are inserted."""
    client.create_collection(
        collection_name=name,
        vectors_config={global_Variable.DENSE: models.VectorParams(size=dense_size, distance=distance)},
        sparse_vectors_config={global_Variable.SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


def build_hybrid_db(db_path: str, collection: str, chunks: list[dict], embed_fn, dense_size: int, batch: int = 64) -> None:
    """chunks = [{"id": int|str, "text": str, "payload": dict}]. Stores the text under payload['page_content']."""
    client = QdrantClient(path=db_path)
    create_hybrid_collection(client, collection, dense_size)
    avgdl = sum(len(tokenize(c["text"])) for c in chunks) / max(len(chunks), 1) or 1.0
    for s in range(0, len(chunks), batch):
        part = chunks[s:s + batch]
        dense = embed_fn([c["text"] for c in part])
        client.upsert(collection, wait=True, points=[
            models.PointStruct(id=c["id"], vector={global_Variable.DENSE: dense[i], global_Variable.SPARSE: encode_doc(c["text"], avgdl)},
                               payload={**c.get("payload", {}), "page_content": c["text"]})
            for i, c in enumerate(part)])
    client.close()


def _flat(payload: dict | None) -> dict:
    payload = payload or {}
    meta = payload.get("metadata")
    return {**meta, **payload} if isinstance(meta, dict) else payload


def _text_of(payload: dict | None) -> str:
    p = _flat(payload)
    for k in global_Variable.TEXT_CANDIDATES:
        if k in p:
            return p[k] or ""
    return ""


def convert_collection(src: QdrantClient, src_name: str, dst: QdrantClient, dst_name: str, batch: int = 256) -> int:
    """Copy a dense-only collection into a hybrid one. Dense vectors and payloads are copied as they are."""
    info = src.get_collection(src_name)
    vecs = info.config.params.vectors
    old_name, p = (next(iter(vecs)), vecs[next(iter(vecs))]) if isinstance(vecs, dict) else (None, vecs)
    create_hybrid_collection(dst, dst_name, p.size, p.distance)

    # pass 1: average chunk length (needed for BM25 length normalisation)
    total, n, offset = 0, 0, None
    while True:
        pts, offset = src.scroll(src_name, limit=1000, offset=offset, with_payload=True, with_vectors=False)
        for pt in pts:
            total += len(tokenize(_text_of(pt.payload)))
            n += 1
        if offset is None:
            break
    avgdl = (total / n) if n else 1.0

    # pass 2: copy
    done, offset = 0, None
    while True:
        pts, offset = src.scroll(src_name, limit=batch, offset=offset, with_payload=True, with_vectors=True)
        if pts:
            dst.upsert(dst_name, wait=True, points=[
                models.PointStruct(
                    id=pt.id,
                    vector={global_Variable.DENSE: pt.vector[old_name] if isinstance(pt.vector, dict) else pt.vector,
                            global_Variable.SPARSE: encode_doc(_text_of(pt.payload), avgdl)},
                    payload=pt.payload) for pt in pts])
            done += len(pts)
            if done % 5000 < batch:
                print(f"      {done}/{n} chunks", flush=True)
        if offset is None:
            break
    return done


def discover_dbs(root: str) -> list[str]:
    found = []
    for d, dirs, files in os.walk(root):
        if "meta.json" in files:
            found.append(d)
            dirs[:] = []
    return sorted(found)


def convert_all(src_root: str, dst_root: str, only: str | None = None) -> None:
    for db in discover_dbs(src_root):
        rel = os.path.relpath(db, src_root)
        if only and only.lower() not in rel.lower():
            continue
        out = os.path.join(dst_root, rel)
        if os.path.exists(os.path.join(out, "meta.json")):
            print(f"skip   {rel} (already converted)")
            continue
        os.makedirs(out, exist_ok=True)
        src, dst = QdrantClient(path=db), QdrantClient(path=out)
        try:
            for c in src.get_collections().collections:
                t = time.time()
                print(f"==== {rel} / {c.name}", flush=True)
                n = convert_collection(src, c.name, dst, c.name)
                n_dst = dst.count(c.name).count
                print(f"     {'OK' if n == n_dst else 'MISMATCH'}: copied {n}, new DB holds {n_dst}  ({time.time() - t:.0f}s)", flush=True)
        finally:
            src.close()
            dst.close()


def check(db: str, n: int = 20) -> None:
    """Model-free test: use a chunk's own text as the query; the keyword side must bring that chunk back."""
    client = QdrantClient(path=db)
    print(client.get_collections().collections)
    coll = client.get_collections().collections[0].name
    info = client.get_collection(coll)
    print(f"{coll}: {info.points_count} chunks | dense vectors {list(info.config.params.vectors)} | sparse {list(info.config.params.sparse_vectors or {})}")
    pts, _ = client.scroll(coll, limit=min(n, info.points_count), with_payload=True, with_vectors=False)
    ok = 0
    for p in pts:
        q = encode_query(_text_of(p.payload))
        if q is None:
            continue
        top = client.query_points(coll, query=q, using=global_Variable.SPARSE, limit=3).points
        ok += int(p.id in [t.id for t in top])
    print(f"keyword search returns the chunk itself in its top 3 for {ok} of {len(pts)} sampled chunks (expect nearly all)")
    client.close()


# --------------------------------------------------------------------------
# query side: same interface as hybrid_rag.HybridRetriever, so eval_retrieval.py can use it
# --------------------------------------------------------------------------
class NativeHybridRetriever:
    def __init__(self, client, collection, embed_model, reranker_name=None, embedder=None, reranker=None, query_prefix=""):
        from Retrival import HybrideRetrival as H
        self.H = H
        self.client, self.collection, self.query_prefix = client, collection, query_prefix
        info = client.get_collection(collection)
        if global_Variable.SPARSE not in (info.config.params.sparse_vectors or {}):
            raise SystemExit(f"[{collection}] has no '{global_Variable.SPARSE}' sparse vector. Convert the DB first: python qdrant_hybrid.py convert ...")
        if embedder is None:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(embed_model)
        if reranker is None:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(reranker_name or global_Variable.RERANKER_NAME, max_length=512)
        self.embedder, self.reranker = embedder, reranker

    def retrieve(self, query: str, section=None, source=None):
        conds = []
        if section:
            conds.append(models.FieldCondition(key=global_Variable.SECTION_FIELD, match=models.MatchValue(value=section)))
        if source:
            conds.append(models.FieldCondition(key=global_Variable.SOURCE_FIELD, match=models.MatchValue(value=source)))
        flt = models.Filter(must=conds) if conds else None

        dense_vec = self.embedder.encode(self.query_prefix + query, normalize_embeddings=True).tolist()
        prefetch = [models.Prefetch(query=dense_vec, using=global_Variable.DENSE, limit=global_Variable.DENSE_TOP_K, filter=flt)]
        sparse = encode_query(query)
        if sparse is not None:
            prefetch.append(models.Prefetch(query=sparse, using=global_Variable.SPARSE, limit=global_Variable.BM25_TOP_K, filter=flt))
        res = self.client.query_points(
            collection_name=self.collection, prefetch=prefetch,
            query=models.RrfQuery(rrf=models.Rrf(k=global_Variable.RRF_K)),
            limit=global_Variable.RERANK_POOL, with_payload=True).points
        return [RetrivalClass.Hit(id=str(p.id), text=_text_of(p.payload), payload=_flat(p.payload), rrf_score=float(p.score)) for p in res]

    def rerank(self, query, hits, top_k):
        if not hits:
            return []
        scores = self.reranker.predict([(query, h.text) for h in hits], batch_size=16, show_progress_bar=False)
        for h, s in zip(hits, scores):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: h.rerank_score, reverse=True)[:top_k]

    def search(self, query, section=None, source=None, top_k=None):
        H = self.H
        top_k = top_k or global_Variable.FINAL_TOP_K
        t = RetrivalClass.Timings()
        t0 = time.perf_counter()
        fused = self.retrieve(query, section, source)
        t.ms["retrieve_parallel"] = (time.perf_counter() - t0) * 1000  # embed + dense + sparse + RRF, all in one call
        t.ms["rrf"] = 0.0
        t0 = time.perf_counter()
        top = self.rerank(query, fused, top_k)
        t.ms["rerank"] = (time.perf_counter() - t0) * 1000
        t.ms["total"] = t.ms["retrieve_parallel"] + t.ms["rrf"] + t.ms["rerank"]
        return top, t


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="copy every DB under --src into a hybrid DB under --dst (no re-embedding)")
    c.add_argument("--src", required=True)
    c.add_argument("--dst", required=True)
    c.add_argument("--only", default=None, help="convert only DBs whose path contains this text")
    k = sub.add_parser("check", help="verify the sparse side of one converted DB (needs no model)")
    k.add_argument("--db", required=True)
    a = ap.parse_args()
    if a.cmd == "convert":
        convert_all(a.src, a.dst, a.only)
    else:
        check(a.db)
