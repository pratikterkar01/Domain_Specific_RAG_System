"""
Why does dense search miss easy questions?  Two tests on ONE DB.

TEST A  (self-retrieval): take stored chunks, embed their own text, search. A correctly set-up DB returns
        the same chunk at rank 1 with a score near 1.0. We try several ways of preparing the text
        (no prefix, "passage: ", header removed, ...). The variant that scores ~100% is how the chunks were
        embedded when the DB was built. If NONE reaches ~100%, the stored text is not what was embedded,
        or the wrong model / metric is used.

TEST B  (your questions): for questions with a 'Gold Pages' value, report where the gold page first appears
        in a dense-only search (top 200). Use --difficulty Easy to look only at the easy ones.

Run (example):
    python dense_selftest.py --db ./DB/bge-small-en-v1.5_Embedding/StructureAwareChunking --model BAAI/bge-small-en-v1.5
    python dense_selftest.py --db ./DB/BGE-M3_Embedding/StructureAwareChunking --model BAAI/bge-m3 \
           --test QA_Dataset.xlsx --difficulty Easy --query-prefix ""
"""
import argparse
import random
import re

import pandas as pd

import eval_retrieval as E
import hybrid_rag as H

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True, help="one local Qdrant folder")
ap.add_argument("--model", required=True, help="embedding model of this DB")
ap.add_argument("--n", type=int, default=40, help="chunks sampled for test A")
ap.add_argument("--test", default=None, help="question file for test B (needs a 'Gold Pages' column)")
ap.add_argument("--difficulty", default=None, help="only questions with this Difficulty, e.g. Easy")
ap.add_argument("--query-prefix", default="", help='prefix put before each QUESTION in test B, e.g. "query: "')
ap.add_argument("--max-q", type=int, default=20)
args = ap.parse_args()

client = H.make_client(args.db)
coll = client.get_collections().collections[0].name
size, keys, n_points = H.autodetect_fields(client, coll)
info = client.get_collection(coll)
vecs = info.config.params.vectors
dist = (vecs[H.VECTOR_NAME] if isinstance(vecs, dict) else vecs).distance
print(f"collection {coll}: {n_points} chunks, vector size {size}, distance = {dist}")
model = E.load_embedder(args.model)
dim = (getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension)()
print(f"model {args.model}: {dim} dims  -> {'matches the DB' if dim == size else 'DOES NOT MATCH THE DB: wrong model'}")
if "cosine" not in str(dist).lower():
    print("NOTE: the DB does not use cosine distance. The retriever normalises vectors, so index and query must match.")


def search(vec, k=1):
    return client.query_points(collection_name=coll, query=vec, using=H.VECTOR_NAME, limit=k, with_payload=True).points


# ---------------- TEST A ----------------
points, _ = client.scroll(coll, limit=2000, with_payload=True, with_vectors=False)
random.seed(0)
sample = random.sample(points, min(args.n, len(points)))


def body_only(t):  # drop a "Document: ... Section: ... Page: ..." header block if there is one
    return t.split("\n\n", 1)[-1] if re.match(r"\s*Document:", t) else t


variants = {
    "raw text": lambda t: t,
    '"passage: " + text': lambda t: "passage: " + t,
    '"query: " + text': lambda t: "query: " + t,
    "header removed": body_only,
    '"passage: " + header removed': lambda t: "passage: " + body_only(t),
    "BGE query instruction + text": lambda t: "Represent this sentence for searching relevant passages: " + t,
}
print("\nTEST A  embed a stored chunk's own text and search: is the chunk returned at rank 1?")
best = None
for name, fn in variants.items():
    hits, scores = 0, []
    for p in sample:
        text = H.flat(p.payload).get(H.TEXT_FIELD, "")
        vec = model.encode(fn(text), normalize_embeddings=True).tolist()
        top = search(vec)[0]
        hits += int(top.id == p.id)
        scores.append(top.score)
    rate = hits / len(sample)
    print(f"   {name:34s} rank-1 = {rate:5.0%}   mean top score = {sum(scores) / len(scores):.3f}")
    if best is None or rate > best[1]:
        best = (name, rate)
print(f"\n   -> best way to reproduce the stored vectors: {best[0]} ({best[1]:.0%})")
if best[1] < 0.9:
    print("   !! No variant reaches ~100%. The text stored in the payload is not what was embedded (for example a")
    print("      context header was added only at embedding time), or the model/metric differs. Dense search will")
    print("      be weak until the query side matches the way the chunks were built.")
else:
    print('   Passages were embedded as above. For E5 the matching QUERY prefix is "query: "; for BGE-M3 none;')
    print('   for bge-small-en-v1.5 try "Represent this sentence for searching relevant passages: " (--query-prefix).')

# ---------------- TEST B ----------------
if args.test:
    df = pd.read_excel(args.test) if args.test.lower().endswith((".xlsx", ".xls")) else pd.read_csv(args.test)
    df.columns = [c.strip() for c in df.columns]
    qs = [q for q in E.build_questions(df) if q["gold_pages"] and not q["question"].startswith("[Topic stub")]
    if args.difficulty:
        qs = [q for q in qs if str(q["difficulty"]).strip().lower() == args.difficulty.lower()]
    qs = qs[: args.max_q]
    print(f"\nTEST B  dense-only search, {len(qs)} questions" + (f" (Difficulty = {args.difficulty})" if args.difficulty else "")
          + f", query prefix {args.query_prefix!r}")
    found = 0
    for q in qs:
        vec = model.encode(args.query_prefix + q["question"], normalize_embeddings=True).tolist()
        res = search(vec, 200)
        rank = next((i for i, p in enumerate(res, 1)
                     if (E.src_key(H.flat(p.payload).get(H.SOURCE_FIELD, "")), int(H.flat(p.payload).get("page") or -1)) in q["gold_pages"]), None)
        top = H.flat(res[0].payload)
        found += rank is not None
        print(f"   gold page rank: {str(rank or '>200'):>5} | top-1: {top.get(H.SOURCE_FIELD)} p.{top.get('page')} | {q['question'][:70]}")
    print(f"\n   gold page inside the dense top 200 for {found} of {len(qs)} questions")
client.close()
