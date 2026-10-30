"""
Retrieval-only evaluation (no LLM, no prompting) of the recommended pipeline, for each saved collection:

    BM25 (top-50) + dense (top-50, in parallel) -> RRF (k=60) -> cross-encoder rerank -> top-10

Metrics: Recall@1/3/5/10 (any relevant chunk in the top k), MRR@10, nDCG@10, latency p50 / p95 (ms).
Qdrant is read from a local folder (no server).

Test file: CSV or XLSX with the columns
    Category, Question, Answer / Ground Truth, Source Document, Section / Clause, Query Type, Difficulty
(rows that repeat the same Question are merged; their clauses are treated as alternative correct answers)

Optional column "Gold Pages" (recommended): the page(s) of the corpus PDF that contain the answer, e.g.
    ECBC_2024.pdf:62 ; ECBC_2024.pdf:77-79
Use it with --match page. It is independent of chunk size, so all 9 DBs are judged by the same rule.

DBs are found automatically: every folder under --db-root that is a local Qdrant storage (has meta.json).
The embedding model is inferred from the folder name (BGE-M3_Embedding, bge-small-en-v1.5_Embedding,
multilingual-E5-base_Embedding), so each DB is queried with the model that built it.

Run:
    python eval_retrieval.py --test questions.xlsx --db-root ./DB
"""
from __future__ import annotations

import argparse
import os
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from Retrival import HybrideRetrival as H

KS = (1, 3, 5, 10)
DEBUG_N = 0  # set by --debug
POOL: list = []  # top-5 hits of every DB, for pooled labelling (--export-pool)


# --------------------------------------------------------------------------
# 1. Relevance judgement: does a retrieved chunk answer the question?
# --------------------------------------------------------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


def clause_ids(label: str) -> list[str]:
    """'Section 7.2, Table 7.1' -> ['section 7.2', 'table 7.1']  (also keeps bare '7.2')."""
    out = []
    for kind, num in re.findall(r"(section|clause|table|annex|appendix|figure)?\s*([A-Za-z]?\d+(?:\.\d+)*)", norm(label)):
        out.append((f"{kind} {num}").strip())
        out.append(num)
    return list(dict.fromkeys(out))


def key_numbers(answer: str) -> list[str]:
    """Numbers in the ground-truth answer (e.g. '10.8'), the facts a correct chunk must contain."""
    nums = re.findall(r"\d+(?:\.\d+)?", str(answer or ""))
    # ignore years such as 2017 / 2021: they name the standard, they are not the answer
    nums = [n for n in nums if not re.fullmatch(r"(19|20)\d{2}", n)]
    return list(dict.fromkeys(nums))


OVERLAP = 0.6  # set by --overlap
_STOP = {"that", "this", "with", "from", "have", "been", "which", "their", "there", "under", "into", "also", "such",
         "these", "those", "shall", "where", "when", "than", "more", "most", "each", "other", "only", "some", "used",
         "using", "apply", "applies", "applicable", "between", "within", "building", "buildings", "code", "ecbc",
         "specifies", "required", "requirement", "requirements", "should", "must", "will", "does", "were", "being"}


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", norm(s)) if w not in _STOP}


def src_key(s) -> str:
    return norm(s).replace(".pdf", "").replace(" ", "_")


def parse_gold_pages(s) -> set:
    """'ECBC_2024.pdf:62; ECBC_2024.pdf:77-79' -> {('ecbc_2024', 62), ('ecbc_2024', 77), ...}"""
    out = set()
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return out
    for src, a, b in re.findall(r"([^:;,\n]+?)\s*:\s*(\d+)(?:\s*-\s*(\d+))?", str(s)):
        lo, hi = int(a), int(b) if b else int(a)
        out.update((src_key(src), p) for p in range(lo, hi + 1))
    return out


def build_questions(df: pd.DataFrame) -> list[dict]:
    """Merge rows with the same question; collect all accepted clauses / sources."""
    qs: dict[str, dict] = {}
    for _, r in df.iterrows():
        q = str(r["Question"]).strip()
        e = qs.setdefault(q, {"question": q, "answer": str(r["Answer / Ground Truth"]),
                              "category": r.get("Category", ""), "difficulty": r.get("Difficulty", ""),
                              "qtype": r.get("Query Type", ""), "sources": set(), "clauses": [], "gold_pages": set()})
        e["sources"].add(norm(r.get("Source Document", "")))
        e["gold_pages"] |= parse_gold_pages(r.get("Gold Pages", ""))
        e["clauses"] += clause_ids(r.get("Section / Clause", ""))
    for e in qs.values():
        e["clauses"] = list(dict.fromkeys(e["clauses"]))
        e["numbers"] = key_numbers(e["answer"])
        e["words"] = content_words(e["answer"])
    return list(qs.values())


def has_number(n: str, text: str) -> bool:
    """Whole-number match: '5' must not match inside '2.5', '50' or '2015'."""
    return re.search(rf"(?<![\d.]){re.escape(n)}(?![\d]|\.\d)", text) is not None


def overlap_score(hit: H.Hit, q: dict) -> float:
    """Share of the ground-truth answer's key words found in the chunk (shown in --debug to tune --overlap)."""
    return round(len(q["words"] & content_words(hit.text)) / len(q["words"]), 2) if q["words"] else -1.0


def is_relevant(hit: H.Hit, q: dict, mode: str) -> bool:
    """
    clause : chunk is from the right document AND mentions / is labelled with an accepted clause
    answer : chunk contains every key number of the ground-truth answer
    either : clause OR answer   (default; robust to chunking that splits clause headings away)
    """
    if mode == "page":  # right source file AND a gold page
        try:
            page = int(hit.payload.get("page"))
        except (TypeError, ValueError):
            return False
        return (src_key(hit.payload.get(H.SOURCE_FIELD, "")), page) in q["gold_pages"]
    text = norm(hit.text + " " + hit.payload.get(H.PARENT_TEXT_FIELD, ""))
    section = norm(hit.payload.get(H.SECTION_FIELD) or hit.payload.get("parent_title", ""))
    source = norm(hit.payload.get(H.SOURCE_FIELD, ""))

    src_ok = (not q["sources"]) or any(s and (s in source or source in s or s.split(" ")[0] in source) for s in q["sources"]) \
        or not source  # payload without a source field: do not penalise

    clause_ok = False
    if src_ok:
        for c in q["clauses"]:
            if "." not in c and " " not in c:  # bare '2' or '4' matches almost anything (e.g. 'CO2'); ignore
                continue
            if c == section or (section and re.search(rf"(?<![A-Za-z0-9.]){re.escape(c)}(?![0-9]|\.\d)", section)):
                clause_ok = True
                break
            if " " in c and c in text:  # 'table 7.1' / 'section 7.2' written in the chunk text
                clause_ok = True
                break

    if q["numbers"]:  # numeric answers: every number must appear in the chunk
        answer_ok = all(has_number(n, text) for n in q["numbers"])
    else:  # text answers: enough of the ground-truth answer's key words must appear in the chunk
        answer_ok = bool(q["words"]) and len(q["words"] & content_words(text)) / len(q["words"]) >= OVERLAP

    return {"clause": clause_ok, "answer": answer_ok, "either": clause_ok or answer_ok}[mode]


# --------------------------------------------------------------------------
# 2. Metrics
# --------------------------------------------------------------------------
def metrics_for(ranked_rel: list[bool]) -> dict:
    out = {}
    for k in KS:
        out[f"recall@{k}"] = float(any(ranked_rel[:k]))
    first = next((i for i, r in enumerate(ranked_rel[:10], 1) if r), None)
    out["mrr@10"] = 1.0 / first if first else 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked_rel[:10], 1) if r)
    n_rel = sum(ranked_rel[:10])
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    out["first_rel_rank"] = first or 0
    return out


# --------------------------------------------------------------------------
# 3. Run one collection with the recommended pipeline
# --------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_embedder(name: str):
    m = SentenceTransformer(name, device=DEVICE)
    if DEVICE == "cuda":
        m.half()  # fp16 halves GPU memory (needed to fit BGE-M3 + reranker in 6 GB) and is faster
    return m


def run_collection(client, cfg, questions, mode, embedders, reranker):
    model = cfg["embed_model"]
    if model not in embedders:
        embedders[model] = load_embedder(model)
    size, keys, n = H.autodetect_fields(client, cfg["name"])
    emb = embedders[model]
    dim = (getattr(emb, "get_embedding_dimension", None) or emb.get_sentence_embedding_dimension)()
    if dim != size:
        raise SystemExit(f"[{cfg['label']}] vectors in the DB have {size} dims but {model} produces {dim}. "
                         "This DB was built with a different embedding model.")
    print(f"   {n} chunks | vector size {size} | payload keys: {keys}")
    ret = H.HybridRetriever(client, cfg["name"], model, embedder=embedders[model], reranker=reranker,
                            query_prefix=cfg.get("query_prefix", ""))

    ret.search(questions[0]["question"], top_k=10)  # warm-up: keeps model start-up out of the latency numbers

    rows = []
    t_start = time.time()
    for idx, q in enumerate(questions, 1):
        if idx % 10 == 0:
            el = time.time() - t_start
            print(f"   {idx}/{len(questions)} questions | {el:.0f}s elapsed | ~{el / idx * (len(questions) - idx):.0f}s left", flush=True)
        top, t = ret.search(q["question"], top_k=10)  # parallel BM25 + dense -> RRF -> rerank
        for h in top[:5]:
            POOL.append((q["question"], str(h.payload.get(H.SOURCE_FIELD)), h.payload.get("page"),
                         h.payload.get(H.SECTION_FIELD) or h.payload.get("parent_title"), h.text[:200], cfg["label"]))
        rel = [is_relevant(h, q, mode) for h in top]
        m = metrics_for(rel)
        if DEBUG_N and questions.index(q) < DEBUG_N:
            print(f"\n   Q: {q['question'][:90]}\n      accepted clauses: {q['clauses'][:6]} | answer numbers: {q['numbers']} | sources: {list(q['sources'])[:2]}")
            for i, (h, r) in enumerate(zip(top[:3], rel[:3]), 1):
                print(f"      #{i} relevant={r} overlap={overlap_score(h, q)} (needs {OVERLAP}) section={h.payload.get(H.SECTION_FIELD) or h.payload.get('parent_title')!r} source={h.payload.get(H.SOURCE_FIELD)!r} | {h.text[:110]!r}")
        review = {"expected_answer": q["answer"][:300], "labelled_source": "; ".join(sorted(q["sources"]))}
        for i, (h, r) in enumerate(zip(top[:3], rel[:3]), 1):
            review[f"top{i}_relevant"] = r
            review[f"top{i}_source"] = h.payload.get(H.SOURCE_FIELD)
            review[f"top{i}_page"] = h.payload.get("page")
            review[f"top{i}_section"] = h.payload.get(H.SECTION_FIELD) or h.payload.get("parent_title")
            review[f"top{i}_text"] = h.text[:300]
        rows.append({"collection": cfg["label"], "question": q["question"], "category": q["category"],
                     "difficulty": q["difficulty"], "qtype": q["qtype"],
                     "latency_ms": t.ms["total"], "retrieve_ms": t.ms["retrieve_parallel"],
                     "rerank_ms": t.ms["rerank"], **m, **review})
    return rows


def summarise(detail: pd.DataFrame) -> pd.DataFrame:
    g = detail.groupby("collection")
    agg = {f"recall@{k}": "mean" for k in KS}
    agg.update({"mrr@10": "mean", "ndcg@10": "mean"})
    s = g.agg(agg)
    s["p50_ms"] = g["latency_ms"].median()
    s["p95_ms"] = g["latency_ms"].quantile(0.95)
    s["avg_rerank_ms"] = g["rerank_ms"].mean()
    s["n_questions"] = g.size()
    return s.reset_index().round(3).sort_values("recall@5", ascending=False)


def embedding_for(path: str) -> tuple[str, str]:
    """Infer the query-time embedding model from the folder name (BGE-M3_Embedding, bge-small-en-v1.5_Embedding, ...)."""
    p = path.lower()
    if "bge-m3" in p or "bge_m3" in p:
        return "BAAI/bge-m3", ""
    if "bge-small" in p:
        return "BAAI/bge-small-en-v1.5", ""
    if "e5" in p:
        return "intfloat/multilingual-e5-base", "query: "
    raise SystemExit(f"Cannot infer the embedding model from '{path}'. Add a rule in embedding_for().")


def discover_dbs(root: str) -> list[str]:
    """Every folder under root that is a local Qdrant storage (contains meta.json)."""
    found = []
    for d, dirs, files in os.walk(root):
        if "meta.json" in files:
            found.append(d)
            dirs[:] = []  # do not descend into a storage folder
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="CSV or XLSX test set")
    ap.add_argument("--db-root", default="./DB", help="folder that contains your 9 local Qdrant DBs")
    ap.add_argument("--only", default=None, help="evaluate only DBs whose path contains this text, e.g. Hierarchical")
    ap.add_argument("--match", default="answer", choices=["clause", "answer", "either", "page"],
                    help="answer = ground-truth answer content (default); page = Gold Pages column (recommended); clause = section labels")
    ap.add_argument("--pool", type=int, default=H.RERANK_POOL, help="fused candidates sent to the reranker")
    ap.add_argument("--out", default="retrieval_results")
    ap.add_argument("--overlap", type=float, default=0.6, help="min share of ground-truth answer key words a chunk must contain (text answers)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N questions (quick test run)")
    ap.add_argument("--rerank-max-len", type=int, default=512, help="reranker max tokens per chunk; 256 is about 2x faster on CPU")
    ap.add_argument("--export-pool", action="store_true", help="write label_pool.csv: candidate pages gathered from all DBs, to speed up labelling Gold Pages")
    ap.add_argument("--audit", action="store_true", help="only list the source files inside the DB vs the test file, then exit")
    ap.add_argument("--debug", type=int, default=0, help="print top-3 chunks and the relevance verdict for the first N questions")
    args = ap.parse_args()
    global DEBUG_N
    DEBUG_N = args.debug
    global OVERLAP
    OVERLAP = args.overlap
    H.RERANK_POOL = args.pool

    df = pd.read_excel(args.test) if args.test.lower().endswith((".xlsx", ".xls")) else pd.read_csv(args.test)
    df.columns = [c.strip() for c in df.columns]
    questions = build_questions(df)
    print(f"{len(df)} rows -> {len(questions)} unique questions")
    if args.limit:
        questions = questions[:args.limit]
    n_stub = len(questions)
    questions = [q for q in questions if not q["question"].startswith("[Topic stub") and not q["answer"].strip().startswith("[PENDING")]
    if len(questions) < n_stub:
        print(f"Skipping {n_stub - len(questions)} unfinished placeholder rows ('[Topic stub ...]' / '[PENDING ...]'): they have no answer to retrieve.")
    if args.match == "page":
        n_p = len(questions)
        questions = [q for q in questions if q["gold_pages"]]
        print(f"--match page: {len(questions)} of {n_p} questions have a 'Gold Pages' label; the rest are skipped.")
        if not questions:
            raise SystemExit("No question has a 'Gold Pages' value. Add the column, or use --match answer. Try --export-pool to get candidate pages.")
    n_all = len(questions)
    questions = [q for q in questions if q["sources"] != {"neither"}]
    if len(questions) < n_all:
        print(f"Skipping {n_all - len(questions)} questions labelled Source Document = 'Neither' (no answer exists in the corpus; test those on abstention, not retrieval).")
    if len(questions) < len(df) * 0.5:
        print("WARNING: many rows share the same Question text and were merged. Check the test file for copy-pasted questions.")

    dbs = discover_dbs(args.db_root)
    if args.only:
        key = re.sub(r"[\\/]+", lambda m: os.sep, args.only).lower()  # accept / \ or \\ typed in any shell
        dbs = [d for d in dbs if key in d.lower()]
    if not dbs:
        raise SystemExit(f"No local Qdrant storage (folder with meta.json) found under {args.db_root}"
                         + (f" matching --only '{args.only}'" if args.only else "")
                         + ". Folders with meta.json that exist: " + ", ".join(discover_dbs(args.db_root)[:3]))
    print(f"Found {len(dbs)} DBs:\n  " + "\n  ".join(dbs))

    if args.audit:
        from collections import Counter
        print("\nTEST FILE  Source Document values:")
        for k, v in df["Source Document"].value_counts().items():
            print(f"   {v:3d}  {k}")
        for db in dbs[:1]:
            client = H.make_client(db)
            for c in client.get_collections().collections:
                H.autodetect_fields(client, c.name)
                _, _, payloads = H.load_all_chunks(client, c.name)
                print(f"\nDB {db}  collection {c.name}  source files inside:")
                for k, v in Counter(p.get(H.SOURCE_FIELD) for p in payloads).most_common():
                    print(f"   {v:6d} chunks  {k}")
                print("   parent_title examples:", [p.get("parent_title") for p in payloads[:5]])
            client.close()
        return

    reranker = CrossEncoder(H.RERANKER_NAME, max_length=args.rerank_max_len, device=DEVICE)
    if DEVICE == "cuda":
        reranker.model.half()
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (fp16)", flush=True)
    else:
        print("WARNING: running on CPU (torch.cuda.is_available() is False). Install the CUDA build of PyTorch for 10x+ speed.", flush=True)
    embedders: dict = {}

    all_rows = []
    for db in dbs:
        model, prefix = embedding_for(db)
        label = os.path.relpath(db, args.db_root)
        client = H.make_client(db)
        try:
            for c in client.get_collections().collections:
                cfg = {"name": c.name, "label": label if len(client.get_collections().collections) == 1 else f"{label}/{c.name}",
                       "embed_model": model, "query_prefix": prefix}
                print(f"== {cfg['label']}  ({model})  started {time.strftime('%H:%M:%S')}", flush=True)
                all_rows += run_collection(client, cfg, questions, args.match, embedders, reranker)
        finally:
            client.close()
        # save after every DB so a long run is never lost if you stop it
        pd.DataFrame(all_rows).to_csv(f"{args.out}_per_question.csv", index=False)
        summarise(pd.DataFrame(all_rows)).to_csv(f"{args.out}_summary.csv", index=False)
        print(f"   saved partial results ({len(set(r['collection'] for r in all_rows))} DBs done)", flush=True)

    detail = pd.DataFrame(all_rows)
    summary = summarise(detail)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n", summary.to_string(index=False))
    if args.export_pool and POOL:
        pool = pd.DataFrame(POOL, columns=["question", "source", "page", "section", "snippet", "db"]).fillna({"page": -1})
        agg = (pool.groupby(["question", "source", "page"]).agg(section=("section", "first"), snippet=("snippet", "first"),
                                                              n_dbs=("db", "nunique")).reset_index()
               .sort_values(["question", "n_dbs"], ascending=[True, False]))
        agg.to_csv("label_pool.csv", index=False)
        print("Wrote label_pool.csv (candidate pages per question; the more DBs agree, the more likely the page is right)")
    print(f"\nSaved {args.out}_summary.csv and {args.out}_per_question.csv")
    print("Open the per-question file and filter first_rel_rank == 0 to read the failed queries.")


if __name__ == "__main__":
    main()