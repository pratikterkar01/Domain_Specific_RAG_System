"""
inspect_vector_db.py
--------------------
Shows exactly what is stored inside your local Qdrant vector databases.

For every database found under DB_ROOT it reports:
  * collection name, number of vectors, vector size
  * a content ID (hash of all chunk text) -> same ID = same document content
  * which source/PDF name is stored in the metadata (if any)
  * pages covered, headings, content types (text/table), parent/child counts
  * a preview of real chunks
  * warnings (empty chunks, duplicates, mismatched counts across embeddings)

Usage:
    python inspect_vector_db.py
    python inspect_vector_db.py --db-root "D:\\M.tech\\Research Area\\RAG pipeline\\DB"
    python inspect_vector_db.py --preview 5
    python inspect_vector_db.py --only HierarchicalChunking
    python inspect_vector_db.py --export chunk_dump        # writes one CSV per DB

IMPORTANT: close any other Python process that has these DBs open
(local Qdrant allows only one open client per folder).
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

from qdrant_client import QdrantClient

import warnings
warnings.filterwarnings("ignore", message="Local mode is not recommended")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SOURCE_KEYS = ["source", "document", "document_name", "doc_name", "file_name", "filename", "pdf"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def default_db_root():
    try:
        import global_Variable
        return global_Variable.DB_ROOT
    except Exception:
        return "DB"


def find_qdrant_dirs(root):
    """A local Qdrant folder contains meta.json and a 'collection' folder."""
    found = []
    for cur, dirs, files in os.walk(root):
        if "meta.json" in files and "collection" in dirs:
            found.append(cur)
            dirs[:] = []  # don't descend further
    return sorted(found)


def load_points(client, collection):
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def vector_info(client, name):
    try:
        params = client.get_collection(name).config.params.vectors
        if isinstance(params, dict):
            return ", ".join(f"{k}:{v.size}" for k, v in params.items())
        return str(params.size)
    except Exception:
        return "?"


def split_payload(payload):
    """langchain_qdrant stores {'page_content': ..., 'metadata': {...}}."""
    if "page_content" in payload:
        return payload.get("page_content") or "", payload.get("metadata") or {}
    text = payload.get("text") or payload.get("content") or ""
    return text, payload


def short(text, n=150):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "..."


def sort_key(rec):
    page = rec["meta"].get("page")
    return (page if isinstance(page, int) else -1, rec["text"][:30])


def describe_extra_files(db_path):
    """Marker file, parents.json etc. living next to the Qdrant data."""
    lines = []
    for name in sorted(os.listdir(db_path)):
        full = os.path.join(db_path, name)
        if name in ("meta.json", "collection") or os.path.isdir(full):
            continue
        try:
            size = os.path.getsize(full)
            if name == "parents.json":
                with open(full, encoding="utf-8") as f:
                    n = len(json.load(f))
                lines.append(f"{name}: {n} parent sections")
            elif size <= 200:
                with open(full, encoding="utf-8") as f:
                    lines.append(f"{name}: '{f.read().strip()}'")
            else:
                lines.append(f"{name}: {size} bytes")
        except Exception as e:
            lines.append(f"{name}: unreadable ({e})")
    return lines


# --------------------------------------------------------------------------
# Per-collection analysis
# --------------------------------------------------------------------------
def analyse(points):
    recs = []
    for p in points:
        text, meta = split_payload(p.payload or {})
        recs.append({"id": str(p.id), "text": text, "meta": meta})

    texts = [r["text"] for r in recs]
    h = hashlib.md5()
    for t in sorted(texts):
        h.update(t.encode("utf-8"))
    content_id = h.hexdigest()[:8]

    lengths = [len(t) for t in texts] or [0]
    pages = [r["meta"].get("page") for r in recs]
    valid_pages = [p for p in pages if isinstance(p, int)]

    sources = Counter()
    for r in recs:
        for k in SOURCE_KEYS:
            if r["meta"].get(k):
                sources[str(r["meta"][k])] += 1
                break

    headings = set()
    for r in recs:
        hd = r["meta"].get("heading") or r["meta"].get("parent_title")
        if hd:
            headings.add(hd)

    text_counts = Counter(texts)
    return {
        "recs": recs,
        "content_id": content_id,
        "n": len(recs),
        "chars_min": min(lengths),
        "chars_avg": sum(lengths) // max(len(lengths), 1),
        "chars_max": max(lengths),
        "page_min": min(valid_pages) if valid_pages else None,
        "page_max": max(valid_pages) if valid_pages else None,
        "page_none": len(pages) - len(valid_pages),
        "content_types": Counter(r["meta"].get("content_type", "-") for r in recs),
        "chunk_types": Counter(r["meta"].get("chunk_type", "-") for r in recs),
        "parent_ids": len({r["meta"].get("parent_id") for r in recs if r["meta"].get("parent_id")}),
        "headings": len(headings),
        "sources": sources,
        "empty": sum(1 for t in texts if not t.strip()),
        "duplicates": sum(c - 1 for c in text_counts.values() if c > 1),
    }


def export_csv(out_dir, label, recs):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{label}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["no", "point_id", "page", "content_type", "heading_or_parent",
                    "parent_id", "chars", "text", "all_metadata_json"])
        for i, r in enumerate(sorted(recs, key=sort_key), 1):
            m = r["meta"]
            w.writerow([
                i, r["id"], m.get("page"), m.get("content_type", ""),
                m.get("heading") or m.get("parent_title") or "",
                m.get("parent_id", ""), len(r["text"]), r["text"],
                json.dumps(m, ensure_ascii=False, default=str),
            ])
    return path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-root", default=default_db_root())
    ap.add_argument("--preview", type=int, default=3, help="chunks to preview per DB")
    ap.add_argument("--only", default="", help="only DBs whose path contains this text")
    ap.add_argument("--export", default="", help="folder to write one CSV per DB")
    args = ap.parse_args()

    root = args.db_root
    print(f"DB root: {root}")
    if not os.path.isdir(root):
        print("This folder does not exist.")
        return

    db_dirs = [d for d in find_qdrant_dirs(root) if args.only.lower() in d.lower()]
    if not db_dirs:
        print("No Qdrant databases found.")
        return
    print(f"Found {len(db_dirs)} database folder(s)\n")

    summary = []
    groups = defaultdict(list)  # (doc_prefix, chunking) -> [(embedding, n, content_id, sources)]
    coverage = defaultdict(dict)  # (chunking, pdf) -> {embedding: chunk count}

    for db_path in db_dirs:
        rel = os.path.relpath(db_path, root)
        parts = rel.split(os.sep)
        emb_idx = next((i for i, p in enumerate(parts) if p.endswith("_Embedding")), None)
        if emb_idx is not None:
            doc_prefix = os.sep.join(parts[:emb_idx]) or "(none)"
            embedding = parts[emb_idx].replace("_Embedding", "")
            chunking = os.sep.join(parts[emb_idx + 1:])
        else:
            doc_prefix, embedding, chunking = "(none)", "?", rel

        print("=" * 100)
        print(f"DB: {rel}")
        print("=" * 100)

        try:
            client = QdrantClient(path=db_path)
            collections = [c.name for c in client.get_collections().collections]
        except Exception as e:
            print(f"  Could not open: {type(e).__name__}: {e}")
            print("  (Is another Python process / notebook still using this DB?)\n")
            continue

        if not collections:
            print("  No collections inside (empty or broken DB)\n")
            client.close()
            continue

        for extra in describe_extra_files(db_path):
            print(f"  file  -> {extra}")

        for col in collections:
            try:
                points = load_points(client, col)
            except Exception as e:
                print(f"  [{col}] failed to read: {e}")
                continue

            a = analyse(points)
            print(f"\n  Collection : {col}")
            print(f"  Vectors    : {a['n']}   (vector size: {vector_info(client, col)})")
            print(f"  Content ID : {a['content_id']}   <- same ID across DBs = same chunk text")
            if a["sources"]:
                for s, c in a["sources"].most_common():
                    print(f"  Source PDF : {s}  ({c} chunks)")
            else:
                print("  Source PDF : NOT STORED in metadata (cannot tell which PDF this is)")
            print(f"  Chunk size : min {a['chars_min']} / avg {a['chars_avg']} / max {a['chars_max']} chars")
            pr = f"{a['page_min']}-{a['page_max']}" if a["page_min"] is not None else "none"
            print(f"  Pages      : {pr}   (chunks without page: {a['page_none']})")
            print(f"  Types      : {dict(a['content_types'])}")
            if "parent" in a["chunk_types"] or "child" in a["chunk_types"]:
                print(f"  Hierarchy  : {dict(a['chunk_types'])}, distinct parent_ids: {a['parent_ids']}")
            print(f"  Headings   : {a['headings']} distinct")
            if a["empty"]:
                print(f"  WARNING    : {a['empty']} empty chunks")
            if a["duplicates"]:
                print(f"  WARNING    : {a['duplicates']} duplicate chunk texts")

            if args.preview > 0:
                print(f"\n  Preview ({args.preview} chunks, ordered by page):")
                for r in sorted(a["recs"], key=sort_key)[: args.preview]:
                    m = r["meta"]
                    tag = m.get("heading") or m.get("parent_title") or ""
                    print(f"    - p.{m.get('page')} | {m.get('content_type', '-')} | {short(tag, 40)}")
                    print(f"      {short(r['text'])}")

            if args.export:
                label = f"{rel.replace(os.sep, '__')}__{col}"[:150]
                print(f"\n  Exported -> {export_csv(args.export, label, a['recs'])}")

            src_text = ", ".join(f"{s}({c})" for s, c in a["sources"].most_common()) or "-"
            for s, c in a["sources"].items():
                coverage[(chunking, s)][embedding] = c
            summary.append((doc_prefix, embedding, chunking, col, a["n"], a["content_id"], src_text))
            groups[(doc_prefix, chunking)].append(
                (embedding, a["n"], a["content_id"], dict(a["sources"])))

        client.close()
        print()

    # ---------------- summary table ----------------
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    hdr = f"{'doc folder':<22}{'embedding':<24}{'chunking':<30}{'vectors':>8}  {'content':<9}source"
    print(hdr)
    print("-" * len(hdr))
    for doc, emb, chk, col, n, cid, src in summary:
        print(f"{doc[:21]:<22}{emb[:23]:<24}{chk[:29]:<30}{n:>8}  {cid:<9}{src[:100]}")

    # ---------------- PDF coverage matrix ----------------
    print("\nPDF COVERAGE (chunks of each PDF inside each DB)")
    emb_names = sorted({e for d in coverage.values() for e in d})
    for chunking in sorted({k[0] for k in coverage}):
        print(f"\n  {chunking}")
        print(f"  {'PDF':<42}" + "".join(f"{e[:20]:>22}" for e in emb_names))
        for (chk, src), d in sorted(coverage.items()):
            if chk != chunking:
                continue
            print(f"  {src[:41]:<42}" + "".join(f"{d.get(e, 'MISSING'):>22}" for e in emb_names))

    # ---------------- consistency checks ----------------
    print("\nCHECKS")
    problems = 0
    for (doc, chunking), items in groups.items():
        ids = {i[2] for i in items}
        counts = {i[1] for i in items}
        if len(items) > 1 and (len(ids) > 1 or len(counts) > 1):
            problems += 1
            print(f"  [!] {chunking}: the embedding DBs disagree "
                  f"-> counts {sorted(counts)}, content IDs {sorted(ids)}")
            for emb, n, cid, srcs in items:
                print(f"      {emb:<22} {n:>7} vectors  {srcs or 'no source tags'}")
            print("      Same chunking should give the same PDFs and counts in every embedding DB. "
                  "A PDF is missing or incomplete in one of them -> rerun the experiment.")
    if not any(s[6] != "-" for s in summary):
        problems += 1
        print("  [!] No source/PDF name is stored in chunk metadata, so you cannot tell which "
              "PDF a vector came from. Add metadata['source'] before embedding.")
    if problems == 0:
        print("  All good: counts and content match across embedding models.")


if __name__ == "__main__":
    main()
