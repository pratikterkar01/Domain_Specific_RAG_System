"""
RAGAS evaluation for the building-code RAG chain, driven by your Excel question set.

Two stages, so the slow part is never repeated:
  1. generate : run every question through the chain (rag_chain.ComplianceRAG) and save answer, contexts,
                citations and timings to  <out-dir>/generation_results.jsonl   (resumable)
  2. score    : compute RAGAS metrics + your own retrieval / citation metrics from that file and write
                <out-dir>/rag_evaluation.xlsx  (Results, Summary, By_Category, By_QueryType, By_Difficulty, Config)

Excel columns (names are matched loosely, extra columns are kept):
  #, Category, Question, Answer / Ground Truth, Source Document, Section / Clause, Query Type, Difficulty,
  Gold Pages   e.g. "ECBC_2024.pdf:42"   (also "a.pdf:42-44", "a.pdf:42,43; b.pdf:10")

Install:
    pip install ragas pandas openpyxl openai sentence-transformers

Examples:
    # everything, GPT-4-class judge (recommended), one collection at a time
    python eval_ragas.py --excel eval_questions.xlsx --db-path ./DB_hybrid/BGE-M3_Embedding/StructureAwareChunking \
        --collection ecbc_2024 --only-source "ECBC 2024" --out-dir ./eval_out --judge openai

    # first 10 questions only, generation only
    python eval_ragas.py ... --stage generate --limit 10

    # re-score later without regenerating (e.g. with another judge)
    python eval_ragas.py --excel eval_questions.xlsx --out-dir ./eval_out --stage score --judge openai

IMPORTANT
  * A 4B local model is a poor RAGAS judge. Use a stronger judge (OpenAI / a larger model) for the scores.
  * Gold pages are compared with  payload page + --page-offset.  If your payload stores 0-based pages
    and the Excel stores PDF page numbers, use --page-offset 1.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import time
import warnings
from pathlib import Path

import pandas as pd
import  torch
from Utils import Config

global_Variable = Config.get_settings()
warnings.filterwarnings("ignore", category=DeprecationWarning)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# 1. Excel -> dataframe
# --------------------------------------------------------------------------
_COLS = {  # canonical name -> keywords searched in the (lower-cased) Excel header
    "id": ["#", "id", "no"],
    "category": ["category"],
    "question": ["question"],
    "ground_truth": ["ground truth", "answer"],
    "source_doc": ["source document", "source"],
    "section": ["section", "clause"],
    "query_type": ["query type", "type"],
    "difficulty": ["difficulty"],
    "gold_pages": ["gold page", "gold"],
}


def _find_col(columns, keys):
    low = {c: str(c).strip().lower() for c in columns}
    for k in keys:  # exact match first, then "contains"
        for c, l in low.items():
            if l == k:
                return c
    for k in keys:
        for c, l in low.items():
            if k in l and len(k) > 2:
                return c
    return None


def load_questions(path: str, sheet=0, only_source: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Reads the question sheet and prints how many rows survive each step (so a lost row is never silent)."""
    sheets = pd.ExcelFile(path).sheet_names
    if str(sheet).lower() == "all":  # every sheet of the workbook, stacked
        raw = pd.concat(pd.read_excel(path, sheet_name=None).values(), ignore_index=True)
        used = "ALL sheets"
    else:
        sheet = int(sheet) if str(sheet).isdigit() else sheet
        raw = pd.read_excel(path, sheet_name=sheet)
        used = f"sheet '{sheets[sheet] if isinstance(sheet, int) else sheet}'"
    print(f"workbook sheets: {sheets} | reading {used}")
    cols, taken = {}, set()
    for canon, keys in _COLS.items():
        c = _find_col([x for x in raw.columns if x not in taken], keys)
        if c is not None:
            cols[canon] = c
            taken.add(c)
    for need in ("question", "ground_truth"):
        if need not in cols:
            raise SystemExit(f"Excel has no column for '{need}'. Found columns: {list(raw.columns)}")
    df = pd.DataFrame({k: raw[v] for k, v in cols.items()})
    for opt in ("category", "source_doc", "section", "query_type", "difficulty", "gold_pages"):
        if opt not in df:
            df[opt] = ""
    n_raw = len(df)
    df = df.dropna(subset=["question"]).copy()
    n_q = len(df)
    # The id links questions to saved answers. If '#' is missing or repeats (e.g. numbering restarts per
    # category), rows would overwrite or skip each other, so fall back to the row position.
    if "id" not in df or df["id"].isna().any() or df["id"].astype(str).duplicated().any():
        print("WARNING: '#' column is missing or not unique -> using row numbers as question ids")
        df["id"] = [str(i) for i in range(1, len(df) + 1)]
    df["id"] = df["id"].astype(str)
    n_src = n_q
    if only_source:
        df = df[df["source_doc"].astype(str).str.contains(only_source, case=False, na=False)]
        n_src = len(df)
    df = df.head(limit) if limit else df
    print(f"questions: {n_raw} rows in sheet -> {n_q} with a question -> {n_src} after --only-source "
          f"{only_source!r} -> {len(df)} after --limit {limit}")
    if only_source and n_src < n_q:
        if "source_doc" in cols:
            print("  'Source Document' values found in the sheet:", sorted(set(raw[cols["source_doc"]].dropna().astype(str)))[:10])
    return df


# --------------------------------------------------------------------------
# 2. Gold pages and retrieval / citation metrics (no LLM needed)
# --------------------------------------------------------------------------
def _stem(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name or "").strip()))[0].lower()


def parse_gold(text) -> dict[str, set[int]]:
    """'ECBC_2024.pdf:42-44; X.pdf:7,9' -> {'ecbc_2024': {42,43,44}, 'x': {7,9}}"""
    out: dict[str, set[int]] = {}
    for part in re.split(r"[;\n]+", "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)):
        part = part.strip()
        if ":" not in part:
            continue
        f, pg = part.rsplit(":", 1)
        pg = re.sub(r"\s*[-–]\s*", "-", pg.strip())
        for tok in re.split(r"[,\s]+", pg):
            m = re.fullmatch(r"(\d+)(?:-(\d+))?", tok)
            if m:
                a, b = int(m.group(1)), int(m.group(2) or m.group(1))
                out.setdefault(_stem(f), set()).update(range(a, b + 1))
    return out


def _page_num(p, offset: int):
    try:
        return int(float(p)) + offset
    except (TypeError, ValueError):
        return None


def _match(src, page, gold, tol):
    g = gold.get(_stem(src))
    return page is not None and g is not None and any(abs(page - x) <= tol for x in g)


def retrieval_metrics(result: dict, gold: dict, offset: int = 0, tol: int = 0) -> dict:
    """Is the gold page in what the LLM saw? At what rerank rank? Did the answer cite it?"""
    if not gold:
        return {"gold_in_context": None, "gold_rank": None, "source_in_context": None, "cited_gold": None}
    ctx = sorted(result.get("context_meta", []), key=lambda c: -(c.get("rerank_score") or 0))
    ranks = [i + 1 for i, c in enumerate(ctx) if _match(c["source"], _page_num(c["page"], offset), gold, tol)]
    cited = [c for c in result.get("citations", [])]
    return {
        "gold_in_context": bool(ranks),
        "gold_rank": min(ranks) if ranks else None,
        "source_in_context": any(_stem(c["source"]) in gold for c in ctx),
        "cited_gold": any(_match(c["source"], _page_num(c["page"], offset), gold, tol) for c in cited)
        if not result.get("abstained") else False,
    }


_EXPECT_ABSTAIN = re.compile(r"unanswerable|out[- ]of[- ]scope|not in (the )?(documents?|corpus)|not found in the documents", re.I)


def expects_abstain(row) -> bool:
    return bool(_EXPECT_ABSTAIN.search(f"{row.get('query_type','')} {row.get('category','')} {row.get('ground_truth','')}"))


# --------------------------------------------------------------------------
# 3. Stage 1: generate
# --------------------------------------------------------------------------
def build_rag(args):
    """Builds ComplianceRAG exactly as in the demo of rag_chain.py. Adjust the import paths to your project."""
    from qdrant_client import QdrantClient
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from rag_chain import ComplianceRAG

    try:
        from Prompting.Prompt import RagPrompt
    except ImportError:
        pass
        #from Prompt import RagPrompt
    if args.engine == "native":
        try:
            from Retrival.qdrant_hybrid import NativeHybridRetriever as Retriever
        except ImportError:
            #from qdrant_hybrid import NativeHybridRetriever as Retriever
            pass
    else:
        from Retrival.HybrideRetrival import HybridRetriever as Retriever

    client = QdrantClient(path=args.db_path)

    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEVICE)  # keep VRAM for the LLM
    reranker = CrossEncoder(global_Variable.RERANKER_NAME, max_length=512, device=DEVICE)
    retriever = Retriever(client, args.collection, "BAAI/bge-small-en-v1.5", embedder=embedder,
                                        reranker=reranker)
    rag = ComplianceRAG(retriever, RagPrompt())
    rag.warmup()
    return rag


def run_generate(rag, df: pd.DataFrame, out_file: Path, section_filter: bool = False) -> None:
    """Runs every question through rag.stream() (gives TTFT) and appends one JSON line per question."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_file.exists():
        done = {json.loads(l)["id"] for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = df[~df["id"].isin(done)]
    print(f"generate: {len(done)} already done, {len(todo)} to run")
    with out_file.open("a", encoding="utf-8") as fh:
        for n, (_, row) in enumerate(todo.iterrows(), start=1):
            rec = {"id": row["id"], "question": row["question"]}
            try:
                final = None
                for ev in rag.stream(str(row["question"])):
                    if ev["type"] == "final":
                        final = ev["result"]
                rec.update(
                    answer=final["answer"],
                    abstained=final["abstained"],
                    contexts=[c["text"] for c in final["context"]],
                    context_meta=[{k: c[k] for k in ("n", "source", "section", "page", "rerank_score")} for c in final["context"]],
                    citations=final["citations"],
                    warnings=final["warnings"],
                    unsupported_numbers=final["unsupported_numbers"],
                    timings_ms=final["timings_ms"],
                    error=None,
                )
            except Exception as e:  # keep going; the row is reported with its error
                rec.update(answer="", abstained=False, contexts=[], context_meta=[], citations=[], warnings=["error"],
                           unsupported_numbers=[], timings_ms={}, error=f"{type(e).__name__}: {e}")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 5 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}")


# --------------------------------------------------------------------------
# 4. Stage 2: RAGAS scoring
# --------------------------------------------------------------------------
def make_judge(args):
    """Judge LLM + embeddings for the current RAGAS API (llm_factory, no deprecated evaluate()).
    Returns (llm, embeddings, max_workers)."""
    from openai import AsyncOpenAI
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms import llm_factory

    if args.judge == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set, so the OpenAI judge cannot run. Set it "
                             "(PowerShell: $env:OPENAI_API_KEY='sk-...') or use --judge ollama.")
        client = AsyncOpenAI()
        model = args.judge_model or os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o")
        workers = 8
    else:  # Ollama exposes an OpenAI-compatible endpoint
        client = AsyncOpenAI(base_url=args.ollama_url.rstrip("/") + "/v1", api_key="ollama")
        model = args.judge_model or "qwen3:8b"
        workers = 1  # one local GPU: run judge calls one at a time
    llm = llm_factory(model, provider="openai", client=client, temperature=0)

    from sentence_transformers import SentenceTransformer

    class LocalEmbeddings(BaseRagasEmbedding):
        """BGE-M3 on CPU, used by answer relevancy (question-vs-generated-question similarity)."""

        def __init__(self, name):
            super().__init__()
            self.model = SentenceTransformer(name, device=DEVICE)

        def embed_text(self, text, **kw):
            return self.model.encode(text, normalize_embeddings=True).tolist()

        async def aembed_text(self, text, **kw):
            return await asyncio.to_thread(self.embed_text, text)

        def embed_texts(self, texts, **kw):
            return self.model.encode(list(texts), normalize_embeddings=True).tolist()

        async def aembed_texts(self, texts, **kw):
            return await asyncio.to_thread(self.embed_texts, texts)

    return llm, LocalEmbeddings(args.embed_model), workers


METRIC_CHOICES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "factual_correctness"]


def _metric_fns(names, llm, embeddings):
    """name -> async function(row) using ragas.metrics.collections (the non-deprecated API)."""
    from ragas.metrics.collections import (AnswerRelevancy, ContextPrecisionWithReference, ContextRecall,
                                           FactualCorrectness, Faithfulness)

    fns = {}
    if "faithfulness" in names:        # is the answer supported by the contexts?
        m = Faithfulness(llm=llm)
        fns["faithfulness"] = lambda r, m=m: m.ascore(user_input=r["q"], response=r["a"], retrieved_contexts=r["ctx"])
    if "answer_relevancy" in names:    # does it answer the question?
        m = AnswerRelevancy(llm=llm, embeddings=embeddings)
        fns["answer_relevancy"] = lambda r, m=m: m.ascore(user_input=r["q"], response=r["a"])
    if "context_precision" in names:   # are the useful contexts ranked first?
        m = ContextPrecisionWithReference(llm=llm)
        fns["context_precision"] = lambda r, m=m: m.ascore(user_input=r["q"], reference=r["gt"], retrieved_contexts=r["ctx"])
    if "context_recall" in names:      # do the contexts contain the ground truth?
        m = ContextRecall(llm=llm)
        fns["context_recall"] = lambda r, m=m: m.ascore(user_input=r["q"], retrieved_contexts=r["ctx"], reference=r["gt"])
    if "factual_correctness" in names:  # answer claims vs ground-truth claims (F1)
        m = FactualCorrectness(llm=llm)
        fns["factual_correctness"] = lambda r, m=m: m.ascore(response=r["a"], reference=r["gt"])
    return fns


def score_with_ragas(results: pd.DataFrame, gt: dict, llm, embeddings, metric_names, workers=4, timeout=300) -> pd.DataFrame:
    """results: rows of generation_results.jsonl; gt: id -> ground truth. Returns one row of scores per result row.
    A failed metric call becomes NaN, and the first error message per metric is printed."""
    fns = _metric_fns(metric_names, llm, embeddings)
    rows = [{"q": str(r["question"]), "a": str(r["answer"] or ""), "ctx": [str(c) for c in (r["contexts"] or [])] or [""],
             "gt": str(gt.get(r["id"], ""))} for _, r in results.iterrows()]
    errors: dict[str, list] = {}
    progress = {"done": 0}

    async def run():
        sem = asyncio.Semaphore(workers)

        async def one(row):
            out = {}
            for name, fn in fns.items():
                val = float("nan")
                for attempt in range(2):
                    try:
                        async with sem:
                            res = await asyncio.wait_for(fn(row), timeout)
                        val = float(res.value) if res.value is not None else float("nan")
                        break
                    except Exception as e:
                        if attempt == 1:
                            cnt, first = errors.get(name, [0, None])
                            errors[name] = [cnt + 1, first or f"{type(e).__name__}: {str(e)[:300]}"]
                out[name] = val
            progress["done"] += 1
            if progress["done"] % 5 == 0 or progress["done"] == len(rows):
                print(f"  scored {progress['done']}/{len(rows)}", flush=True)
            return out

        return await asyncio.gather(*[one(r) for r in rows])

    sc = pd.DataFrame(asyncio.run(run()), columns=list(fns))
    sc.index = results.index
    for name, (cnt, first) in errors.items():
        print(f"  {name}: {cnt} call(s) failed. First error: {first}")
    return sc


# --------------------------------------------------------------------------
# 5. Report
# --------------------------------------------------------------------------
def load_results(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty. Run --stage generate first.")
    return pd.DataFrame(rows)


def assemble(questions: pd.DataFrame, results: pd.DataFrame, scores: pd.DataFrame | None, offset=0, tol=0) -> pd.DataFrame:
    df = questions.merge(results, on="id", how="inner", suffixes=("", "_run"))
    ret = []
    for _, r in df.iterrows():
        m = retrieval_metrics(r.to_dict(), parse_gold(r["gold_pages"]), offset, tol)
        ret.append(m)
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(ret)], axis=1)
    t = df["timings_ms"].apply(lambda d: d or {})
    df["retrieval_ms"] = t.apply(lambda d: d.get("total"))
    df["ttft_ms"] = t.apply(lambda d: (d.get("total", 0) or 0) + (d.get("augment", 0) or 0) + (d.get("ttft_after_retrieval") or 0)
                            if d.get("ttft_after_retrieval") is not None else None)
    df["end_to_end_ms"] = t.apply(lambda d: d.get("end_to_end"))
    df["expects_abstain"] = df.apply(lambda r: expects_abstain(r.to_dict()), axis=1)
    df["answered"] = ~df["abstained"].astype(bool)
    df["citation_ok"] = df.apply(lambda r: bool(r["abstained"]) or (len(r["citations"]) > 0 and not r["warnings"]), axis=1)
    if scores is not None:
        scores = scores.copy()
        _SCORE_COLS[:] = [c for c in scores.columns if c != "id"]
        scores["id"] = results["id"].values
        df = df.merge(scores, on="id", how="left")
    return df


_SCORE_COLS: list[str] = []   # names of the RAGAS columns returned in this run (set in assemble)


def _metric_cols(df):
    if _SCORE_COLS:
        return [c for c in _SCORE_COLS if c in df]
    return [c for c in df.columns if c in METRIC_CHOICES]


def summarize(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    rcols = _metric_cols(df)
    cols = rcols + ["gold_in_context", "cited_gold", "answered", "citation_ok", "retrieval_ms", "ttft_ms", "end_to_end_ms"]
    cols = [c for c in cols if c in df]
    num = df.copy()
    for c in ("gold_in_context", "cited_gold", "answered", "citation_ok"):
        if c in num:
            num[c] = pd.to_numeric(num[c].map({True: 1.0, False: 0.0}), errors="coerce")
    for c in cols:
        num[c] = pd.to_numeric(num[c], errors="coerce")
    if by is None:
        out = num[cols].mean().to_frame("mean").T
        out.insert(0, "n_questions", len(num))
        p95 = num["end_to_end_ms"].quantile(0.95) if "end_to_end_ms" in num else None
        out["end_to_end_p95_ms"] = p95
        out.index = ["ALL"]
    else:
        out = num.groupby(by)[cols].mean()
        out.insert(0, "n_questions", num.groupby(by).size())
    return out.round(3)


def write_report(df: pd.DataFrame, path: Path, config: dict) -> None:
    from openpyxl.styles import Alignment, Font

    show = ["id", "category", "query_type", "difficulty", "question", "ground_truth", "answer", "abstained", "expects_abstain",
            "gold_pages", "gold_in_context", "gold_rank", "source_in_context", "cited_gold", "citation_ok", "warnings",
            "unsupported_numbers"] + _metric_cols(df) + ["retrieval_ms", "ttft_ms", "end_to_end_ms", "error"]
    res = df[[c for c in show if c in df]].copy()
    for c in ("warnings", "unsupported_numbers"):
        if c in res:
            res[c] = res[c].apply(lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else v)
    extra = {}
    exp = df[df["expects_abstain"]]
    if len(exp):
        extra["abstain_recall_on_unanswerable"] = round(float(exp["abstained"].mean()), 3)
    ans = df[~df["expects_abstain"]]
    if len(ans):
        extra["wrongly_abstained_on_answerable"] = round(float(ans["abstained"].mean()), 3)
    overall = summarize(df)
    for k, v in extra.items():
        overall[k] = v
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        res.round(3).to_excel(xw, sheet_name="Results", index=False)
        overall.to_excel(xw, sheet_name="Summary")
        for sheet, col in (("By_Category", "category"), ("By_QueryType", "query_type"), ("By_Difficulty", "difficulty")):
            if col in df and df[col].astype(str).str.strip().ne("").any():
                summarize(df, col).to_excel(xw, sheet_name=sheet)
        pd.DataFrame(list(config.items()), columns=["setting", "value"]).to_excel(xw, sheet_name="Config", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for col in ws.columns:
                letter = col[0].column_letter
                header = str(col[0].value or "")
                ws.column_dimensions[letter].width = 60 if header in ("question", "ground_truth", "answer") else min(max(len(header) + 2, 12), 28)
            if ws.title == "Results":
                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")


# --------------------------------------------------------------------------
# 6. CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--excel", required=True)
    ap.add_argument("--sheet", default=0, help="sheet name or number, or 'all' to read every sheet")
    ap.add_argument("--only-source", default=None, help="keep rows whose 'Source Document' contains this text (match it to the collection)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="./eval_out")
    ap.add_argument("--stage", choices=["generate", "score", "all"], default="all")
    # generation
    ap.add_argument("--db-path")
    ap.add_argument("--collection")
    ap.add_argument("--engine", choices=["native", "manual"], default="native")
    ap.add_argument("--embed-model", default=global_Variable.USED_EMBEDDING_MODELS)
    ap.add_argument("--reranker", default=global_Variable.RERANKER_NAME)
    # scoring
    ap.add_argument("--judge", choices=["openai", "ollama"], default="ollama")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--metrics", nargs="+", default=METRIC_CHOICES, choices=METRIC_CHOICES)
    ap.add_argument("--page-offset", type=int, default=0, help="added to payload page numbers before comparing with Gold Pages")
    ap.add_argument("--page-tol", type=int, default=0, help="accept pages within +/- this many pages of a gold page")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen_file = out / "generation_results.jsonl"
    questions = load_questions(a.excel, a.sheet, a.only_source, a.limit)
    print(f"{len(questions)} questions loaded")

    if a.stage == "generate":
        print("NOTE: --stage generate only creates generation_results.jsonl. RAGAS scores need --stage score (or all).")
    if a.stage in ("generate", "all"):
        if not (a.db_path and a.collection):
            raise SystemExit("--db-path and --collection are required for the generate stage")
        run_generate(build_rag(a), questions, gen_file)

    if a.stage in ("score", "all"):
        all_results = load_results(gen_file)
        results = all_results[all_results["id"].isin(questions["id"])].reset_index(drop=True)
        gt = dict(zip(questions["id"], questions["ground_truth"]))
        ok = results[results["error"].isna()] if "error" in results else results
        print(f"scoring funnel: {len(questions)} questions -> {len(results)} generated -> {len(ok)} without error")
        if len(ok) < len(results):
            print("  rows that failed in generation (not scored):", results.loc[~results.index.isin(ok.index), "id"].tolist()[:20])
        if len(results) < len(questions):
            print("  not generated yet:", len(questions) - len(results), "-> run --stage generate again (it resumes)")
        if len(ok) == 0:  # nothing to score: say why and stop instead of writing an empty report
            print("\nNOTHING TO SCORE.")
            if len(results) == 0:
                print(f"  {gen_file} has {len(all_results)} saved answers, but none match the {len(questions)} questions selected now.")
                print("  saved ids    :", all_results["id"].astype(str).tolist()[:8])
                print("  question ids :", questions["id"].astype(str).tolist()[:8])
                print("  Usual causes: --only-source / --limit / --sheet differ from the generate run, or the sheet's '#' column changed.")
                print("  Fix: use the same options as the generate run, or delete the .jsonl file and run --stage generate again.")
            else:
                errs = results["error"].dropna().astype(str)
                print(f"  all {len(results)} generated rows failed. First error messages:")
                for e in errs.drop_duplicates().head(3):
                    print("   -", e[:300])
                print("  Fix the cause (e.g. Ollama not running, wrong import path, wrong collection), delete the .jsonl file, run --stage generate again.")
            raise SystemExit(1)
        llm, emb, workers = make_judge(a)
        t0 = time.time()
        scores = score_with_ragas(ok.reset_index(drop=True), gt, llm, emb, a.metrics, workers=workers)
        print(f"RAGAS scoring took {time.time() - t0:.0f}s")
        valid = scores.notna().sum()
        print("valid scores per metric:", {k: f"{int(v)}/{len(scores)}" for k, v in valid.items()})
        if (valid == 0).any() or len(scores.columns) == 0:
            print("WARNING: no valid score for: " + ", ".join(valid[valid == 0].index) + "\n"
                  "  Usual causes: wrong/missing API key, a judge model that cannot return valid JSON (small local\n"
                  "  models), or timeouts. Scroll up for 'Exception raised in Job' / 'failed to parse output' lines.")
        scores_full = scores.reindex(range(len(ok)))
        df = assemble(questions, ok.reset_index(drop=True), scores_full, a.page_offset, a.page_tol)
        path = out / "rag_evaluation.xlsx"
        write_report(df, path, {"excel": a.excel, "collection": a.collection, "judge": f"{a.judge}:{a.judge_model}",
                                "metrics": ", ".join(a.metrics), "page_offset": a.page_offset, "page_tol": a.page_tol,
                                "questions": len(questions), "scored": len(ok)})
        print(f"report written: {path}")
        print(summarize(df).T.to_string())


if __name__ == "__main__":
    main()