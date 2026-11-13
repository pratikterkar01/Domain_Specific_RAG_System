"""
Augmentation + generation for the building-code RAG, orchestrated with LangChain (LCEL).

    question
      -> retriever.search()            your hybrid retriever (BM25 + BGE-M3, RRF, cross-encoder rerank)
      -> evidence gate                 weak evidence -> "Not found in the documents." (no LLM call)
      -> augment                       parent sections, de-duplicated, capped, best first and last, numbered [n]
      -> your prompt | LLM | parser    grounded + citation + abstain (prompt text comes from your Prompt.py / config)
      -> verify                        cited [n] exist, numbers appear in the cited context, build citation list
      -> result dict

Works with ANY retriever object that has
    search(query, section=None, source=None, top_k=None) -> (hits, timings)
where each hit has  .id  .text  .payload (dict)  .rerank_score
(this is true for HybridRetriever in HybrideRetrival.py and NativeHybridRetriever in qdrant_hybrid.py).

Install:
    pip install langchain-core langchain-ollama
    pip install langchain-openai          # only if you want GPT-4 as the LLM
    ollama pull qwen3:4b                  # Q4_K_M by default

Use:
    from Prompt import RagPrompt                    # your prompt class (Prompt.py)
    rag = ComplianceRAG(retriever, RagPrompt())
    result = rag.chain.invoke({"question": "What is the minimum window-to-wall ratio for Climate Zone 2?"})
    print(result["answer"], result["citations"])

    # streaming (FastAPI / Angular):
    for ev in rag.stream("What U-value must my roof meet?"):
        if ev["type"] == "token": ...        # send ev["text"] to the browser
        else:                      ...       # ev["result"] is the same dict as chain.invoke()
"""
from __future__ import annotations

import os
import re
import time
import torch
from Utils import Config
from dataclasses import dataclass
from typing import  Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

global_variable = Config.get_settings()
# Optional: reuse your project settings for payload field names (falls back to defaults).
try:
    from Utils import Config

    _cfg = Config.get_settings()
except Exception:  # running outside your project
    _cfg = None


def _cfgv(name: str, default):
    """Read a setting from your project config; use the default if it is missing or the config fails."""
    if _cfg is None:
        return default
    try:
        value = getattr(_cfg, name)
    except Exception:  # missing name, or a config object that raises something other than AttributeError
        try:
            value = _cfg[name]  # dict-style config
        except Exception:
            return default
    return default if value is None else value


ABSTAIN = _cfgv("ABSTAIN_TEXT", "Not found in the documents.")   # same setting your Prompt.py uses


# --------------------------------------------------------------------------
# Settings (override with environment variables or by passing your own)
# --------------------------------------------------------------------------
@dataclass
class RagSettings:
    # LLM
    provider: str = os.getenv("RAG_LLM_PROVIDER", "ollama")         # "ollama" | "openai"
    ollama_model: str = os.getenv("RAG_OLLAMA_MODEL", "qwen3:4b")
    ollama_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    openai_model: str = os.getenv("RAG_OPENAI_MODEL", "gpt-4o")     # set to the GPT-4 model you use
    num_ctx: int = int(os.getenv("RAG_NUM_CTX", "4096"))            # raise if you send bigger parents
    max_new_tokens: int = int(os.getenv("RAG_MAX_NEW_TOKENS", "400"))
    keep_alive: str = os.getenv("RAG_KEEP_ALIVE", "30m")            # keep the model loaded in VRAM
    # Retrieval -> augmentation
    retrieve_top_k: int = 5                                         # hits asked from the retriever (after rerank)
    max_parents: int = 3                                            # parent sections sent to the LLM
    parent_max_chars: int = 2400                                    # ~600 tokens per parent
    # Evidence gate. CrossEncoder gives 0-1 scores for bge-reranker-v2-m3 (sigmoid). TUNE on your golden set.
    min_rerank_score: float = float(os.getenv("RAG_MIN_RERANK", "0.05"))
    # Payload field names
    source_field: str = _cfgv("SOURCE_FIELD", "source")
    section_field: str = _cfgv("SECTION_FIELD", "section")
    parent_id_field: str = _cfgv("PARENT_ID_FIELD", "parent_id")
    parent_text_field: str = _cfgv("PARENT_TEXT_FIELD", "parent_text")
    page_keys: tuple = ("page", "page_number", "page_no", "pageno")


# --------------------------------------------------------------------------
# The prompt itself lives in YOUR Prompt.py / config (SYSTEM_PROMPT, HUMAN_PROMPT). Only this small
# optional instruction is defined here: it is appended to the question when style="plain".
# --------------------------------------------------------------------------
STYLES = {
    "technical": "",
    "plain": "Explain in plain English for a non-engineer, but keep every number and every [n] citation exact.",
}

_THINK = re.compile(r"<think>.*?</think>", re.S)
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_NUM = re.compile(r"\d+(?:\.\d+)*")


# --------------------------------------------------------------------------
# Accept a ChatPromptTemplate OR your RagPrompt class instance
# --------------------------------------------------------------------------
def _resolve_prompt(p):
    if isinstance(p, Runnable):  # already a ChatPromptTemplate / runnable
        return p
    for attr in ("prompt", "chat_prompt", "template"):  # attribute holding the template
        v = getattr(p, attr, None)
        if isinstance(v, Runnable):
            return v
    for meth in ("get_prompt", "build_prompt", "_build_prompt"):  # method returning the template
        f = getattr(p, meth, None)
        if callable(f):
            v = f()
            if isinstance(v, Runnable):
                return v
    raise TypeError(
        f"{type(p).__name__} is not a prompt template and has no .prompt attribute or get_prompt()/build_prompt() "
        "method returning one. Pass a ChatPromptTemplate, or add `self.prompt = ChatPromptTemplate...` to your class."
    )


# --------------------------------------------------------------------------
# LLM factory
# --------------------------------------------------------------------------
def make_llm(s: RagSettings):
    if s.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=s.openai_model, temperature=0, max_tokens=s.max_new_tokens, streaming=True)
    from langchain_ollama import ChatOllama

    kw = dict(
        model=s.ollama_model,
        base_url=s.ollama_url,
        temperature=0,
        num_ctx=s.num_ctx,
        num_predict=s.max_new_tokens,
        keep_alive=s.keep_alive,
    )
    try:  # Qwen3 "thinking" must be off for latency; the parameter exists in recent langchain-ollama
        return ChatOllama(**kw, reasoning=False)
    except Exception:
        return ChatOllama(**kw)


# --------------------------------------------------------------------------
# The RAG object
# --------------------------------------------------------------------------
class ComplianceRAG:
    def __init__(self, retriever, prompt, llm=None, settings: RagSettings | None = None):
        """prompt: your RagPrompt() object from Prompt.py (or a ChatPromptTemplate)."""
        self.s = settings or RagSettings()
        self.retriever = retriever
        self.prompt = _resolve_prompt(prompt)   # RagPrompt object -> its ChatPromptTemplate
        self.llm = llm or make_llm(self.s)
        self._gen = RunnableLambda(self._prompt_inputs) | self.prompt | self.llm | StrOutputParser()
        self._prepare = RunnableLambda(self._retrieve_and_augment)
        self.chain: Runnable = self._build_chain()

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _coerce(inp) -> dict:
        if isinstance(inp, str):
            inp = {"question": inp}
        return {
            "question": inp["question"],
            "section": inp.get("section"),
            "source": inp.get("source"),
            "style": inp.get("style", "technical"),
        }

    def _page(self, payload: dict):
        for k in self.s.page_keys:
            if payload.get(k) not in (None, ""):
                return payload[k]
        return None

    def _fit(self, parent: str, child: str, max_chars: int) -> str:
        """Cap a parent section but keep the window around the matched child chunk."""
        if len(parent) <= max_chars:
            return parent
        pos = parent.find(child[:100]) if child else -1
        pos = max(pos, 0)
        start = max(0, pos - max(0, max_chars - len(child)) // 2)
        start = max(0, min(start, len(parent) - max_chars))
        end = start + max_chars
        return ("…" if start > 0 else "") + parent[start:end] + ("…" if end < len(parent) else "")

    # ---- step 1+2+3: retrieve, gate, augment --------------------------------
    def _retrieve_and_augment(self, inp: dict) -> dict:
        s = self.s
        inp = self._coerce(inp)
        t_start = time.perf_counter()
        hits, timings = self.retriever.search(
            inp["question"], section=inp["section"], source=inp["source"], top_k=s.retrieve_top_k
        )
        timings_ms = dict(getattr(timings, "ms", {}) or {})

        t0 = time.perf_counter()
        gate_pass = bool(hits) and float(getattr(hits[0], "rerank_score", 1.0)) >= s.min_rerank_score

        context, seen = [], set()
        for h in hits:  # hits are already best-first
            payload = h.payload or {}
            key = payload.get(s.parent_id_field) or h.id
            if key in seen:
                continue
            seen.add(key)
            parent = payload.get(s.parent_text_field) or h.text
            context.append(
                {
                    "source": payload.get(s.source_field, "unknown source"),
                    "section": payload.get(s.section_field, ""),
                    "page": self._page(payload),
                    "rerank_score": float(getattr(h, "rerank_score", 0.0)),
                    "text": self._fit(parent, h.text or "", s.parent_max_chars),
                }
            )
            if len(context) == s.max_parents:
                break

        # "lost in the middle": best first, second best last, weakest in the middle
        if len(context) > 2:
            context = [context[0]] + context[2:] + [context[1]]
        for n, c in enumerate(context, start=1):
            c["n"] = n

        blocks = []
        for c in context:
            page = f" | p.{c['page']}" if c["page"] is not None else ""
            sec = f" | {c['section']}" if c["section"] else ""
            blocks.append(f"[{c['n']}] ({c['source']}{sec}{page})\n{c['text']}")
        timings_ms["augment"] = (time.perf_counter() - t0) * 1000

        return {
            **inp,
            "t_start": t_start,
            "gate_pass": gate_pass,
            "context": context,
            "context_text": "\n\n".join(blocks),
            "timings": timings_ms,
        }

    def _prompt_inputs(self, st: dict) -> dict:
        """Give the prompt only the variables it declares (works with your Prompt.py: {context}, {question})."""
        style = STYLES.get(st["style"], "")
        question = st["question"]
        wanted = set(getattr(self.prompt, "input_variables", []) or [])
        if style and "style" not in wanted:  # your prompt has no {style}: add the instruction to the question
            question = f"{question}\n{style}"
        values = {"context": st["context_text"], "question": question, "style": style}
        return {k: v for k, v in values.items() if not wanted or k in wanted}

    # ---- step 4: generate (LCEL) --------------------------------------------
    def _build_chain(self) -> Runnable:
        def mark(key):
            def _f(st):
                st.setdefault("_t", {})[key] = time.perf_counter()
                return st

            return RunnableLambda(_f)

        generate = (
            mark("g0")
            | RunnablePassthrough.assign(raw_answer=self._gen)
            | mark("g1")
        )
        abstain = RunnableLambda(lambda st: {**st, "raw_answer": ABSTAIN})
        branch = RunnableBranch((lambda st: not st["gate_pass"], abstain), generate)
        return self._prepare | branch | RunnableLambda(self._finalize)

    # ---- step 5: verify + build the result ------------------------------------
    def _finalize(self, st: dict) -> dict:
        raw = _THINK.sub("", st["raw_answer"]).strip()
        abstained = ABSTAIN.lower().rstrip(". ") in raw.lower()   # same rule as your Prompt.py postprocess
        by_n = {c["n"]: c for c in st["context"]}

        cited = []
        for grp in _CITE.findall(raw):
            for x in grp.split(","):
                n = int(x)
                if n not in cited:
                    cited.append(n)
        valid = [n for n in cited if n in by_n]
        invalid = [n for n in cited if n not in by_n]

        warnings = []
        unsupported = []
        if not abstained:
            if not valid:
                warnings.append("no_valid_citation")
            if invalid:
                warnings.append(f"invalid_citation:{invalid}")
            cited_text = " ".join(by_n[n]["text"] for n in valid)
            body = _CITE.sub("", raw)
            unsupported = [x for x in dict.fromkeys(_NUM.findall(body)) if x not in cited_text]
            if unsupported:
                warnings.append("numbers_not_in_cited_context")

        t = dict(st["timings"])
        g = st.get("_t", {})
        if "g0" in g and "g1" in g:
            t["generate"] = (g["g1"] - g["g0"]) * 1000
        if st.get("ttft_ms") is not None:
            t["ttft_after_retrieval"] = st["ttft_ms"]
            t["generate"] = st.get("gen_total_ms", t.get("generate"))
        t["end_to_end"] = (time.perf_counter() - st["t_start"]) * 1000

        return {
            "question": st["question"],
            "answer": raw,
            "abstained": abstained,
            "citations": [
                {k: by_n[n][k] for k in ("n", "source", "section", "page", "rerank_score")} for n in valid
            ]
            if not abstained
            else [],
            # same keys as your Prompt.py postprocess(), so existing evaluation code keeps working
            "cited_ids": valid,
            "invalid_citations": invalid,
            "no_citation": (not abstained) and not valid,
            "sources": [{k: c[k] for k in ("source", "section", "page")} for c in st["context"]],
            "unsupported_numbers": unsupported,
            "warnings": warnings,
            "needs_review": bool(warnings),
            "context": st["context"],
            "timings_ms": {k: round(v, 1) for k, v in t.items() if v is not None},
        }

    # ---- public API -------------------------------------------------------------
    def ask(self, question: str, section=None, source=None, style="technical") -> dict:
        return self.chain.invoke({"question": question, "section": section, "source": source, "style": style})

    def stream(self, question: str, section=None, source=None, style="technical") -> Iterator[dict]:
        """Yields {"type": "token", "text": ...} while generating, then {"type": "final", "result": {...}}."""
        st = self._prepare.invoke({"question": question, "section": section, "source": source, "style": style})
        if not st["gate_pass"]:
            st["raw_answer"] = ABSTAIN
            yield {"type": "token", "text": ABSTAIN}
        else:
            parts, t0, first = [], time.perf_counter(), None
            for tok in self._gen.stream(st):
                if first is None:
                    first = (time.perf_counter() - t0) * 1000
                parts.append(tok)
                yield {"type": "token", "text": tok}
            st["raw_answer"] = "".join(parts)
            st["ttft_ms"] = first
            st["gen_total_ms"] = (time.perf_counter() - t0) * 1000
        yield {"type": "final", "result": self._finalize(st)}

    def warmup(self) -> None:
        """Load the model into VRAM and prime the static prefix. Call once at application start-up."""
        try:
            self._gen.invoke({"question": "warmup", "context_text": "[1] (warmup)\nwarmup", "style": "technical"})
        except Exception as e:  # never block start-up
            print(f"warmup skipped: {e}")


# --------------------------------------------------------------------------
# Demo with your retrievers (adjust the import paths to your project)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from qdrant_client import QdrantClient
    from sentence_transformers import CrossEncoder, SentenceTransformer

    # from Retrival.qdrant_hybrid import NativeHybridRetriever as Retriever   # native hybrid inside Qdrant
    from Retrival import qdrant_hybrid as Q

    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="local Qdrant folder")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--stream", action="store_true")
    a = ap.parse_args()

    client = QdrantClient(path=a.path)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEVICE)                    # keep VRAM for the LLM
    reranker = CrossEncoder(global_variable.RERANKER_NAME, max_length=512, device=DEVICE)
    retriever = Q.NativeHybridRetriever(client, a.collection, "BAAI/bge-small-en-v1.5", embedder=embedder, reranker=reranker)

    from Prompting.Prompt import RagPrompt                    # your prompt class; adjust the import path
    rag = ComplianceRAG(retriever, RagPrompt())
    rag.warmup()
    if a.stream:
        for ev in rag.stream(a.query):
            if ev["type"] == "token":
                print(ev["text"], end="", flush=True)
            else:
                print("\n", ev["result"]["citations"], ev["result"]["timings_ms"])
    else:
        out = rag.chain.invoke({"question": a.query})
        print(out["answer"], "\n", out["citations"], "\n", out["warnings"], "\n", out["timings_ms"])