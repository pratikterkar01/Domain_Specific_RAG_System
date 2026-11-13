from pydantic_settings import  BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    ## server setting
    HOST: str = "127.0.0.1"
    PORT: int = 8080
    DEBUG: bool = True

    # ChromaDB
    CHROMA_COLLECTION: str = "building_docs"
    CHROMA_PERSIST_DIR: str = "D:\\AiEnergyAssitanceForBuilding\\AiBuildingAssistanceApi\\ChromeDB"
    CURRENT_DIR:str  = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT: str = os.path.dirname(CURRENT_DIR)
    # 3. Create your DB path cleanly
    DB_ROOT: str = os.path.join(PROJECT_ROOT, "DB")
    PARENTS_FILE:str = "parents.json"
    MANIFEST_FILE:str = "ingest_manifest.json"
    EMBEDDING_MODELS:dict = {
        "BGE-M3": "BAAI/bge-m3",
        "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
        "multilingual-E5-base": "intfloat/multilingual-e5-base",
    }

    CHUNKING_FOLDERS:dict = {
        "structure_aware": "StructureAwareChunking",
        "hierarchical": "HierarchicalChunking",
        "context_aware": "ContextuaTokenAwareChunking",
    }



    # --------------------------------------------------------------------------
    # CONFIG: change these to match how your 9 collections were written
    # --------------------------------------------------------------------------
    TEXT_FIELD:str = "text"  # payload key holding the child chunk text
    PARENT_ID_FIELD:str = "parent_id"  # payload key linking a child to its parent section
    PARENT_TEXT_FIELD:str = "parent_text"  # payload key with the full parent section (if stored)
    SOURCE_FIELD:str = "source"  # e.g. document name, used for citations
    SECTION_FIELD :str= "section"  # e.g. "6.5.1", used for citations
    VECTOR_NAME:str | None = None  # None for an unnamed vector, or e.g. "dense"

    DENSE_TOP_K:int = 50
    BM25_TOP_K :int= 50
    RRF_K:int = 60
    RERANK_POOL:int = 30  # how many fused candidates the cross-encoder scores
    FINAL_TOP_K:int = 5
    RERANKER_NAME:str = "BAAI/bge-reranker-v2-m3"
    TEXT_CANDIDATES:tuple = ("text", "page_content", "content", "chunk_text", "chunk")

    ABSTAIN_TEXT:str = "Not found in the documents."

    # ---- 1. Static prefix (system prompt + few-shot) ----
    # Kept identical on every request so the server can cache it (prefix caching).
    SYSTEM_PROMPT:str = f"""You are an assistant that answers questions about building codes \
    and energy standards using ONLY the numbered context clauses provided.

    Rules:
    1. Use only the information in the context. Do not use outside knowledge.
    2. After every claim, cite the supporting clause as [n], e.g. [1] or [2][3].
    3. Copy numbers, units and thresholds exactly as written in the context.
    4. If the context does not contain enough information to answer, reply exactly:
       "{ABSTAIN_TEXT}"
    5. Do not guess or invent clause numbers. Cite only the [n] labels shown.
    6. Be concise. Answer in at most 5 sentences.

    Examples (illustrative format only; the values are placeholders):

    Context:
    [1] (Doc A, Sec 1.1) The minimum value of X for category P is 10 units.
    [2] (Doc A, Sec 1.2) Category Q has no minimum requirement for X.
    Question: What is the minimum X for category P?
    Answer: The minimum X for category P is 10 units [1].

    Context:
    [1] (Doc B, Sec 2.1) Equipment Y shall be tested annually.
    Question: What is the maximum operating temperature of equipment Y?
    Answer: {ABSTAIN_TEXT}
    """

    # ---- 2. Dynamic part (context + question) ----
    HUMAN_PROMPT:str = """Context:
    {context}

    Question: {question}
    Answer:"""
    STYLES :dict= {
        "technical": "",
        "plain": "Explain in plain English for a non-engineer, but keep every number and every [n] citation exact.",
    }
    class ConfigDict:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()