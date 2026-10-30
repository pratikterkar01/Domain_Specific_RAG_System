from pydantic_settings import  BaseSettings
from pydantic import ConfigDict
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
    VECTOR_NAME:str = None  # None for an unnamed vector, or e.g. "dense"

    DENSE_TOP_K:int = 50
    BM25_TOP_K :int= 50
    RRF_K:int = 60
    RERANK_POOL:int = 30  # how many fused candidates the cross-encoder scores
    FINAL_TOP_K:int = 5
    RERANKER_NAME:str = "BAAI/bge-reranker-v2-m3"


    class ConfigDict:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()