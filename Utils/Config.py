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
    EMBEDDING_MODELS:dict = {
        "BGE-M3_Embeding": "BAAI/bge-m3",
        "bge-small-en-v15_Embeding": "BAAI/bge-small-en-v1.5",
        "multilingual-E5-base_Embeding": "intfloat/multilingual-e5-base",
    }

    CHUNKING_FOLDERS:dict = {
        "structure_aware": "StructureAwareChunking",
        "hierarchical": "HierarchicalChunking",
        "context_aware": "ContextuaTokenAwareChunking",
    }

    class ConfigDict:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()