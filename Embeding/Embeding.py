from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import os

## class work for converting the chunks to the embedings and store it in the vector DB (quadrant)
class Embeding:
    def __init__(self):
        pass

    def embed(self,model_name):
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": "cuda"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

        return  embeddings


    def vectore_store(self,embeddings,langchain_chunks,collection_name,db_path):
        os.makedirs(db_path, exist_ok=True)
        vector_store = QdrantVectorStore.from_documents(
            documents=langchain_chunks,
            embedding=embeddings,
            path=db_path,
            collection_name=collection_name
        )
        # Your version had no return, so the store was lost. Return it.
        return vector_store

