from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

## class work for converting the chunks to the embedings and store it in the vector DB (quadrant)
class Embeding:
    def __init__(self):
        pass

    def embed(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={
                "device": "cuda"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )


    def vectore_store(self,embeddings,langchain_chunks):
        vector_store = QdrantVectorStore.from_documents(
            documents=langchain_chunks,
            embedding=embeddings,
            path="D:\\M.tech\\Research Area\\RAG pipeline\\DB\\ParentChildChunk",
            collection_name="contextual_token"
        )