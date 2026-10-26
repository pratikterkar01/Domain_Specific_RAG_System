import gc
import os

import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models


class OOMSafeEmbeddings(Embeddings):
    """Wraps an embedding model. On CUDA out-of-memory it halves the batch size
    and retries the same batch, so a build keeps going instead of failing."""

    def __init__(self, base, batch_size=16):
        self.base = base
        self.batch_size = batch_size

    def embed_documents(self, texts):
        out, i = [], 0
        while i < len(texts):
            batch = texts[i:i + self.batch_size]
            try:
                out.extend(self.base.embed_documents(batch))
                i += len(batch)
            except torch.cuda.OutOfMemoryError:
                Embeding.clear_cuda()
                if self.batch_size == 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                print(f"  CUDA OOM -> retrying with batch size {self.batch_size}")
        return out

    def embed_query(self, text):
        return self.base.embed_query(text)


## Converts chunks to embeddings and stores them in the vector DB (Qdrant, local mode).
## One DB folder can hold many PDFs: every chunk carries metadata["source"].
class Embeding:
    # Local Qdrant allows ONE open client per folder, so clients are cached
    # here and reused for every PDF that goes into the same DB.
    _clients = {}

    def __init__(self):
        pass

    @staticmethod
    def clear_cuda():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ model
    def embed(self, model_name, device=None, batch_size=16, max_seq_length=512):
        # falls back to CPU automatically if CUDA is not available
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Embedding device: {device}")

        model_kwargs = {"device": device}
        if device.startswith("cuda"):
            # fp16 halves the weight memory (BGE-M3: ~2.2 GB -> ~1.1 GB)
            model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

        base = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={
                "batch_size": batch_size,
                "normalize_embeddings": True,
            },
        )
        # BGE-M3 defaults to 8192 tokens, which is what caused the CUDA OOM
        base._client.max_seq_length = max_seq_length

        return OOMSafeEmbeddings(base, batch_size=batch_size)

    # ---------------------------------------------------------------- Qdrant
    def get_client(self, db_path):
        if db_path not in Embeding._clients:
            os.makedirs(db_path, exist_ok=True)
            Embeding._clients[db_path] = QdrantClient(path=db_path)
        return Embeding._clients[db_path]

    @staticmethod
    def _source_filter(source):
        return models.Filter(must=[
            models.FieldCondition(key="metadata.source",
                                  match=models.MatchValue(value=source))
        ])

    def count_source(self, db_path, collection_name, source):
        """How many vectors of ONE pdf are in the collection."""
        client = self.get_client(db_path)
        if not client.collection_exists(collection_name):
            return 0
        return client.count(collection_name=collection_name,
                            count_filter=self._source_filter(source),
                            exact=True).count

    def delete_source(self, db_path, collection_name, source):
        """Remove only ONE pdf's vectors; other PDFs stay untouched."""
        client = self.get_client(db_path)
        if client.collection_exists(collection_name):
            client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(filter=self._source_filter(source)),
            )

    def vectore_store(self, embeddings, langchain_chunks, collection_name, db_path,
                      source=None, batch=256):
        """Adds chunks to the collection (creates it if missing).
        If `source` is given, that PDF's old vectors are removed first, so a
        rerun never creates duplicates and other PDFs are never touched."""
        client = self.get_client(db_path)

        if not client.collection_exists(collection_name):
            size = len(embeddings.embed_query("dimension probe"))
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=size,
                                                   distance=models.Distance.COSINE),
            )

        if source is not None:
            self.delete_source(db_path, collection_name, source)

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        for i in range(0, len(langchain_chunks), batch):
            vector_store.add_documents(langchain_chunks[i:i + batch])

        return vector_store

    def open_store(self, embeddings, collection_name, db_path):
        """Open an existing DB for searching (no building)."""
        return QdrantVectorStore(
            client=self.get_client(db_path),
            collection_name=collection_name,
            embedding=embeddings,
        )