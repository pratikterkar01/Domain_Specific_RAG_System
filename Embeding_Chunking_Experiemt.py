import os
import glob
import torch
import hashlib
import re
import json
from Utils import Config
from Chunker import Ingestion,StructureAwareChunking,ParentChildChunking,ContextualTokenAwareChunking
from Embeding.Embeding import Embeding
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

global_Variable = Config.get_settings()

dbRoot = global_Variable.DB_ROOT
PARENTS_FILE = "parents.json"
MANIFEST_FILE = "ingest_manifest.json"


# ============================== small helpers ===============================
def doc_slug(document_name):
    base = os.path.splitext(os.path.basename(str(document_name)))[0]
    base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")[:25] or "doc"
    return f"{base}_{hashlib.md5(str(document_name).encode('utf-8')).hexdigest()[:6]}"


def chunks_fingerprint(chunks):
    h = hashlib.md5()
    for c in chunks:
        h.update(c.page_content.encode("utf-8"))
    return h.hexdigest()


def load_manifest(db_path):
    try:
        with open(os.path.join(db_path, MANIFEST_FILE), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_manifest(db_path, manifest):
    os.makedirs(db_path, exist_ok=True)
    tmp = os.path.join(db_path, MANIFEST_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(tmp, os.path.join(db_path, MANIFEST_FILE))


def save_parents(db_path, parents, source):
    """Merge this PDF's parents into parents.json (other PDFs are kept)."""
    path = os.path.join(db_path, PARENTS_FILE)
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data = {pid: d for pid, d in data.items() if d["metadata"].get("source") != source}
    for p in parents:
        data[p.metadata["parent_id"]] = {"page_content": p.page_content,
                                         "metadata": p.metadata}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_parents(db_path):
    with open(os.path.join(db_path, PARENTS_FILE), encoding="utf-8") as f:
        data = json.load(f)
    return {pid: Document(**d) for pid, d in data.items()}


def parents_present(db_path, source):
    try:
        with open(os.path.join(db_path, PARENTS_FILE), encoding="utf-8") as f:
            return any(d["metadata"].get("source") == source for d in json.load(f).values())
    except (FileNotFoundError, json.JSONDecodeError):
        return False


# ============================ the main function =============================
def execute_all_combinations(documen_parse, document_name):
    embedding_service = Embeding()
    db_root = global_Variable.DB_ROOT
    slug = doc_slug(document_name)
    print(f"\n##### Document: {document_name}")

    parent_cache = {}

    def build_hierarchical():
        parents, children = (
            ParentChildChunking.ParentChildChunking()
            .create_parent_child_chunks(documen_parse)
        )
        # parent_1, child_1 ... repeat in every PDF -> make ids unique per PDF
        for p in parents:
            p.metadata["parent_id"] = f"{slug}::{p.metadata['parent_id']}"
            p.metadata["source"] = document_name
        for c in children:
            c.metadata["parent_id"] = f"{slug}::{c.metadata['parent_id']}"
            c.metadata["child_id"] = f"{slug}::{c.metadata['child_id']}"
        parent_cache["HierarchicalChunking"] = parents
        return children  # embed ONLY children

    chunk_factories = {
        "ContextualTokenAwareChunking": lambda:
        ContextualTokenAwareChunking.ContextualTokenAwareChunking()
        .contextual_token_aware_chunking(documen_parse, document_name),
        "HierarchicalChunking": build_hierarchical,
        "StructureAwareChunking": lambda:
        StructureAwareChunking.StructureAwareChunking()
        .structure_aware_chunking(documen_parse),
    }
    chunk_cache = {}

    def get_chunks(name):
        if name not in chunk_cache:
            print(f"Generating chunks: {name}")
            chunks = chunk_factories[name]()
            for c in chunks:
                c.metadata["source"] = document_name
            chunk_cache[name] = chunks
        return chunk_cache[name]

    embedding_models = {
        "BGE-M3": "BAAI/bge-m3",
        "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
        "multilingual-E5-base": "intfloat/multilingual-e5-base",
    }

    results = {"done": [], "skipped": [], "failed": {}}
    vector_stores = {}

    for embedding_folder, model_name in embedding_models.items():
        embeddings = None  # load the model only if something needs building

        for chunking_name in chunk_factories:
            combination_name = f"{embedding_folder}_{chunking_name}"
            collection_name = combination_name.replace("-", "_")
            db_path = os.path.join(db_root, f"{embedding_folder}_Embedding", chunking_name)
            is_hier = chunking_name == "HierarchicalChunking"

            try:
                chunks = get_chunks(chunking_name)
                fingerprint = chunks_fingerprint(chunks)

                # ---- resume check: only for THIS pdf inside this DB ----
                entry = load_manifest(db_path).get(document_name)
                if (
                        entry
                        and entry.get("fingerprint") == fingerprint
                        and entry.get("count") == len(chunks)
                        and embedding_service.count_source(
                    db_path, collection_name, document_name) == len(chunks)
                        and (not is_hier or parents_present(db_path, document_name))
                ):
                    print(f"Skipping (already complete): {combination_name}")
                    results["skipped"].append(combination_name)
                    continue

                if embeddings is None:
                    print(f"\nLoading embedding: {embedding_folder}")
                    embeddings = embedding_service.embed(model_name=model_name)

                print(f"\nCreating: {combination_name}  ({len(chunks)} chunks)")
                print(f"Database path: {db_path}")

                # forget the old manifest entry of THIS pdf; vectore_store()
                # removes its old vectors and adds the new ones
                manifest = load_manifest(db_path)
                manifest.pop(document_name, None)
                save_manifest(db_path, manifest)

                vector_store = embedding_service.vectore_store(
                    embeddings=embeddings,
                    langchain_chunks=chunks,
                    collection_name=collection_name,
                    db_path=db_path,
                    source=document_name,
                )

                if is_hier:
                    save_parents(db_path, parent_cache["HierarchicalChunking"], document_name)

                # manifest entry LAST = "this pdf is fully in the DB"
                manifest = load_manifest(db_path)
                manifest[document_name] = {"fingerprint": fingerprint, "count": len(chunks)}
                save_manifest(db_path, manifest)

                vector_stores[combination_name] = vector_store
                results["done"].append(combination_name)
                print(f"Completed: {combination_name}")

            except Exception as e:
                print(f"FAILED: {combination_name} -> {type(e).__name__}: {e}")
                results["failed"][combination_name] = str(e)
                try:  # remove only this pdf's partial vectors, keep other PDFs
                    embedding_service.delete_source(db_path, collection_name, document_name)
                except Exception:
                    pass
                embedding_service.clear_cuda()
                continue

        embeddings = None
        embedding_service.clear_cuda()

    print(f"\n=== Summary for {document_name} ===")
    print(f"Built:   {len(results['done'])}")
    print(f"Skipped: {len(results['skipped'])}")
    print(f"Failed:  {len(results['failed'])}")
    for name, err in results["failed"].items():
        print(f"  - {name}: {err}")
    if results["failed"]:
        print("Rerun the script to retry only the failed combinations.")

    return vector_stores, results



def collect_pdfs(source):
    """Accept a folder, a single PDF path, or a list of PDF paths."""
    if isinstance(source, (list, tuple)):
        pdfs = list(source)
    elif os.path.isdir(source):
        pdfs = sorted(glob.glob(os.path.join(source, "*.pdf")))
    else:
        pdfs = [source]
    missing = [p for p in pdfs if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"PDF not found: {missing}")
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {source}")
    return pdfs

def run_experiment():
    txt_files = sorted(Ingestion.PARSED_DIR.glob("*.txt"))

    for path in txt_files:
        documen_parse = Ingestion.Ingestion.load_parsed_document(path)
        document_name = f"{path.stem}.pdf"  # same name as the original PDF

        execute_all_combinations(documen_parse, document_name)


def parse_and_save_all_pdfs():
    pdf_Names = collect_pdfs("D:\\AiEnergyAssitanceForBuilding\\PdfResourc\\Used")
    #Ingestion.Ingestion.save_parsed_document(pdf_Names[0])
    for pdf in pdf_Names[1:]:
        Ingestion.Ingestion.save_parsed_document(pdf)  # skips if JSON already exists

    print(f"Done. Parsed {len(pdf_Names)} PDFs.")


if __name__ == "__main__":
    #parse_and_save_all_pdfs()
    pdf_names = run_experiment()
    #run_experiment()


