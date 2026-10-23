import os
from Utils import Config
from Chunker import Ingestion,StructureAwareChunking,ParentChildChunking,ContextualTokenAwareChunking
from Embeding.Embeding import Embeding
import glob
import gc
import shutil
import torch
from qdrant_client import QdrantClient
import json
from langchain_core.documents import Document
global_Variable = Config.get_settings()

dbRoot = global_Variable.DB_ROOT
PARENTS_FILE = "parents.json"

def save_parents(db_path, parents):
    os.makedirs(db_path, exist_ok=True)
    data = {
        p.metadata["parent_id"]: {
            "page_content": p.page_content,
            "metadata": p.metadata,
        }
        for p in parents
    }
    with open(os.path.join(db_path, PARENTS_FILE), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_parents(db_path):
    with open(os.path.join(db_path, PARENTS_FILE), encoding="utf-8") as f:
        data = json.load(f)
    return {pid: Document(**d) for pid, d in data.items()}
def execute_all_combinations(documen_parse, document_name):
    embedding_service = Embeding()
    db_root = global_Variable.DB_ROOT

    parent_cache = {}   # NEW: parents for hierarchical chunking

    def build_hierarchical():                      # NEW
        parents, children = (
            ParentChildChunking.ParentChildChunking()
            .create_parent_child_chunks(documen_parse)
        )
        parent_cache["HierarchicalChunking"] = parents
        return children                            # embed ONLY children

    chunk_factories = {
        "ContextualTokenAwareChunking": lambda:
            ContextualTokenAwareChunking.ContextualTokenAwareChunking()
            .contextual_token_aware_chunking(documen_parse, document_name),
        "HierarchicalChunking": build_hierarchical,   # CHANGED
        "StructureAwareChunking": lambda:
            StructureAwareChunking.StructureAwareChunking()
            .structure_aware_chunking(documen_parse),
    }
    chunk_cache = {}

    def get_chunks(name):
        if name not in chunk_cache:
            print(f"Generating chunks: {name}")
            chunk_cache[name] = chunk_factories[name]()
        return chunk_cache[name]

    embedding_models = {
        "BGE-M3": "BAAI/bge-m3",
        "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
        "multilingual-E5-base": "intfloat/multilingual-e5-base",
    }

    results = {"done": [], "skipped": [], "failed": {}}
    vector_stores = {}

    for embedding_folder, model_name in embedding_models.items():
        embeddings = None

        for chunking_name in chunk_factories:
            combination_name = f"{embedding_folder}_{chunking_name}"
            collection_name = combination_name.replace("-", "_")
            db_path = os.path.join(
                db_root, f"{embedding_folder}_Embedding", chunking_name
            )
            is_hier = chunking_name == "HierarchicalChunking"   # NEW

            try:
                chunks = get_chunks(chunking_name)   # now always a flat list of Documents

                # Resume check: hierarchical also needs its parents file   # CHANGED
                parents_ok = (not is_hier) or os.path.exists(
                    os.path.join(db_path, PARENTS_FILE)
                )
                if parents_ok and is_combination_complete(
                    db_path, collection_name, len(chunks)
                ):
                    print(f"Skipping (already complete): {combination_name}")
                    results["skipped"].append(combination_name)
                    continue

                reset_combination(db_path)

                if embeddings is None:
                    print(f"\nLoading embedding: {embedding_folder}")
                    embeddings = embedding_service.embed(model_name=model_name)

                print(f"\nCreating: {combination_name}")
                print(f"Database path: {db_path}")

                vector_store = embedding_service.vectore_store(
                    embeddings=embeddings,
                    langchain_chunks=chunks,
                    collection_name=collection_name,
                    db_path=db_path,
                )

                # NEW: save parents BEFORE the success marker
                if is_hier:
                    save_parents(db_path, parent_cache["HierarchicalChunking"])

                with open(os.path.join(db_path, MARKER), "w") as f:
                    f.write("ok")

                vector_stores[combination_name] = vector_store
                results["done"].append(combination_name)
                print(f"Completed: {combination_name}")

            except Exception as e:
                print(f"FAILED: {combination_name} -> {type(e).__name__}: {e}")
                results["failed"][combination_name] = str(e)
                reset_combination(db_path)
                free_gpu()
                continue

        if embeddings is not None:      # CHANGED: avoid del on None edge cases
            del embeddings
        free_gpu()

    print("\n=== Summary ===")
    print(f"Built:   {len(results['done'])}")
    print(f"Skipped: {len(results['skipped'])}")
    print(f"Failed:  {len(results['failed'])}")
    for name, err in results["failed"].items():
        print(f"  - {name}: {err}")
    if results["failed"]:
        print("Rerun the script to retry only the failed combinations.")

    return vector_stores, results


MARKER = "_COMPLETE"


def is_combination_complete(db_path, collection_name, expected_count):
    """True only if the DB folder, the collection, and all points exist."""
    if not os.path.isdir(db_path):
        return False

    # Fast path: marker written after a fully successful build
    if os.path.exists(os.path.join(db_path, MARKER)):
        return True

    # Fallback: verify inside Qdrant (covers DBs built before the marker existed)
    client = None
    try:
        client = QdrantClient(path=db_path)
        if not client.collection_exists(collection_name):
            return False
        count = client.count(collection_name, exact=True).count
        return count >= expected_count
    except Exception:
        return False
    finally:
        if client is not None:
            client.close()   # local Qdrant locks the folder; always release


def reset_combination(db_path):
    """Remove a partial/corrupt DB so it can be rebuilt cleanly."""
    if os.path.isdir(db_path):
        print(f"  Removing incomplete DB: {db_path}")
        shutil.rmtree(db_path, ignore_errors=True)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    Ingestion.Ingestion.save_parsed_document(pdf_Names[0])
    '''for pdf in pdf_Names[:2]:
        Ingestion.Ingestion.save_parsed_document(pdf) ''' # skips if JSON already exists

    print(f"Done. Parsed {len(pdf_Names)} PDFs.")

def run_experiment():
    txt_files = sorted(Ingestion.PARSED_DIR.glob("*.txt"))
    documen_parse = Ingestion.Ingestion.load_parsed_document(txt_files[0])
    document_name = f"{txt_files[0].stem}.pdf"
    execute_all_combinations(documen_parse, document_name)

    '''for json_path in json_files:
        documen_parse = Ingestion.Ingestion.load_parsed_document(json_path)
        document_name = f"{json_path.stem}.pdf"   # same name as the original PDF

        execute_all_combinations(documen_parse, document_name)'''
if __name__ == "__main__":
    #parse_and_save_all_pdfs()
    #pdf_names = run_experiment()
    run_experiment()


