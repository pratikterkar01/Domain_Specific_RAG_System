# -------------------------  Chunking part of the RAG (Structure Aware chunking) -------------------------------------
from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document


PDF_PATH = "D:\\AiEnergyAssitanceForBuilding\\PdfResourc\\Used\\ECBC_HVAC_TIP_SHEET_2011.pdf"


def parse_pdf(pdf_path: str):
    """Parse PDF using Docling."""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document


def structure_aware_chunking(doc):
    """
    Create structure-aware chunks from a DoclingDocument.

    The chunking logic:
    - Keeps headings associated with their content.
    - Preserves document/page metadata where available.
    - Treats tables separately.
    - Creates a new chunk when a new heading/section is encountered.
    """

    chunks = []

    current_heading = []
    current_content = []
    current_page = None

    def save_chunk():
        if not current_content:
            return

        content = "\n".join(current_content).strip()

        if not content:
            return

        metadata = {
            "heading": " > ".join(current_heading),
            "page": current_page,
            "content_type": "text"
        }

        chunks.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

    # Iterate through Docling document items
    for item, level in doc.iterate_items():

        item_type = item.__class__.__name__

        # -------------------------
        # Heading
        # -------------------------
        if item_type in ["SectionHeaderItem", "TitleItem"]:

            # Save previous section
            save_chunk()

            current_content.clear()

            text = getattr(item, "text", "").strip()

            if text:
                # Maintain heading hierarchy
                if level is not None:

                    current_heading = current_heading[:level]

                    current_heading.append(text)

                else:
                    current_heading.append(text)

            continue

        # -------------------------
        # Table
        # -------------------------
        if item_type == "TableItem":

            # Save text before table
            save_chunk()
            current_content.clear()

            try:
                table_text = item.export_to_markdown(doc)

            except Exception:
                table_text = str(item)

            chunks.append(
                Document(
                    page_content=table_text,
                    metadata={
                        "heading": " > ".join(current_heading),
                        "page": current_page,
                        "content_type": "table"
                    }
                )
            )

            continue

        # -------------------------
        # Normal text
        # -------------------------
        text = getattr(item, "text", None)

        if text:

            text = text.strip()

            if text:
                current_content.append(text)

        # -------------------------
        # Page information
        # -------------------------
        prov = getattr(item, "prov", None)

        if prov:
            try:
                current_page = prov[0].page_no
            except Exception:
                pass

    # Save final chunk
    save_chunk()

    return chunks

def create_parent_child_chunks(doc):

    parents = []
    current_parent = None
    parent_content = []

    parent_id = 0
    child_id = 0

    for item, level in doc.iterate_items():

        item_type = item.__class__.__name__

        text = getattr(item, "text", None)

        if text:
            text = text.strip()

        # -------------------------------------------------
        # New heading
        # -------------------------------------------------

        if item_type in ["SectionHeaderItem", "TitleItem"]:

            # Save previous parent
            if current_parent is not None:

                current_parent["content"] = "\n".join(
                    parent_content
                ).strip()

                parents.append(current_parent)

            # Create new parent
            parent_id += 1

            current_parent = {
                "parent_id": f"parent_{parent_id}",
                "title": text,
                "level": level,
                "content": ""
            }

            parent_content = []

            continue

        # -------------------------------------------------
        # Normal text
        # -------------------------------------------------

        if text:

            parent_content.append(text)

    # -----------------------------------------------------
    # Save final parent
    # -----------------------------------------------------

    if current_parent is not None:

        current_parent["content"] = "\n".join(
            parent_content
        ).strip()

        parents.append(current_parent)


    # -----------------------------------------------------
    # Create child chunks
    # -----------------------------------------------------

    child_documents = []

    for parent in parents:

        content = parent["content"]

        # Simple paragraph-based child splitting
        paragraphs = [
            p.strip()
            for p in content.split("\n")
            if p.strip()
        ]

        for paragraph in paragraphs:

            child_id += 1

            child_documents.append(
                Document(
                    page_content=paragraph,
                    metadata={
                        "chunk_type": "child",
                        "child_id": f"child_{child_id}",
                        "parent_id": parent["parent_id"],
                        "parent_title": parent["title"],
                        "parent_level": parent["level"]
                    }
                )
            )

    return parents, child_documents



def main():

    doc = parse_pdf(PDF_PATH)

    chunks = structure_aware_chunking(doc)

    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:10]):

        print("\n" + "=" * 80)
        print(f"CHUNK {i + 1}")

        print("\nMetadata:")
        print(chunk.metadata)

        print("\nContent:")
        print(chunk.page_content[:1000])
    return chunks



result  =  main()
print(len(result[0]))