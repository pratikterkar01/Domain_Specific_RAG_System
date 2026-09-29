from langchain_core.documents import Document

class StructureAwareChunking:

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