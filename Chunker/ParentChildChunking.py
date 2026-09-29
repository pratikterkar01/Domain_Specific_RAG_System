from langchain_core.documents import Document

class ParentChildChunking:
    def __init__(self):
        pass
    def get_page_number(self,item):
        prov = getattr(item, "prov", None)

        if prov:
            try:
                return prov[0].page_no
            except Exception:
                pass

        return None

    def create_parent_child_chunks(self,doc):

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