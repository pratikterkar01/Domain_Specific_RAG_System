from langchain_core.documents import Document


class ParentChildChunking:

    def get_page_number(self, item):
        prov = getattr(item, "prov", None)
        if prov:
            try:
                return prov[0].page_no
            except Exception:
                pass
        return None

    def create_parent_child_chunks(self, doc):
        parents = []          # list of Document (parent)
        children = []         # list of Document (child)

        parent_id = 0
        child_id = 0

        current = None        # dict holding the in-progress parent

        def start_parent(title, level, page):
            nonlocal parent_id
            parent_id += 1
            return {
                "parent_id": f"parent_{parent_id}",
                "title": title,
                "level": level,
                "page": page,
                "items": [],   # list of (text, page, content_type)
            }

        def finish_parent(p):
            nonlocal child_id
            if p is None or not p["items"]:
                return

            full_text = "\n".join(t for t, _, _ in p["items"]).strip()
            if not full_text:
                return

            parents.append(
                Document(
                    page_content=full_text,
                    metadata={
                        "chunk_type": "parent",
                        "parent_id": p["parent_id"],
                        "title": p["title"],
                        "level": p["level"],
                        "page": p["page"],
                    },
                )
            )

            for text, page, ctype in p["items"]:
                child_id += 1
                children.append(
                    Document(
                        page_content=text,
                        metadata={
                            "chunk_type": "child",
                            "content_type": ctype,
                            "child_id": f"child_{child_id}",
                            "parent_id": p["parent_id"],
                            "parent_title": p["title"],
                            "parent_level": p["level"],
                            "page": page,
                        },
                    )
                )

        for item, level in doc.iterate_items():
            item_type = item.__class__.__name__
            page = self.get_page_number(item)

            # ---- Heading: start a new parent ----
            if item_type in ("SectionHeaderItem", "TitleItem"):
                finish_parent(current)
                title = (getattr(item, "text", "") or "").strip()
                current = start_parent(title, level, page)
                continue

            # Content before any heading -> create an intro parent
            if current is None:
                current = start_parent("Introduction", 0, page)

            # ---- Table ----
            if item_type == "TableItem":
                try:
                    table_text = item.export_to_markdown(doc)
                except Exception:
                    table_text = str(item)
                if table_text.strip():
                    current["items"].append((table_text, page, "table"))
                continue

            # ---- Normal text ----
            text = (getattr(item, "text", None) or "").strip()
            if text:
                current["items"].append((text, page, "text"))

        finish_parent(current)

        return parents, children