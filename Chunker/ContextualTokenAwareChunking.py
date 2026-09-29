from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
import re



TARGET_TOKENS = 400
MAX_TOKENS = 600
OVERLAP_TOKENS = 60

class ContextualTokenAwareChunking:
    # ------------------------ Context + token aware chunking -----------------------------------
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

    def tokenize(self,text):
        return re.findall(
            r"\w+|[^\w\s]",
            text,
            re.UNICODE
        )

    def detokenize(self,tokens):
        text = " ".join(tokens)
        # Fix spaces before punctuation
        text = re.sub(
            r"\s+([.,!?;:%)\]])",
            r"\1",
            text
        )
        # Fix spaces after opening brackets
        text = re.sub(
            r"([\(\[]) +",
            r"\1",
            text
        )
        return text

    def token_aware_split(self,text,target_tokens=TARGET_TOKENS,max_tokens=MAX_TOKENS,overlap_tokens=OVERLAP_TOKENS):

        tokens = self.tokenize(text)
        if len(tokens) <= max_tokens:
            return [text]
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(
                start + target_tokens,
                len(tokens)
            )
            chunk_tokens = tokens[start:end]
            chunk = self.detokenize(chunk_tokens)
            chunks.append(chunk)
            if end >= len(tokens):
                break
            # Move backward to create overlap
            start = end - overlap_tokens
        return chunks

    def create_context(self,document_name,parent_title,parent_level,page,content_type):
        context_parts = []
        if document_name:
            context_parts.append(f"Document: {document_name}")
        if parent_title:
            context_parts.append(f"Section: {parent_title}")
        if parent_level is not None:
            context_parts.append(f"Section Level: {parent_level}")
        if page is not None:
            context_parts.append(f"Page: {page}")
        if content_type:
            context_parts.append(f"Content Type: {content_type}")
        return "\n".join(context_parts)

    def contextual_token_aware_chunking(self,doc,document_name ):
        chunks = []
        current_parent = None
        current_parent_content = []
        parent_counter = 0
        child_counter = 0
        # --------------------------------------------------------
        # Save current parent
        # --------------------------------------------------------

        def save_parent():
            nonlocal current_parent
            nonlocal current_parent_content
            if current_parent is None:
                return
            current_parent["content"] = (
                current_parent_content.copy()
            )
            process_parent(
                current_parent
            )
            current_parent = None
            current_parent_content = []
        # --------------------------------------------------------
        # Process one parent section
        # --------------------------------------------------------
        def count_tokens(self,text):
            return len(self.tokenize(text))

        def process_parent(self,parent):
            nonlocal child_counter
            for element in parent["content"]:
                content_type = element["type"]
                page = element["page"]
                content = element["content"]
                # ==============================================
                # TEXT
                # ==============================================
                if content_type == "text":
                    text_chunks = self.token_aware_split(
                        content
                    )
                    for text_chunk in text_chunks:
                        child_counter += 1
                        context = self.create_context(document_name=document_name,parent_title=parent["title"],parent_level=parent["level"],page=page,content_type="text" )
                        contextual_content = (context + "\n\n" + text_chunk)
                        chunks.append(
                            Document(
                                page_content=contextual_content,
                                metadata={
                                    "chunk_type": "child",
                                    "content_type": "text",
                                    "chunk_id":f"child_{child_counter}",
                                    "parent_id":parent["parent_id"],
                                    "parent_title":parent["title"],
                                    "parent_level":parent["level"],
                                    "page":page,
                                    "document":document_name,
                                    "token_count":count_tokens(contextual_content)
                                }

                            )

                        )

                # ==============================================
                # TABLE
                # ==============================================
                elif content_type == "table":
                    child_counter += 1
                    context = self.create_context(
                        document_name=document_name,
                        parent_title=parent["title"],
                        parent_level=parent["level"],
                        page=page,
                        content_type="table"
                    )
                    contextual_content = (
                            context
                            + "\n\n"
                            + content
                    )
                    chunks.append(
                        Document(
                            page_content=contextual_content,
                            metadata={"chunk_type": "child",
                                "content_type": "table",
                                "chunk_id": f"child_{child_counter}",
                                "parent_id":parent["parent_id"],
                                "parent_title":parent["title"],
                                "parent_level":parent["level"],
                                "page":page,
                                "document":document_name,
                                "token_count":count_tokens(contextual_content)
                            }
                        )
                    )

        # --------------------------------------------------------
        # Iterate Docling items
        # --------------------------------------------------------

        for item, level in doc.iterate_items():
            item_type = item.__class__.__name__
            page = self.get_page_number(item)
            # ====================================================
            # HEADING
            # ====================================================
            if item_type in [
                "SectionHeaderItem",
                "TitleItem"
            ]:
                heading = getattr(item,"text","")
                if not heading:
                    continue
                heading = heading.strip()
                save_parent()
                parent_counter += 1
                current_parent = {
                    "parent_id":f"parent_{parent_counter}",
                    "title": heading,
                    "level":  level,
                    "page":   page,
                    "content":  []
                }
                continue

            # ====================================================
            # TABLE
            # ====================================================
            if item_type == "TableItem":
                if current_parent is None:
                    parent_counter += 1
                    current_parent = {"parent_id": f"parent_{parent_counter}",
                        "title":"Document Content",
                        "level": 0,
                        "page":  page,
                        "content": []
                    }

                try:
                    table_markdown = (
                        item.export_to_markdown(doc)
                    )
                except Exception:
                    table_markdown = str(item)
                current_parent_content.append({
                    "type": "table",
                    "content": table_markdown,
                    "page": page

                })
                continue
            # ====================================================
            # TEXT
            # ====================================================
            text = getattr(item,"text",None)
            if text:
                text = text.strip()
                if text:
                    if current_parent is None:
                        parent_counter += 1
                        current_parent = {"parent_id":f"parent_{parent_counter}",
                            "title":"Document Introduction",
                            "level":0,
                            "page": page,
                            "content": []
                        }

                    current_parent_content.append({
                        "type": "text",
                        "content": text,
                        "page":  page
                    })
        # Save final parent
        save_parent()

        return chunks