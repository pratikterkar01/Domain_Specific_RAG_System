from pathlib import Path
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument
import json
import pickle
PARSED_DIR = Path(r"D:\M.tech\Research Area\RAG pipeline\Pdf_Doc_Parsing")


class Ingestion:

    @staticmethod
    def parse_pdf(pdf_path: str) -> DoclingDocument:
        """Parse PDF using Docling."""
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document

    @staticmethod
    def parse_pdf(pdf_path: str) -> DoclingDocument:
        converter = DocumentConverter()
        return converter.convert(pdf_path).document

    ## save in the text format
    @staticmethod
    def save_parsed_document(pdf_path: str, parsed_dir: Path = PARSED_DIR) -> Path:
        parsed_dir = Path(parsed_dir)
        parsed_dir.mkdir(parents=True, exist_ok=True)
        txt_path = parsed_dir / f"{Path(pdf_path).stem}.txt"

        if txt_path.exists():
            print(f"Already parsed, skipping: {txt_path.name}")
            return txt_path

        doc = Ingestion.parse_pdf(pdf_path)
        txt_path.write_text(doc.model_dump_json(), encoding="utf-8")
        print(f"Saved parsed document: {txt_path}")
        return txt_path

    @staticmethod
    def load_parsed_document(txt_path: Path) -> DoclingDocument:
        text = Path(txt_path).read_text(encoding="utf-8")
        return DoclingDocument.model_validate_json(text)