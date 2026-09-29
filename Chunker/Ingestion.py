from docling.document_converter import DocumentConverter

class Ingestion:
    def parse_pdf(pdf_path: str):
        """Parse PDF using Docling."""
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document