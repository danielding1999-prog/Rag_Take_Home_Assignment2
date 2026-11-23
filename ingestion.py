from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PDFExtractor:
    """Extract PDF documents without chunking or embedding them."""

    def load_pdf(self, pdf_path: str | Path) -> list[Document]:
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file: {path}")

        return PyPDFLoader(str(path)).load()

    def load_directory(self, pdf_directory: str | Path) -> list[Document]:
        directory = Path(pdf_directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"PDF directory not found: {directory}")

        pdf_files = sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".pdf"
            ),
            key=lambda path: path.name.lower(),
        )

        documents: list[Document] = []
        for pdf_file in pdf_files:
            documents.extend(self.load_pdf(pdf_file))
        return documents

    def load_paths(self, pdf_paths: Iterable[str | Path]) -> list[Document]:
        documents: list[Document] = []
        for pdf_path in pdf_paths:
            path = Path(pdf_path)
            if path.is_dir():
                documents.extend(self.load_directory(path))
            else:
                documents.extend(self.load_pdf(path))
        return documents
