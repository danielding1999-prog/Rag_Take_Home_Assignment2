import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from chunking import RecursiveSplitter
from embedding import EmbeddingGenerator
from ingestion import PDFExtractor
from vector_store import VectorStore

PROJECT_DIRECTORY = Path(__file__).resolve().parent


def ingest_pdf_source(source: Path) -> int:
    """Extract, chunk, embed, and store PDFs from a file or directory."""
    source = source.expanduser().resolve()
    documents = PDFExtractor().load_paths([source])
    if not documents:
        raise ValueError(f"No PDF documents were found at: {source}")

    chunks = RecursiveSplitter().split(documents)
    if not chunks:
        raise ValueError(f"No text chunks were created from: {source}")

    embeddings = EmbeddingGenerator().embed_documents(chunks)
    ids = [
        hashlib.sha256(
            (
                f"{chunk.metadata.get('source', '')}:"
                f"{chunk.metadata.get('page', '')}:"
                f"{index}:{chunk.page_content}"
            ).encode("utf-8")
        ).hexdigest()
        for index, chunk in enumerate(chunks)
    ]
    vector_store = VectorStore(
        persist_directory=PROJECT_DIRECTORY / "chroma_db",
    )
    vector_store.add_documents(
        chunks,
        embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
        ids=ids,
    )
    return len(chunks)


def launch_frontend() -> None:
    """Start the Streamlit client from the project directory."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_DIRECTORY / "frontend.py"),
        ],
        cwd=PROJECT_DIRECTORY,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index PDFs and run the requirement assistant."
    )
    parser.add_argument(
        "--ingest",
        type=Path,
        metavar="PDF_OR_DIRECTORY",
        default=PROJECT_DIRECTORY / "data",
        help=(
            "PDF file or directory of PDFs to add before launch "
            "(default: the project's data directory)."
        ),
    )
    args = parser.parse_args()

    chunk_count = ingest_pdf_source(args.ingest)
    print(f"Indexed {chunk_count} chunks from {args.ingest}.")

    launch_frontend()


if __name__ == "__main__":
    main()
