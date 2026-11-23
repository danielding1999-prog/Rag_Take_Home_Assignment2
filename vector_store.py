from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
import numpy as np
from langchain_core.documents import Document


class VectorStore:
    """Store and retrieve embedded chunks in a local persistent Chroma collection."""

    def __init__(
        self,
        persist_directory: str | Path = "chroma_db",
        collection_name: str = "documents",
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
        )

    def add_documents(
        self,
        chunks: Iterable[Document | str],
        embeddings: np.ndarray,
        metadatas: Iterable[Mapping[str, Any] | None] | None = None,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        chunks = list(chunks)
        if not chunks:
            return []
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("The number of chunks must match the number of embeddings")

        texts = [
            chunk.page_content if isinstance(chunk, Document) else chunk
            for chunk in chunks
        ]

        generated_ids = list(ids) if ids is not None else [str(uuid4()) for _ in chunks]
        if len(generated_ids) != len(chunks):
            raise ValueError("The number of ids must match the number of chunks")

        metadata_list = (
            [dict(metadata) if metadata is not None else {} for metadata in metadatas]
            if metadatas is not None
            else [{} for _ in chunks]
        )
        if len(metadata_list) != len(chunks):
            raise ValueError("The number of metadata entries must match the number of chunks")

        self.collection.add(
            ids=generated_ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata_list,
        )
        return generated_ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        embedding = np.asarray(query_embedding).reshape(-1).tolist()
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=dict(where) if where is not None else None,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self.collection.count()
