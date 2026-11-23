from collections.abc import Iterable

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Generate embeddings for chunked text with all-MiniLM-L6-v2."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(
        self,
        chunks: Iterable[Document | str],
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        texts = [
            chunk.page_content if isinstance(chunk, Document) else chunk
            for chunk in chunks
        ]
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))

        return self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
        )

    def embed_query(
        self,
        query: str,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        if not query.strip():
            raise ValueError("query must not be empty")

        return self.model.encode(
            query,
            normalize_embeddings=normalize_embeddings,
        )
