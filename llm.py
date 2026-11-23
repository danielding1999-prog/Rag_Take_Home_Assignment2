import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from embedding import EmbeddingGenerator
from vector_store import VectorStore

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    metadata: Mapping[str, Any]


class RAGAssistant:
    """Retrieve relevant stored chunks and use them to answer with Groq."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        top_k: int = 5,
        vector_store: VectorStore | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if api_key is None:
            load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to the environment or a .env file."
            )
        if not model.strip():
            raise ValueError("model must not be empty")

        self.api_key = api_key
        self.model = model
        self.top_k = top_k
        self.vector_store = (
            vector_store if vector_store is not None else VectorStore()
        )
        self.embedding_generator = (
            embedding_generator
            if embedding_generator is not None
            else EmbeddingGenerator()
        )

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Return the closest stored chunks for a non-empty query."""
        if not query.strip():
            raise ValueError("query must not be empty")

        query_embedding = self.embedding_generator.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=self.top_k)
        documents = results.get("documents")
        metadatas = results.get("metadatas")

        if not documents or not documents[0]:
            return []

        chunks: list[RetrievedChunk] = []
        for index, content in enumerate(documents[0]):
            if not isinstance(content, str) or not content.strip():
                continue

            metadata: Mapping[str, Any] = {}
            if metadatas and metadatas[0] and index < len(metadatas[0]):
                candidate = metadatas[0][index]
                if isinstance(candidate, Mapping):
                    metadata = candidate
            chunks.append(RetrievedChunk(content=content, metadata=metadata))

        return chunks

    def answer(self, query: str) -> str:
        """Generate a grounded answer using the closest chunks as context."""
        chunks = self.retrieve(query)
        if not chunks:
            return "I couldn't find relevant information in the indexed documents."

        context_parts: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            source_details = [
                f"{key}: {chunk.metadata[key]}"
                for key in ("source", "page", "title")
                if chunk.metadata.get(key) is not None
            ]
            source = f" ({', '.join(source_details)})" if source_details else ""
            context_parts.append(
                f"[Source {index}]{source}\n{chunk.content}"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clear, helpful assistant. Answer the user's question "
                    "using only the supplied document context. Give a well-rounded "
                    "response: explain the key points, include relevant nuance, and "
                    "organize the answer when useful. Cite supporting passages with "
                    "their [Source N] labels. If the context does not support an "
                    "answer, say what is missing instead of guessing.\n\n"
                    "Document context:\n" + "\n\n".join(context_parts)
                ),
            },
            {"role": "user", "content": query},
        ]
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
            },
            timeout=(10, 90),
        )
        response.raise_for_status()

        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Groq returned an unexpected response.")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Groq returned no answer choices.")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError("Groq returned a malformed answer choice.")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Groq returned a malformed answer message.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Groq returned an empty answer.")
        return content.strip()
