import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq
from pydantic import ConfigDict

from embedding import EmbeddingGenerator
from vector_store import VectorStore

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_TOP_K = 3


class ChromaRetriever(BaseRetriever):
    """Adapt the project's Chroma vector store to a LangChain retriever."""

    vector_store: VectorStore
    embedding_generator: EmbeddingGenerator
    top_k: int = DEFAULT_TOP_K

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        _ = run_manager
        query_embedding = self.embedding_generator.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=self.top_k)
        documents = results.get("documents")
        metadatas = results.get("metadatas")

        if not documents or not documents[0]:
            return []

        retrieved: list[Document] = []
        for index, content in enumerate(documents[0]):
            if not isinstance(content, str) or not content.strip():
                continue

            metadata: dict[str, Any] = {}
            if metadatas and metadatas[0] and index < len(metadatas[0]):
                candidate = metadatas[0][index]
                if isinstance(candidate, Mapping):
                    metadata = dict(candidate)
            metadata["source_id"] = index + 1
            retrieved.append(Document(page_content=content, metadata=metadata))

        return retrieved


class RAGOrchestrator:
    """Build and invoke the retrieval-augmented Groq answer chain."""

    def __init__(
        self,
        api_key: str | None = None,
        persist_directory: str | Path = "./chroma_db",
        collection_name: str = "documents",
        top_k: int = DEFAULT_TOP_K,
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
                "GROQ_API_KEY is not set. Add it to the .env file or environment."
            )

        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStore(
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
        )
        self.embedding_generator = (
            embedding_generator
            if embedding_generator is not None
            else EmbeddingGenerator()
        )
        retriever = ChromaRetriever(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
            top_k=top_k,
        )

        llm = ChatGroq(
            model=DEFAULT_MODEL,
            temperature=0.1,
            max_retries=2,
            api_key=api_key,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the question using only the supplied document context. "
                    "Give a clear, well-rounded response and cite relevant passages "
                    "using their [Source N] labels. If the context does not contain "
                    "the answer, say so instead of guessing.\n\n"
                    "Context:\n{context}",
                ),
                ("human", "{input}"),
            ]
        )
        document_prompt = PromptTemplate.from_template(
            "[Source {source_id}]\n{page_content}"
        )
        document_chain = create_stuff_documents_chain(
            llm,
            prompt,
            document_prompt=document_prompt,
        )
        self.chain = create_retrieval_chain(retriever, document_chain)

    def invoke(self, query: str) -> dict[str, Any]:
        """Return the answer and retrieved source documents for a query."""
        if not query.strip():
            raise ValueError("query must not be empty")
        return self.chain.invoke({"input": query})
