from typing import Any

import streamlit as st
from groq import APIError

from orchestrator import RAGOrchestrator


@st.cache_resource
def get_orchestrator() -> RAGOrchestrator:
    return RAGOrchestrator()


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    with st.expander("Retrieved source chunks"):
        for source in sources:
            label = f"Source {source['source_id']}"
            details = [str(source["source"])]
            if source["page"] is not None:
                details.append(f"page {source['page']}")
            st.markdown(f"**{label}** — {', '.join(details)}")
            st.text(source["content"])


def main() -> None:
    st.set_page_config(page_title="Requirement Assistant")
    st.title("Requirement Assistant")
    st.caption("Ask a question about the indexed requirements.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    query = st.chat_input("Enter your question")
    if not query or not query.strip():
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the requirements and preparing an answer..."):
                result = get_orchestrator().invoke(query.strip())
        except (APIError, RuntimeError) as error:
            st.error(f"Unable to answer the query: {error}")
            return

        answer = result["answer"]
        source_documents = result.get("context", [])
        sources = [
            {
                "source_id": document.metadata.get("source_id", index),
                "source": document.metadata.get("source", "Unknown source"),
                "page": document.metadata.get("page"),
                "content": document.page_content,
            }
            for index, document in enumerate(source_documents, start=1)
        ]

        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
