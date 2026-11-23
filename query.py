def get_user_query(prompt: str = "Enter your question: ") -> str:
    """Prompt for a non-empty query from the terminal."""
    while True:
        query = input(prompt).strip()
        if query:
            return query
        print("Please enter a question.")


def request_query_once() -> str:
    """Request one user query and return it."""
    return get_user_query()


def request_queries() -> list[str]:
    """Prompt the user for queries until they enter 'quit' or 'exit'."""
    queries: list[str] = []

    while True:
        query = get_user_query()
        if query.lower() in {"quit", "exit"}:
            break
        queries.append(query)

    return queries


if __name__ == "__main__":
    print(request_query_once())
