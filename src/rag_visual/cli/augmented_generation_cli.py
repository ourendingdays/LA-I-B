# Custom Modules
from src.rag_visual.lib.inverted_index import load_movies
from src.rag_visual.lib.hybrid_search import HybridSearch

# Data Science Libraries
from openai import OpenAI
from dotenv import load_dotenv

# Standard Libraries
import argparse
import os


load_dotenv()

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize search results")
    summarize_parser.add_argument("query", type=str, help="Search query")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            # Searching
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, k=60, limit=5)

            # Formatting docs for the prompt
            docs = ""
            titles = []
            for _, data in results:
                doc = data["doc"]
                titles.append(doc["title"])
                docs += f"{doc['title']} - {doc.get('description', '')[:200]}\n"

            # Generating answer
            api_key = os.environ.get("OPENROUTER_API_KEY")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
                Your task is to provide a natural-language answer to the user's query based on documents retrieved from our movie database.
                Provide a comprehensive answer that addresses the user's query.

                Query: {query}

                Documents:
                {docs}

                Answer:"""

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )

            answer = response.choices[0].message.content.strip()

            # Results
            print("Search Results:")
            for title in titles:
                print(f"- {title}")
            print(f"\nRAG Response:\n{answer}")

        case "summarize":
            query = args.query

            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, k=60, limit=args.limit)

            formatted_results = ""
            titles = []
            for _, data in results:
                doc = data["doc"]
                titles.append(doc["title"])
                formatted_results += f"{doc['title']} - {doc.get('description', '')[:200]}\n"

            api_key = os.environ.get("OPENROUTER_API_KEY")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results.

        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information.

        This should be tailored to Webflyx users. Webflyx is a movie streaming service.

        Query: {query}

        Search results:
        {formatted_results}

        Provide a comprehensive 3-4 sentence answer that combines information from multiple results."""

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )

            summary = response.choices[0].message.content.strip()

            print("Search Results:")
            for title in titles:
                print(f"  - {title}")
            
            print(f"\nLLM Summary:\n{summary}")


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()