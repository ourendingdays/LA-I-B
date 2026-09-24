# Data Science Libraries
from openai import OpenAI

# Standard Libraries
from dotenv import load_dotenv
import os
import time

load_dotenv()

def spell_correct(query: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{
            "role": "user",
            "content": f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.
                User query: "{query}"
            """
        }],
    )
    return response.choices[0].message.content.strip()

def rewrite_query(query: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{
            "role": "user",
            "content": f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
        }],
    )
    return response.choices[0].message.content.strip()

def expand_query(query: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{
            "role": "user",
            "content": f"""Expand the user-provided movie search query below with related terms.

            Add synonyms and related concepts that might appear in movie descriptions.
            Keep expansions relevant and focused.
            Output only the additional terms; they will be appended to the original query.

            Examples:
            - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
            - "action movie with bear" -> "action thriller bear chase fight adventure"
            - "comedy with bear" -> "comedy funny bear humor lighthearted"

            User query: "{query}"
        """
        }],
    )
    return response.choices[0].message.content.strip()

def rerank_individual(query: str, results: list, limit: int) -> list:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    print(f"Re-ranking top {len(results)} results using individual method...")

    for _, data in results:
        doc = data["doc"]
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[{
                "role": "user",
                "content": f"""Rate how well this movie matches the search query.

                Query: "{query}"
                Movie: {doc.get("title", "")} - {doc.get("description", "")[:200]}

                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Output ONLY the number in your response, no other text or explanation.

                Score:"""
            }],
        )
        try:
            score = float(response.choices[0].message.content.strip())
        except ValueError:
            score = 0.0
        data["rerank_score"] = score
        time.sleep(3)

    results.sort(key=lambda x: x[1]["rerank_score"], reverse=True)
    return results[:limit]