# Data Science Libraries
from openai import OpenAI
from sentence_transformers import CrossEncoder

# Standard Libraries
from dotenv import load_dotenv
import json
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

def rerank_batch(query: str, results: list, limit: int) -> list:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    print(f"Re-ranking top {len(results)} results using batch method...")

    # build the movie list string
    doc_list_str = ""
    doc_map = {}
    for doc_id, data in results:
        doc = data["doc"]
        doc_list_str += f"ID: {doc['id']} - {doc.get('title', '')} - {doc.get('description', '')[:200]}\n"
        doc_map[doc["id"]] = (doc_id, data)

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{
            "role": "user",
            "content": f"""Rank the movies listed below by relevance to the following search query.

            Query: "{query}"

            Movies:
            {doc_list_str}

            Return the movie IDs in order of relevance, best match first.

            Your response must be a raw JSON array of integers.
            Do not wrap the JSON in Markdown. Do not use a ```json code block.
            Do not include any explanatory text.

            For example:
            [75, 12, 34, 2, 1]

            Ranking:"""
        }],
    )

    try:
        ranked_ids = json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        return results[:limit]

    # rebuild results in ranked order
    ranked_results = []
    for rank, movie_id in enumerate(ranked_ids, 1):
        if movie_id in doc_map:
            doc_id, data = doc_map[movie_id]
            data["rerank_rank"] = rank
            ranked_results.append((doc_id, data))

    # add any results not in the LLM response at the end
    ranked_movie_ids = set(ranked_ids)
    for doc_id, data in results:
        if data["doc"]["id"] not in ranked_movie_ids:
            data["rerank_rank"] = len(ranked_results) + 1
            ranked_results.append((doc_id, data))

    return ranked_results[:limit]

def rerank_cross_encoder(query: str, results: list, limit: int) -> list:
    print(f"Re-ranking top {len(results)} results using cross_encoder method...")

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu")

    pairs = []
    for _, data in results:
        doc = data["doc"]
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

    scores = cross_encoder.predict(pairs)

    for i, (_, data) in enumerate(results):
        data["cross_encoder_score"] = float(scores[i])

    results.sort(key=lambda x: x[1]["cross_encoder_score"], reverse=True)
    return results[:limit]