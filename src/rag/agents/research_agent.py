# Custom Libraries
from src.rag.clients.embedding_client import EmbeddingClient
from src.rag.clients.hugging_face_client import HuggingFaceClient
from src.rag.documents.splitters import chunk_passages, split_sentences
from src.rag.config import load_config

# Data Science and NLP Libraries
from bs4 import BeautifulSoup
from ddgs import DDGS 
import numpy as np

# Other Libraries
import random
import re
import requests
import time
import urllib.parse


class WebResearchAgent(HuggingFaceClient):
    def __init__(self, embedding_client: EmbeddingClient = None):
        super().__init__()
        # Geting a list of available models that work with the Hugging Face Inference API right now - they rotate availability on the free tier, so some may be down at any given time.
        data = load_config("src/rag/configs/rag_simple.yaml")
        MODELS_TO_TEST = data["model"].get("instruct_completion_models", [])

        self.available_models = self.get_working_models(MODELS_TO_TEST)
        self.model = random.choice(self.available_models)

        self.embedding_client = embedding_client or EmbeddingClient()

    def unwrap_ddg(self, url: str) -> str:
        """Extracts real URL from DuckDuckGo redirect wrapper.

        Args:
            url (str): The URL to unwrap.
        Returns:
            str: The unwrapped URL, or the original URL if it cannot be unwrapped.
        """
        try:
            parsed = urllib.parse.urlparse(url)
            if "duckduckgo.com" in parsed.netloc:
                qs = urllib.parse.parse_qs(parsed.query)
                uddg = qs.get("uddg")
                if uddg:
                    return urllib.parse.unquote(uddg[0])
        except Exception:
            pass
        return url

    def search_web(self, query: str, max_results: int = 5) -> list[str]:
        """Searches the web and return a list of URLs.

        Args:
            query       (str): The search query.
            max_results (int): Maximum number of search results to return.
        Returns:
            list[str]: A list of URLs from the search results.
        """
        urls = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r.get("href") or r.get("url")
                if not url:
                    continue
                url = self.unwrap_ddg(url) # Cleaning up DDG redirect links
                urls.append(url)
        return urls

    def fetch_text(self, url: str, timeout: int = 10) -> str:
        """Fetches and cleans text content from a URL.

        Args:
            url     (str): The URL to fetch.
            timeout (int): Timeout for the request in seconds.
        Returns:
            str: Cleaned text content from the URL, or an empty string if fetching fails.
        """
        headers = {"User-Agent": "Mozilla/5.0 (research-agent)"}
        try:
            r = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
            if r.status_code != 200:
                return ""
            ct = r.headers.get("content-type", "")
            if "html" not in ct.lower(): # Skipping non-HTML content
                return ""
            
            soup = BeautifulSoup(r.text, "html.parser")
            
            # Removing all annoying tags
            for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "iframe", "nav", "aside"]):
                tag.extract()
                
            # Getting all paragraph text
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            text = " ".join([p for p in paragraphs if p])
            
            if text.strip():
                # Cleaning up whitespace
                return re.sub(r"\s+", " ", text).strip()
                
            # --- Fallback logic if <p> tags fail ---
            meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            if meta and meta.get("content"):
                return meta["content"].strip()
            if soup.title and soup.title.string:
                return soup.title.string.strip()
                
        except Exception:
            return "" # Fails silently
        return ""

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Computes cosine similarity between two vectors.

        Args:
            a (np.ndarray): First vector.
            b (np.ndarray): Second vector.
        Returns:
            float: Cosine similarity score between -1 and 1.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    def _extractive_summary(self, q_emb: np.ndarray, top_passages: list[dict], summary_sentences: int = 3) -> str:
        """Generates summary using extractive method (no LLM needed).

        Args:
            q_emb           (np.ndarray): The embedding of the query.
            top_passages    (list[dict]): List of top passages with their URLs.
        Returns:
            str: A summary generated from the top passages.
        If the top passages are empty or do not contain the answer, returns a default message.
        """
        sentences = []
        for tp in top_passages:
            for s in split_sentences(tp["passage"]):
                sentences.append({"sent": s, "url": tp["url"]})

        if not sentences:
            return "No summary could be generated."

        sent_texts = [s["sent"] for s in sentences]
        sent_embs = self.embedding_client.encode(sent_texts)
        sent_sims = [self.cosine(e, q_emb) for e in sent_embs]

        top_sent_idx = np.argsort(sent_sims)[::-1][:summary_sentences]
        chosen = [sentences[idx] for idx in top_sent_idx]

        seen = set()
        lines = []
        for s in chosen:
            key = s["sent"].lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{s['sent']} (Source: {s['url']})")
        return " ".join(lines)


    def run(self, query: str, use_llm_summary: bool = False, passages_per_page: int= 4, 
            top_passages: int = 5, summary_sentences: int = 3, max_results: int = 5) -> dict:
        """Main method to run the web research agent: performs web search, fetches and chunks documents, creates embeddings, ranks passages, and optionally summarizes them.
        
        Args:
            query               (str)   : The search query.
            use_llm_summary     (bool)  : Whether to use LLM for summarization.
            passages_per_page   (int)   : Number of passages per page.
            top_passages        (int)   : Number of top passages to consider.
            summary_sentences   (int)   : Number of sentences in the summary.
            max_results         (int)   : Maximum number of search results to fetch.
        Returns:
            dict: Contains the query, top passages, summary, and elapsed time.
        """
        # Constants
        start = time.time()
        
        # Starting the search
        urls = self.search_web(query = query, max_results = max_results)
        print(f"Found {len(urls)} urls.")
        
        # Fetch & Chunk
        docs = []
        for u in urls:
            txt = self.fetch_text(u)
            if not txt:
                continue
            chunks = chunk_passages(txt, max_words=120)
            for c in chunks[:passages_per_page]:
                docs.append({"url": u, "passage": c})
        
        if not docs:
            print("No documents fetched.")
            return {"query": query, "passages": [], "summary": ""}
        
        # Embedding
        print(f"Embedding {len(docs)} passages...")
        texts = [d["passage"] for d in docs]
        emb_texts = self.embedding_client.encode(texts)
        q_emb = self.embedding_client.encode([query])[0]

        # Ranking by similarity
        sims = [self.cosine(e, q_emb) for e in emb_texts]
        top_idx = np.argsort(sims)[::-1][:top_passages]
        top_passages = [{
            "url": docs[i]["url"],
            "passage": docs[i]["passage"],
            "score": float(sims[i])
        } for i in top_idx]

        # Summarization
        if use_llm_summary:
            context_str = "\n\n".join(f"[Source: {p['url']}]\n{p['passage']}" for p in top_passages)
            prompt = "Based on the research passages below, provide a clear, concise summary that answers the question. Cite the sources."
            llm_context = f"Research Passages:\n{context_str}\n\nSummary:"
            llm_query = f"Question: {query}"
            summary = self.ask_model(query=llm_query, prompt=prompt, context=llm_context, model=self.model, max_tokens=300)
        else:
            summary = self._extractive_summary(q_emb=q_emb, top_passages=top_passages, summary_sentences=summary_sentences)

        elapsed = time.time() - start
        return {
            "query": query,
            "passages": top_passages,
            "summary": summary,
            "time": elapsed
        }


    
if __name__ == "__main__":
    agent = WebResearchAgent()
    config = load_config("src/rag/configs/web_agent.yaml")

    q = "What causes the long heat waves in Europe and how they originate?"
    print(f"\nResearching: {q}\n")
    out = agent.run(query=q, use_llm_summary = config["web_research_agent"]["use_llm_summary"], 
                    passages_per_page = config["web_research_agent"]["passages_per_page"], 
                    top_passages = config["web_research_agent"]["top_passages"], 
                    summary_sentences = config["web_research_agent"]["summary_sentences"], 
                    max_results = config["web_search"]["max_results"],)

    print("\nTop passages:")
    for p in out["passages"]:
        print(f"  score {p['score']:.3f} | {p['url']}")
        print(f"  {p['passage'][:150]}...\n")

    print("--- Summary ---")
    print(out["summary"])
    print("---------------")
    print(f"\nDone in {out['time']:.1f}s")