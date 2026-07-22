# Custom Modules
from src.rag.hugging_face_client import HuggingFaceClient
from src.rag.utils import load_config

# Data Science Libraries
import networkx as nx

# Standard Libraries
import json
import random
from typing import Dict, List, Optional


TRIPLE_EXTRACTION_PROMPT = """You are an expert knowledge graph builder.
Extract entities and relationships from the text.
Return ONLY a JSON list. No explanation, no markdown, no backticks.
Each item must contain:
- "head": source entity
- "relation": relationship
- "tail": target entity
Output JSON:"""

ENTITY_EXTRACTION_PROMPT_TEMPLATE = """Given this question and this list of known entities, return ONLY the single entity name from the list that the question is most about. Return just the entity name, nothing else, no punctuation.

Entities: {entities}"""

ANSWER_PROMPT = "Answer the question based only on the following context. If the context doesn't contain the answer, say so."


class GraphRAG(HuggingFaceClient):
    def __init__(self):
        super().__init__()
        self.KG: Optional[nx.DiGraph] = None
        self.triples: Optional[List[Dict[str, str]]] = None

    def extract_triples(self, text: str, model: str) -> List[Dict[str, str]]:
        """
        Extracts (head, relation, tail) triples from text using the LLM.

        Args:
            text  (str) : Input text from which to extract triples.
            model (str) : Model to use for extraction.

        Returns:
            list: Extracted triples as dicts with 'head', 'relation', 'tail' keys.
        """
        response = self.ask_model(prompt=TRIPLE_EXTRACTION_PROMPT, query=text, context="", model=model)

        # Cleaning up in case the model wraps the JSON in backticks / a "json" language tag
        response = response.strip().strip('`').strip()
        if response.lower().startswith('json'):
            response = response[4:].strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model returned invalid JSON for triple extraction: {response!r}") from e

    def build_knowledge_graph(self, text: str, model: str) -> tuple[nx.DiGraph, List[Dict[str, str]]]:
        """
        Builds a directed knowledge graph from text, storing it on the instance.

        Args:
            text  (str) : Input text from which to build the knowledge graph.
            model (str) : Model to use for extracting triples.

        Returns:
            tuple: (the built nx.DiGraph, the extracted triples)
        """
        self.triples = self.extract_triples(text=text, model=model)

        self.KG = nx.DiGraph()
        for item in self.triples:
            head = item.get("head")
            tail = item.get("tail")
            relation = item.get("relation")

            if head and tail:
                self.KG.add_node(head)
                self.KG.add_node(tail)
                self.KG.add_edge(head, tail, label=relation)

        return self.KG, self.triples

    def retrieve_graph_context(self, entity: str, max_depth: int = 2) -> str:
        """
        Retrieves context from the stored knowledge graph via multi-hop traversal
        starting at `entity`.

        Args:
            entity    (str) : The entity to start traversal from.
            max_depth (int) : Maximum traversal depth. Default is 2.

        Returns:
            str: Retrieved context as a period-joined string of facts.
        """
        if self.KG is None:
            raise ValueError("No knowledge graph built yet. Call build_knowledge_graph() first.")

        if entity not in self.KG.nodes:
            return ""

        context = set()
        visited_nodes = set()

        def dfs(node, depth):
            if depth > max_depth:
                return
            visited_nodes.add(node)

            for neighbor in self.KG.successors(node):
                relation = self.KG.get_edge_data(node, neighbor)["label"]
                context.add(f"{node} {relation} {neighbor}")
                if neighbor not in visited_nodes:
                    dfs(neighbor, depth + 1)

            for predecessor in self.KG.predecessors(node):
                relation = self.KG.get_edge_data(predecessor, node)["label"]
                context.add(f"{predecessor} {relation} {node}")
                if predecessor not in visited_nodes:
                    dfs(predecessor, depth + 1)

        dfs(entity, 1)
        return ". ".join(context)

    def extract_entity_from_question(self, question: str, model: str) -> str:
        """
        Identifies which known entity in the stored graph a question is most about.

        Args:
            question (str) : The user's question.
            model    (str) : Model to use for entity extraction.

        Returns:
            str: The best-matching entity name from the graph.
        """
        if self.KG is None:
            raise ValueError("No knowledge graph built yet. Call build_knowledge_graph() first.")

        prompt = ENTITY_EXTRACTION_PROMPT_TEMPLATE.format(entities=list(self.KG.nodes()))
        entity = self.ask_model(prompt=prompt, query=question, context="", model=model, max_tokens=30)
        return entity.strip().strip('."\'')

    def graph_rag_answer(self, question: str, model: str, entity: Optional[str] = None, max_depth: int = 3) -> str:
        """
        Answers a question using multi-hop graph context. If `entity` isn't given,
        it's auto-extracted from the question against the stored graph's nodes.

        Args:
            question  (str)           : The user's question.
            model     (str)           : Model to use for the final answer.
            entity    (str, optional) : Starting entity. Auto-extracted if omitted.
            max_depth (int)           : Maximum traversal depth. Default is 3.

        Returns:
            str: Generated answer from the LLM.
        """
        if entity is None:
            entity = self.extract_entity_from_question(question=question, model=model)

        graph_context = self.retrieve_graph_context(entity=entity, max_depth=max_depth)

        if not graph_context:
            return f"No information about '{entity}' was found in the knowledge graph."

        return self.ask_model(prompt=ANSWER_PROMPT, query=question, context=graph_context, model=model, max_tokens=200)


if __name__ == "__main__":
    graph_rag = GraphRAG()

    config_data = load_config("src/rag/configs/rag_simple.yaml")
    models_to_test = config_data['model'].get("instruct_completion_models", [])
    available_models = graph_rag.get_working_models(models_to_test)
    model = random.choice(available_models)

    text = (
        "The Moon orbits Earth. The Moon has an atmosphere called the Exosphere. "
        "Apollo 11 landed on the Moon. The Moon has a crater named the South Pole-Aitken Basin. "
        "Earth's Moon is classified as a natural satellite."
    )
    question = "On which natural satellite did Apollo land?"

    _, _ = graph_rag.build_knowledge_graph(text=text, model=model)
    answer = graph_rag.graph_rag_answer(question=question, model=model)

    print(f"\nFinal Answer:\n{answer}")