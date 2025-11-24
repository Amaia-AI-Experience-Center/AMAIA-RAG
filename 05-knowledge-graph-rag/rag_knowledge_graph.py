import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set

import networkx as nx
import openai
from dotenv import load_dotenv

load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")
client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")

# Path to the folder where this script lives
script_dir = Path(__file__).parent

# Path to the graph data inside the 'data' folder
graph_data_path = script_dir / "data" / "graph_data.json"


class KnowledgeGraphRAG:
    """Knowledge Graph-based Retrieval Augmented Generation system."""

    def __init__(self, graph_data_path: Path):
        """Initialize the knowledge graph from JSON data."""
        self.graph = nx.DiGraph()
        self.entities = {}
        self.relationships = {}

        # Load graph data
        with open(graph_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Build the graph
        self._build_graph(data)

    def _build_graph(self, data: Dict):
        """Build NetworkX graph from entities and relationships."""
        # Add entities as nodes
        for entity in data['entities']:
            entity_id = entity['id']
            self.entities[entity_id] = entity

            # Add node with all properties
            self.graph.add_node(
                entity_id,
                name=entity['name'],
                type=entity['type'],
                properties=entity.get('properties', {})
            )

        # Add relationships as edges
        for rel in data['relationships']:
            rel_id = rel['id']
            self.relationships[rel_id] = rel

            # Add edge with relationship properties
            self.graph.add_edge(
                rel['source'],
                rel['target'],
                rel_id=rel_id,
                rel_type=rel['type'],
                properties=rel.get('properties', {})
            )

    def search_entities(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for entities by name or properties.
        Returns top_k most relevant entities.
        """
        query_lower = query.lower()
        query_tokens = [token for token in re.split(r"\W+", query_lower) if token]
        if not query_tokens:
            return []
        matches = []

        for entity_id, entity in self.entities.items():
            score = 0

            # Check name match
            entity_name = entity['name'].lower()
            name_matches = sum(1 for token in query_tokens if token in entity_name)
            score += 10 * name_matches

            # Check type match
            entity_type = entity['type'].lower()
            type_matches = sum(1 for token in query_tokens if token in entity_type)
            score += 5 * type_matches

            # Check properties
            for key, value in entity.get('properties', {}).items():
                key_lower = str(key).lower()
                value_lower = str(value).lower()
                prop_matches = sum(
                    1
                    for token in query_tokens
                    if token in key_lower or token in value_lower
                )
                score += 3 * prop_matches

            if score > 0:
                matches.append({
                    'entity': entity,
                    'score': score
                })

        # Sort by score and return top_k
        matches.sort(key=lambda x: x['score'], reverse=True)
        return [m['entity'] for m in matches[:top_k]]

    def get_entity_context(self, entity_id: str, depth: int = 1) -> Dict:
        """
        Get an entity and its neighborhood (connected entities and relationships).
        depth controls how many hops away to include.
        """
        if entity_id not in self.entities:
            return None

        context = {
            'entity': self.entities[entity_id],
            'relationships': []
        }

        # Get outgoing relationships
        for target in self.graph.successors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, target)
            context['relationships'].append({
                'type': edge_data['rel_type'],
                'direction': 'outgoing',
                'target': self.entities[target],
                'properties': edge_data.get('properties', {})
            })

        # Get incoming relationships
        for source in self.graph.predecessors(entity_id):
            edge_data = self.graph.get_edge_data(source, entity_id)
            context['relationships'].append({
                'type': edge_data['rel_type'],
                'direction': 'incoming',
                'source': self.entities[source],
                'properties': edge_data.get('properties', {})
            })

        return context

    def traverse_graph(self, start_entities: List[str], max_depth: int = 2) -> List[Dict]:
        """
        Traverse the graph from starting entities up to max_depth.
        Returns all visited entities and relationships.
        """
        visited = set()
        results = []

        def dfs(entity_id: str, depth: int):
            if depth > max_depth or entity_id in visited:
                return

            visited.add(entity_id)
            context = self.get_entity_context(entity_id, depth=1)
            if context:
                results.append(context)

            # Continue traversal
            if depth < max_depth:
                for neighbor in self.graph.successors(entity_id):
                    dfs(neighbor, depth + 1)
                for neighbor in self.graph.predecessors(entity_id):
                    dfs(neighbor, depth + 1)

        for entity_id in start_entities:
            dfs(entity_id, 0)

        return results

    def format_context_for_llm(self, contexts: List[Dict]) -> str:
        """Format graph contexts into a readable string for the LLM."""
        if not contexts:
            return "No relevant information found in the knowledge graph."

        formatted = "Knowledge Graph Information:\n\n"

        for i, ctx in enumerate(contexts, 1):
            entity = ctx['entity']
            formatted += f"Entity #{i}: {entity['name']} ({entity['type']})\n"

            # Add entity properties
            if entity.get('properties'):
                formatted += "  Properties:\n"
                for key, value in entity['properties'].items():
                    formatted += f"    - {key}: {value}\n"

            # Add relationships
            if ctx['relationships']:
                formatted += "  Relationships:\n"
                for rel in ctx['relationships']:
                    if rel['direction'] == 'outgoing':
                        formatted += f"    - {rel['type']} -> {rel['target']['name']} ({rel['target']['type']})\n"
                    else:
                        formatted += f"    - {rel['type']} <- {rel['source']['name']} ({rel['source']['type']})\n"

                    # Add relationship properties if they exist
                    if rel.get('properties'):
                        for key, value in rel['properties'].items():
                            formatted += f"      {key}: {value}\n"

            formatted += "\n"

        return formatted


# Initialize the knowledge graph
kg_rag = KnowledgeGraphRAG(graph_data_path)

QUERY_REWRITE_SYSTEM_MESSAGE = """
You are a helpful assistant that rewrites user questions into keyword queries
for searching a knowledge graph about technology, programming languages, frameworks, and concepts.

Extract the key entities, concepts, or topics the user is asking about.
Focus on specific names of languages, frameworks, libraries, organizations, or technical concepts.


Respond with ONLY the keyword query (2-6 words).
"""

SYSTEM_MESSAGE = """
You are a helpful assistant that answers questions using a knowledge graph about
technology, programming languages, frameworks, libraries, and related concepts.

You must base your answers on the knowledge graph data provided in the context.
If the information is not in the knowledge graph, say so clearly.
Use the relationships and properties to provide comprehensive and accurate answers.
"""

messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

print("Knowledge Graph RAG System")
print("=" * 50)
print(f"Loaded {len(kg_rag.entities)} entities and {len(kg_rag.relationships)} relationships")
print("=" * 50)
print("\nAvailable entity types:", set(e['type'] for e in kg_rag.entities.values()))
print("\nAvailable relationship types:", set(r['type'] for r in kg_rag.relationships.values()))
print("\nType 'quit' or 'exit' to stop.\n")

while True:
    question = input("\nYour question: ")

    if question.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break

    if not question.strip():
        continue

    # Rewrite the query to extract key entities/concepts
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.05,
        messages=[
            {"role": "system", "content": QUERY_REWRITE_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": f"New user question: {question}\n\nConversation history: {messages}",
            },
        ],
    )
    search_query = response.choices[0].message.content
    print(f"\n[Search Query]: {search_query}")

    # Search for relevant entities in the knowledge graph
    entities = kg_rag.search_entities(search_query, top_k=5)
    print(f"[Found Entities]: {len(entities)} matches")

    if entities:
        # Get context for each entity (entity + its immediate relationships)
        contexts = []
        for entity in entities:
            context = kg_rag.get_entity_context(entity['id'])
            if context:
                contexts.append(context)

        # Format the graph context for the LLM
        graph_context = kg_rag.format_context_for_llm(contexts)
        print(f"\n[Graph Context]:\n{graph_context[:300]}..." if len(graph_context) > 300 else f"\n[Graph Context]:\n{graph_context}")
    else:
        graph_context = "No relevant information found in the knowledge graph."
        print(f"\n[Graph Context]: {graph_context}")

    # Use the graph context to generate a response
    messages.append({
        "role": "user",
        "content": f"{question}\n\nKnowledge Graph Context:\n{graph_context}"
    })

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.3,
        messages=messages
    )

    llm_response = response.choices[0].message.content
    messages.append({"role": "assistant", "content": llm_response})

    print(f"\n[Response]:")
    print(llm_response)
