# Knowledge Graph RAG

A generic Knowledge Graph-based Retrieval Augmented Generation (RAG) system that uses graph structures to represent and query interconnected entities and relationships.

## Overview

This module implements a Knowledge Graph RAG system that:
- Stores entities and their relationships in a directed graph structure
- Enables semantic search across entities based on properties and relationships
- Provides contextual information by traversing graph connections
- Uses LLM query rewriting for improved search relevance
- Supports conversational interactions with graph-based context

## Architecture

### Components

1. **Knowledge Graph Structure**
   - Entities: Nodes representing concepts, objects, or entities
   - Relationships: Directed edges connecting entities
   - Properties: Metadata attached to entities and relationships

2. **Graph Operations**
   - Entity search: Find entities by name, type, or properties
   - Context retrieval: Get entity with its immediate relationships
   - Graph traversal: Navigate multi-hop connections
   - Context formatting: Transform graph data into LLM-readable format

3. **RAG Pipeline**
   - Query rewriting: Extract key concepts from user questions
   - Entity search: Find relevant nodes in the knowledge graph
   - Context assembly: Gather entity and relationship information
   - LLM generation: Answer questions using graph context

## Data Format

The knowledge graph is stored in JSON format with two main components:

### Entities

```json
{
  "id": "unique_identifier",
  "name": "Entity Name",
  "type": "EntityType",
  "properties": {
    "key1": "value1",
    "key2": "value2"
  }
}
```

### Relationships

```json
{
  "id": "unique_identifier",
  "source": "source_entity_id",
  "target": "target_entity_id",
  "type": "RELATIONSHIP_TYPE",
  "properties": {
    "description": "relationship description",
    "strength": "primary|supporting"
  }
}
```

## Features

### 1. Entity Search
Finds entities matching search queries by:
- Name matching
- Type matching
- Property matching
- Relevance scoring

### 2. Contextual Retrieval
For each entity, retrieves:
- Entity properties
- Outgoing relationships (what this entity relates to)
- Incoming relationships (what relates to this entity)
- Connected entity details

### 3. Query Rewriting
Uses LLM to transform natural language questions into keyword queries optimized for graph search.

### 4. Conversational Memory
Maintains conversation history to:
- Provide context-aware responses
- Support follow-up questions
- Enable multi-turn dialogues

## Usage

### Running the System

```bash
python rag_knowledge_graph.py
```

### Environment Variables

Required in `.env` file:
- `GITHUB_TOKEN`: API key for GitHub AI Models
- `GITHUB_MODEL` (optional): Model name, defaults to "openai/gpt-4o"
- `API_HOST` (optional): Defaults to "github"

### Example Interactions

**Question**: "What is Python used for?"
- System finds Python entity
- Retrieves relationships (USED_IN Machine Learning, Data Science, etc.)
- Formats context with connected frameworks and libraries
- LLM generates comprehensive answer based on graph data

**Question**: "Tell me about machine learning frameworks"
- System finds Machine Learning entity
- Identifies related frameworks (TensorFlow, etc.)
- Retrieves framework details and relationships
- Provides structured information about ML frameworks

**Question**: "What libraries work with Python for data science?"
- System finds Python and Data Science entities
- Traverses graph to find libraries (Pandas, NumPy, etc.)
- Shows relationships and dependencies
- Explains each library's role

## Sample Data

The included `data/graph_data.json` contains a generic knowledge graph about:
- Programming languages (Python, JavaScript)
- Frameworks (TensorFlow, Django, React)
- Libraries (NumPy, Pandas)
- Fields (Machine Learning, Web Development, Data Science)
- Organizations (OpenAI)
- Models and concepts

This can be easily replaced with domain-specific data.

## Customization

### Adding Your Own Data

1. Create or modify `data/graph_data.json`
2. Define entities with relevant types and properties
3. Specify relationships between entities
4. Update the query rewrite prompt if needed for your domain

### Extending the Graph

The graph structure is generic and can represent:
- Technical documentation (APIs, services, dependencies)
- Business knowledge (products, customers, transactions)
- Scientific data (research papers, citations, authors)
- Any domain with interconnected concepts

### Modifying Search Behavior

Adjust the `search_entities()` method to:
- Change scoring weights
- Add fuzzy matching
- Include semantic similarity
- Filter by entity types

### Customizing Context Retrieval

Modify `get_entity_context()` or `traverse_graph()` to:
- Change traversal depth
- Filter relationship types
- Include specific property patterns
- Optimize for performance

## Technologies

- **NetworkX**: Graph data structure and algorithms
- **OpenAI API**: LLM for query rewriting and response generation
- **Python**: Core implementation language
- **JSON**: Data storage format

## Advantages of Knowledge Graph RAG

1. **Structured Relationships**: Explicitly models connections between entities
2. **Multi-hop Reasoning**: Can traverse relationships for deeper insights
3. **Semantic Context**: Provides rich contextual information beyond simple text retrieval
4. **Scalability**: Efficient graph algorithms for large knowledge bases
5. **Interpretability**: Clear provenance of information through graph paths
6. **Flexibility**: Generic structure adaptable to any domain

## Future Enhancements

Potential improvements:
- Graph database integration (Neo4j, ArangoDB)
- Semantic embeddings for entities
- Path-finding algorithms for complex queries
- Graph visualization
- Incremental graph updates
- Entity resolution and disambiguation
- Subgraph extraction for focused contexts
- Hybrid search combining text and graph

## Comparison with Other RAG Approaches

| Feature | Knowledge Graph RAG | Vector RAG | CSV/Text RAG |
|---------|-------------------|------------|--------------|
| Structure | Explicit relationships | Semantic embeddings | Unstructured text |
| Reasoning | Multi-hop traversal | Similarity search | Keyword matching |
| Context | Rich, connected | Relevant chunks | Matching rows/docs |
| Setup | Requires graph modeling | Embedding generation | Simple indexing |
| Best for | Interconnected data | Semantic similarity | Simple queries |

## License

Part of the AMAIA-RAG project.
