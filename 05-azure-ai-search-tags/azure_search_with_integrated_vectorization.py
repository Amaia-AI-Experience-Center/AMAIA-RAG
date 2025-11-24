"""
Ejercicio: Azure AI Search con Integrated Vectorization (Vectorización Integrada)

Este ejercicio demuestra cómo usar Azure AI Search con vectorización integrada:
1. Azure genera automáticamente los embeddings (no los calculas en local)
2. Solo subes el texto plano de los documentos
3. Azure usa Azure OpenAI para crear los embeddings de forma consistente
4. Mantiene la consistencia entre indexación y búsqueda

VENTAJAS vs. Embeddings Locales:
✅ No necesitas generar embeddings localmente
✅ Azure sabe qué modelo usa (lo configuras en el índice)
✅ Consistencia garantizada entre indexación y búsqueda
✅ Actualizaciones automáticas cuando cambias el modelo
✅ Menos código, menos errores

Requisitos:
- Azure AI Search service
- Azure OpenAI service con un deployment de embeddings
- Variables de entorno en .env:
  - AZURE_SEARCH_ENDPOINT
  - AZURE_SEARCH_KEY
  - AZURE_OPENAI_ENDPOINT
  - AZURE_OPENAI_KEY
  - AZURE_OPENAI_EMBEDDING_DEPLOYMENT (ej: "text-embedding-3-small")
"""

import json
import os
import pathlib
import time
from typing import List, Dict, Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    AzureOpenAIModelName,
)
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv(override=True)

# Configuración Azure AI Search
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = "documents-integrated-vectorization"

# Configuración Azure OpenAI (para que Azure genere los embeddings)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Clientes de Azure AI Search
credential = AzureKeyCredential(AZURE_SEARCH_KEY)
index_client = SearchIndexClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=credential)
search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=credential)


def create_search_index_with_vectorization():
    """
    Crea un índice con VECTORIZACIÓN INTEGRADA.
    
    La diferencia clave: configuramos un AzureOpenAIVectorizer que le dice a Azure
    qué modelo de Azure OpenAI usar para generar embeddings automáticamente.
    """
    print(f"\n🔧 Creando índice con vectorización integrada '{INDEX_NAME}'...")

    # Configurar el vectorizer (le dice a Azure cómo generar embeddings)
    vectorizer = AzureOpenAIVectorizer(
        vectorizer_name="myVectorizer",
        parameters=AzureOpenAIVectorizerParameters(
            resource_url=AZURE_OPENAI_ENDPOINT,
            deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_key=AZURE_OPENAI_KEY,
            model_name="text-embedding-ada-002",
        ),
    )
    print(f"  ✓ Vectorizer configurado:")
    print(vectorizer)

    # Definir los campos del índice
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            sortable=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SimpleField(
            name="category",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="tags",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
            facetable=True,
        ),
        # Campo vectorial: Azure lo genera automáticamente del campo 'content'
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,  # Dimensiones de text-embedding-3-small
            vector_search_profile_name="myProfile",
        ),
    ]

    # Configurar búsqueda vectorial con el vectorizer
    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="myProfile",
                algorithm_configuration_name="myAlgorithm",
                vectorizer_name="myVectorizer",  # 👈 Asocia el vectorizer al profile
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(name="myAlgorithm")
        ],
        vectorizers=[vectorizer],  # 👈 Agrega el vectorizer al índice
    )

    # Crear el índice
    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
    )

    # Eliminar índice si ya existe
    try:
        index_client.delete_index(INDEX_NAME)
        print(f"  ✓ Índice anterior eliminado")
    except Exception:
        pass

    # Crear nuevo índice
    index_client.create_index(index)
    print(f"  ✓ Índice '{INDEX_NAME}' creado con vectorización integrada")
    print(f"  ✓ Azure generará embeddings usando: {AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")
    print(f"  ✓ Campos: id, content, category, source, tags, contentVector")


def index_documents():
    """
    Indexa documentos SIN generar embeddings localmente.
    
    IMPORTANTE: Solo subimos el texto plano. Azure genera los embeddings automáticamente.
    """
    print(f"\n📝 Indexando documentos (Azure generará los embeddings)...")

    # Cargar documentos
    documents = json.load(
        open(pathlib.Path(__file__).parent / "data/documents_with_metadata.json", "r", encoding="utf-8")
    )

    # Preparar documentos para indexar
    docs_to_index = []
    for doc in documents:
        print(f"  Preparando: {doc['id']} - Categoría: {doc['category']}")

        # ⚠️ NOTA: NO generamos embeddings aquí - Azure lo hace automáticamente
        doc_to_index = {
            "id": doc["id"],
            "content": doc["content"],  # 👈 Solo el texto
            "category": doc["category"],
            "source": doc["source"],
            "tags": doc["tags"],
            # contentVector NO se incluye - Azure lo genera automáticamente
        }
        docs_to_index.append(doc_to_index)

    # Subir documentos (Azure generará los embeddings)
    result = search_client.upload_documents(documents=docs_to_index)
    print(f"  ✓ {len(docs_to_index)} documentos subidos")
    print(f"  ✓ Azure está generando los embeddings en background...")
    
    # Esperar a que los documentos se indexen y vectoricen completamente
    print(f"  ⏳ Esperando a que la vectorización complete...")
    time.sleep(10)  # Más tiempo porque Azure necesita generar los embeddings

    return documents


def search_by_tag(category: str = None, source: str = None, tags: List[str] = None):
    """
    Búsqueda filtrada por tags usando filtros OData
    """
    filters = []

    if category:
        filters.append(f"category eq '{category}'")

    if source:
        filters.append(f"source eq '{source}'")

    if tags:
        tag_filters = [f"tags/any(t: t eq '{tag}')" for tag in tags]
        filters.append(f"({' or '.join(tag_filters)})")

    filter_expression = " and ".join(filters) if filters else None

    print(f"\n🔍 Búsqueda con filtros:")
    if category:
        print(f"  - Categoría: {category}")
    if source:
        print(f"  - Fuente: {source}")
    if tags:
        print(f"  - Tags: {tags}")

    if filter_expression:
        print(f"  Expresión de filtro: {filter_expression}")

    # Realizar búsqueda
    results = search_client.search(
        search_text="*",
        filter=filter_expression,
        select=["id", "content", "category", "source", "tags"],
        top=3
    )

    print(f"\n📋 Resultados:")
    count = 0
    for result in results:
        count += 1
        print(f"\n  {count}. ID: {result['id']}")
        print(f"     Categoría: {result['category']}")
        print(f"     Fuente: {result['source']}")
        print(f"     Tags: {', '.join(result['tags'])}")
        print(f"     Contenido: {result['content'][:100]}...")

    if count == 0:
        print("  ⚠️  No se encontraron resultados")

    return count


def hybrid_search(query: str, category: str = None, tags: List[str] = None):
    """
    Búsqueda híbrida con vectorización integrada.
    
    IMPORTANTE: Usamos VectorizableTextQuery en lugar de VectorizedQuery.
    Esto le dice a Azure: "toma este texto, genera el embedding usando el modelo
    que configuramos en el índice, y búscalo".
    """
    
    # Construir filtros
    filters = []
    if category:
        filters.append(f"category eq '{category}'")
    if tags:
        tag_filters = [f"tags/any(t: t eq '{tag}')" for tag in tags]
        filters.append(f"({' or '.join(tag_filters)})")

    filter_expression = " and ".join(filters) if filters else None

    print(f"\n🔍 Búsqueda híbrida (con vectorización integrada):")
    print(f"  - Query: '{query}'")
    if category:
        print(f"  - Categoría: {category}")
    if tags:
        print(f"  - Tags: {tags}")

    # 👈 CLAVE: VectorizableTextQuery - Azure genera el embedding automáticamente
    vector_query = VectorizableTextQuery(
        text=query,  # Solo pasamos el texto
        k_nearest_neighbors=5,
        fields="contentVector"
    )

    # Realizar búsqueda híbrida
    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],  # Azure vectoriza el query automáticamente
        filter=filter_expression,
        select=["id", "content", "category", "source", "tags"],
        top=5
    )

    print(f"\n📋 Resultados (ordenados por relevancia):")
    count = 0
    for result in results:
        count += 1
        score = result.get('@search.score', 0)
        print(f"\n  {count}. ID: {result['id']} (Score: {score:.4f})")
        print(f"     Categoría: {result['category']}")
        print(f"     Tags: {', '.join(result['tags'])}")
        print(f"     Contenido: {result['content'][:150]}...")

    if count == 0:
        print("  ⚠️  No se encontraron resultados")

    return count


def vector_only_search(query: str):
    """
    Búsqueda PURAMENTE vectorial.
    Ignora coincidencias de palabras clave (search_text=None).
    Sirve para demostrar que la búsqueda entiende significados, no solo palabras.
    """
    print(f"\n🧠 Búsqueda SOLO Vectorial (Semántica pura):")
    print(f"  - Query: '{query}'")

    vector_query = VectorizableTextQuery(
        text=query,
        k_nearest_neighbors=5,
        fields="contentVector"
    )

    # search_text=None desactiva la búsqueda por palabras clave (BM25)
    results = search_client.search(
        search_text=None,  
        vector_queries=[vector_query],
        select=["id", "content", "category"],
        top=3
    )

    print(f"\n📋 Resultados semánticos:")
    count = 0
    for result in results:
        count += 1
        score = result.get('@search.score', 0)
        print(f"\n  {count}. ID: {result['id']} (Score: {score:.4f})")
        print(f"     Contenido: {result['content'][:100]}...")

    if count == 0:
        print("  ⚠️  No se encontraron resultados")

def main():
    """Función principal"""

    print("=" * 70)
    print("  Azure AI Search - Vectorización Integrada con Tags")
    print("=" * 70)

    # 1. Crear índice con vectorización integrada
    #create_search_index_with_vectorization()

    # 2. Indexar documentos (Azure genera los embeddings)
    #documents = index_documents()

    # 3. Ejemplo 1: Buscar por categoría
    #search_by_tag(category="negocios")

    # 4. Ejemplo 2: Buscar documentos de salud
    #search_by_tag(category="salud")

    # 5. Ejemplo 3: Buscar con tag específico
    #search_by_tag(tags=["ia"])

    # 6. Ejemplo 4: Búsqueda híbrida sobre tecnología
    #hybrid_search("lenguajes de programación", category="tecnologia")

    # 7. Ejemplo 5: Búsqueda híbrida sobre salud
    #hybrid_search("cómo mejorar la salud del corazón", category="salud")

    # 8. Ejemplo 6: Búsqueda sobre métricas
    hybrid_search("jefe")

    vector_only_search("running") 

    # 9. Ejemplo 7: Búsqueda con filtro de tags
    hybrid_search("búsqueda de información", tags=["rag"])

    print("\n" + "=" * 70)
    print("  ✅ Ejercicio completado!")
    print("=" * 70)
    print("\n💡 Ventajas de la vectorización integrada:")
    print("  1. ✅ No calculas embeddings localmente")
    print("  2. ✅ Azure sabe qué modelo usa (configurado en el índice)")
    print("  3. ✅ Consistencia automática entre indexación y búsqueda")
    print("  4. ✅ Menos código, menos posibilidad de errores")
    print("  5. ✅ Si cambias de modelo, solo actualizas la configuración del índice")
    print("\n🔑 Diferencias clave:")
    print("  - Embeddings locales: tú generas, subes vectores, Azure solo almacena")
    print("  - Vectorización integrada: subes texto, Azure genera y almacena vectores")


if __name__ == "__main__":
    main()
