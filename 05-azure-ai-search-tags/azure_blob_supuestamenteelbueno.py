"""
Azure AI Search + Blob Storage + Indexers + Skillset (Vectorización Integrada)

Correcciones aplicadas:
1. Se añadió un 'SearchIndexerSkillset' con 'AzureOpenAIEmbeddingSkill'.
2. Se configuró el Indexer para usar ese Skillset.
3. Se añadieron 'output_field_mappings' al Indexer para guardar el vector generado en el índice.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any, Dict, List

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    IndexingParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    # Nuevos imports necesarios para el Skillset
    SearchIndexerSkillset,
    AzureOpenAIEmbeddingSkill,
    InputFieldMappingEntry,
    OutputFieldMappingEntry
)
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
load_dotenv(override=True)

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
# Nota: text-embedding-ada-002 y text-embedding-3-small usan 1536 dimensiones por defecto
AZURE_OPENAI_EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER", "documents-vector-demo")
BLOB_PREFIX = "json-docs"

INDEX_NAME = "documents-integrated-vectorization-indexer-1"
DATA_SOURCE_NAME = "documents-blob-datasource"
SKILLSET_NAME = "documents-vector-skillset"  # Nuevo
INDEXER_NAME = "documents-blob-indexer"
VECTOR_PROFILE_NAME = "blobProfile"
ALGORITHM_NAME = "blobHnsw"
VECTORIZER_NAME = "blobVectorizer"

DATA_FILE = pathlib.Path(__file__).parent / "data" / "documents_with_metadata.json"

index_client: SearchIndexClient | None = None
indexer_client: SearchIndexerClient | None = None
search_client: SearchClient | None = None


def _validate_env_variables() -> None:
    missing = [
        var
        for var in [
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_KEY",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            "AZURE_STORAGE_CONNECTION_STRING",
        ]
        if os.getenv(var) is None
    ]
    if missing:
        raise EnvironmentError(f"Faltan variables de entorno: {', '.join(missing)}")


def _init_clients() -> None:
    global index_client, indexer_client, search_client
    credential = AzureKeyCredential(AZURE_SEARCH_KEY)
    index_client = SearchIndexClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=credential)
    indexer_client = SearchIndexerClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=credential)
    search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=credential)


def _require_clients() -> None:
    if not all([index_client, indexer_client, search_client]):
        raise RuntimeError("Los clientes de Azure Search no están inicializados")


def load_documents() -> List[Dict[str, Any]]:
    if not DATA_FILE.exists():
        # Generamos datos dummy si el archivo no existe para probar
        return [
            {"id": "1", "content": "El running es excelente para la salud cardiovascular y ayuda a adelgazar.", "category": "Deportes", "source": "blob", "tags": ["salud", "running"]},
            {"id": "2", "content": "El aprendizaje automático convierte datos en vectores matemáticos.", "category": "Tecnología", "source": "blob", "tags": ["AI", "math"]},
        ]
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def sync_documents_to_blob(documents: List[Dict[str, Any]]) -> None:
    print("\n🚚 Subiendo documentos al contenedor de Blob Storage...")
    blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service.get_container_client(BLOB_CONTAINER_NAME)
    try:
        container_client.create_container()
        print(f"  ✓ Contenedor '{BLOB_CONTAINER_NAME}' creado")
    except ResourceExistsError:
        print(f"  • Contenedor '{BLOB_CONTAINER_NAME}' ya existe, se reutiliza")

    for doc in documents:
        blob_name = f"{BLOB_PREFIX}/{doc['id']}.json"
        data = json.dumps(doc, ensure_ascii=False).encode("utf-8")
        container_client.upload_blob(name=blob_name, data=data, overwrite=True)
        print(f"  ✓ Blob actualizado: {blob_name}")


def create_index_with_integrated_vectorization() -> None:
    _require_clients()
    print("\n🧱 Creando índice con configuración vectorial (para Query-Time)...")

    # Este vectorizador se usa cuando haces la CONSULTA (search), no durante la ingesta.
    vectorizer = AzureOpenAIVectorizer(
        vectorizer_name=VECTORIZER_NAME,
        parameters=AzureOpenAIVectorizerParameters(
            resource_url=AZURE_OPENAI_ENDPOINT,
            deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_key=AZURE_OPENAI_KEY,
            model_name=AZURE_OPENAI_EMBEDDING_MODEL_NAME,
        ),
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True, sortable=True),
        SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True, sortable=True),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(
            name="tags",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536, # Asegurar que coincida con el modelo
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
    ]

    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME,
                algorithm_configuration_name=ALGORITHM_NAME,
                vectorizer_name=VECTORIZER_NAME,
            )
        ],
        algorithms=[HnswAlgorithmConfiguration(name=ALGORITHM_NAME)],
        vectorizers=[vectorizer],
    )

    index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)

    try:
        index_client.delete_index(INDEX_NAME)
        print("  • Índice anterior eliminado")
    except Exception:
        pass

    index_client.create_index(index)
    print(f"  ✓ Índice '{INDEX_NAME}' listo")


def create_blob_data_source() -> None:
    _require_clients()
    print("\n🔌 Creando data source del contenedor...")

    # Limpieza preventiva
    try:
        indexer_client.delete_indexer(INDEXER_NAME)
    except Exception: pass

    container = SearchIndexerDataContainer(name=BLOB_CONTAINER_NAME, query=f"{BLOB_PREFIX}")
    data_source = SearchIndexerDataSourceConnection(
        name=DATA_SOURCE_NAME,
        type="azureblob",
        connection_string=AZURE_STORAGE_CONNECTION_STRING,
        container=container,
        description="Blob container con documentos JSON",
    )

    try:
        indexer_client.delete_data_source_connection(DATA_SOURCE_NAME)
    except Exception: pass

    indexer_client.create_data_source_connection(data_source)
    print(f"  ✓ Data source '{DATA_SOURCE_NAME}' lista")


def create_skillset() -> None:
    """
    Crea el Skillset que realiza la vectorización durante la ingesta de documentos.
    """
    _require_clients()
    print("\n🧠 Creando Skillset para vectorización (Ingestion-Time)...")

    # Habilidad para convertir texto a embedding usando Azure OpenAI
    embedding_skill = AzureOpenAIEmbeddingSkill(
        name="embedding-skill",
        description="Genera embeddings del contenido usando Azure OpenAI",
        context="/document",  # El contexto raíz es el documento JSON
        resource_url=AZURE_OPENAI_ENDPOINT,
        deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_key=AZURE_OPENAI_KEY,
        model_name=AZURE_OPENAI_EMBEDDING_MODEL_NAME,
        inputs=[
            # Mapeamos el campo 'content' del JSON al input 'text' del modelo
            InputFieldMappingEntry(name="text", source="/document/content")
        ],
        outputs=[
            # Guardamos el resultado en un campo temporal 'vector_data'
            OutputFieldMappingEntry(name="embedding", target_name="vector_data")
        ]
    )

    skillset = SearchIndexerSkillset(
        name=SKILLSET_NAME,
        description="Skillset para generar embeddings automáticos",
        skills=[embedding_skill]
    )

    try:
        indexer_client.delete_skillset(SKILLSET_NAME)
    except Exception: pass

    indexer_client.create_skillset(skillset)
    print(f"  ✓ Skillset '{SKILLSET_NAME}' creado")


def create_blob_indexer() -> None:
    _require_clients()
    print("\n⚙️  Creando indexer con Skillset vinculado...")
    
    indexer = SearchIndexer(
        name=INDEXER_NAME,
        data_source_name=DATA_SOURCE_NAME,
        target_index_name=INDEX_NAME,
        skillset_name=SKILLSET_NAME,  # <--- IMPORTANTE: Vincular el Skillset
        description="Indexer para vectorización integrada desde blobs",
        parameters=IndexingParameters(configuration={"parsingMode": "json"}),
        # Mapeamos los campos que salen del Skillset hacia el Índice
        output_field_mappings=[
            # Tomamos el campo temporal 'vector_data' y lo escribimos en 'contentVector'
            OutputFieldMappingEntry(source_field_name="/document/vector_data", target_field_name="contentVector")
        ]
        # Nota: Los campos simples (id, content, category) se mapean automáticamente 
        # si los nombres en el JSON coinciden con los nombres en el Index.
    )

    try:
        indexer_client.delete_indexer(INDEXER_NAME)
    except Exception: pass

    indexer_client.create_indexer(indexer)
    print(f"  ✓ Indexer '{INDEXER_NAME}' listo")


def run_indexer_and_wait(timeout_seconds: int = 180) -> None:
    _require_clients()
    print("\n▶️  Ejecutando indexer y esperando a que finalice...")
    indexer_client.run_indexer(INDEXER_NAME)

    waited = 0
    poll_interval = 5
    while waited < timeout_seconds:
        status = indexer_client.get_indexer_status(INDEXER_NAME)
        last_result = status.last_result
        
        if last_result and last_result.status == "success":
            print(f"  ✓ Indexer completado. Docs procesados: {last_result.item_count}")
            return
        elif last_result and last_result.status == "transientFailure":
             print(f"  ⚠️ Fallo transitorio: {last_result.error_message}")
        elif last_result and last_result.status == "error":
            raise RuntimeError(f"Indexer falló: {last_result.error_message}")

        print(f"  • Estado: {status.status} (esperados {waited}/{timeout_seconds}s)")
        time.sleep(poll_interval)
        waited += poll_interval

    raise TimeoutError("El indexer no completó dentro del tiempo esperado")


def vector_only_query(query: str) -> None:
    _require_clients()
    print(f"\n🔎 Buscando: '{query}'")

    # Azure Search usará la configuración del índice (vectorizer) para convertir
    # este texto en vector y compararlo con los vectores guardados por el skillset.
    vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=3, fields="contentVector")
    
    results = search_client.search(
        search_text=None, 
        vector_queries=[vector_query], 
        select=["id", "content", "category"], 
        top=3
    )

    print(f"  Resultados:")
    found = False
    for result in results:
        found = True
        snippet = result["content"][:100].replace("\n", " ")
        score = result.get("@search.score", 0.0)
        print(f"   ★ Score {score:.4f} | ID: {result['id']} | {snippet}...")
    
    if not found:
        print("   (No se encontraron resultados)")


def main() -> None:
    try:
        _validate_env_variables()
        _init_clients()

        documents = load_documents()
        sync_documents_to_blob(documents)

        # 1. Crear índice (define cómo buscar y almacenar)
        create_index_with_integrated_vectorization()
        
        # 2. Crear Data Source (define de dónde leer)
        create_blob_data_source()
        
        # 3. Crear Skillset (define cómo transformar/vectorizar data entrante)
        create_skillset()
        
        # 4. Crear Indexer (une Source + Skillset + Index)
        create_blob_indexer()

        # 5. Correr y probar
        run_indexer_and_wait()

        vector_only_query("running para adelgazar")
        vector_only_query("matematicas y vectores")

        print("\n✅ Flujo completado exitosamente.")

    except Exception as e:
        print(f"\n❌ Error en la ejecución: {e}")

if __name__ == "__main__":
    main()