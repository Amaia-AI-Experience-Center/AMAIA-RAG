import os
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# Configuración
STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "documentos"
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = "documentos-index-0"
OPENAI_ENDPOINT =os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"

# Cliente de Azure OpenAI
openai_client = AzureOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_API_KEY,
    api_version="2024-02-01"
)

def generar_embedding(texto):
    """Genera embeddings usando Azure OpenAI"""
    response = openai_client.embeddings.create(
        input=texto,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def subir_documentos_blob():
    """Sube documentos a Azure Blob Storage"""
    blob_service_client = BlobServiceClient.from_connection_string(
        STORAGE_CONNECTION_STRING
    )
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    
    # Crear contenedor si no existe
    try:
        container_client.create_container()
    except:
        pass
    
    # Subir archivos desde carpeta local
    archivos = ["documento1.txt", "documento2.txt"]
    for archivo in archivos:
        with open(archivo, "rb") as data:
            blob_client = container_client.get_blob_client(archivo)
            blob_client.upload_blob(data, overwrite=True)
            print(f"Subido: {archivo}")

def crear_indice_busqueda():
    """Crea el índice de búsqueda con campos vectoriales"""
    index_client = SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=AzureKeyCredential(SEARCH_API_KEY)
    )
    
    # Definir campos del índice
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),
        SearchableField(
            name="contenido",
            type=SearchFieldDataType.String,
            searchable=True
        ),
        SearchableField(
            name="nombre_archivo",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True
        ),
        SearchField(
            name="contenido_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="mi-perfil-vector"
        )
    ]
    
    # Configurar búsqueda vectorial
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="mi-algoritmo-hnsw")
        ],
        profiles=[
            VectorSearchProfile(
                name="mi-perfil-vector",
                algorithm_configuration_name="mi-algoritmo-hnsw"
            )
        ]
    )
    
    # Crear índice
    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search
    )
    
    index_client.create_or_update_index(index)
    print(f"Índice '{INDEX_NAME}' creado")

def indexar_documentos():
    """Lee documentos de Blob Storage, genera embeddings e indexa"""
    # Leer documentos de Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(
        STORAGE_CONNECTION_STRING
    )
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    
    documentos = []
    for blob in container_client.list_blobs():
        blob_client = container_client.get_blob_client(blob.name)
        contenido = blob_client.download_blob().readall().decode('utf-8')
        
        # Generar embedding
        embedding = generar_embedding(contenido)
        
        documentos.append({
            "id": blob.name.replace(".", "_"),
            "contenido": contenido,
            "nombre_archivo": blob.name,
            "contenido_vector": embedding
        })
    
    # Indexar en Azure AI Search
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_API_KEY)
    )
    
    result = search_client.upload_documents(documents=documentos)
    print(f"Indexados {len(documentos)} documentos")

def busqueda_hibrida(query_texto):
    """Ejecuta una búsqueda híbrida (texto + vectorial)"""
    # Generar embedding para la consulta
    query_embedding = generar_embedding(query_texto)
    
    # Cliente de búsqueda
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_API_KEY)
    )
    
    # Crear consulta vectorial
    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=3,
        fields="contenido_vector"
    )
    
    # Ejecutar búsqueda híbrida
    results = search_client.search(
        search_text=query_texto,  # Búsqueda de texto
        vector_queries=[vector_query],  # Búsqueda vectorial
        select=["id", "contenido", "nombre_archivo"],
        top=5
    )
    
    print(f"\nResultados para: '{query_texto}'\n")
    for result in results:
        print(f"Score: {result['@search.score']:.4f}")
        print(f"Archivo: {result['nombre_archivo']}")
        print(f"Contenido: {result['contenido'][:200]}...")
        print("-" * 80)

# Flujo completo
if __name__ == "__main__":
    # 1. Subir documentos a Blob Storage
    print("1. Subiendo documentos a Blob Storage...")
    subir_documentos_blob()
    
    # 2. Crear índice de búsqueda
    print("\n2. Creando índice de búsqueda...")
    crear_indice_busqueda()
    
    # 3. Indexar documentos con embeddings
    print("\n3. Indexando documentos con embeddings...")
    indexar_documentos()
    
    # 4. Realizar búsqueda híbrida
    print("\n4. Ejecutando búsqueda híbrida...")
    busqueda_hibrida("running")
