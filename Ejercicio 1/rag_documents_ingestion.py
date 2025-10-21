import json
import os
import pathlib

import azure.identity
import openai
import pymupdf4llm
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")
client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")

data_dir = pathlib.Path(os.path.dirname(__file__)) / "data"
filenames = ["California_carpenter_bee.pdf", "Centris_pallida.pdf", "Western_honey_bee.pdf", "Aphideater_hoverfly.pdf"]
all_chunks = []
for filename in filenames:
    # Extract text from the PDF file
    md_text = pymupdf4llm.to_markdown(data_dir / filename)

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o", chunk_size=500, chunk_overlap=125
    )
    texts = text_splitter.create_documents([md_text])
    file_chunks = [{"id": f"{filename}-{(i + 1)}", "text": text.page_content} for i, text in enumerate(texts)]

    # Generate embeddings using openAI SDK for each text
    for file_chunk in file_chunks:
        file_chunk["embedding"] = (
            client.embeddings.create(model="text-embedding-3-small", input=file_chunk["text"]).data[0].embedding
        )
    all_chunks.extend(file_chunks)

# Save the documents with embeddings to a JSON file
with open("rag_ingested_chunks.json", "w") as f:
    json.dump(all_chunks, f, indent=4)
