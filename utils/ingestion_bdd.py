import chromadb
import logging
import os
from utils.chunk_files import chunk_folder
from utils.utils import embedding_function

def embed_chunks_to_chroma(chunks: list[str], metadatas: list[dict], collection_name: str, collection_path: str):    
    client = chromadb.PersistentClient(path=collection_path)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)
       # Ajout à la collection (ajustement des ids si nécessaire)
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=[f"doc_{i}" for i in range(len(chunks))]
    )

def ingest_to_chroma(folder_path, collection_name, collection_path):
    chunks, metadatas = chunk_folder(folder_path)
    embed_chunks_to_chroma(chunks, metadatas, collection_name, collection_path)

folder_path = r"Agent\Markdown_data"
collection_name = "STAT_NON_PARAM_v"
collection_path = r"Agent\DBv"

ingest_to_chroma(folder_path, collection_name, collection_path)

