import json
from utils.ingestion_bdd import embed_chunks_to_chroma

import chromadb
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


def load_chunks_from_json(json_path):
    """
    arg : json_path (path des données chunkées stockées au format json)
    return : le contenu des chunks et des metadonnées chacun sous forme de liste
    """
    chunks = []
    metadatas = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.pop("content", None)
            metadata = data.pop("metadata", None)
            if not text:
                continue
            chunks.append(text)
            metadatas.append(metadata)
    return chunks, metadatas

def ingest_to_chroma(json_path):
    """
    vectorise les chunks avec chromadb et historise les metadonnées
    """
    collection_name = "UVSQ_DOCS"
    collection_path = "Agent/info_UVSQ/DBv"
    chunks, metadatas = load_chunks_from_json(json_path)
    print("Ingestion en cours...")
    embed_chunks_to_chroma(chunks, metadatas, collection_name, collection_path)
    print("Done")

json_path = r"Agent\info_UVSQ\data\uvsq_chunks.json"
ingest_to_chroma(json_path = json_path)