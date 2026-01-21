from langchain_text_splitters import MarkdownHeaderTextSplitter
import os

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

def md_chunker(md_file_path, headers_to_split_on=headers_to_split_on):
    # ouvre le markdown
    with open(md_file_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    # split et chunk le md
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    chunks = splitter.split_text(md_text)

    return chunks


def chunk_folder(folder_path: str):
    all_chunks = []
    metadatas = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            file_path = os.path.join(folder_path, filename)
            print(f"Chunking: {file_path}")
            chunks = md_chunker(file_path)

            for i, chunk in enumerate(chunks):
                # Si chunk est un Document (par ex. de langchain)
                all_chunks.append(chunk.page_content)
                metadatas.append({
                    "source": filename,
                    "chunk_id": i,
                    **chunk.metadata  # ajoute aussi les métadonnées du splitter, si existantes
                })

    return all_chunks, metadatas