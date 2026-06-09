# ingest.py — loads Aman's profile into ChromaDB

import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"
BATCH_SIZE = 20


def ingest():
    print("Loading profile data...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print("No documents found in data/. Make sure aman.md exists.")
        return

    print(f"Loaded {len(documents)} document(s). Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"Embedding batch {i // BATCH_SIZE + 1}/{(len(chunks) - 1) // BATCH_SIZE + 1}...")
        if db is None:
            db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_PATH,
            )
        else:
            db.add_documents(batch)
        time.sleep(1)

    print(f"Done. {len(chunks)} chunks saved to {CHROMA_PATH}/")


if __name__ == "__main__":
    ingest()
