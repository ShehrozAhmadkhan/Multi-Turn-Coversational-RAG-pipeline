from pinecone import Pinecone
from dotenv import load_dotenv
import os
from load_pdf_data import load_data
from embeddings import batch_based_embeddings
from upserting import batch_based_upserting

load_dotenv()

def word_based_chunking(data,chunk_size,overlap):
    start = 0
    end = chunk_size
    step = chunk_size - overlap
    temp = data.split(" ")
    chunks = []

    while start < len(temp):
        chunk = temp[start:end]
        updated_chunk = " ".join(chunk)
        chunks.append(updated_chunk)

        start += step
        end += step

    return chunks


