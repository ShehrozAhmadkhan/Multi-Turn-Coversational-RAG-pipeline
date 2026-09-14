from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("multi-turn-rag-pipeline")


def batch_based_upserting(index,embeds,chunks,batches = 100):
    optimized_vectors = []

    for i in range(len(embeds)):
        optimized_vectors.append({"id" : f"chunk-{i}",
                                  "values" : embeds[i],
                                  "metadata" : {"text" : chunks[i]}})

    for i in range(0,len(optimized_vectors),batches):
        batch = optimized_vectors[i:i+batches]
        index.upsert(vectors = batch)

    return "Done"



    

