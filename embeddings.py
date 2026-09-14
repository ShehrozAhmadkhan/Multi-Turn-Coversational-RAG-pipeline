from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def batch_based_embeddings(chunks, batches = 100):

    all_embeddings = []

    for i in range(0,len(chunks),batches):
        batch = chunks[i:i+batches]

        response = client.embeddings.create(model = "text-embedding-3-small", input = batch, dimensions = 1024)

        for i in response.data:
            all_embeddings.append(i.embedding)

    return all_embeddings