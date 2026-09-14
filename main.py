from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone

from load_pdf_data import load_data
from chunking import word_based_chunking
from embeddings import batch_based_embeddings
from upserting import batch_based_upserting

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("multi-turn-rag-pipeline")

"""
data = load_data("catalog.pdf")
#print(data[:500])

chunks = word_based_chunking(data,100,20)
#print(chunks[0])

embeds = batch_based_embeddings(chunks)
#print(embeds[0])

upsert = batch_based_upserting(index,embeds,chunks)
print(upsert)

print(index.describe_index_stats())
"""

entry = 5
history = []

def save_entry(history,entry,user_question,assistant_answer):
    history.append({"role":"user","content":user_question})
    history.append({"role":"assistant","content":assistant_answer})

    max_chat = entry*2
    if len(history) > max_chat:
        history = history[-max_chat:]

    return history

while True:

    user_question = input("I am your helpfull Assistant! Ask questions related to fccu 26-27 catalog and to exit enter 'quit' : ")

    if user_question.lower() == "quit":
        break

    else:

        if len(history) == 0:

            try:
                user_question_embed1 = client.embeddings.create(model = "text-embedding-3-small", input = user_question, dimensions = 1024)
                user_question_vector1 = user_question_embed1.data[0].embedding

            except Exception as e:
                print(f"Error: {e}")
                continue

            try:
                vdb_query1 = index.query(vector = user_question_vector1, top_k = 5, include_metadata = True)

                if (len(vdb_query1.matches) == 0) or vdb_query1.matches[0].score < 0.3:
                    print("The question is out base knowledge/Domain! Ask question relevant to fccu 26-27 catalog.")
                    continue

                else:
                    context1 = []
                    for i in vdb_query1.matches:
                        context1.append(i.metadata["text"])

                    context1 = "\n".join(context1)

            except Exception as e:
                print(f"Error: {e}")
                continue

            try:
                prompt1 = f""" You are an helpfull assistant. You goal is to generate a well structured and clean answer using question and context. If the question is in english generate answer in englis, If the question is in roman urdu generate answer in roman urdu, you must not include hindi words!
                question : {user_question}
                context : {context1}"""

                llm1 = client.chat.completions.create(model = "gpt-4o-mini", messages = [{"role" : "user", "content" : prompt1}])
                answer1 = llm1.choices[0].message.content
                history = save_entry(history,entry,user_question,answer1)
                print(answer1)

            except Exception as e:
                print(f"Error: {e}")
                continue

        elif len(history) != 0:

            try:
                
                prompt2 = f"""You are a helpful assistant whose only job is to reformulate the user's current question into a clear, standalone search query.

                Use the conversation history to understand references in the current question.

                IMPORTANT RULE:
                If the current question contains a reference such as "their", "those", "these", "that", "it", "that course", "those courses", "in ke", or similar words, identify exactly what that reference refers to from the PREVIOUS ASSISTANT ANSWER.

                Do NOT automatically refer to the main course or subject mentioned in the previous USER QUESTION.

                For example:
                Previous user question: What are the prerequisites for COMP 200?
                Previous assistant answer: The prerequisites are COMP 111 and COMP 113.
                Current question: What are their credit hours?
                Correct reformulated query: What are the credit hours of COMP 111 and COMP 113?

                Another example:
                Previous user question: What is the course code for Operating Systems?
                Previous assistant answer: The course code is COMP 301.
                Current question: What are the prerequisites for that course?
                Correct reformulated query: What are the prerequisites for COMP 301?

                Preserve all important entities from the previous assistant answer when they are the subject of the current question.

                If the current question does not contain a reference, simply reformulate it without changing its meaning.

                If the question is in English, generate the reformulated query in English.
                If the question is in Roman Urdu, generate the reformulated query in Roman Urdu.
                Do not use Hindi words.

                Return ONLY the reformulated query. Do not answer the question.

                Current question: {user_question}
                Conversation history: {history}"""

                
                llm2 = client.chat.completions.create(model = "gpt-4o-mini", messages = [{"role" : "user", "content" : prompt2}])
                reformulated_query = llm2.choices[0].message.content

            except Exception as e:
                print(f"Error: {e}")
                continue

            try:
                response = client.embeddings.create(model = "text-embedding-3-small",input = reformulated_query,dimensions = 1024)
                reformulated_query_vector = response.data[0].embedding

            except Exception as e:
                print(f"Error: {e}")
                continue

            try:
                vdb_query2 = index.query(vector = reformulated_query_vector, top_k = 5, include_metadata = True)

                if (len(vdb_query2.matches) == 0) or vdb_query2.matches[0].score < 0.3:
                    print("The question is out base knowledge/Domain! Ask question relevant to fccu 26-27 catalog.")
                    continue

                else:
                    context2 = []
                    for i in vdb_query2.matches:
                        context2.append(i.metadata["text"])

                    context2 = "\n".join(context2)

            except Exception as e:
                print(f"Error: {e}")
                continue

            try:
                prompt3 = f"""You are an helpfull assistant. Your job is to generate a well structured and clean answer. If the question is in english generate answer in engliah, if the question is in roman urdu generate answer in roman urdu, you must no include hindi words!
                Answer only the specific question asked. Do not list unrelated courses, credit totals, or extra information unless the user asks for them. Use only the relevant information from the context.
                Use conversation history only to understand the current question. Do not answer previous questions or include information about other courses unless the current question asks for it. If the current question asks about a specific course, answer only about that course.
                question: {user_question}
                reformulated question: {reformulated_query}
                context: {context2}
                history: {history}"""

                llm3 = client.chat.completions.create(model = "gpt-4o-mini", messages = [{"role": "user", "content":prompt3}])
                answer2 = llm3.choices[0].message.content
                history = save_entry(history,entry,user_question,answer2)
                print(answer2)

            except Exception as e:
                print(f"Error: {e}")
                continue

