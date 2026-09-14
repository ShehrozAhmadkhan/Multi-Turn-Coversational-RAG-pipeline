# Multi-Turn Conversational RAG Pipeline

This project implements a **multi-turn conversational RAG (Retrieval-Augmented Generation) pipeline** for answering questions from the FCCU 2026–27 academic catalog.

The pipeline allows users to ask questions about the catalog and continue the conversation using follow-up questions such as *"What are their credit hours?"* or *"What are the prerequisites for that course?"*

## Pipeline

The project follows these main steps:

**PDF → Text Extraction → Chunking → Embeddings → Pinecone → Retrieval → Query Reformulation → LLM → Answer**

### 1. Load PDF

* Uses **PyPDF2** to extract text from the FCCU 2026–27 catalog PDF.

### 2. Chunking

* Splits the extracted text into smaller word-based chunks.
* Uses a chunk size of **100 words** with **20 words overlap**.

### 3. Embeddings

* Generates vector embeddings using OpenAI's `text-embedding-3-small` model.
* Embeddings are generated in batches.

### 4. Vector Database

* Stores the generated vectors in **Pinecone**.
* Each vector contains the corresponding text as metadata.

### 5. Query Retrieval

* The user's question is converted into an embedding.
* Pinecone retrieves the most relevant chunks from the catalog.

### 6. Multi-Turn Query Reformulation

* Conversation history is used to understand follow-up questions.
* References such as **"their"**, **"that course"**, and similar expressions are reformulated into standalone search queries.

For example:

```text
User: What are the prerequisites for COMP 200?
Assistant: The prerequisites are COMP 111 and COMP 113.

User: What are their credit hours?

Reformulated query:
What are the credit hours of COMP 111 and COMP 113?
```

### 7. Answer Generation

* The retrieved context, current question, reformulated question, and conversation history are provided to **GPT-4o-mini**.
* The model generates the final answer based on the retrieved catalog information.

## Technologies Used

* Python
* OpenAI API
* Pinecone
* PyPDF2
* python-dotenv
* GPT-4o-mini
* `text-embedding-3-small`

## Project Structure

```text
Multi-Turn-Coversational-RAG-pipeline/
│
├── main.py
├── load_pdf_data.py
├── chunking.py
├── embeddings.py
├── upserting.py
├── catalog.pdf
└── .env
```

## Features

* PDF-based question answering
* Vector similarity search
* Multi-turn conversations
* Follow-up question understanding
* Query reformulation
* English and Roman Urdu support
* Context-based answer generation

## Note

This project is built as a learning project to understand how a **multi-turn conversational RAG pipeline** works from scratch using OpenAI and Pinecone.
