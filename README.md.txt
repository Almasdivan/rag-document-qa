# RAG-Based Document Question Answering System

A beginner-friendly implementation of **Retrieval-Augmented Generation (RAG)** using LangChain, FAISS, Hugging Face embeddings, and an open-source Large Language Model (LLM).

This project demonstrates how to build a question-answering system that retrieves relevant information from external text documents and generates context-aware responses using an LLM.

## Overview

Traditional LLMs generate responses based on their pretrained knowledge. RAG enhances this process by retrieving relevant information from an external knowledge source before generating a response.

This implementation allows users to ask questions about the information available in `about_rag.txt`. The system retrieves relevant text chunks and passes them as context to the language model to generate answers.

## Architecture

1. **Document Loading:** Loads the knowledge source using LangChain's `TextLoader`.
2. **Text Chunking:** Splits documents into smaller chunks using `RecursiveCharacterTextSplitter`, with a chunk size of 300 and overlap of 50.
3. **Embedding Generation:** Converts text chunks into vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Vector Storage:** Stores embeddings in a FAISS vector database.
5. **Information Retrieval:** Retrieves relevant document chunks based on semantic similarity to the user's query.
6. **Context Preparation:** Combines the retrieved information with the user's question.
7. **Response Generation:** Uses the `Qwen/Qwen2.5-1.5B-Instruct` model to generate an answer based on the retrieved context.

## Tech Stack

- **Programming Language:** Python
- **RAG Framework:** LangChain
- **Vector Database:** FAISS
- **Embedding Model:** all-MiniLM-L6-v2
- **LLM:** Qwen2.5-1.5B-Instruct
- **Model Framework:** Hugging Face Transformers
- **Embedding Library:** Sentence Transformers

## Project Structure

```text
rag-document-qa/
│
├── data/
│   └── about_rag.txt
│
├── rag_ai.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Almasdivan/rag-document-qa.git
cd rag-document-qa
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your Knowledge Source

Place your text document inside the `data` directory:

```text
data/about_rag.txt
```

The application uses this document as its knowledge source.

### 5. Run the Application

```bash
python rag_ai.py
```

The Hugging Face model and embedding model may be downloaded automatically during the first execution.

## Example Query

```python
response = rag_chain("What is GCP?")
print(response)
```

The system retrieves relevant information from the document and generates a response using the retrieved context.

## Key Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Text Chunking and Overlapping
- Vector Embeddings
- FAISS Vector Search
- Context-Based LLM Response Generation
- LangChain Integration with Hugging Face Models

## Future Enhancements

- Support for PDF and multiple document formats.
- Implement conversational memory for multi-turn interactions.
- Build an interactive chatbot interface using Streamlit.
- Experiment with different embedding models and vector databases.
- Improve retrieval using similarity thresholds and reranking.

---

**Author:** Almasdivan K

**GitHub:** [Almasdivan](https://github.com/Almasdivan)