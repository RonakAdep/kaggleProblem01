# EU AI Act — Hybrid RAG System

An end-to-end **Retrieval-Augmented Generation (RAG)** system for querying the **EU AI Act** using dense retrieval, sparse retrieval, hybrid search, cross-encoder reranking, and local LLM generation.

The project explores how combining multiple retrieval techniques can improve the quality and relevance of context provided to an LLM.

## 🚀 RAG Pipeline

```text
EU AI Act PDF
      │
      ▼
   Docling
      │
      ▼
Markdown Extraction & Cleaning
      │
      ▼
Hierarchical Chunking
      │
      ▼
Recursive Character Splitting
      │
      ▼
┌─────────────────────────────────────┐
│         Multiple Retrievers         │
│                                     │
│  Dense        BM25        SPLADE    │
│  BGE-M3       Sparse      Sparse    │
└─────────────────────────────────────┘
      │
      ▼
Reciprocal Rank Fusion (RRF)
      │
      ▼
Cross-Encoder Reranking
      │
      ▼
Top Relevant Chunks
      │
      ▼
Qwen3-1.7B
      │
      ▼
Final Answer
```

## 🧠 Components

### 1. Document Processing

The EU AI Act PDF is processed using **Docling** and exported as Markdown.

The extracted Markdown is cleaned by:

* Removing unnecessary escaped characters
* Removing excessive blank lines
* Cleaning malformed/fake Markdown tables
* Removing consecutive duplicate lines

### 2. Chunking

The document is first split according to its Markdown structure using:

* Chapter
* Article
* Section
* Subsection
* Sub-subsection

The resulting documents are then further divided using `RecursiveCharacterTextSplitter`.

```text
Chunk size:    1000
Chunk overlap: 200
```

This combines structural information from the document with smaller retrieval-friendly chunks.

### 3. Dense Retrieval

Dense embeddings are generated using:

```text
BAAI/bge-m3
```

The embeddings are stored in **ChromaDB** using an HNSW index with cosine similarity.

Configuration used:

```text
space:           cosine
ef_construction: 200
max_neighbors:   32
```

### 4. BM25 Retrieval

Traditional lexical retrieval is performed using BM25.

This helps retrieve documents containing important exact terms that semantic retrieval may not prioritize.

### 5. SPLADE Retrieval

Sparse neural retrieval is performed using:

```text
sparse-encoder-testing/splade-bert-tiny-nq-onnx
```

This provides another sparse retrieval signal alongside BM25.

### 6. Hybrid Search

The project combines three retrieval strategies:

```text
Dense Retrieval
      +
BM25 Retrieval
      +
SPLADE Retrieval
```

The results are combined using **Reciprocal Rank Fusion (RRF)**.

This allows documents that rank highly across different retrieval methods to receive stronger combined rankings.

### 7. Cross-Encoder Reranking

The top results from the hybrid retrieval stage are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

The cross-encoder evaluates the query-document pairs directly and produces a relevance score.

The highest-scoring chunks are then selected as the final context.

### 8. Generation

The final context is passed to a local language model:

```text
Qwen/Qwen3-1.7B
```

The model is instructed to answer using only the retrieved context.

This keeps the generation stage grounded in the source document.

## 🔍 Example Query

The notebook demonstrates the following type of question:

> If a company develops an AI model trained with 10^26 FLOPs and integrates it into a customer-facing chatbot sold in the EU, what obligations does the company face under the AI Act?

The query is passed through the complete retrieval pipeline:

```text
Query
 ↓
Dense Retrieval
 ↓
BM25
 ↓
SPLADE
 ↓
RRF Fusion
 ↓
Cross-Encoder Reranking
 ↓
Relevant Context
 ↓
Qwen3-1.7B
 ↓
Answer
```

## 🛠️ Technologies Used

* Python
* LangChain
* LangChain Docling
* Docling
* ChromaDB
* Sentence Transformers
* BGE-M3
* BM25
* SPLADE
* Reciprocal Rank Fusion (RRF)
* Cross-Encoder
* Hugging Face Transformers
* Qwen3-1.7B
* PyTorch

## 📦 Installation

Install the required packages:

```bash
pip install langchain langchain-core langchain-community
pip install langchain-docling
pip install docling
pip install langchain-text-splitters
pip install chromadb
pip install rank-bm25
pip install sentence-transformers
pip install transformers
pip install torch
```

## ▶️ Running the Project

The project is implemented as a Jupyter Notebook.

1. Clone the repository.

2. Install the required dependencies.

3. Place the EU AI Act PDF in the expected location.

4. Open:

```text
kagglepractice_(1).ipynb
```

5. Run the notebook cells sequentially.

> **Note:** The notebook currently uses a local/Colab-style file path for the EU AI Act PDF. Update the `FILE_PATH` variable according to your environment.

## 📁 Project Structure

```text
kaggleProblem01/
│
├── kagglepractice_(1).ipynb
├── README.md
└── .gitignore
```

## 🎯 Project Goal

The main goal of this project is to understand and implement the major components of a modern RAG system rather than relying on a single retrieval method.

The project covers:

```text
Document Ingestion
        ↓
Document Cleaning
        ↓
Chunking
        ↓
Embeddings
        ↓
Vector Search
        ↓
Sparse Search
        ↓
Hybrid Retrieval
        ↓
RRF
        ↓
Reranking
        ↓
Context Construction
        ↓
LLM Generation
```

## 🔮 Future Improvements

Possible improvements include:

* Add a proper evaluation dataset
* Measure retrieval performance using Recall@K and MRR
* Evaluate answer quality using RAG-specific metrics
* Experiment with different chunking strategies
* Compare different embedding models
* Optimize retrieval latency
* Add metadata-aware filtering
* Add a conversational interface
* Build an API around the RAG pipeline
* Add source citations to generated answers
* Move from notebook experimentation to a modular production pipeline

## 📌 Status

**Completed:** Core RAG pipeline

* Document processing
* Chunking
* Dense retrieval
* BM25 retrieval
* SPLADE retrieval
* Hybrid retrieval
* RRF
* Cross-encoder reranking
* Local LLM generation

This project is primarily an exploration and implementation of the individual components that make up a modern RAG pipeline.
