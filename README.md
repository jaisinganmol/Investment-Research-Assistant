# Investment Research Assistant

A Retrieval-Augmented Generation (RAG) system for intelligent Q&A over PDF documents using HuggingFace embeddings, ChromaDB vector storage, and OpenAI's GPT-4.

## Features

- **Advanced Embeddings**: BAAI/bge-large-en-v1.5 model for high-quality semantic representations
- **Optimized Vector Search**: ChromaDB with tuned HNSW indexing for fast similarity search
- **Smart Document Processing**: Recursive text splitting with overlap for context preservation
- **Persistent Storage**: Vector database persists to disk for reuse across sessions

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install sentence-transformers langchain-community langchain-text-splitters \
            chromadb openai python-dotenv pypdf ipykernel pydantic-settings
```

## Requirements

```
sentence-transformers
langchain-community
langchain-text-splitters
chromadb
openai
python-dotenv
pypdf
ipykernel
pydantic-settings
```

## Configuration

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Jupyter Notebook Mode

```bash
jupyter notebook
```

Open `rag.ipynb` and run blocks sequentially (2-5 minutes total):

1. Setup & API Key validation
2. Load embedding model (~500MB download first time)
3. Define embedding function
4. Load & chunk PDF document
5. Generate embeddings for all chunks
6. Initialize ChromaDB
7. Create vector collection
8. Insert documents into database
9. Define retrieval function
10. Run example RAG query

### Python Script Mode

```python
from rag import rag_query

# Ask questions about your documents
answer = rag_query("What are the main findings in the report?")
print(answer)
```

### Using Your Own Documents

```python
# Remote PDF
loader = PyPDFLoader("https://example.com/document.pdf")

# Local PDF
loader = PyPDFLoader("./data/your_document.pdf")
```

## Project Structure

```
rag-system/
├── rag.ipynb               # Jupyter notebook
├── rag.py                  # Python script version
├── .env                    # API keys
├── pyproject.toml          # Project dependencies
├── README.md
└── chroma_db_data/         # Vector database storage (auto-created)
```

## Architecture

```
User Query
    |
    v
+------------------+
|  Embed Query     |  (HuggingFace)
+------------------+
    |
    v
+------------------+
|  Vector Search   |  (ChromaDB)
+------------------+
    |
    v
+------------------+
| Retrieve Context |  (Top-K Results)
+------------------+
    |
    v
+------------------+
| Generate Answer  |  (GPT-4)
+------------------+
    |
    v
+------------------+
|  Final Response  |
+------------------+
```

## License

MIT Open Source License
