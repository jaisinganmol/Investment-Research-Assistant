# Investment Research Assistant using ChromaDB

A complete Retrieval-Augmented Generation (RAG) system using HuggingFace embeddings, ChromaDB vector storage, and OpenAI's GPT-4 for intelligent document Q&A.

## Overview

This project demonstrates a production-ready RAG pipeline that:
- Loads and processes PDF documents
- Generates semantic embeddings using HuggingFace models
- Stores vectors in ChromaDB for efficient similarity search
- Retrieves relevant context for user queries
- Generates accurate answers using GPT-4

## Features

- **Advanced Embeddings** - BAAI/bge-large-en-v1.5 model for high-quality vector representations
- **Optimized Vector Search** - ChromaDB with tuned HNSW indexing for fast retrieval
- **Smart Chunking** - Recursive text splitting with overlap for context preservation
- **Clean Architecture** - Modular functions for easy customization and extension
- **Persistent Storage** - Vector database saved to disk for reuse

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd rag-pipeline

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install sentence-transformers
pip install langchain-community
pip install langchain-text-splitters
pip install chromadb
pip install openai
pip install python-dotenv
pip install pypdf
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running the Jupyter Notebook

1. **Start Jupyter Notebook**
```bash
jupyter notebook
```

2. **Open `rag.ipynb`** in your browser

3. **Run blocks sequentially** (Shift + Enter for each cell):

   - **Block 1**: Setup & API Key validation
   - **Block 2**: Load embedding model (~500MB download first time)
   - **Block 3**: Define embedding function
   - **Block 4**: Load & chunk PDF document (10-30 seconds)
   - **Block 5**: Generate embeddings for all chunks (30-60 seconds)
   - **Block 6**: Initialize ChromaDB
   - **Block 7**: Create vector collection
   - **Block 8**: Insert documents into database
   - **Block 9**: Define retrieval function
   - **Block 10**: Run example RAG query

4. **Expected total time**: 2-5 minutes for complete pipeline

### Quick Start (Python Script)

```python
# Run the complete pipeline
python rag.py
```

### Custom Queries

```python
from rag import rag_query

# Ask questions about your documents
answer = rag_query("What are the main findings in the report?")
print(answer)
```

### Using Your Own Documents

Replace the PDF URL in the code:

```python
loader = PyPDFLoader("path/to/your/document.pdf")
```

Or load local files:

```python
loader = PyPDFLoader("./data/your_document.pdf")
```

## Architecture

```
┌─────────────────────────────────────────────┐
│            User Query                       │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────▼─────────┐
         │  Embed Query      │
         │  (HuggingFace)    │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │  Vector Search    │
         │  (ChromaDB)       │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │ Retrieve Context  │
         │ (Top-K Results)   │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │  Generate Answer  │
         │  (GPT-4)          │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │  Final Response   │
         └───────────────────┘
```

## Project Structure

```
rag-pipeline/
├── rag.ipynb               # Jupyter notebook (run block by block)
├── rag.py                  # Python script version
├── .env                    # API keys (gitignored)
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── chroma_db_data/        # Vector database storage (auto-created)
```

## Configuration Options

### Chunking Parameters

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,        # Characters per chunk
    chunk_overlap=20       # Overlap between chunks
)
```

### Retrieval Settings

```python
def get_retrieved_context(query_text, n_results=5):
    # n_results: Number of similar chunks to retrieve
```

### ChromaDB Index Tuning

```python
metadata={
    "hnsw:space": "cosine",           # Distance metric
    "hnsw:construction_ef": 200,       # Index build quality
    "hnsw:search_ef": 100,            # Search quality
    "hnsw:M": 16                      # Graph connectivity
}
```

## API Costs

**Embeddings**: Free (HuggingFace model runs locally)

**OpenAI API**:
- GPT-4o: ~$5 per 1M input tokens, ~$15 per 1M output tokens
- Typical query: $0.01-0.05 per question

## Performance

- **Document Processing**: ~2-5 seconds per page
- **Embedding Generation**: ~0.1 seconds per chunk (CPU)
- **Vector Search**: <100ms for thousands of documents
- **End-to-End Query**: 2-5 seconds

## Troubleshooting

### Running in Jupyter Notebook

**Block fails with import error**
```bash
# Install missing package in notebook cell
!pip install <package-name>
```

**"Kernel died" or memory error**
```python
# Reduce chunk size or process in smaller batches
chunk_size=200
# Or restart kernel: Kernel → Restart & Clear Output
```

**ChromaDB already exists warning**
```python
# Delete existing database to start fresh
!rm -rf chroma_db_data/
# Then re-run from Block 6
```

**Want to re-run with different PDF**
- Change URL in Block 4
- Delete `chroma_db_data/` folder
- Re-run from Block 4 onward

### General Issues

### "OPENAI_API_KEY not found"
Ensure `.env` file exists with valid API key:
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### ChromaDB Permission Errors
```bash
# Delete and recreate the database
rm -rf chroma_db_data/
# Re-run notebook from Block 6
```

### Out of Memory Errors
Reduce chunk size or batch processing:
```python
chunk_size=200  # Smaller chunks
```

### Slow Embedding Generation
Use GPU acceleration:
```bash
pip install sentence-transformers[cuda]
```

## Advanced Usage

### Batch Processing

```python
queries = [
    "What is the company's revenue?",
    "Who are the key executives?",
    "What are the main risks?"
]

for query in queries:
    answer = rag_query(query)
    print(f"Q: {query}\nA: {answer}\n")
```

### Custom Embedding Models

```python
# Use a different model
embedding_model = SentenceTransformer('all-mpnet-base-v2')
```

### Multi-Document Support

```python
documents = []
for pdf_path in ["doc1.pdf", "doc2.pdf", "doc3.pdf"]:
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())
```

## Limitations

- Requires OpenAI API access (paid service)
- Context window limited by LLM (GPT-4: ~128k tokens)
- First-run downloads ~500MB embedding model
- Accuracy depends on document quality and chunking strategy

## Future Enhancements

- [ ] Support for multiple document formats (DOCX, TXT, HTML)
- [ ] Hybrid search (keyword + semantic)
- [ ] Answer citation with source page numbers
- [ ] Web UI with Streamlit/Gradio
- [ ] Conversation memory for follow-up questions
- [ ] Support for Claude, Gemini, and other LLMs

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - feel free to use in your own projects

## Acknowledgments

- **HuggingFace** - BAAI/bge-large-en-v1.5 embedding model
- **ChromaDB** - High-performance vector database
- **LangChain** - Document processing utilities
- **OpenAI** - GPT-4 language model

## Support

For issues or questions:
- Open a GitHub issue
- Email: your-email@example.com
- Documentation: [Link to docs]

---

**Built with ❤️ using modern RAG techniques**