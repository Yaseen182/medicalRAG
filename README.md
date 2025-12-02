# Medical RAG Chatbot Backend

Production-ready medical chatbot backend using RAG (Retrieval-Augmented Generation) with local Qwen3 4B model and LangChain.

## Features

- 🏥 **Medical Knowledge Base**: Combines MedQA, MedDialog, HealthSearchQA, and LiveQA Medical datasets
- 🤖 **Local LLM**: Uses Qwen2.5-3B-Instruct model (no API keys required)
- 🔍 **RAG Pipeline**: Intelligent retrieval with vector similarity search
- 🚀 **Production Ready**: Error handling, logging, and monitoring
- ⚡ **Fast**: FAISS vector store for efficient similarity search
- 🔒 **Secure**: No data sent to external APIs
- 💾 **Redis Caching**: Intelligent query caching for faster responses
- 📜 **Conversation History**: Track and retrieve past conversations per session
- 🔄 **Batch Processing**: Process multiple questions simultaneously

## Architecture

```
├── data/
│   ├── raw/              # Source datasets
│   ├── processed/        # Preprocessed data
│   └── vector_store/     # FAISS/Chroma index
├── logs/                 # Application logs
├── app.py               # Flask API server
├── rag_pipeline.py      # RAG implementation
├── data_loader.py       # Dataset loader
├── config.py            # Configuration management
└── logger.py            # Logging setup
```

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)
- 10GB+ disk space for models and data

## Installation

1. **Clone and navigate to project:**
```bash
cd d:\rag
```

2. **Create virtual environment:**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Setup environment variables:**
```bash
copy .env.example .env
# Edit .env with your settings
```

5. **Create necessary directories:**
```bash
mkdir data\raw, data\processed, data\vector_store, logs
```

## Configuration

Edit `.env` file:

```env
# API Configuration
FLASK_ENV=production
SECRET_KEY=your-secure-secret-key
API_PORT=5000

# Local LLM (Qwen3 4B)
LLM_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
LLM_DEVICE=cuda  # or 'cpu'
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Store
VECTOR_STORE_TYPE=faiss
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
```

## Usage

### 1. Load and Process Data

```bash
python data_loader.py
```

This downloads and preprocesses all medical datasets.

### 2. Build Vector Store

```bash
python rag_pipeline.py
```

This creates embeddings and builds the FAISS index.

### 3. Start API Server

```bash
python app.py
```

Server runs on `http://localhost:5000`

### For Production (with Gunicorn):

```bash
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 300 app:app
```

## API Endpoints

### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-29T10:00:00",
  "pipeline_initialized": true
}
```

### 2. Chat Query
```bash
POST /api/v1/chat
Content-Type: application/json

{
  "question": "What are the symptoms of diabetes?",
  "include_sources": true
}
```

Response:
```json
{
  "question": "What are the symptoms of diabetes?",
  "answer": "Based on medical knowledge, the main symptoms of diabetes include...",
  "sources": [...],
  "timestamp": "2025-11-29T10:00:00",
  "processing_time": 2.34
}
```

### 3. Similarity Search
```bash
POST /api/v1/search
Content-Type: application/json

{
  "query": "diabetes treatment",
  "top_k": 5
}
```

Response:
```json
{
  "query": "diabetes treatment",
  "results": [
    {
      "content": "...",
      "metadata": {...}
    }
  ],
  "count": 5,
  "timestamp": "2025-11-29T10:00:00"
}
```

### 4. Rebuild Index
```bash
POST /api/v1/rebuild-index
Content-Type: application/json

{
  "confirm": true
}
```

### 5. System Stats
```bash
GET /api/v1/stats
```

## Testing with cURL

```bash
# Health check
curl http://localhost:5000/health

# Chat query
curl -X POST http://localhost:5000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What are the symptoms of diabetes?\", \"include_sources\": true}"

# Search
curl -X POST http://localhost:5000/api/v1/search \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"diabetes treatment\", \"top_k\": 5}"
```

## Performance Optimization

### GPU Acceleration
- Requires CUDA-compatible GPU
- Set `LLM_DEVICE=cuda` in `.env`
- Significantly faster inference

### CPU Only
- Set `LLM_DEVICE=cpu` in `.env`
- Slower but works without GPU
- Requires more RAM

### Memory Optimization
- Use smaller chunk sizes for limited RAM
- Reduce `TOP_K_RESULTS` for faster queries
- Consider using quantized models


⚠️ **note:** It’s recommended to have Redis properly set up, as it is required for the caching layer to function correctly and improve the system’s response speed.


## Datasets

The chatbot combines four medical datasets:

1. **MedQA**: Medical licensing exam questions
2. **MedDialog**: Doctor-patient conversations
3. **HealthSearchQA**: Health search queries with answers
4. **LiveQA Medical**: Consumer health questions

## Monitoring

Logs are written to:
- Console: INFO level
- File (`logs/app.log`): DEBUG level (JSON format)

Monitor with:
```bash
tail -f logs/app.log
```

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use CPU instead of GPU
- Decrease chunk size

### Slow Inference
- Enable GPU acceleration
- Use smaller model
- Reduce `TOP_K_RESULTS`

### Model Download Issues
- Set `HF_HOME` environment variable
- Use mirror: `HF_ENDPOINT=https://hf-mirror.com`

## License

This project is for educational purposes. Please ensure compliance with dataset licenses.

## Support

For issues or questions, check the logs in `logs/app.log`
