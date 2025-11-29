"""
Flask API for Medical RAG Chatbot
Production-ready REST API with error handling, logging, and caching
"""
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import logging
from typing import Dict, Any, List, Optional
import traceback
from functools import wraps
import time
from datetime import datetime
import hashlib
import json
import uuid
import redis
from concurrent.futures import ThreadPoolExecutor

from config import get_settings, create_directories
from logger import setup_logging, get_logger
from rag_pipeline import MedicalRAGPipeline, initialize_rag_pipeline
from data_loader import MedicalDataLoader, prepare_documents_for_rag

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
config = get_settings()
app.config['SECRET_KEY'] = config.SECRET_KEY

# Setup logging
setup_logging(config.LOG_LEVEL, config.LOG_FILE)
logger = get_logger(__name__)

# Create necessary directories
create_directories()

# Global RAG pipeline instance
rag_pipeline: MedicalRAGPipeline = None

# Redis clients
redis_cache_client: Optional[redis.Redis] = None
redis_history_client: Optional[redis.Redis] = None

# Thread pool for batch processing
executor = ThreadPoolExecutor(max_workers=4)


def initialize_redis():
    """Initialize Redis connections"""
    global redis_cache_client, redis_history_client
    
    if not config.REDIS_ENABLED:
        logger.info("Redis is disabled")
        return
    
    try:
        logger.info("Connecting to Redis...")
        redis_cache_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            password=config.REDIS_PASSWORD,
            decode_responses=True
        )
        redis_cache_client.ping()
        
        redis_history_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.HISTORY_DB,
            password=config.REDIS_PASSWORD,
            decode_responses=True
        )
        redis_history_client.ping()
        
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Continuing without caching.")
        redis_cache_client = None
        redis_history_client = None


def initialize_pipeline():
    """Initialize RAG pipeline on startup"""
    global rag_pipeline
    
    try:
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = initialize_rag_pipeline(config)
        
        # Try to load existing vector store
        try:
            rag_pipeline.load_vector_store(store_type=config.VECTOR_STORE_TYPE)
            logger.info("Vector store loaded successfully")
        except FileNotFoundError:
            logger.warning("No existing vector store found. Building from scratch...")
            
            # Load and process data
            loader = MedicalDataLoader(config.DATA_DIR, config.PROCESSED_DATA_DIR)
            documents = loader.load_processed_data()
            prepared_docs = prepare_documents_for_rag(documents)
            
            # Create vector store
            chunked_docs = rag_pipeline.chunk_documents(prepared_docs)
            rag_pipeline.create_vector_store(chunked_docs, store_type=config.VECTOR_STORE_TYPE)
            
            logger.info("Vector store created successfully")
        
        logger.info("RAG pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        logger.error(traceback.format_exc())
        raise


def error_handler(f):
    """Decorator for error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": str(e),
                "message": "An error occurred processing your request"
            }), 500
    return decorated_function


def timing_decorator(f):
    """Decorator to log execution time"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{f.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return decorated_function


def get_cache_key(question: str) -> str:
    """Generate cache key from question"""
    return f"cache:{hashlib.md5(question.lower().strip().encode()).hexdigest()}"


def get_from_cache(question: str) -> Optional[Dict[str, Any]]:
    """Get cached response"""
    if not redis_cache_client:
        return None
    
    try:
        cache_key = get_cache_key(question)
        cached = redis_cache_client.get(cache_key)
        if cached:
            logger.info(f"Cache hit for question: {question[:50]}...")
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Cache retrieval error: {e}")
    
    return None


def set_cache(question: str, result: Dict[str, Any]):
    """Cache response"""
    if not redis_cache_client:
        return
    
    try:
        cache_key = get_cache_key(question)
        redis_cache_client.setex(
            cache_key,
            config.CACHE_TTL,
            json.dumps(result)
        )
        logger.info(f"Cached response for: {question[:50]}...")
    except Exception as e:
        logger.warning(f"Cache storage error: {e}")


def get_session_id() -> str:
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


def save_to_history(session_id: str, question: str, answer: str, sources: List = None):
    """Save conversation to history"""
    if not redis_history_client:
        return
    
    try:
        history_key = f"history:{session_id}"
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources or []
        }
        
        redis_history_client.lpush(history_key, json.dumps(message))
        redis_history_client.ltrim(history_key, 0, config.MAX_HISTORY_PER_SESSION - 1)
        redis_history_client.expire(history_key, 86400 * 7)  # 7 days
        
        logger.info(f"Saved to history for session: {session_id[:8]}...")
    except Exception as e:
        logger.warning(f"History storage error: {e}")


def get_history(session_id: str, limit: int = 10) -> List[Dict]:
    """Get conversation history"""
    if not redis_history_client:
        return []
    
    try:
        history_key = f"history:{session_id}"
        messages = redis_history_client.lrange(history_key, 0, limit - 1)
        return [json.loads(msg) for msg in messages]
    except Exception as e:
        logger.warning(f"History retrieval error: {e}")
        return []


def process_single_query(question: str, include_sources: bool = False) -> Dict[str, Any]:
    """Process a single query (used for batch processing)"""
    try:
        # Check cache
        cached_result = get_from_cache(question)
        if cached_result:
            return {
                "question": question,
                "answer": cached_result["answer"],
                "sources": cached_result.get("sources", []) if include_sources else [],
                "cached": True,
                "processing_time": 0
            }
        
        # Query RAG pipeline
        start_time = time.time()
        result = rag_pipeline.query(question)
        processing_time = time.time() - start_time
        
        # Cache result
        set_cache(question, result)
        
        return {
            "question": question,
            "answer": result["answer"],
            "sources": result.get("source_documents", []) if include_sources else [],
            "cached": False,
            "processing_time": round(processing_time, 2)
        }
    except Exception as e:
        logger.error(f"Error processing query '{question}': {e}")
        return {
            "question": question,
            "error": str(e),
            "processing_time": 0
        }


@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline_initialized": rag_pipeline is not None
    }), 200


@app.route('/api/v1/chat', methods=['POST'])
@error_handler
@timing_decorator
def chat():
    """
    Main chat endpoint for medical queries
    
    Request body:
    {
        "question": "What are the symptoms of diabetes?",
        "include_sources": true  (optional, default: false)
    }
    
    Response:
    {
        "question": "...",
        "answer": "...",
        "sources": [...],  (if include_sources=true)
        "timestamp": "...",
        "processing_time": 1.23
    }
    """
    if rag_pipeline is None:
        return jsonify({
            "error": "RAG pipeline not initialized",
            "message": "The system is still initializing. Please try again in a moment."
        }), 503
    
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({
            "error": "Missing required field",
            "message": "Please provide a 'question' in the request body"
        }), 400
    
    question = data['question']
    include_sources = data.get('include_sources', False)
    
    if not question or not question.strip():
        return jsonify({
            "error": "Empty question",
            "message": "Question cannot be empty"
        }), 400
    
    logger.info(f"Received question: {question[:100]}...")
    
    # Get session ID
    session_id = get_session_id()
    
    # Check cache first
    cached_result = get_from_cache(question)
    if cached_result:
        response = {
            "question": question,
            "answer": cached_result["answer"],
            "sources": cached_result.get("source_documents", []) if include_sources else [],
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": 0,
            "cached": True
        }
        
        # Save to history
        save_to_history(session_id, question, cached_result["answer"], 
                       cached_result.get("source_documents", []) if include_sources else None)
        
        return jsonify(response), 200
    
    start_time = time.time()
    
    # Query RAG pipeline
    result = rag_pipeline.query(question)
    
    processing_time = time.time() - start_time
    
    # Cache the result
    set_cache(question, result)
    
    # Prepare response
    response = {
        "question": result["question"],
        "answer": result["answer"],
        "timestamp": datetime.utcnow().isoformat(),
        "processing_time": round(processing_time, 2),
        "cached": False
    }
    
    if include_sources:
        response["sources"] = result["source_documents"]
    
    # Save to history
    save_to_history(session_id, question, result["answer"], 
                   result["source_documents"] if include_sources else None)
    
    logger.info(f"Query processed successfully in {processing_time:.2f}s")
    
    return jsonify(response), 200


@app.route('/api/v1/search', methods=['POST'])
@error_handler
@timing_decorator
def search():
    """
    Similarity search endpoint (without LLM generation)
    
    Request body:
    {
        "query": "diabetes symptoms",
        "top_k": 5  (optional, default: 5)
    }
    
    Response:
    {
        "query": "...",
        "results": [...],
        "count": 5,
        "timestamp": "..."
    }
    """
    if rag_pipeline is None:
        return jsonify({
            "error": "RAG pipeline not initialized"
        }), 503
    
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({
            "error": "Missing required field",
            "message": "Please provide a 'query' in the request body"
        }), 400
    
    query = data['query']
    top_k = data.get('top_k', 5)
    
    if not query or not query.strip():
        return jsonify({
            "error": "Empty query"
        }), 400
    
    if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
        return jsonify({
            "error": "Invalid top_k",
            "message": "top_k must be an integer between 1 and 20"
        }), 400
    
    logger.info(f"Received search query: {query[:100]}...")
    
    # Perform similarity search
    results = rag_pipeline.similarity_search(query, k=top_k)
    
    response = {
        "query": query,
        "results": results,
        "count": len(results),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return jsonify(response), 200


@app.route('/api/v1/rebuild-index', methods=['POST'])
@error_handler
def rebuild_index():
    """
    Rebuild vector store index
    Admin endpoint to rebuild the index from source data
    
    Request body:
    {
        "confirm": true
    }
    """
    data = request.get_json()
    
    if not data or not data.get('confirm'):
        return jsonify({
            "error": "Confirmation required",
            "message": "Set 'confirm': true to rebuild the index"
        }), 400
    
    logger.info("Starting index rebuild...")
    
    try:
        # Load and process data
        loader = MedicalDataLoader(config.DATA_DIR, config.PROCESSED_DATA_DIR)
        documents = loader.load_all_datasets()
        prepared_docs = prepare_documents_for_rag(documents)
        
        # Create new vector store
        global rag_pipeline
        if rag_pipeline is None:
            rag_pipeline = initialize_rag_pipeline(config)
        
        chunked_docs = rag_pipeline.chunk_documents(prepared_docs)
        rag_pipeline.create_vector_store(chunked_docs, store_type=config.VECTOR_STORE_TYPE)
        
        # Setup QA chain
        rag_pipeline.setup_qa_chain()
        
        logger.info("Index rebuilt successfully")
        
        return jsonify({
            "message": "Index rebuilt successfully",
            "documents_processed": len(documents),
            "chunks_created": len(chunked_docs),
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        raise


@app.route('/api/v1/history', methods=['GET'])
@error_handler
def get_conversation_history():
    """
    Get conversation history for current session
    
    Query params:
        limit: Number of messages to retrieve (default: 10)
    
    Response:
    {
        "session_id": "...",
        "messages": [...],
        "count": 10
    }
    """
    session_id = get_session_id()
    limit = request.args.get('limit', 10, type=int)
    
    if limit < 1 or limit > 100:
        return jsonify({
            "error": "Invalid limit",
            "message": "Limit must be between 1 and 100"
        }), 400
    
    history = get_history(session_id, limit)
    
    return jsonify({
        "session_id": session_id,
        "messages": history,
        "count": len(history),
        "timestamp": datetime.utcnow().isoformat()
    }), 200


@app.route('/api/v1/history', methods=['DELETE'])
@error_handler
def clear_history():
    """
    Clear conversation history for current session
    
    Response:
    {
        "message": "History cleared",
        "session_id": "..."
    }
    """
    if not redis_history_client:
        return jsonify({
            "error": "History not available",
            "message": "Redis is not enabled"
        }), 503
    
    session_id = get_session_id()
    history_key = f"history:{session_id}"
    
    try:
        redis_history_client.delete(history_key)
        logger.info(f"Cleared history for session: {session_id[:8]}...")
        
        return jsonify({
            "message": "History cleared successfully",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise


@app.route('/api/v1/batch', methods=['POST'])
@error_handler
def batch_process():
    """
    Process multiple questions in batch
    
    Request body:
    {
        "questions": ["question 1", "question 2", ...],
        "include_sources": false
    }
    
    Response:
    {
        "results": [...],
        "total": 5,
        "successful": 5,
        "failed": 0,
        "total_time": 10.5,
        "timestamp": "..."
    }
    """
    if rag_pipeline is None:
        return jsonify({
            "error": "RAG pipeline not initialized"
        }), 503
    
    data = request.get_json()
    
    if not data or 'questions' not in data:
        return jsonify({
            "error": "Missing required field",
            "message": "Please provide 'questions' array in the request body"
        }), 400
    
    questions = data['questions']
    include_sources = data.get('include_sources', False)
    
    if not isinstance(questions, list):
        return jsonify({
            "error": "Invalid format",
            "message": "'questions' must be an array"
        }), 400
    
    if len(questions) == 0:
        return jsonify({
            "error": "Empty questions array"
        }), 400
    
    if len(questions) > 20:
        return jsonify({
            "error": "Too many questions",
            "message": "Maximum 20 questions per batch"
        }), 400
    
    logger.info(f"Processing batch of {len(questions)} questions")
    
    start_time = time.time()
    
    # Process questions in parallel
    results = list(executor.map(
        lambda q: process_single_query(q, include_sources),
        questions
    ))
    
    total_time = time.time() - start_time
    
    successful = sum(1 for r in results if 'error' not in r)
    failed = len(results) - successful
    
    response = {
        "results": results,
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "total_time": round(total_time, 2),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    logger.info(f"Batch processed: {successful} successful, {failed} failed in {total_time:.2f}s")
    
    return jsonify(response), 200


@app.route('/api/v1/cache/clear', methods=['POST'])
@error_handler
def clear_cache():
    """
    Clear all cached responses
    Admin endpoint
    
    Response:
    {
        "message": "Cache cleared",
        "keys_deleted": 42
    }
    """
    if not redis_cache_client:
        return jsonify({
            "error": "Cache not available",
            "message": "Redis is not enabled"
        }), 503
    
    try:
        # Find all cache keys
        keys = redis_cache_client.keys("cache:*")
        count = 0
        
        if keys:
            count = redis_cache_client.delete(*keys)
        
        logger.info(f"Cleared {count} cache entries")
        
        return jsonify({
            "message": "Cache cleared successfully",
            "keys_deleted": count,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise


@app.route('/api/v1/cache/stats', methods=['GET'])
@error_handler
def cache_stats():
    """
    Get cache statistics
    
    Response:
    {
        "cache_enabled": true,
        "cached_items": 42,
        "memory_usage": "2.5 MB"
    }
    """
    if not redis_cache_client:
        return jsonify({
            "cache_enabled": False,
            "message": "Redis is not enabled"
        }), 200
    
    try:
        keys = redis_cache_client.keys("cache:*")
        info = redis_cache_client.info('memory')
        
        return jsonify({
            "cache_enabled": True,
            "cached_items": len(keys),
            "memory_usage_bytes": info.get('used_memory', 0),
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise


@app.route('/api/v1/stats', methods=['GET'])
@error_handler
def get_stats():
    """
    Get system statistics
    
    Response:
    {
        "pipeline_status": "initialized",
        "vector_store_path": "...",
        "model_name": "...",
        "timestamp": "..."
    }
    """
    cache_enabled = redis_cache_client is not None
    history_enabled = redis_history_client is not None
    
    stats = {
        "pipeline_status": "initialized" if rag_pipeline else "not_initialized",
        "vector_store_path": config.VECTOR_STORE_PATH,
        "model_name": config.LLM_MODEL_NAME,
        "embedding_model": config.EMBEDDING_MODEL,
        "device": config.LLM_DEVICE,
        "top_k_results": config.TOP_K_RESULTS,
        "cache_enabled": cache_enabled,
        "history_enabled": history_enabled,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return jsonify(stats), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint does not exist"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500


if __name__ == '__main__':
    logger.info("Starting Medical RAG Chatbot API...")
    logger.info(f"Configuration: {config.dict()}")
    
    # Initialize Redis
    initialize_redis()
    
    # Initialize pipeline before starting server
    initialize_pipeline()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=config.API_PORT,
        debug=(config.FLASK_ENV == 'development')
    )
