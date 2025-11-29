"""
Configuration management for the Medical RAG Chatbot
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    FLASK_APP: str = "app.py"
    FLASK_ENV: str = "development"
    SECRET_KEY: str = "default-secret-key-change-me"
    API_PORT: int = 5000
    
    # Local LLM Configuration
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"
    LLM_DEVICE: str = "cuda"
    LLM_MAX_LENGTH: int = 2048
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE: str = "chroma"
    VECTOR_STORE_PATH: str = "./data/vector_store"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # RAG Configuration
    TOP_K_RESULTS: int = 5
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 2000
    
    # Data Sources
    DATA_DIR: str = "./data/raw"
    PROCESSED_DATA_DIR: str = "./data/processed"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_ENABLED: bool = True
    CACHE_TTL: int = 3600  # Cache time-to-live in seconds (1 hour)
    
    # Conversation History
    MAX_HISTORY_PER_SESSION: int = 50
    HISTORY_DB: int = 1  # Use a separate Redis DB for history
    
    # Rate Limiting
    RATE_LIMIT: str = "100 per hour"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


def get_settings() -> Settings:
    """Get application settings"""
    return Settings()


def create_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        "data/raw",
        "data/processed",
        "data/vector_store",
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
