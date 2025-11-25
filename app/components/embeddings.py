from app.config.config import USE_OLLAMA, OLLAMA_EMBEDDING_MODEL, OLLAMA_BASE_URL
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embedding_model():
    """
    Load embeddings based on USE_OLLAMA configuration.
    - If USE_OLLAMA=true: Use Ollama embeddings (local)
    - If USE_OLLAMA=false: Use HuggingFace embeddings (for Docker)
    """
    try:
        if USE_OLLAMA:
            logger.info(f"Loading Ollama embeddings with model: {OLLAMA_EMBEDDING_MODEL}")
            from langchain_ollama import OllamaEmbeddings
            
            embeddings = OllamaEmbeddings(
                model=OLLAMA_EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL
            )
            logger.info("Ollama embeddings loaded successfully")
            return embeddings
        else:
            logger.info(f"Loading HuggingFace embeddings model")
            from langchain_huggingface import HuggingFaceEmbeddings
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("HuggingFace embedding model loaded successfully")
            return embeddings
    except Exception as e:
        error_message = CustomException("Failed to load embedding model", e)
        logger.error(str(error_message))
        raise error_message