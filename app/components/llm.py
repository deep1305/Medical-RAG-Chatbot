from app.config.config import USE_OLLAMA, OLLAMA_LLM_MODEL, OLLAMA_BASE_URL, HF_TOKEN, HUGGINGFACE_REPO_ID
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm():
    """
    Load LLM based on USE_OLLAMA configuration.
    - If USE_OLLAMA=true: Use Ollama LLM (local)
    - If USE_OLLAMA=false: Use HuggingFace LLM (for Docker)
    """
    try:
        if USE_OLLAMA:
            logger.info(f"Loading Ollama LLM with model: {OLLAMA_LLM_MODEL}")
            from langchain_ollama import ChatOllama
            
            llm = ChatOllama(
                model=OLLAMA_LLM_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.7
            )
            logger.info("Ollama LLM loaded successfully")
            return llm
        else:
            logger.info(f"Loading HuggingFace LLM model: {HUGGINGFACE_REPO_ID}")
            from langchain_huggingface import HuggingFaceEndpoint
            
            if not HF_TOKEN:
                raise ValueError("HF_TOKEN is required for HuggingFace models")
            
            llm = HuggingFaceEndpoint(
                repo_id=HUGGINGFACE_REPO_ID,
                huggingfacehub_api_token=HF_TOKEN,
                temperature=0.3,
                max_new_tokens=512, 
                return_full_text=False
            )
            logger.info("HuggingFace LLM loaded successfully")
            return llm
    except Exception as e:
        error_message = CustomException("Failed to load LLM model", e)
        logger.error(str(error_message))
        raise error_message