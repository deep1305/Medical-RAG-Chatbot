import os
from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)


def load_vector_store():
    try:
        embedding_model = get_embedding_model()
        if os.path.exists(DB_FAISS_PATH):
            logger.info(f"Loading existing vector store from {DB_FAISS_PATH}")
            return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        
        else:
            logger.warning(f"Vector store not found at {DB_FAISS_PATH}, creating new one")
    
    except Exception as e:
        error_message = CustomException("Failed to load vector store", e)
        logger.error(str(error_message))
        raise error_message

#For creating a new vector store
def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks were found", None)

        logger.info("Generating a new vector store")

        embedding_model = get_embedding_model()

        db = FAISS.from_documents(text_chunks, embedding_model)

        logger.info("Saving the vector store to the database")

        db.save_local(DB_FAISS_PATH)

        logger.info("Vector store saved successfully")

        return db
    
    except Exception as e:
        error_message = CustomException("Failed to create new vector store", e)
        logger.error(str(error_message))
        raise error_message