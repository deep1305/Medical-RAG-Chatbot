import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.common.custom_exception import CustomException
from app.common.logger import get_logger
from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException("Data path does not exist", None)
        
        logger.info(f"Loading PDF files from {DATA_PATH}")
        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader) #It will only load the pdf files from the data path

        documents = loader.load()

        if not documents:
            logger.warning("No PDFs found in the data path")
        else:
            logger.info(f"Successfully loaded {len(documents)} documents from {DATA_PATH}")

        return documents
    
    except Exception as e:
        error_message = CustomException("Failed to load PDF files", e)
        logger.error(str(error_message))
        raise error_message
    

def create_text_chunks(documents):
    try:
        if not documents:
            raise CustomException("No documents to create text chunks", None)
        
        logger.info(f"Creating text chunks for {len(documents)} documents")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        text_chunks = text_splitter.split_documents(documents)

        logger.info(f"Successfully created {len(text_chunks)} text chunks")

        return text_chunks
    
    except Exception as e:
        error_message = CustomException("Failed to create text chunks", e)
        logger.error(str(error_message))
        raise error_message