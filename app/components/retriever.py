from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import HUGGINGFACE_REPO_ID, OLLAMA_LLM_MODEL, OLLAMA_BASE_URL, HF_TOKEN


logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep your answer concise and relevant to the medical context provided.

Context: {context}

Chat History: {chat_history}

Question: {question}

Helpful Answer:"""

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE, 
        input_variables=["context", "chat_history", "question"]
    )


def create_qa_chain(chat_history=[]):
    try:
        logger.info("Loading vector store for context")
        db = load_vector_store()
        
        if db is None:
            raise CustomException("Vector store not found", None)
        
        llm = load_llm()

        if llm is None:
            raise CustomException("LLM not loaded", None)

        # Create memory for conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Add existing chat history to memory
        for i in range(0, len(chat_history), 2):
            if i + 1 < len(chat_history):
                memory.save_context(
                    {"question": chat_history[i]["content"]},
                    {"answer": chat_history[i+1]["content"]}
                )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": set_custom_prompt()}
        )
        
        logger.info("Conversational QA chain created successfully")

        return qa_chain
    except Exception as e:
        error_message = CustomException("Failed to create QA chain", e)
        logger.error(str(error_message))
        raise error_message

