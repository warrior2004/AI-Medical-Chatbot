import os
import sys
from pathlib import Path
from langchain_huggingface import HuggingFaceEndpoint
from langchain.llms.base import LLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from src.logger import logging
from src.exception import CustomException
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, PrivateAttr

load_dotenv()

# Load environment variables
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

if not HF_TOKEN:
    logging.error("HF_TOKEN is not set. Please check your environment variables.")
    raise ValueError("HF_TOKEN is missing!")

# Step 1: Create a Custom LLM Wrapper for Hugging Face API
logging.info("Initializing Hugging Face LLM wrapper...")

class HuggingFaceLLM(LLM, BaseModel):
    model_id: str = Field(default=HUGGINGFACE_REPO_ID)
    token: str = Field(default=HF_TOKEN)
    temperature: float = Field(default=0.5)
    _client: InferenceClient = PrivateAttr()

    def __init__(self, model_id: str, token: str, temperature: float = 0.5):
        super().__init__()
        self._client = InferenceClient(model=model_id, token=token)
        self.temperature = temperature

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._client.text_generation(prompt, max_new_tokens=512, temperature=self.temperature)
        return response.strip()

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_id": self._client.model, "temperature": self.temperature}

    @property
    def _llm_type(self) -> str:
        return "huggingface_custom"
    
# Instantiate the LLM
llm = HuggingFaceLLM(model_id=HUGGINGFACE_REPO_ID, token=HF_TOKEN)
logging.info("LLM model initialized successfully!")

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

logging.info("Setting up custom prompt...")
def set_custom_prompt(custom_prompt_template):
    try:
        return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    except Exception as e:
        logging.error("Error in setting custom prompt")
        raise CustomException(e, sys)

custom_prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
logging.info("Custom prompt set successfully")

# Step 3: Load FAISS vector database
logging.info("Loading FAISS database...")
try:
    DB_FAISS_PATH = Path("vectorstore/db_faiss")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    logging.info("Successfully loaded FAISS database")
except Exception as e:
    logging.error("Error in loading FAISS database")
    raise CustomException(e, sys)

# Step 4: Create QA chain
logging.info("Creating QA chain...")
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': custom_prompt}
    )
    logging.info("QA chain created successfully")
except Exception as e:
    logging.error("Error in creating QA chain")
    raise CustomException(e, sys)

# Step 5: Process user query
try:
    user_query = input("Write your query here: ")
    response = qa_chain.invoke({'query': user_query})

    print("\nRESULT: ", response.get("result", "No result returned"))
    print("\nSOURCE DOCUMENTS: ", response.get("source_documents", "No source documents found"))

except Exception as e:
    logging.error("Error in processing user query")
    raise CustomException(e, sys)