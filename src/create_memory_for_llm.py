import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.logger import logging
from src.exception import CustomException

# Step 1: Load raw PDF(s)
DATA_PATH = Path("data/")
logging.info("loading pdf files")
def load_pdf_files(data):
    try:
        loader = DirectoryLoader(data,
                                glob='*.pdf',
                                loader_cls=PyPDFLoader)
        
        documents=loader.load()
        return documents
    except Exception as e:
        logging.error("Error in loading PDF files")
        raise CustomException(e,sys)

documents=load_pdf_files(data=DATA_PATH)
logging.info("loaded pdf files successfully")

# Step 2: Create Chunks
logging.info("Creating chunks")
def create_chunks(extracted_data):
    try:
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                    chunk_overlap=50)
        text_chunks=text_splitter.split_documents(extracted_data)
        return text_chunks
    
    except Exception as e:
        logging.error("Error in creating chunks")
        raise CustomException(e,sys)

text_chunks=create_chunks(extracted_data=documents)
logging.info("Chunks created successfully")

# Step 3: Create Vector Embeddings 
logging.info("Creating Vector Embeddings")
def get_embedding_model():
    try:
        embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return embedding_model
    except Exception as e:
        logging.error("Error in creating Vector Embeddings")
        raise CustomException(e,sys)

logging.info("Vector Embedding created successfully")
embedding_model=get_embedding_model()

# Step 4: Store embeddings in FAISS
logging.info("Storing embedding in FAISS")
try:
    DB_FAISS_PATH=Path("vectorstore/db_faiss")
    db=FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    logging.info("Embeddings stored successfully")
    
except Exception as e:
    logging.error("Error in storing embeddings")
    raise CustomException(e,sys)