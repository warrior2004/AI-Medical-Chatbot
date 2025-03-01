import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")

DB_FAISS_PATH = Path("vectorstore/db_faiss")

def get_vectorstore():
    """Load FAISS vector database with embeddings."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        print(f"Error loading FAISS database: {e}")
        return None

def set_custom_prompt():
    """Define a custom prompt for the chatbot."""
    custom_prompt_template = """
    Use the pieces of information provided in the context to answer the user's question.
    If you don’t know the answer, just say that you don’t know—don’t make up an answer.
    Only provide information from the given context.

    Context: {context}
    Question: {question}

    Start your answer directly. No small talk.
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, hf_token):
    """Load the Hugging Face model using HuggingFaceEndpoint (compatible with LangChain)."""
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            huggingfacehub_api_token=hf_token,
            temperature=0.5,
            max_length=512
        )
        return llm
    except Exception as e:
        print(f"Error initializing Hugging Face model: {e}")
        return None

@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests from users."""
    data = request.json
    user_prompt = data.get("prompt", "")

    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return jsonify({"error": "Failed to load the vector store"}), 500

        llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
        if llm is None:
            return jsonify({"error": "Failed to load the language model"}), 500
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,  # Correctly passing the LLM object
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )

        response = qa_chain.invoke({'query': user_prompt})
        result = response.get("result", "No response generated.")
        source_documents = response.get("source_documents", [])

        return jsonify({"result": result, "source_documents": str(source_documents)})

    except Exception as e:
        print(f"Error in chat processing: {e}")
        return jsonify({"error": str(e)}), 500

