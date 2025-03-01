import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name = "AI_Medical_Chatbot"

list_of_folders = ["vectorstore","data"]
list_of_files = [
    f"src/__init__.py",
    f"src/connect_memory_with_llm.py",
    f"src/create_memory_for_llm.py",
    f"src/logger.py",
    f"src/exception.py",
    f"templates/index.html",
    f"static/styles.css",
    f"medibot.py",
    f"setup.py",
    "Dockerfile",
    ".dockerignore",
    "wscgi.py"
]

for folder in list_of_folders:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        logging.info(f"Folder '{folder}' created successfully.")
    else:
        logging.info(f"Folder '{folder}' already exists.")

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"creating directory:{filedir} for the file {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
        logging.info(f"Creating empty file:{filepath}")

    else:
        logging.info(f"{filename} is already exists")