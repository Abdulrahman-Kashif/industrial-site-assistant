import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Paths
DATA_PATH = "data/"
DB_PATH = "db/"

def create_vector_db():
    print("--- 1. Loading Documents ---")
    # Load all .txt files from the data directory
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    print("--- 2. Splitting Text ---")
    # Split text into chunks of 500 characters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print("--- 3. Creating Embeddings & Vector DB ---")
    # Use a lightweight CPU model for embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create and save the Chroma database
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=DB_PATH
    )
    
    print(f"--- Finished! Vector DB saved to {DB_PATH} ---")

if __name__ == "__main__":
    # Ensure the db folder exists
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    
    create_vector_db()