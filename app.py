import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA

# --- Page Config ---
st.set_page_config(
    page_title="Industrial Site Assistant",
    page_icon="🏗️",
    layout="centered"
)

# --- Header ---
st.title("🏗️ Industrial Site Assistant (KSA)")
st.caption("Powered by Local Llama 3 & RAG | Private & Secure")

# --- 1. Load Resources (Cached for speed) ---
@st.cache_resource
def load_resources():
    print("--- Loading Vector DB & Model ---")
    # Embedding Model (Must match the one used in ingestion)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the existing Database
    vector_db = Chroma(persist_directory="db/", embedding_function=embedding_model)
    
    # Initialize Local Llama 3
    llm = Ollama(model="llama3:8b-instruct-q4_K_M")
    
    # Create the RAG Chain (The "Brain" that connects them)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 2}), # Retrieve top 2 matching chunks
        return_source_documents=True
    )
    return qa_chain

# Initialize the chain
try:
    qa_chain = load_resources()
    st.success("✅ System Ready: Knowledge Base Loaded")
except Exception as e:
    st.error(f"Failed to load system: {e}")
    st.stop()

# --- 2. Chat Interface ---
# Initialize chat history if empty
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I have access to the Site Safety Manuals. Ask me anything."}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. Handle User Input ---
if prompt := st.chat_input("Ask about safety protocols or machinery..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching manuals..."):
            try:
                # Run the RAG Chain
                response = qa_chain.invoke({"query": prompt})
                result = response["result"]
                source_docs = response["source_documents"]
                
                # Display the answer
                st.markdown(result)
                
                # Show sources (Proof of work)
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Source {i+1}:** {doc.page_content[:200]}...")

                # Save assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": result})
            
            except Exception as e:
                st.error(f"An error occurred: {e}")