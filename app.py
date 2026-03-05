import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic.agents import create_react_agent
from langchain_classic.agents import AgentExecutor
from langchain_core.prompts import PromptTemplate

# --- Page Config ---
st.set_page_config(
    page_title="Industrial Site Assistant",
    page_icon="🏗️",
    layout="centered"
)

# --- Header ---
st.title("🏗️ Industrial Site Assistant (KSA)")
st.caption("Powered by Local Llama 3 | RAG + Math Agent | Private & Secure")

# --- 1. Define the Math Tool ---
@tool
def calculate_material_cost(input_string: str) -> str:
    """Useful for calculating the total financial cost of materials.
    Action Input must be a SINGLE string with two numbers separated by a comma: quantity, price.
    Example Action Input: 500, 15
    """
    try:
        cleaned = input_string.replace("quantity=", "").replace("price_per_unit=", "").strip()
        quantity_str, price_str = cleaned.split(",")
        total = float(quantity_str.strip()) * float(price_str.strip())
        return f"The exact calculated cost is {total} SAR."
    except Exception as e:
        return "Error: Tell the user you need exactly two numbers separated by a comma."

# --- 2. Load Resources & Initialize Agent ---
@st.cache_resource
def load_agent():
    print("--- Booting up Najm Agent ---")
    
    # A. Setup the Brain (LLM)
    llm = Ollama(model="llama3:8b-instruct-q4_K_M")
    
    # B. Setup the Memory (Vector DB)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="db/", embedding_function=embedding_model)
    
    # C. Create the Retriever Tool (Turns RAG into a tool the Agent can use)
    retriever_tool = create_retriever_tool(
        vector_db.as_retriever(search_kwargs={"k": 2}),
        name="search_safety_manuals",
        description="Searches and returns excerpts from the industrial safety and machinery manuals. Use this when the user asks about protocols, specs, or rules."
    )
    
    # D. Give the Agent its Toolbox
    tools = [calculate_material_cost, retriever_tool]
    
    # E. The Agent's Internal Monologue Prompt
    template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following strict format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the exact inputs to the action separated by a comma
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''
    
    prompt = PromptTemplate.from_template(template)
    
    # F. Assemble the Executor
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return agent_executor

# Initialize the Agent
try:
    agent_executor = load_agent()
    st.success("✅ System Ready: Najm Agent Online")
except Exception as e:
    st.error(f"Failed to load system: {e}")
    st.stop()

# --- 3. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am Najm. I can read the site safety manuals OR calculate material costs. How can I help?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Handle User Input ---
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Najm is thinking... (Check VS Code terminal to see my thoughts)"):
            try:
                # Invoke the Agent!
                response = agent_executor.invoke({"input": prompt})
                result = response["output"]
                
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            
            except Exception as e:
                st.error(f"An error occurred: {e}")