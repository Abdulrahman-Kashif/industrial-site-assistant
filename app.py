from langchain_community.llms import Ollama

# 1. Initialize the Local LLM (The Bridge)
print("--- Initializing Llama 3 ---")
llm = Ollama(model="llama3:8b-instruct-q4_K_M")

# 2. Define the Prompt
prompt = "You are an AI assistant for a Saudi industrial site. Introduce yourself in one sentence."

# 3. Invoke the Model
print("--- Sending Request to AI ---")
response = llm.invoke(prompt)

# 4. Print the Result
print(f"\nResponse:\n{response}")