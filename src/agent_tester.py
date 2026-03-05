from langchain_community.llms import Ollama
from langchain.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# --- 1. Define the Tool ---
# The @tool decorator tells LangChain this function can be used by the AI.
# The docstring (text inside """) is CRITICAL. The AI reads it to know when to use the tool.
@tool
def calculate_material_cost(input_string: str) -> str:
    """Useful for calculating the total financial cost of materials.
    Action Input must be a SINGLE string with two numbers separated by a comma: quantity, price.
    Example Action Input: 500, 15
    """
    try:
        # Clean up the AI's text and split it into two variables
        cleaned = input_string.replace("quantity=", "").replace("price_per_unit=", "").strip()
        quantity_str, price_str = cleaned.split(",")
        
        # Do the math
        total = float(quantity_str.strip()) * float(price_str.strip())
        return f"The exact calculated cost is {total} SAR."
    except Exception as e:
        return "Error: Tell the user you need exactly two numbers separated by a comma."

# --- 2. Initialize the Local Model ---
llm = Ollama(model="llama3:8b-instruct-q4_K_M")

# --- 3. The "Brain" (ReAct Prompt) ---
# This forces the LLM to think logically: Thought -> Action -> Observation
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

# --- 4. Assemble the Agent ---
tools = [calculate_material_cost]
agent = create_react_agent(llm, tools, prompt)

# handle_parsing_errors=True is crucial for local models in case they format output slightly wrong
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

if __name__ == "__main__":
    print("--- Asking Najm to do Industrial Math ---")
    question = "Calculate the cost for 500 bags of cement at 15 SAR each."
    
    response = agent_executor.invoke({"input": question})
    print("\n--- Final Output ---")
    print(response["output"])