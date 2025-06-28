from typing import List, TypedDict
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


#This the alternate way witout using chatgpt openai api key using llama3.2 with 3B paraeters
# To run this you need to download llama using 'ollama run llama3.2'
# or if downloaded run it using 'ollama run llama3.2'

# Define your state
class AgentState(TypedDict):
    message: List[HumanMessage]

# Use Ollama LLM (running locally)
llm = ChatOllama(model="llama3.2")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["message"])
    print(f"\nAI: {response.content}\n")
    return state

# Set up LangGraph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# CLI Chat Loop
user_input = input("Enter: ")
while user_input.lower() != "exit":
    agent.invoke({"message": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
