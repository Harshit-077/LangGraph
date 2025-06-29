from typing import List, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    #messages_ai: List[AIMessage] ##we can do loke this i.e individually or we can do it in one only

llm = ChatOllama(model="llama3.2")

def process(state: AgentState) -> AgentState:
    """This node will solve the request"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content = response.content))
    print(f"\n AI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START,"process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history=[]

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content = user_input))
    result = agent.invoke({"messages": conversation_history})
    print(result["messages"])
    conversation_history = result["messages"]
    user_input = input("Enter: ")


#To write the conversation history we can use database or text file
#Example with text file

with open ("logging.txt","w") as file:
    file.write("Your Conversation Log:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message,AIMessage):
            file.write(f"AI: {message.content}\n")
        else:
            file.write("End of Conversation Log\n")

print("Conversation saved to logging.txt")


