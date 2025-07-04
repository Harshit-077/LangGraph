from typing import TypedDict, Annotated, Sequence
#Annotated - provides additional context without affecting the type itself
#Sequence - To automatically hanle the state updates for the sequences such as by adding new messages to a chat history
from langchain_core.messages import BaseMessage # Foundational class for all message type in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id 
from langchain_core.messages import SystemMessage # Message to provide instructions to LLM
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages #add message is a reducer function
from langgraph.graph import StateGraph, END 
from langgraph.prebuilt import ToolNode

# Reducer Function
# Rules that controls how updates from nodes are combined with the existing state.
# Tells us how to merge new data into the current state

#Without a reducer, updates would have replaced the existing value entirely

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
    """This is an addition function that adds two numbers"""
    return a+b
@tool
def subtract(a: int, b: int):
    """This is an addition function that adds two numbers"""
    return a-b
@tool
def multiply(a: int, b: int):
    """This is an addition function that adds two numbers"""
    return a*b

tools = [add, subtract, multiply]

model = ChatOllama(model = 'llama3.2').bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    # response = model.invoke(["You are my AI Assistant, please answer my query to the best of your ability."])
    # return {"messages":[response]} # We can write like this
    # OR

    system_prompt = SystemMessage(content = "You are my AI Assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages":[response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent",model_call)

tool_node = ToolNode(tools = tools)
graph.add_node("tools",tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    },
)

graph.add_edge("tools","our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages":[("user","first add 40 + 12 and then multiply it by 6")]}
print_stream(app.stream(inputs,stream_mode="values"))
