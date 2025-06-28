from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="llama3.2")
response = llm.invoke([HumanMessage(content="What is the capital of France?")])
print(response.content)
