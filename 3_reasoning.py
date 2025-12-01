from langchain.messages import HumanMessage
from langchain_core.messages import ChatMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="deepseek-r1:8b",
                 reasoning=True
                 ) #, reasoning=True, llama3.1:8b, deepseek-r1:8b

messages = [
    #ChatMessage(role="control", content="thinking"),
    HumanMessage("What is 3^3?"),
]

response = llm.invoke(messages)
print("Respuesta completa: ",response, sep='\n')
print("Razonamiento: ",response.additional_kwargs['reasoning_content'], sep='\n')
print("Respuesta: ",response.content, sep='\n')