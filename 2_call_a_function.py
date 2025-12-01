from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool
from typing import List
from datetime import datetime
from langchain.agents import create_agent

# Use one of 'human', 'user', 'ai', 'assistant', 'function', 'tool', 'system', or 'developer'.

@tool
def get_datetime() -> str:
    """
    Devuelve la fecha y hora actual como una cadena de texto.

    Returns:
        str: La fecha y hora actual en formato ISO-8601, por ejemplo "2025-11-15 12:45:30.123456".
    
    Ejemplo:
        >>> get_datetime()
        '2025-11-15 12:45:30.123456'
    """
    return str(datetime.now())


llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    validate_model_on_init=True,
    # other params...
)

# Crear el agente con tu LLM + tools
agent = create_agent(model=llm, tools=[get_datetime],
    system_prompt= "Eres un asistente, que ayuda al usuario a responder sus dudas. usa la respuesta de la herramienta y si hay datos innecesarios omitelos, solo responde la pregunta."
)

messages = [
    ("human", "Qué hora es?"),
]

result = llm.invoke(messages) #ai_msg

r = agent.invoke({"messages": [
    {"role": "user", "content": "Qué hora es??"}
]})

#if isinstance(result, AIMessage) and result.tool_calls:
#    print(result.tool_calls)

res = get_datetime.invoke({})

print("Respuesta Tool: ", res, sep='\n')

print("Respuesta modelo: ",result.content, sep='\n')

print("Respuesta Agente: ",r['messages'][-1].content, sep='\n')
