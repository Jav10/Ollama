#from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
#from langchain_community.tools import DuckDuckGoSearchResults
#from langchain_ollama import ChatOllama
#from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
#from langchain.tools import tool
#
## Opción de time,Significado,Descripción
## d,Día (Day),Resultados publicados en las últimas 24 horas.
## w,Semana (Week),Resultados publicados en la última semana.
## m,Mes (Month),Resultados publicados en el último mes.
## y,Año (Year),Resultados publicados en el último año.
## (Ninguno),Sin filtro,"Incluye resultados de todos los tiempos. Si omites el parámetro time o lo dejas como una cadena vacía, no se aplica ningún filtro de tiempo."
#
## 1. Definición de la herramienta de búsqueda libre
#search = DuckDuckGoSearchRun(
#    name="buscador_web", # Nombre que usará el LLM
#    description="Útil para buscar información reciente en internet."
#)
#wrapper = DuckDuckGoSearchAPIWrapper(region="es-mx", max_results=5) #time="w"
#
#search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="text", output_format="list") # text , news
#
#@tool
#def buscador_web(query):
#    """
#    Searches for the given query using the DuckDuckGo search engine and returns a list of results.
#
#    Args:
#        query (str): Search terms to look for.
#
#    Returns:
#        list: A list of search results.
#    """
#    return search.invoke(query)
#
##print(search.invoke("Obama"))
#
#
#researcher_llm = ChatOllama(model="llama3.1:8b", temperature=0, validate_model_on_init=True).bind_tools([buscador_web])
#
#messages = [
#    SystemMessage(content="You are an assistant. Use tool to get information on internet."),
#    HumanMessage(content="Who is Jhonny Deep?") # Add a user prompt
#]
#
#result = researcher_llm.invoke(messages)
#
#print(result)
#
##if result.tool_calls:
##    for tool in result.tool_calls:
##        if tool["name"] == "buscador_web":
##            query = tool["args"]["query"]
##            tool_output = buscador_web(query)
##
##            # Agregamos la respuesta de la herramienta
##            messages.append(result)
##            messages.append(
##                ToolMessage(
##                    tool_call_id=tool["id"],
##                    content=str(tool_output)
##                )
##            )
##
##            # 3. Llamamos de nuevo al LLM para obtener la respuesta final
##            final_response = researcher_llm.invoke(messages)
##            print(final_response.content)
##else:
##    # Si no pidió tool, imprimimos resultado directo
##    print(result.content)

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent

import requests
from bs4 import BeautifulSoup
import re


# Herramienta con el decorador oficial de LangChain
@tool
def browser(query: str):
    """Busca información usando DuckDuckGo."""
    # Configuración del buscador
    info = []
    for region in ["us-en", "es-mx"]:
        wrapper = DuckDuckGoSearchAPIWrapper(region=region, max_results=5) # es-mx , time="w"
        search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="text", output_format="list")
        info.extend(search.invoke(query))
    return info

# Tu modelo Ollama
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    validate_model_on_init=True
)

# Crear el agente con tu LLM + tools
agent = create_agent(
    model=llm,
    tools=[browser],
    system_prompt= "Eres un asistente, que ayuda al usuario a responder sus dudas. usa la respuesta de la herramienta y si hay datos innecesarios omitelos, solo responde la pregunta."
)




def obtener_contenido(url: str):
    """Extrae el texto útil de una página web eliminando elementos no relevantes como footers, aside, ads y popups."""
    
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    # ░░░ Tags completos que se eliminarán del DOM ░░░
    tags_eliminar = [
        "footer", "aside", "nav", "header", "script", "style",
        "form", "noscript",  "picture", "video",
        "audio","iframe", "svg","figure"
    ]

    for tag in tags_eliminar:
        for elemento in soup.find_all(tag):
            elemento.decompose()

    # ░░░ BONUS: Eliminar bloques por clase (ads, banners, popups) ░░░
    clases_eliminar = [
        #"ad", "ads", "advertisement", "sponsor", "promo", "banner",
        #"popup", "pop-up", "cookie", "gdpr", "subscribe", "newsletter"
    ]

    for classname in clases_eliminar:
        for elemento in soup.find_all(
            class_=lambda c: c and classname in c.lower()
        ):
            elemento.decompose()

    # ░░░ Obtener texto visible ░░░
    texto = soup.get_text() # separator="\n"

    # ░░░ Limpiar líneas en blanco ░░░
    texto = re.sub(r"\n\s*\n+", "\n", texto).strip()

    return texto

# Ejecutar la consulta
result = agent.invoke({"messages": [
    #{"role": "ai", "content": obtener_contenido('https://realpython.com/python314-new-features/')},
    {"role": "user", "content": "cuales son las nuevas caracteristicas de python 3.14?"}
]})

print(result)

print(result["messages"][-1].content)