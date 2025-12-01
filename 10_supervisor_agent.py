from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from json import dump


class AgentState(TypedDict):
    """
    Estado compartido que rastrea la información a través del flujo de LangGraph.
    """
    input: str          # ⬅️ La pregunta o tarea inicial del usuario.
    research_results: str # ⬅️ Contenido/Datos recopilados por el investigador.
    response: str       # ⬅️ La respuesta final sintetizada por el escritor.
    next_node: str      # ⬅️ La decisión del supervisor sobre el siguiente agente a ejecutar.
    researcher_count: int      # ⬅️ Conteo del nodo researcher.
    # --- Tokens por agente ---
    tokens: dict[str, dict[str, int]]

def save(data, filename="tokens_usados.json"):
    with open(filename, "w", encoding="utf-8") as f:
        dump(data, f, indent=4, ensure_ascii=False)

@tool
def search_tool(query):
    """Busca información usando DuckDuckGo."""
    # Configuración del buscador
    info = []
    for region in ["us-en", "es-mx"]:
        wrapper = DuckDuckGoSearchAPIWrapper(region=region, time='m', max_results=5) # es-mx
        search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="text", output_format="list")
        info.extend(search.invoke(query))
    return info


# Definir modelos
supervisor_llm = ChatOllama(model="llama3.1:8b", validate_model_on_init=True)
researcher_llm = ChatOllama(model="llama3.1:8b", temperature=0, validate_model_on_init=True)
writer_llm = ChatOllama(model="qwen3:8b", temperature=0.0, validate_model_on_init=True)  # Un modelo más económico para la redacción

research_system_prompt = """
Eres un agente investigador experto.
utiliza los datos que te devuelve la herramienta para hacer la investigación.
"""

writer_system_prompt = """
Eres un redactor experto.
Tu tarea es sintetizar la investigación proporcionada en una respuesta final coherente, profesional y amigable para el usuario, responde siempre en español.
"""

research_agent_executor = create_agent(
    model = researcher_llm, 
    tools = [search_tool],
    system_prompt=research_system_prompt
)


writer_agent_executor = create_agent(
    model = writer_llm,
    system_prompt=writer_system_prompt
)

def supervisor_agent(state: AgentState):
    """
    Decide el siguiente paso (agente) basándose en la tarea y el estado actual.
    """
    print("--- 👮‍♂️ Supervisor: Usando Llama3.1:8b para supervisar... ---")
    # Si NO hay investigación → forzar investigador
    if not state.get("research_results") or state.get("research_results") == "":
        return {"next_node": "investigador", 'researcher_count': state.get("researcher_count", 0)}
    
    if state.get("researcher_count", 0) == 3:
        return {**state, "next_node": "escritor"}

    system_prompt = (
        "Eres un Agente Supervisor. Tu tarea es analizar la 'input' y "
        "decidir qué agente debe ejecutarse a continuación: 'investigador', 'escritor', o 'FIN'. "
        "Responde SOLO con el nombre del nodo seleccionado."
        """Responde SOLO una palabra exacta:
        - 'investigador'
        - 'escritor'
        - 'FIN'
        No inventes variaciones ni sinónimos, asegurate que la información sea suficiente, si no es suficiente envia: 'investigador'
        """
    )
    
    response = supervisor_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Tarea: {state}"),
    ])
    
    state["tokens"]["supervisor"]["in"] = state["tokens"]["supervisor"]["in"] + response.usage_metadata["input_tokens"]
    state["tokens"]["supervisor"]["out"] = state["tokens"]["supervisor"]["out"] + response.usage_metadata["output_tokens"]
    
    decision = response.content.strip().lower()
    
    if decision == 'escribir':
        decision = 'escritor'
    elif decision == 'investigar':
        decision = 'investigador'

    return {**state, "next_node": decision, 'researcher_count': state.get("researcher_count", 0), "research_results": state.get("research_results", "")}


def investigador_agent(state: AgentState) -> AgentState:
    """
    Utiliza el LLM Llama3.1:8b y una herramienta de búsqueda para recopilar datos
    basándose en la 'input' del estado.
    """
    print("--- 🕵️ Investigador: Usando Llama3.1:8b para buscar información... ---")
    
    task = state["input"]
    print("task: ", task)
    print("research_count: ", state.get("researcher_count", 0))
    
    # Invocación real al executor
    #invoke({"input": task})
    research_output = research_agent_executor.invoke({"messages": [{"role": "user", "content": task}]})
    
    print("Salida investigador: ")
    print(research_output)
    
    state["tokens"]["investigador"]["in"] = state["tokens"]["investigador"]["in"] + research_output["messages"][1].usage_metadata["input_tokens"]
    state["tokens"]["investigador"]["out"] = state["tokens"]["investigador"]["out"] + research_output["messages"][1].usage_metadata["output_tokens"]
    
    state["tokens"]["investigador"]["in"] = state["tokens"]["investigador"]["in"] + research_output["messages"][3].usage_metadata["input_tokens"]
    state["tokens"]["investigador"]["out"] = state["tokens"]["investigador"]["out"] + research_output["messages"][3].usage_metadata["output_tokens"]
    
    research_results = research_output["messages"][3].content
    
    # Actualizar el estado con los resultados y señalar que debe volver al supervisor
    return {
        **state,
        "research_results": (state.get("research_results", "") + "\n\n" + research_results).strip(), 
        "next_node": "supervisor",
        'researcher_count': state.get("researcher_count", 0) + 1
    }



# 4. Definir la función del agente escritor
def escritor_agent(state: AgentState) -> AgentState:
    """
    Sintetiza los resultados de la investigación en una respuesta final.
    """
    print("--- ✍️ Escritor: Generando respuesta final con Qwen 3 ---")
    
    # Obtener los datos necesarios del estado compartido
    task = state["input"]
    research = state["research_results"]
    
    # 5. Invocar la cadena de redacción
    # Pasamos los datos del estado a la cadena para llenar el prompt
    final_response_content = writer_agent_executor.invoke({"messages": [{"role": "user", "content": f"Tarea original: {task}\n\nInvestigación: {research}"}]})
    
    #print(final_response_content)
    state["tokens"]["escritor"]["in"] = state["tokens"]["escritor"]["in"] + final_response_content['messages'][1].usage_metadata["input_tokens"]
    state["tokens"]["escritor"]["out"] = state["tokens"]["escritor"]["out"] + final_response_content['messages'][1].usage_metadata["output_tokens"]
    # 6. Actualizar el estado con la respuesta final
    return {
        **state,
        "response": final_response_content['messages'][1].content,
        "next_node": "FIN" # Señalamos que la tarea se completó
    }


# 1. Inicializar el grafo
workflow = StateGraph(AgentState)

# 2. Añadir los agentes (Nodos)
# Asume que 'investigador_agent' y 'escritor_agent' son funciones que toman 'state' y retornan 'state'
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("investigador", investigador_agent) 
workflow.add_node("escritor", escritor_agent)

# Definir la Entrada
workflow.set_entry_point("supervisor")

# Definir las Transiciones (Lógica Condicional)
# El Supervisor decide a dónde ir después.

workflow.add_conditional_edges(
    "supervisor",             # Nodo de partida
    lambda state: state["next_node"], # Función que obtiene la decisión del supervisor
    {                         # Mapeo de decisiones a nodos de destino
        "investigador": "investigador",
        "escritor": "escritor",
        "FIN": END
    }
)

# Definir el Loop (Opcional, para ciclos de trabajo)
# Después de un agente, regresa al supervisor para ver si se necesita el siguiente paso
workflow.add_edge("investigador", "supervisor")
#workflow.add_edge("escritor", "supervisor")


# 6. Compilar y Usar
app = workflow.compile()

initial_state = {
    "input": "Cuales son las nuevas caracteristicas de python 3.14?",
    "research_results": "",
    "response": "",
    "next_node": "supervisor",
    "researcher_count": 0,

    # TOKEN TRACKING
    "tokens": {
        "supervisor": {"in": 0, "out": 0},
        "investigador": {"in": 0, "out": 0},
        "escritor": {"in": 0, "out": 0},
    }
}

# Ejecutar el Supervisor
final_state = app.invoke(
    initial_state, #{"input": "Cuales son las nuevas caracteristicas de python 3.14?"}, 
    # El 'END' indica que se deben ejecutar todas las ramas del grafo
    # hasta que un nodo finalice en END.
    {"configurable": {"thread_id": "test_run"}} # Opcional: Define un thread_id
)

# Imprimir la respuesta final del agente escritor
save(final_state['tokens'])
print("\n" + "="*50)
print("✨ RESPUESTA FINAL DEL AGENTE ESCRITOR ✨")
print("="*50)
print(final_state["response"])