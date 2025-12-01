from langchain.agents import create_agent
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, AIMessage, SystemMessage

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.types import Command 

with open(r"docs\pass.txt", "r", encoding="utf-8") as f:
    uri = f.read()

# LLM local
llm = ChatOllama(model="qwen3:8b", # qwen3:8b, llama3.1:8b
                temperature=0,
                validate_model_on_init=True,
                top_k = 40,
                top_p = 0.8
                #reasoning=True
                ) # llama3.1:8b no tiene reasoning

# Conexión SQL
db = SQLDatabase.from_uri(f"{uri}"
    "?sslmode=require"
    "&sslrootcert=server-ca_respaldo.pem"
    "&sslcert=client-cert_respaldo.pem"
    "&sslkey=client-key_respaldo.pem")

query_tool = QuerySQLDataBaseTool(db=db)

# Crear agente
system_prompt = (
    "Eres un analista de datos experto.\n"
    "Debes generar SQL seguro, eficiente y con filtros apropiados.\n"
    "Evita consultas peligrosas y explica siempre tu razonamiento final."
    "Usa unicamente la vista data_orders_view_COPY o data_quotations_view_COPY segun corresponda."
)

agent = create_agent(
    model = llm,
    tools=[query_tool],
    system_prompt=system_prompt,
    middleware=[ 
        HumanInTheLoopMiddleware( 
            interrupt_on={"sql_db_query": True}, 
            description_prefix="Tool execution pending approval", 
        ), 
    ], 
    checkpointer=InMemorySaver(), 
)

question = """
Genera el query para ver la informacion de quotations, quiero ver la conversion de cada refaccionaria (su identificador es organization_id y el nombre name_refa),
la conversion la obtenemos con el numero de cotizaciones totales (quotation_id por cotizacion) y por otro lado el recuento de las que fueron aceptadas (usamos status='Aceptada')
entonces obtenemos la conversion aceptadas/total para obtener la conversion, solo para noviembre de 2025 (fecha_creacion) con ese query usa la herramienta para obtener informacion y darme tu analisis.
"""

config = {"configurable": {"thread_id": "1"}} 

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]}, 
    config=config, 
    stream_mode="values",):
    if "messages" in step:
        step["messages"][-1].pretty_print()
    elif "__interrupt__" in step: 
        print("INTERRUPTED:") 
        interrupt = step["__interrupt__"][0] 
        for request in interrupt.value["action_requests"]: 
            print(request["description"]) 
    else:
        pass

# Aprobar la ejecución del query y reanudar
final_response_message = None
for step in agent.stream(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
    stream_mode="values",
):
    if "messages" in step:
        # El último mensaje contendrá el resultado de la herramienta o la respuesta final del LLM
        current_message = step["messages"][-1]
        current_message.pretty_print()

        # Si es un mensaje de 'AI' (del LLM) con contenido, es la respuesta final
        if isinstance(current_message, AIMessage) and current_message.content:
            final_response_message = current_message.content
    elif "__interrupt__" in step:
        # Esto no debería ocurrir en este segundo stream si el agente funciona correctamente
        pass

print("\n" + "=" * 50)
if final_response_message:
    print("## 🎉 Respuesta Final Analizada por el LLM:")
    print(final_response_message)
else:
    print("No se encontró la respuesta final del LLM después de la aprobación.")

