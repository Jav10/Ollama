from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

# Tool para obtener la informacion
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    docs = vector_store.similarity_search(query, k=6)
    content = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in docs
    )
    return content, docs

# Modelo
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    validate_model_on_init=True,
    # other params...
)

# Cargar datos
loader = TextLoader(r'docs\historia resumen.txt', encoding='utf-8')

# print(loader.load())

# Generar embeddings y cargarlos en memoria
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore(embeddings)

# Cargar tu archivo .txt
#with open(r"docs\historia resumen.txt", "r", encoding="utf-8") as f:
#    texto = f.read()

# Dividir en chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(loader.load()[0].page_content) # loader.load() o texto
print("Total de chunks:", len(chunks))

print("Los chunks:", chunks)

# agregar informacion al vector_store
document_ids = vector_store.add_texts(texts=chunks)  # Para pdfs .add_documents(documents=chunks)

print("Total de documentos: ", len(document_ids),"ID's documentos: ",document_ids[:3])

vector = vector_store.store[document_ids[0]]#["embedding"]
print(vector)
print("ID del documento:", vector['id'])
print("Vector:", vector['vector'])
print("Dimensiones:", len(vector['vector']))
print("Texto: ", vector['text'])


# Crear el agente
tools = [retrieve_context]

# Instrucciones
prompt = (
    "Tu tienes acceso a una tool que regresa contexto de un texto."
    "Usa la herramienta para ayudar a responder las consultas del usuario."
    "Responde siempre en español."
    "Solo regresa la respuesta."
)
agent = create_agent(llm, tools, system_prompt=prompt)

query = (
    "Quién es el héroe?\n\n"
    "Cuáles son las habilidades de sus amigos?."
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    print("********** MENSAJES **********")
    print(event["messages"])
    event["messages"][-1].pretty_print()