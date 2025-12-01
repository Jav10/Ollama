from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader

# Cargar datos
loader = TextLoader(r'docs\historia resumen.txt', encoding='utf-8')

print('Texto: ',loader.load())

# Generar embeddings y cargarlos en memoria
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore(embeddings)

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


query = "Quién es el héroe?" # Consulta del usuario

# La Vector Store convierte esta 'query' en un vector y busca los documentos más similares.
docs_encontrados = vector_store.similarity_search(query, k=3, score_threshold=0.8) # k=2 para obtener los 2 resultados más relevantes

# 5. Imprimir los resultados
print(f"Consulta: '{query}'\n")
print("--- Resultados de la Búsqueda Semántica ---")

for i, doc in enumerate(docs_encontrados):
    print(f"Resultado {i+1}: {doc.page_content}")