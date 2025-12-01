from langchain_ollama import ChatOllama
from langchain.tools import tool


llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    validate_model_on_init=True,
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

ai_msg = llm.invoke(messages)

print(ai_msg)
print(ai_msg.usage_metadata, ai_msg.content, sep='\n')