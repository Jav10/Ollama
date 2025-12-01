from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama


# --- MODELO OLLAMA ---
llm = ChatOllama(
    model="deepseek-r1:8b",
    temperature=0,
    
)

# Run the agent
conversation_thread = [SystemMessage("Eres un asistente muy útil")] #[{"role": "user", "content": "How are you?"}]

x=True
while x:
    user = input("User: ")
    if user.lower() == "exit":
        x = False
        break
    conversation_thread.append(HumanMessage(user))
    
    res = llm.invoke(conversation_thread) # what is the time righ now ?
    ai_msg = res.content
    print("AI: ",ai_msg)
    conversation_thread.append(AIMessage(ai_msg))
    # MemorySaver

print("\n\nHilo de conversación\n")
print(conversation_thread)
