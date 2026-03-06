from langchain_ollama import OllamaLLM

# Inicializar el modelo de Ollama (llama3.1)
llm = OllamaLLM(model="llama3.1")

# Definir la petición
prompt = "Cuentame como hacer configuraciones básicas de llamadas a Ollama en local"

# Invocar al modelo y mostrar la respuesta
print(f"Enviando petición: '{prompt}'...\n")
respuesta = llm.invoke(prompt)

print("Respuesta:")
print(respuesta)
