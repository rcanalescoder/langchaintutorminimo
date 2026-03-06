"""
Introducción:
Este script demuestra cómo utilizar LangChain para ejecutar múltiples tareas en paralelo 
sobre un mismo texto de entrada utilizando un modelo de lenguaje local (Ollama). 

En concreto, el programa realiza las siguientes acciones:
1. Inicializa un modelo de lenguaje (Llama 3.1) a través de Ollama.
2. Define dos plantillas de prompt (instrucciones) diferentes:
   - Una para resumir el texto en una sola frase.
   - Otra para extraer tres ideas o puntos clave del texto.
3. Crea un "Pipeline paralelo" (RunnableParallel) que ejecuta ambas tareas al mismo tiempo.
4. Pasa un texto de prueba al pipeline y muestra el resultado combinado de ambas tareas.

Conceptos clave de LangChain utilizados:
- OllamaLLM: Integración de LangChain para usar modelos locales mediante Ollama.
- ChatPromptTemplate: Una forma de estructurar y formatear los mensajes (prompts) que se envían al modelo.
- RunnableParallel: Un componente de LangChain Expression Language (LCEL) que permite ejecutar múltiples "Runnables" (como cadenas o modelos) en paralelo y devolver la salida como un diccionario.
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

# 1. Inicialización del modelo de lenguaje
# Aquí instanciamos el modelo "llama3.1" que debe estar ejecutándose localmente a través de Ollama.
# Este objeto 'llm' será el encargado de procesar nuestras peticiones y generar las respuestas de texto.
llm = OllamaLLM(model="llama3.1")

# 2. Definición del Prompt para el resumen
# Creamos una plantilla de chat que define el comportamiento esperado del modelo.
# "system": Da la instrucción general de cómo debe actuar el modelo (resumir en una frase).
# "user": Es el espacio reservado (variable {text}) donde inyectaremos el texto a analizar.
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Resume el texto en una frase."),
    ("user", "{text}")
])

# 3. Definición del Prompt para extraer puntos clave
# Similar al anterior, pero esta vez la instrucción del sistema pide extraer tres ideas clave.
# Reutilizamos la variable {text} para pasar el mismo contenido a ambas tareas.
keypoints_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extrae tres ideas clave."),
    ("user", "{text}")
])

# 4. Creación del Pipeline paralelo utilizando RunnableParallel (LCEL)
# RunnableParallel nos permite ejecutar múltiples cadenas (chains) simultáneamente.
# En este caso, construimos un diccionario con dos tareas principales:
#   - 'resumen': Conecta (usando el operador pipe '|') el prompt de resumen con el modelo (llm).
#   - 'ideas_clave': Conecta el prompt de ideas clave con el mismo modelo (llm).
# Cuando se invoque 'chain', ambas tareas se procesarán en paralelo con la misma entrada.
chain = RunnableParallel(
    resumen = summary_prompt | llm,
    ideas_clave = keypoints_prompt | llm
)

# 5. Ejecución del pipeline
# Usamos el método invoke() para pasar el diccionario de variables de entrada.
# En este caso proporcionamos el texto bajo la clave "text", la cual será reemplazada en {text} en ambos prompts.
resultado = chain.invoke({
    "text": "LangChain es un framework para aplicaciones con LLM mientras LangGraph permite construir agentes y flujos complejos."
})

# 6. Mostrar el resultado
# Imprimimos la salida en la consola. El resultado será un diccionario con las claves 'resumen' y 'ideas_clave',
# donde cada valor será el texto generado por el modelo Llama 3.1 para su tarea respectiva.
print(resultado)