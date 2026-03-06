"""
Introducción:
Este script muestra un ejemplo avanzado de LangChain Expression Language (LCEL) 
implementando un flujo de trabajo de "Generación y Revisión" (Actor-Evaluator pattern).

Objetivo:
- Utilizar un primer modelo de lenguaje para generar un borrador sobre un tema técnico.
- Pasar ese borrador a un segundo modelo distinto para que lo revise, critique y mejore.
- Demostrar una arquitectura más realista, modular y estructurada para crear aplicaciones más robustas.

Conceptos clave de LangChain utilizados:
- ChatOllama: Interfaz para interactuar con modelos de chat locales vía Ollama.
- ChatPromptTemplate: Creación de prompts estructurados con roles (system, user).
- StrOutputParser: Un analizador para extraer el texto limpio de la respuesta del LLM (eliminando metadatos).
- RunnableLambda: Permite integrar funciones personalizadas de Python (como 'preparar_revision') dentro de nuestra cadena o pipeline (LCEL).
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# 1. Inicialización de dos modelos distintos
# ---------------------------------------------------------

# Inicializamos el "Modelo Generador". 
# Usamos "llama3.1" con una temperatura de 0.3 (ligeramente creativo pero controlado)
# Su propósito es crear una respuesta inicial (borrador) con el contenido técnico.
generator_llm = ChatOllama(
    model="llama3.1",
    temperature=0.3
)

# Inicializamos el "Modelo Revisor".
# Simulamos usar un modelo diferente ("mistral" u otro disponible) con temperatura 0.1.
# Una temperatura baja asegura que sus respuestas sean muy deterministas, analíticas y precisas, ideal para revisión de código o texto.
reviewer_llm = ChatOllama(
    model="mistral",
    temperature=0.1
)

# Creamos un parser para extraer únicamente la cadena de texto de la respuesta del modelo,
# ignorando la estructura del objeto 'AIMessage' que devuelve LangChain por defecto.
parser = StrOutputParser()


# ---------------------------------------------------------
# 2. Prompt de generación inicial (Borrador)
# ---------------------------------------------------------

# Construimos la instrucción para el primer modelo.
# Le asignamos el rol de un "experto en frameworks de IA" para la generación.
# La variable de entrada es {question}, que contiene la duda original del usuario.
generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en frameworks de IA. Responde de forma técnica, clara y bien estructurada."),
    ("user", "{question}")
])


# ---------------------------------------------------------
# 3. Prompt de revisión (Criterios de mejora)
# ---------------------------------------------------------

# Construimos la instrucción para el segundo modelo (revisor).
# Le damos instrucciones estrictas sobre cómo debe actuar y evaluar el texto.
# Requerimos dos variables de entrada: 
#   - {question}: Para darle contexto sobre qué se preguntó originalmente.
#   - {draft}: El texto generado previamente por el primer modelo.
review_prompt = ChatPromptTemplate.from_messages([
    ("system", """
Eres un revisor técnico senior.

Tu tarea es mejorar la respuesta recibida.

Criterios:
- mantener exactitud técnica
- mejorar claridad
- eliminar redundancias
- organizar mejor las ideas
- conservar un tono profesional
- no inventar información nueva si no está justificada
"""),
    ("user", """
Pregunta original:
{question}

Respuesta inicial:
{draft}

Devuelve una versión revisada y mejorada.
""")
])


# ---------------------------------------------------------
# 4. Primera cadena: Generar borrador
# ---------------------------------------------------------

# Expresión LCEL que define la primera fase de nuestro pipeline.
# El flujo es: (1) Inyectar variables en el Prompt de generación -> (2) Ejecutar el modelo generador -> (3) Limpiar la salida como texto.
# Nota: Esta cadena NO se ejecuta todavía, solo se define.
draft_chain = generation_prompt | generator_llm | parser


# ---------------------------------------------------------
# 5. Función puente: Prepara los datos para la revisión
# ---------------------------------------------------------

def preparar_revision(data):
    """
    Función de utilidad (RunnableLambda) en nuestro flujo de LCEL.
    Recibe los datos de entrada, ejecuta la primera cadena (draft_chain) 
    y construye un diccionario (contexto) para la segunda fase.
    """
    # Ejecutamos síncronamente la generación del borrador usando la pregunta.
    draft = draft_chain.invoke({
        "question": data["question"]
    })

    # Devolvemos un diccionario con las dos variables que necesita 'review_prompt'.
    return {
        "question": data["question"],
        "draft": draft
    }


# ---------------------------------------------------------
# 6. Cadena final: Flujo completo (Generación -> Puente -> Revisión)
# ---------------------------------------------------------

# Componente principal que coordina todo el proceso usando LCEL.
# 1. `RunnableLambda(preparar_revision)`: Punto de entrada que genera el borrador y mapea las variables.
# 2. `review_prompt`: Toma el diccionario (question y draft) y construye el prompt final.
# 3. `reviewer_llm`: Ejecuta el prompt final (revisión de texto).
# 4. `parser`: Convierte la salida del modelo a texto puro.
chain = (
    RunnableLambda(preparar_revision)
    | review_prompt
    | reviewer_llm
    | parser
)


# ---------------------------------------------------------
# 7. Ejecución
# ---------------------------------------------------------

# Definimos una pregunta técnica de prueba
pregunta = "Explica la diferencia entre LangChain y LangGraph"

print(f"Enviando petición: '{pregunta}'...\n")

# Invocamos la cadena principal (flujo completo).
# Automáticamente pasará por la generación del borrador, y este se pasará al modelo de revisión.
respuesta = chain.invoke({
    "question": pregunta
})

# Mostramos el resultado final ya procesado por el segundo modelo.
print("Respuesta final revisada:")
print(respuesta)