"""
Ejemplo de cadena larga con LCEL, mostrando por consola cada fase del proceso
y comparando su longitud con una llamada básica al modelo.

Objetivo didáctico:
- Mostrar que una sola llamada directa al LLM suele producir una respuesta limitada
- Mostrar que, si dividimos el problema en varias fases, podemos construir una
  respuesta mucho más extensa, estructurada y útil

Flujo general:
1. Se hace una llamada básica al modelo para tener una referencia inicial
2. Se normaliza la entrada del usuario
3. Se genera un esquema básico en JSON
4. Se desarrolla cada apartado por separado
5. Se construye un borrador largo
6. Se revisa el borrador para eliminar repeticiones
7. Se comparan las longitudes de las distintas salidas

Puntos didácticos importantes:
- Se usa ChatOllama para trabajar con mensajes de sistema y usuario
- Se usa LCEL para encadenar fases del pipeline
- Se escapan las llaves del JSON en el prompt con {{ y }}
  para que LangChain no las interprete como variables
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import json


# =========================================================
# 1. FUNCIONES AUXILIARES DE MEDICIÓN
# =========================================================

def count_words(text):
    """
    Cuenta palabras de forma aproximada separando por espacios.
    """
    if not text:
        return 0
    return len(text.split())


def print_text_metrics(label, text):
    """
    Muestra métricas sencillas de longitud para un texto.

    Se muestran:
    - número de caracteres
    - número aproximado de palabras
    """
    chars = len(text) if text else 0
    words = count_words(text)

    print(f"\n[MÉTRICAS] {label}")
    print(f"  - Caracteres: {chars}")
    print(f"  - Palabras aprox.: {words}")


def safe_json_dumps(data):
    """
    Convierte un objeto Python a JSON legible, preservando tildes y caracteres.
    """
    return json.dumps(data, indent=2, ensure_ascii=False)


# =========================================================
# 2. CONFIGURACIÓN GENERAL
# =========================================================

print("\n" + "=" * 100)
print("INICIALIZANDO ENTORNO")
print("=" * 100)

# Modelo principal.
# Se puede cambiar por otro modelo de Ollama que tengas descargado.
llm = ChatOllama(
    model="llama3.1",
    temperature=0.2
)

# Parser básico para convertir la salida del modelo en texto.
parser = StrOutputParser()

print("Modelo cargado correctamente: llama3.1")
print("Temperatura configurada: 0.2")
print("Parser de salida: StrOutputParser")
print()


# =========================================================
# 3. LLAMADA BÁSICA DE REFERENCIA
# =========================================================

basic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Eres un experto en frameworks de IA.
Responde de forma clara y útil, pero en una única respuesta directa.
"""
    ),
    (
        "user",
        "{topic}"
    )
])

basic_chain = basic_prompt | llm | parser


# =========================================================
# 4. PROMPT DE NORMALIZACIÓN DE LA ENTRADA
# =========================================================

normalization_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Eres un asistente especializado en preparar consultas para un sistema de IA.

Tu tarea es normalizar el tema introducido por el usuario.

Reglas:
- Corrige pequeñas ambigüedades o formulaciones pobres
- Haz el tema más claro y más útil para su desarrollo
- Mantén la intención original
- Devuelve solo el tema normalizado
"""
    ),
    (
        "user",
        "Tema original del usuario: {topic}"
    )
])

normalization_chain = normalization_prompt | llm | parser


# =========================================================
# 5. PROMPT PARA GENERAR EL ESQUEMA EN JSON
# =========================================================
# IMPORTANTE:
# Las llaves del JSON de ejemplo deben escaparse con {{ y }}
# porque, si no, LangChain intentará interpretarlas como variables.

outline_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Eres un experto en estructurar contenido educativo y técnico.

Tu tarea es crear un esquema básico y útil sobre un tema.

Devuelve únicamente JSON válido.
No añadas explicaciones, comentarios ni bloques markdown.
No escribas ```json ni texto antes o después.

La salida debe seguir exactamente esta estructura:

{{
  "tema": "string",
  "apartados": [
    "string",
    "string",
    "string"
  ]
}}

Criterios:
- Genera entre 4 y 6 apartados
- Los apartados deben ser distintos entre sí
- Los apartados deben cubrir el tema de forma progresiva
- Los nombres de los apartados deben ser claros y concretos
"""
    ),
    (
        "user",
        "Genera un esquema básico sobre este tema: {normalized_topic}"
    )
])

outline_chain = outline_prompt | llm | parser


# =========================================================
# 6. PROMPT PARA DESARROLLAR CADA APARTADO
# =========================================================

section_expansion_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Eres un experto en explicar temas técnicos de forma clara, estructurada y útil.

Tu tarea es desarrollar un apartado concreto dentro de un tema más amplio.

Criterios:
- Explica el apartado con claridad
- Mantén foco en el apartado pedido
- Evita repetir ideas genéricas que pertenezcan a otros apartados
- Escribe un texto sustancial y pedagógico
- Usa un tono profesional
"""
    ),
    (
        "user",
        """
Tema general: {topic}

Apartado a desarrollar: {section}

Desarrolla este apartado con algo más de detalle que una respuesta breve.
"""
    )
])

section_expansion_chain = section_expansion_prompt | llm | parser


# =========================================================
# 7. PROMPT DE REVISIÓN Y ELIMINACIÓN DE REPETICIONES
# =========================================================

dedup_review_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Eres un editor técnico.

Tu tarea es revisar un contenido estructurado y mejorar su calidad.

Objetivos:
- Eliminar repeticiones entre apartados
- Evitar solapamientos innecesarios
- Mantener la estructura y el sentido del contenido
- Mejorar claridad y cohesión
- Conservar un tono profesional y pedagógico

Devuelve el resultado final ya limpio y bien organizado.
"""
    ),
    (
        "user",
        """
Revisa este contenido y elimina repeticiones o ideas redundantes:

{draft}
"""
    )
])

dedup_review_chain = dedup_review_prompt | llm | parser


# =========================================================
# 8. FUNCIÓN: LLAMADA BÁSICA DE COMPARACIÓN
# =========================================================

def run_basic_reference(data):
    """
    Ejecuta una llamada simple al modelo para comparar el tamaño y la
    profundidad de la respuesta frente al pipeline largo.
    """
    print("\n" + "-" * 100)
    print("FASE 0: LLAMADA BÁSICA DE REFERENCIA")
    print("-" * 100)

    topic = data["topic"]
    print(f"Consulta directa enviada al modelo: {topic}")

    basic_response = basic_chain.invoke({
        "topic": topic
    }).strip()

    print("\nRespuesta básica generada:")
    print(basic_response)

    print_text_metrics("Respuesta básica", basic_response)

    return {
        "topic": topic,
        "basic_response": basic_response
    }


# =========================================================
# 9. FUNCIÓN DE NORMALIZACIÓN
# =========================================================

def normalize_input(data):
    """
    Recibe el tema original del usuario y genera una versión
    más clara y mejor preparada para el resto del pipeline.
    """
    print("\n" + "-" * 100)
    print("FASE 1: NORMALIZACIÓN DE LA ENTRADA")
    print("-" * 100)

    original_topic = data["topic"]

    print(f"Tema original recibido: {original_topic}")

    normalized_topic = normalization_chain.invoke({
        "topic": original_topic
    }).strip()

    print(f"Tema normalizado generado: {normalized_topic}")

    print_text_metrics("Tema normalizado", normalized_topic)

    return {
        "topic": original_topic,
        "basic_response": data["basic_response"],
        "normalized_topic": normalized_topic
    }


# =========================================================
# 10. FUNCIÓN PARA GENERAR EL ESQUEMA JSON
# =========================================================

def build_outline(data):
    """
    Genera un esquema básico en formato JSON a partir del tema normalizado.

    Si el modelo devuelve un JSON incorrecto, se aplica un fallback
    para que la cadena no se rompa.
    """
    print("\n" + "-" * 100)
    print("FASE 2: GENERACIÓN DEL ESQUEMA BÁSICO EN JSON")
    print("-" * 100)

    print(f"Tema normalizado de entrada: {data['normalized_topic']}")

    raw_outline = outline_chain.invoke({
        "normalized_topic": data["normalized_topic"]
    }).strip()

    print("\nSalida cruda del modelo para el esquema:")
    print(raw_outline)
    print_text_metrics("Esquema JSON crudo", raw_outline)

    # Limpieza básica por si el modelo devuelve markdown.
    cleaned_outline = raw_outline.replace("```json", "").replace("```", "").strip()

    print("\nSalida limpiada antes de parsear JSON:")
    print(cleaned_outline)
    print_text_metrics("Esquema JSON limpio", cleaned_outline)

    try:
        outline = json.loads(cleaned_outline)
        print("\nJSON parseado correctamente.")
    except json.JSONDecodeError as e:
        print("\nNo se pudo parsear el JSON generado por el modelo.")
        print(f"Detalle del error: {e}")
        print("Se aplicará un fallback local para continuar con la ejecución.")

        outline = {
            "tema": data["normalized_topic"],
            "apartados": [
                "Introducción",
                "Conceptos clave",
                "Diferencias principales",
                "Casos de uso",
                "Conclusión"
            ]
        }

    print("\nEsquema final que usará la cadena:")
    print(safe_json_dumps(outline))
    print_text_metrics("Esquema final serializado", safe_json_dumps(outline))

    return {
        "topic": data["topic"],
        "basic_response": data["basic_response"],
        "normalized_topic": data["normalized_topic"],
        "outline": outline
    }


# =========================================================
# 11. FUNCIÓN PARA DESARROLLAR LOS APARTADOS
# =========================================================

def expand_sections(data):
    """
    Recorre los apartados del esquema y desarrolla cada uno
    por separado.
    """
    print("\n" + "-" * 100)
    print("FASE 3: DESARROLLO INDIVIDUAL DE CADA APARTADO")
    print("-" * 100)

    outline = data["outline"]
    topic = outline.get("tema", data["normalized_topic"])
    sections = outline.get("apartados", [])

    print(f"Tema base para desarrollar: {topic}")
    print(f"Número de apartados detectados: {len(sections)}")

    expanded_sections = []

    total_partial_chars = 0
    total_partial_words = 0

    for index, section in enumerate(sections, start=1):
        print("\n" + "." * 100)
        print(f"Desarrollando apartado {index}/{len(sections)}: {section}")
        print("." * 100)

        content = section_expansion_chain.invoke({
            "topic": topic,
            "section": section
        }).strip()

        print("Contenido generado para este apartado:")
        print(content)

        chars = len(content)
        words = count_words(content)
        total_partial_chars += chars
        total_partial_words += words

        print_text_metrics(f"Apartado {index}: {section}", content)

        expanded_sections.append({
            "titulo": section,
            "contenido": content,
            "chars": chars,
            "words": words
        })

    print("\nTodos los apartados han sido desarrollados correctamente.")
    print("\nSuma de tamaños de todos los desarrollos parciales:")
    print(f"  - Caracteres totales parciales: {total_partial_chars}")
    print(f"  - Palabras totales parciales: {total_partial_words}")

    return {
        "topic": data["topic"],
        "basic_response": data["basic_response"],
        "normalized_topic": data["normalized_topic"],
        "outline": outline,
        "expanded_sections": expanded_sections,
        "partials_total_chars": total_partial_chars,
        "partials_total_words": total_partial_words
    }


# =========================================================
# 12. FUNCIÓN PARA CONSTRUIR EL BORRADOR COMPLETO
# =========================================================

def build_draft(data):
    """
    Convierte los apartados desarrollados en un borrador único
    con formato tipo Markdown.
    """
    print("\n" + "-" * 100)
    print("FASE 4: CONSTRUCCIÓN DEL BORRADOR COMPLETO")
    print("-" * 100)

    topic = data["outline"].get("tema", data["normalized_topic"])
    expanded_sections = data["expanded_sections"]

    print(f"Título principal del borrador: {topic}")
    print(f"Se van a integrar {len(expanded_sections)} apartados en el documento final.")

    blocks = [f"# {topic}\n"]

    for section in expanded_sections:
        blocks.append(f"## {section['titulo']}\n{section['contenido']}\n")

    draft = "\n".join(blocks)

    print("\nBorrador completo construido:")
    print(draft)

    print_text_metrics("Borrador completo", draft)

    return {
        "topic": data["topic"],
        "basic_response": data["basic_response"],
        "normalized_topic": data["normalized_topic"],
        "outline": data["outline"],
        "expanded_sections": expanded_sections,
        "partials_total_chars": data["partials_total_chars"],
        "partials_total_words": data["partials_total_words"],
        "draft": draft
    }


# =========================================================
# 13. FUNCIÓN DE REVISIÓN FINAL
# =========================================================

def review_and_deduplicate(data):
    """
    Envía el borrador completo al modelo para que revise el texto,
    elimine repeticiones y mejore la cohesión general.
    """
    print("\n" + "-" * 100)
    print("FASE 5: REVISIÓN FINAL Y ELIMINACIÓN DE REPETICIONES")
    print("-" * 100)

    print("Se enviará el borrador al revisor para pulir el contenido.")
    print("\nVista previa del borrador que entra en revisión:")
    print(data["draft"])

    final_answer = dedup_review_chain.invoke({
        "draft": data["draft"]
    }).strip()

    print("\nResultado devuelto por la fase de revisión:")
    print(final_answer)

    print_text_metrics("Respuesta final revisada", final_answer)

    basic_chars = len(data["basic_response"])
    final_chars = len(final_answer)
    basic_words = count_words(data["basic_response"])
    final_words = count_words(final_answer)

    ratio_chars = round(final_chars / basic_chars, 2) if basic_chars > 0 else 0
    ratio_words = round(final_words / basic_words, 2) if basic_words > 0 else 0

    print("\nComparativa directa entre llamada básica y respuesta final:")
    print(f"  - Caracteres respuesta básica: {basic_chars}")
    print(f"  - Caracteres respuesta final:  {final_chars}")
    print(f"  - Multiplicador por caracteres: x{ratio_chars}")
    print(f"  - Palabras respuesta básica: {basic_words}")
    print(f"  - Palabras respuesta final:  {final_words}")
    print(f"  - Multiplicador por palabras: x{ratio_words}")

    return {
        "topic": data["topic"],
        "basic_response": data["basic_response"],
        "normalized_topic": data["normalized_topic"],
        "outline": data["outline"],
        "expanded_sections": data["expanded_sections"],
        "partials_total_chars": data["partials_total_chars"],
        "partials_total_words": data["partials_total_words"],
        "draft": data["draft"],
        "final_answer": final_answer,
        "basic_chars": basic_chars,
        "basic_words": basic_words,
        "final_chars": final_chars,
        "final_words": final_words,
        "ratio_chars": ratio_chars,
        "ratio_words": ratio_words
    }


# =========================================================
# 14. CONSTRUCCIÓN DE LA CADENA LARGA CON LCEL
# =========================================================

print("\n" + "=" * 100)
print("CONSTRUYENDO CADENA LCEL")
print("=" * 100)

chain = (
    RunnableLambda(run_basic_reference)
    | RunnableLambda(normalize_input)
    | RunnableLambda(build_outline)
    | RunnableLambda(expand_sections)
    | RunnableLambda(build_draft)
    | RunnableLambda(review_and_deduplicate)
)

print("Cadena construida correctamente.")
print("Orden de ejecución:")
print("0. Llamada básica de referencia")
print("1. Normalización")
print("2. Esquema JSON")
print("3. Desarrollo de apartados")
print("4. Construcción de borrador")
print("5. Revisión final")
print()


# =========================================================
# 15. EJECUCIÓN DEL EJEMPLO
# =========================================================

user_topic = "Explica la diferencia entre LangChain y LangGraph"

print("\n" + "=" * 100)
print("EJECUCIÓN DEL PIPELINE")
print("=" * 100)
print(f"Tema enviado por el usuario: {user_topic}")

result = chain.invoke({
    "topic": user_topic
})


# =========================================================
# 16. MOSTRAR RESULTADOS FINALES
# =========================================================

print("\n" + "=" * 100)
print("RESUMEN FINAL DE RESULTADOS")
print("=" * 100)

print("\nTEMA ORIGINAL:")
print(result["topic"])

print("\nRESPUESTA BÁSICA:")
print(result["basic_response"])
print_text_metrics("Respuesta básica final registrada", result["basic_response"])

print("\nTEMA NORMALIZADO:")
print(result["normalized_topic"])
print_text_metrics("Tema normalizado final registrado", result["normalized_topic"])

print("\nESQUEMA GENERADO:")
print(safe_json_dumps(result["outline"]))
print_text_metrics("Esquema serializado final registrado", safe_json_dumps(result["outline"]))

print("\nDESARROLLOS PARCIALES:")
for index, section in enumerate(result["expanded_sections"], start=1):
    print("\n" + "~" * 100)
    print(f"Apartado {index}: {section['titulo']}")
    print("Contenido:")
    print(section["contenido"])
    print(f"Caracteres: {section['chars']}")
    print(f"Palabras aprox.: {section['words']}")

print("\nSUMA DE LOS DESARROLLOS PARCIALES:")
print(f"  - Caracteres totales parciales: {result['partials_total_chars']}")
print(f"  - Palabras totales parciales: {result['partials_total_words']}")

print("\nBORRADOR PREVIO A LA REVISIÓN:")
print(result["draft"])
print_text_metrics("Borrador previo a revisión", result["draft"])

print("\nRESPUESTA FINAL REVISADA:")
print(result["final_answer"])
print_text_metrics("Respuesta final revisada", result["final_answer"])

print("\nCOMPARATIVA FINAL ENTRE ENFOQUES:")
print(f"  - Llamada básica -> caracteres: {result['basic_chars']}")
print(f"  - Pipeline largo -> caracteres: {result['final_chars']}")
print(f"  - Multiplicador de tamaño por caracteres: x{result['ratio_chars']}")
print(f"  - Llamada básica -> palabras: {result['basic_words']}")
print(f"  - Pipeline largo -> palabras: {result['final_words']}")
print(f"  - Multiplicador de tamaño por palabras: x{result['ratio_words']}")

print("\nCONCLUSIÓN DIDÁCTICA:")
print(
    "La llamada básica ofrece una respuesta directa, pero el pipeline por fases "
    "permite ampliar el tema, estructurarlo mejor y multiplicar el tamaño total "
    "de la respuesta al desarrollar cada apartado por separado y revisar el resultado final."
)

print("\n" + "=" * 100)
print("FIN DE LA EJECUCIÓN")
print("=" * 100)