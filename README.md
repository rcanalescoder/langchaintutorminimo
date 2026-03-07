# langchaintutorminimo

Tutorial básico y progresivo de LangChain utilizando modelos locales mediante Ollama. Este repositorio contiene varios ejemplos prácticos que aumentan en complejidad, demostrando diferentes capacidades del framework (principalmente el uso de **LCEL** o LangChain Expression Language).

## Contenido del Proyecto

A continuación se describe el objetivo y la estructura de cada uno de los archivos fuente incluidos en el proyecto:

### 1. `1langchainOllamaminimo.py`
- **Objetivo**: Es el punto de entrada más básico al uso de LangChain con modelos locales. Demuestra cómo instanciar un modelo y realizar una consulta directa.
- **Estructura**:
  - Inicializa el modelo `llama3.1` usando `OllamaLLM`.
  - Define una cadena de texto simple con la pregunta (prompt directo).
  - Invoca al modelo con `invoke()` y muestra la respuesta cruda por la consola.

### 2. `2langchainparser.py`
- **Objetivo**: Mostrar cómo ejecutar múltiples tareas simultáneas sobre el mismo texto de entrada utilizando un **Pipeline paralelo** de LCEL.
- **Estructura**:
  - Define dos plantillas de prompts distintas (`ChatPromptTemplate`): una para resumir y otra para extraer ideas clave.
  - Utiliza `RunnableParallel` para conectar cada instrucción al modelo (`OllamaLLM`).
  - Al ejecutar, procesa ambas tareas de forma paralela y devuelve un diccionario estructurado con los resultados de cada acción.

### 3. `3langchaindosmodelos.py`
- **Objetivo**: Implementar un flujo de trabajo avanzado de tipo "Generación y Revisión" (Actor-Evaluator).
- **Estructura**:
  - Inicializa dos modelos distintos de `ChatOllama` (uno para generar creativamente y otro más determinista para revisar).
  - Define instrucciones para ambas fases (borrador inicial y revisión crítica).
  - Encadena ambos flujos integrando una función personalizada (`preparar_revision`) mediante `RunnableLambda`, conectando la salida del primer LLM como parte del contexto para el segundo LLM.
  - Genera una respuesta final mejorada y refinada.

### 4. `4langchainRespuestasLargas.py`
- **Objetivo**: Demostrar la utilidad de dividir una petición compleja en múltiples etapas usando LCEL, combatiendo las respuestas limitadas que suelen dar los LLMs en una sola llamada (Zero-Shot).
- **Estructura**: Es un pipeline de múltiples fases perfectamente delimitadas y medibles:
  1. *Llamada básica*: Como punto de comparación.
  2. *Normalización*: Aclarar y mejorar el tema de entrada.
  3. *Esquema JSON*: Generar un índice estructurado sobre el tema.
  4. *Desarrollo individual*: Generar contenido extenso para cada apartado del esquema por separado.
  5. *Construcción de borrador*: Unir todo el texto en un formato Markdown.
  6. *Revisión final*: Analizar el borrador completo eliminando repeticiones.
  - El resultado compara las métricas (longitud y palabras) de una llamada directa vs el flujo orquestado.
