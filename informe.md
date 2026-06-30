# Informe Milestone 1 — Agente con herramientas

**Materia:** Agentes — Maestría en Inteligencia Artificial (UdeSA)
**Milestone:** M1 — Bucle del agente y herramientas

---

## 1. Introducción

### 1.1 Contexto: Agentes basados en LLM

Un **agente** es un sistema que utiliza un modelo de lenguaje (LLM) como motor de razonamiento y le da la capacidad de actuar sobre el mundo a través de **herramientas** (functions/tools). A diferencia de un chatbot convencional que solo genera texto, un agente puede:

- Decidir cuándo necesita información externa o realizar un cómputo.
- Invocar la herramienta adecuada con los parámetros correctos.
- Observar el resultado y decidir si necesita más acciones o si ya puede responder.

Este patrón se conoce como **ReAct** (Reasoning + Acting): el LLM alterna entre razonar sobre qué hacer y actuar ejecutando herramientas, en un bucle iterativo hasta llegar a una respuesta final.

### 1.2 Objetivo del Milestone 1

Construir un agente funcional capaz de:

- Registrar herramientas con una interfaz tipada.
- Ejecutar un bucle de razonamiento-acción (ReAct loop).
- Respetar límites de iteraciones para evitar bucles infinitos.
- Manejar errores sin romperse (herramientas desconocidas, fallos de ejecución).

### 1.3 Stack tecnológico

| Componente | Tecnología | Rol |
|---|---|---|
| Modelo de lenguaje | **Ollama + llama3.1** (8B parámetros, local) | Motor de razonamiento y decisión |
| Framework del agente | **mia_agents** (provisto por la cátedra) | Protocolos, tipos, cliente LLM, CLI |
| Código del alumno | **student_framework/** | Implementación del agente y herramientas |
| Definición de herramientas | **Pydantic** (`Annotated` + `Field`) | Generación automática de JSON Schema |
| Testing | **pytest** + `MockLLMClient` | Tests deterministas sin modelo real |
| Lenguaje | **Python 3.14** | Todo el proyecto |

---

## 2. Arquitectura del sistema

### 2.1 Vista general

El sistema se compone de tres capas bien separadas:

1. **Capa de herramientas**: funciones Python puras que realizan operaciones concretas (calcular, leer archivos, etc.).
2. **Capa del agente**: el bucle ReAct que orquesta la interacción entre el usuario, el LLM y las herramientas.
3. **Capa del LLM Client**: abstracción que traduce entre el formato interno del framework y la API del proveedor (Ollama, Bedrock, o un mock para tests).

### 2.2 Descripcion del diagrama de arquitectura (cajas y flechas)

```
Descripción del diagrama para su armado:

Cajas principales (de izquierda a derecha):

[Usuario] --> [CLI / build_agent()] --> [MyAgent] --> [LLMClient] --> [Proveedor LLM]
                                           |                              |
                                           v                              |
                                     [Registro de                         |
                                      Herramientas]                       |
                                      _tools{}                            |
                                      _schemas{}                          |
                                           |                              |
                                           v                              |
                                     [Herramientas]                       |
                                     - calculator                         |
                                     - (file_reader)                      |
                                     - (tool libre)                       |
                                                                          v
                                                                   [Ollama local]
                                                                   [llama3.1 8B]

Flechas del flujo principal:
1. Usuario --"mensaje"--> CLI
2. CLI --"user_message"--> MyAgent.run()
3. MyAgent --"chat(messages, tools)"--> LLMClient
4. LLMClient --"request"--> Ollama (llama3.1)
5. Ollama --"LLMResponse"--> LLMClient --> MyAgent
6. Si hay tool_calls: MyAgent --"ejecuta"--> Herramienta --> resultado --> MyAgent
7. MyAgent --"chat(messages + resultado)"--> LLMClient (segunda llamada)
8. LLMClient --"request"--> Ollama
9. Ollama --"respuesta final (texto)"--> LLMClient --> MyAgent
10. MyAgent --"AgentResult"--> CLI --> Usuario
```

### 2.3 Descripcion del diagrama de flujo del bucle del agente

```
Descripción del diagrama para su armado:

Inicio: run(user_message) es invocado.

[Inicio]
    |
    v
[Armar messages = [{role: "user", content: user_message}]]
    |
    v
[Preparar tools = lista de ToolSchemas registrados]
    |
    v
[iteración = 0]
    |
    v
<¿iteración < max_iterations?> --NO--> [Devolver AgentResult con lo acumulado]
    |
   SI
    |
    v
[Llamar self._llm.chat(messages, tools, system)]
    |
    v
[Recibir LLMResponse]
    |
    v
<¿Hay tool_calls en la respuesta?> --NO--> [Devolver AgentResult(answer=content, steps)]
    |                                                        FIN
   SI
    |
    v
[Agregar mensaje assistant con tool_calls a messages]
    |
    v
[Para cada tool_call:]
    |
    v
    <¿Existe la herramienta?> --NO--> [Registrar AgentStep con error]
        |                                     |
       SI                                     |
        |                                     |
        v                                     |
    [Parsear arguments (JSON)]                |
        |                                     |
        v                                     |
    [Ejecutar herramienta(**kwargs)]           |
        |                                     |
      <¿Éxito?>                               |
       /     \                                |
     SI       NO                              |
      |        |                              |
      v        v                              |
  [AgentStep  [AgentStep                      |
   output=r]   error=e]                       |
      |        |                              |
      v        v                              v
    [Agregar mensaje {role: "tool", content: resultado/error} a messages]
        |
        v
    [iteración += 1]
        |
        v
    [Volver al inicio del loop]
```

### 2.4 Descripción del diagrama de secuencia (ejemplo con calculadora)

```
Descripción del diagrama para su armado:

Participantes (de izquierda a derecha):
- Usuario
- MyAgent
- LLMClient
- Ollama (llama3.1)
- calculator()

Secuencia temporal (de arriba a abajo):

1. Usuario -> MyAgent: run("¿Cuánto es 15 * 7?")
2. MyAgent -> MyAgent: messages = [{role: "user", content: "¿Cuánto es 15 * 7?"}]
3. MyAgent -> LLMClient: chat(messages, tools=[calculator_schema], system="Eres un asistente útil.")
4. LLMClient -> Ollama: POST /api/chat (model=llama3.1, messages, tools)
5. Ollama -> LLMClient: response {tool_calls: [{name: "calculator", arguments: {left_operand: 15, right_operand: 7, operator: "*"}}]}
6. LLMClient -> MyAgent: LLMResponse(content=None, tool_calls=[ToolCall(...)])
7. MyAgent -> MyAgent: Agrega mensaje assistant con tool_calls a messages
8. MyAgent -> calculator: calculator(left_operand=15, right_operand=7, operator="*")
9. calculator -> MyAgent: "105"
10. MyAgent -> MyAgent: Registra AgentStep(tool_name="calculator", tool_output="105")
11. MyAgent -> MyAgent: Agrega {role: "tool", content: "105"} a messages
12. MyAgent -> LLMClient: chat(messages [3 elementos], tools=[calculator_schema], system=...)
13. LLMClient -> Ollama: POST /api/chat (messages con resultado de la tool)
14. Ollama -> LLMClient: response {content: "El resultado de 15 × 7 es 105."}
15. LLMClient -> MyAgent: LLMResponse(content="El resultado de 15 × 7 es 105.", tool_calls=[])
16. MyAgent -> MyAgent: No hay tool_calls → fin del bucle
17. MyAgent -> Usuario: AgentResult(answer="El resultado de 15 × 7 es 105.", steps=[AgentStep(...)])
```

---

## 3. Diseño de la interfaz de herramientas

### 3.1 El patrón completo: de función Python a herramienta del LLM

El diseño de herramientas en este framework sigue un flujo en cuatro etapas. Cada etapa transforma la información de una forma a otra:

**Etapa 1 — Definición**: el desarrollador escribe una función Python con tipos anotados.

```python
def calculator(
    left_operand: Annotated[float, Field(description="Primer operando numérico.")],
    right_operand: Annotated[float, Field(description="Segundo operando numérico.")],
    operator: Annotated[str, Field(description="Operador aritmético: +, -, * o /")],
) -> str:
    """Calcula el resultado de una operación aritmética entre dos números."""
```

Cada parte tiene un propósito:

| Elemento | Qué aporta | Quién lo consume |
|---|---|---|
| `Annotated[float, ...]` | Tipo del parámetro (se traduce a `"type": "number"` en JSON Schema) | `ToolSchema.from_callable` |
| `Field(description="...")` | Descripción del parámetro para el LLM | `ToolSchema.from_callable` |
| Docstring de la función | Descripción general de la herramienta | `ToolSchema.from_callable` |
| `-> str` | Tipo de retorno (las herramientas siempre devuelven string) | El agente |

**Etapa 2 — Generación del schema**: `ToolSchema.from_callable(calculator)` inspecciona la firma de la función y genera:

```python
ToolSchema(
    name="calculator",
    description="Calcula el resultado de una operación aritmética entre dos números.",
    parameters={
        "type": "object",
        "properties": {
            "left_operand": {"type": "number", "description": "Primer operando numérico."},
            "right_operand": {"type": "number", "description": "Segundo operando numérico."},
            "operator": {"type": "string", "description": "Operador aritmético: +, -, * o /"}
        },
        "required": ["left_operand", "right_operand", "operator"]
    }
)
```

Internamente, `from_callable` usa `inspect.signature()` y `get_type_hints()` para extraer los parámetros, construye un modelo Pydantic dinámico con `create_model()`, y llama a `model_json_schema()` para obtener el JSON Schema estándar.

**Etapa 3 — Registro en el agente**: `agent.register_tool(calculator, calculator_schema)` almacena:

- `self._tools["calculator"]` → la función callable (para ejecutarla cuando el LLM la pida).
- `self._schemas["calculator"]` → el `ToolSchema` (para pasárselo al LLM en cada llamada).

**Etapa 4 — Exposición al LLM**: en cada llamada a `run()`, el agente pasa los schemas al LLM:

```python
self._llm.chat(messages=messages, tools=list(self._schemas.values()), ...)
```

El `LLMClient` aplica `to_llm_spec()` sobre cada `ToolSchema` y lo traduce al formato nativo del proveedor. Para Ollama, eso produce:

```json
{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Calcula el resultado de una operación aritmética...",
        "parameters": { ... }
    }
}
```

### 3.2 Descripción del diagrama de transformación del schema

```
Descripción del diagrama para su armado:

Es un flujo horizontal que muestra cómo una función Python se transforma
paso a paso hasta llegar al proveedor LLM:

[Función Python]           [ToolSchema]              [LLM Spec]              [Formato Ollama]
 con Annotated    --->    (name, description,   ---> to_llm_spec()   --->   {"type": "function",
 + Field                   parameters: {})            devuelve dict          "function": {...}}
 + docstring
                  ^                              ^                     ^
                  |                              |                     |
          from_callable()                 register_tool()         _format_tools()
          (inspección de                  (guarda en el           (LLMClient traduce
           firma Python)                   agente)                al proveedor)

Cada flecha representa una transformación:
- from_callable: Python types → JSON Schema (via Pydantic)
- register_tool: almacena callable + schema en dicts internos
- to_llm_spec: ToolSchema → dict genérico {name, description, parameters}
- _format_tools: dict genérico → formato específico de Ollama/Bedrock
```

### 3.3 La abstracción de proveedores

El `LLMClient` actúa como capa de abstracción entre el agente y el proveedor concreto:

```
MyAgent  --->  LLMClient (protocolo: solo chat())
                    |
               ¿Qué proveedor?
                /           \
         OllamaProvider    BedrockProvider
              |                  |
         API Ollama         API AWS Converse
         (local)            (cloud)
```

El agente nunca sabe qué proveedor hay debajo. Solo depende del protocolo `LLMClient` que tiene un único método:

```python
def chat(messages, tools, system, temperature, response_format) -> LLMResponse
```

Esto permite:

- **Desarrollo local**: usar Ollama con llama3.1 sin costos ni internet.
- **Producción/evaluación**: cambiar a AWS Bedrock modificando solo variables de entorno.
- **Testing**: inyectar un `MockLLMClient` que devuelve respuestas deterministas.

El cambio de proveedor se hace **sin tocar una línea del agente ni de las herramientas**.

---

## 4. Herramientas implementadas

### 4.1 Calculadora simple

| Campo | Valor |
|---|---|
| **Archivo** | `student_framework/tools/calculator.py` |
| **Nombre del schema** | `calculator` |
| **Descripción para el LLM** | "Calcula el resultado de una operación aritmética entre dos números." |

**Parámetros:**

| Parámetro | Tipo | Descripción |
|---|---|---|
| `left_operand` | `float` | Primer operando numérico |
| `right_operand` | `float` | Segundo operando numérico |
| `operator` | `str` | Operador aritmético: `+`, `-`, `*` o `/` |

**Operadores soportados:**

| Operador | Operación | Ejemplo |
|---|---|---|
| `+` | Suma | `calculator(10, 3, "+")` → `"13"` |
| `-` | Resta | `calculator(10, 3, "-")` → `"7"` |
| `*` | Multiplicación | `calculator(10, 3, "*")` → `"30"` |
| `/` | División | `calculator(10, 3, "/")` → `"3.3333333333333335"` |

**Manejo de errores:**

| Caso | Respuesta |
|---|---|
| Operador no soportado (ej. `%`, `^`) | `"Error: operador '%' no soportado. Usa +, -, * o /"` |
| División por cero | `"Error: división por cero."` |

**Decisiones de diseño:**

- Los resultados enteros se devuelven sin decimales (`"105"` en vez de `"105.0"`) para que la respuesta del LLM sea más natural.
- Los operadores se despachan mediante un diccionario de lambdas, evitando cadenas de `if/elif` y el uso inseguro de `eval()`.
- La función devuelve mensajes de error como strings (no lanza excepciones), para que el LLM pueda interpretar el error y comunicarlo al usuario.

### 4.2 Lector de archivos (pendiente)

Herramienta que recibirá una ruta a un archivo de texto y devolverá su contenido. Pendiente de implementación.

### 4.3 Herramienta libre (pendiente)

Herramienta a elección. Pendiente de definición e implementación.

---

## 5. El bucle del agente en detalle

### 5.1 Estructura de datos del flujo de mensajes

El bucle mantiene una lista `messages` que crece con cada interacción. A continuación se muestra cómo evoluciona en un ejemplo con una invocación de herramienta:

**Estado inicial (antes de la primera llamada al LLM):**

```python
messages = [
    {"role": "user", "content": "¿Cuánto es 15 * 7?"}
]
```

**Después de recibir un tool_call del LLM (antes de la segunda llamada):**

```python
messages = [
    {"role": "user", "content": "¿Cuánto es 15 * 7?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_abc123", "function": {
            "name": "calculator",
            "arguments": "{\"left_operand\": 15, \"right_operand\": 7, \"operator\": \"*\"}"
        }}
    ]},
    {"role": "tool", "tool_call_id": "call_abc123", "content": "105"}
]
```

Tres roles participan en la conversación:

| Rol | Quién lo genera | Contenido |
|---|---|---|
| `user` | El usuario (mensaje inicial) | La pregunta o instrucción |
| `assistant` | El LLM | Texto libre y/o `tool_calls` |
| `tool` | El agente (resultado de ejecutar la herramienta) | El string devuelto por la función |

### 5.2 Condiciones de terminación

El bucle puede terminar de dos maneras:

1. **Respuesta final del LLM**: el LLM responde con `content` (texto) y sin `tool_calls`. Este es el caso normal y exitoso.
2. **Límite de iteraciones**: se alcanza `max_iterations` (por defecto 10). Esto previene bucles infinitos donde el LLM sigue pidiendo herramientas sin converger. Aun en este caso, se devuelve un `AgentResult` válido con los steps acumulados.

### 5.3 Manejo de errores

El agente maneja tres tipos de error sin romperse:

| Escenario | Qué pasa | Resultado |
|---|---|---|
| **Herramienta desconocida** | El LLM pide una herramienta que no existe | Se registra un `AgentStep` con `error` no nulo y se envía un mensaje de error al LLM para que pueda corregirse |
| **Error de ejecución** | La herramienta lanza una excepción | Se captura, se registra un `AgentStep` con `error`, y se envía el error al LLM |
| **Bucle infinito** | El LLM sigue pidiendo herramientas indefinidamente | Se corta al alcanzar `max_iterations` y se devuelve un `AgentResult` con lo acumulado |

En los tres casos, `agent.run()` **siempre devuelve un `AgentResult` válido**. Nunca lanza una excepción al usuario.

---

## 6. Tipos de datos del framework

### 6.1 Descripción del diagrama de clases (dataclasses)

```
Descripción del diagrama para su armado:

Es un diagrama de clases con las dataclasses del framework y sus relaciones:

┌──────────────────────────────────┐
│         AgentResult              │
├──────────────────────────────────┤
│ answer: str                      │  ← Respuesta final al usuario
│ steps: list[AgentStep]           │  ← Historial de herramientas invocadas
│ error: str | None                │  ← Error general (si hubo)
│ input_tokens: int | None         │  ← Tokens consumidos (entrada)
│ output_tokens: int | None        │  ← Tokens consumidos (salida)
└──────────┬───────────────────────┘
           │ contiene 0..*
           v
┌──────────────────────────────────┐
│          AgentStep               │
├──────────────────────────────────┤
│ tool_name: str | None            │  ← Nombre de la herramienta invocada
│ tool_input: str | None           │  ← JSON con los argumentos enviados
│ tool_output: str | None          │  ← String devuelto por la herramienta
│ error: str | None                │  ← Error (None si fue exitoso)
└──────────────────────────────────┘


┌──────────────────────────────────┐
│         LLMResponse              │
├──────────────────────────────────┤
│ content: str | None              │  ← Texto generado por el LLM
│ tool_calls: list[ToolCall]       │  ← Herramientas que el LLM quiere invocar
│ input_tokens: int | None         │
│ output_tokens: int | None        │
│ raw_response: dict | None        │  ← Payload crudo del proveedor
└──────────┬───────────────────────┘
           │ contiene 0..*
           v
┌──────────────────────────────────┐
│          ToolCall                │
├──────────────────────────────────┤
│ id: str                          │  ← Identificador único de la llamada
│ name: str                        │  ← Nombre de la herramienta a invocar
│ arguments: str                   │  ← JSON con los argumentos
└──────────────────────────────────┘


┌──────────────────────────────────┐
│         ToolSchema               │
├──────────────────────────────────┤
│ name: str                        │  ← Nombre de la herramienta
│ description: str                 │  ← Descripción para el LLM
│ parameters: dict                 │  ← JSON Schema de los parámetros
├──────────────────────────────────┤
│ from_callable(fn) → ToolSchema   │  ← Genera schema desde firma Python
│ to_llm_spec() → dict            │  ← Serializa para el LLMClient
└──────────────────────────────────┘


Relaciones:
- MyAgent tiene 0..* ToolSchema (en _schemas)
- MyAgent tiene 0..* Callable (en _tools), uno por cada ToolSchema
- MyAgent.run() produce 1 AgentResult
- Cada iteración del loop produce 1 LLMResponse
- Cada LLMResponse puede contener 0..* ToolCall
- Cada ToolCall ejecutada produce 1 AgentStep
- Todos los AgentStep se acumulan en el AgentResult final
```

---

## 7. Testing y validación

### 7.1 Estrategia de testing

Los tests de conformidad usan un `MockLLMClient` que devuelve respuestas predeterminadas, eliminando la dependencia del modelo real. Esto permite:

- Tests **deterministas** (siempre el mismo resultado).
- Ejecución **rápida** (sin esperar al LLM).
- Ejecución **en cualquier máquina** (sin Ollama ni API keys).

### 7.2 Tests de conformidad M1

| Test | Qué verifica | Estado |
|---|---|---|
| `test_build_agent_factory_exists` | `build_agent(config)` devuelve un objeto que satisface el protocolo `Agent` | PASA |
| `test_run_returns_agent_result` | `agent.run("hola")` devuelve un `AgentResult` | PASA |
| `test_no_tool_no_loop` | Si el LLM responde con texto sin tool_calls, se devuelve ese texto en un solo turno, con `steps == []` y una sola llamada al LLM | PASA |
| `test_register_tool_signature` | `register_tool(callable, ToolSchema)` expone la herramienta al LLM en el argumento `tools` de `chat()` | PASA |
| `test_tool_is_executed_when_called` | Cuando el LLM emite un tool_call, el agente ejecuta el callable, registra un `AgentStep` con el `tool_name` correcto, y devuelve la respuesta final del LLM | PASA |

### 7.3 Cómo funciona el MockLLMClient

```python
mock = MockLLMClient([
    LLMResponse(content=None, tool_calls=[ToolCall("c1", "calculator", '{"left_operand":2,...}')]),
    LLMResponse(content="La respuesta es 4."),
])
```

Cada llamada a `mock.chat()` consume la siguiente respuesta de la lista. Además, registra todos los argumentos recibidos en `mock.calls` para que los tests puedan verificar qué envió el agente al LLM.

---

## 8. Ejecución local

### 8.1 Infraestructura

El agente corre completamente local usando:

- **Ollama** como servidor de inferencia (puerto 11434).
- **llama3.1** (8B parámetros, ~4.9 GB) como modelo de lenguaje.
- **Contexto de 16384 tokens** (configurado en `OllamaProvider`, suficiente para conversaciones con múltiples turnos de herramientas).

### 8.2 Flujo de ejecución desde la terminal

```bash
.venv/bin/python -m mia_agents.cli run --message "¿Cuánto es 15 * 7?"
```

Produce:

```json
{
  "answer": "El resultado de 15 × 7 es 105.",
  "steps": [
    {
      "tool_name": "calculator",
      "tool_input": "{\"left_operand\": 15, \"operator\": \"*\", \"right_operand\": 7}",
      "tool_output": "105",
      "error": null
    }
  ],
  "error": null,
  "input_tokens": null,
  "output_tokens": null
}
```

---

## 9. Limitaciones conocidas

### 9.1 Limitaciones del modelo local

- **Calidad del tool calling**: llama3.1 (8B) es menos confiable que modelos más grandes para emitir tool_calls bien formados. Puede alucinar nombres de herramientas o enviar JSON malformado. El agente maneja estos casos sin romperse, pero la experiencia del usuario puede no ser ideal.
- **Velocidad de inferencia**: depende del hardware disponible (CPU/GPU). En máquinas sin GPU, cada llamada al LLM puede tardar varios segundos.
- **Ventana de contexto**: aunque se configuran 16384 tokens, conversaciones muy largas con muchas invocaciones de herramientas podrían acercarse al límite.

### 9.2 Limitaciones de la implementación actual

- **Sin historial entre llamadas a `run()`**: en M1, cada llamada es independiente. El agente no recuerda conversaciones anteriores (esto se implementa en M2).
- **Sin `structured_call()`**: la salida estructurada con validación y reparación es un stub que lanza `NotImplementedError` (se implementa en M2).
- **Sin conteo de tokens**: `input_tokens` y `output_tokens` en `AgentResult` quedan como `None` (se implementa en M2).
- **Sin reintentos ante fallos transitorios**: si Ollama falla (timeout, error de red), el agente no reintenta la llamada.

### 9.3 Limitaciones del diseño de herramientas

- Las herramientas solo pueden devolver `str`. No hay soporte para tipos complejos, binarios o streaming.
- No hay mecanismo de autorización: cualquier herramienta registrada está disponible para el LLM en todas las llamadas.
- El LLM decide cuándo y cómo invocar una herramienta. Si el modelo no entiende bien la descripción del schema, puede no usarla o usarla incorrectamente.

---

## 10. Herramientas implementadas (M1 completo)

### 10.1 Conjunto de herramientas

Con esta actualización, el Milestone 1 cuenta con las tres herramientas obligatorias:

| Herramienta | Archivo | Descripción |
|---|---|---|
| `calculator` | `student_framework/tools/calculator.py` | Operaciones aritméticas: `+`, `-`, `*`, `/` |
| `read_text_file` | `student_framework/tools/file_reader.py` | Lectura de archivos de texto plano (UTF-8) |
| `thermo_converter` | `student_framework/tools/thermo_converter.py` | Conversor de unidades termodinámicas (presión, volumen, temperatura, energía, masa, sustancia, constante R) |

Las tres se registran en `build_agent()` (`student_framework/__init__.py`).

### 10.2 Lector de archivos (`read_text_file`)

Esta herramienta implementa el patrón de E/S restringida pedido por la consigna. Su diseño:

- Solo acepta archivos de texto plano con codificación UTF-8.
- Limita el tamaño a 100 KB para no exceder la ventana de contexto del LLM.
- Maneja explícitamente todos los errores posibles (archivo inexistente, directorio, binario, sin permisos) devolviendo mensajes descriptivos como `str` en lugar de lanzar excepciones.
- No utiliza librerías externas: solo `pathlib` de la biblioteca estándar de Python.

El valor pedagógico es directo: permite al agente observar información externa que no está en el prompt ni en el conocimiento del LLM, extendiendo su capacidad de razonamiento más allá del contexto recibido.

### 10.3 Conversor de unidades termodinámicas (`thermo_converter`, herramienta libre)

Se eligió `thermo_converter` como herramienta libre. Convierte entre unidades de la misma magnitud física, cubriendo las siete categorías relevantes en termodinámica:

| Categoría | Unidades soportadas |
|---|---|
| Presión | Pa, hPa, kPa, MPa, GPa, bar, mbar, atm, psi, mmHg, torr, inHg |
| Volumen | m3, L, mL, cm3, dm3, ft3, in3, gal |
| Temperatura | K, C (°C), F (°F), R (Rankine) |
| Energía | J, kJ, MJ, cal, kcal, BTU, Wh, kWh, eV, erg |
| Masa | kg, g, mg, ug, lb, oz, t |
| Sustancia | mol, mmol, umol, nmol, kmol |
| Constante R | J\_mol\_K, kJ\_mol\_K, cal\_mol\_K, kcal\_mol\_K, Latm\_mol\_K, Lbar\_mol\_K, m3Pa\_mol\_K, BTU\_lbmol\_R |

**Decisiones de diseño:**

1. **Conversión de temperatura no lineal**: K ↔ °C ↔ °F ↔ Rankine se manejan con funciones de conversión explícitas (a través de Kelvin como pivote), no con factores multiplicativos.
2. **Constante de gas ideal R como categoría propia**: las unidades compuestas de R (J/(mol·K), L·atm/(mol·K), etc.) se tratan como una magnitud separada con sus propios factores de conversión derivados de R = 8.314 462 J/(mol·K). Esto permite al agente expresar R en el sistema de unidades que el problema requiera.
3. **Detección de errores entre categorías**: si el LLM intenta convertir entre categorías distintas (e.g., `atm` → `J`), la herramienta devuelve un mensaje de error descriptivo en lugar de producir un resultado sin sentido.
4. **Sin librerías externas**: la implementación usa solo Python puro (dicts, `match`, aritmética flotante), sin `pint` ni similares.

**Motivación pedagógica**: el LLM puede confundir factores de conversión o no recordar valores exactos (e.g., 1 atm = 101 325 Pa). Delegar la conversión a Python garantiza exactitud y permite al agente resolver problemas de termodinámica (PV = nRT, ciclos de Carnot, entalpías) sin errores numéricos.

### 10.4 Principio de diseño: sin frameworks externos

No se utilizó ninguna librería de agentes (LangChain, CrewAI, AutoGen, LlamaIndex, Haystack ni similares). El objetivo de la materia es estudiar la implementación manual del ciclo ReAct:

```
LLM devuelve tool_call
    → agente ejecuta callable
    → resultado se agrega al historial como mensaje role=tool
    → LLM vuelve a llamar con contexto actualizado
    → repite hasta texto sin tool_calls
```

Este ciclo es visible línea por línea en `student_framework/agent.py`. Ocultarlo detrás de una librería eliminaría el valor de aprendizaje del TP.
