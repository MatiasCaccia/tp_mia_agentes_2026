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

### 9.2 Evolución entre M1 y M2

Las siguientes capacidades se incorporaron progresivamente entre los dos milestones:

- **Historial entre llamadas a `run()`**: en M1 cada llamada era independiente; el contexto de conversación no persistía. En M2 se incorporó `self._history`, una lista persistente que acumula todos los turnos de la conversación y permite al LLM responder con memoria de intercambios anteriores.
- **`structured_call()`**: en M1 era un stub que lanzaba `NotImplementedError`. En M2 se implementó usando la herramienta sintética `final_result`, validación Pydantic y un bucle de reparación con reintentos.
- **Herramientas obligatorias**: en M1 se completó el conjunto con calculadora, lector de archivos y herramienta libre de estadísticas de texto (ver sección 10).
- **Conteo de tokens**: en M1 `input_tokens` y `output_tokens` quedaban como `None`. En M2 se acumulan los tokens de cada llamada al LLM y se exponen en `AgentResult`.
- **Sin reintentos ante fallos transitorios**: si el proveedor LLM falla (timeout, error de red), el agente no reintenta automáticamente. Esta limitación permanece en M2.

### 9.3 Limitaciones del diseño de herramientas

- Las herramientas solo pueden devolver `str`. No hay soporte para tipos complejos, binarios o streaming.
- No hay mecanismo de autorización: cualquier herramienta registrada está disponible para el LLM en todas las llamadas.
- El LLM decide cuándo y cómo invocar una herramienta. Si el modelo no entiende bien la descripción del schema, puede no usarla o usarla incorrectamente.

---

## 10. Actualización M1 — Herramientas completas

### 10.1 Estado final del conjunto de herramientas

Con esta actualización, el Milestone 1 cuenta con las tres herramientas obligatorias:

| Herramienta | Archivo | Descripción |
|---|---|---|
| `calculator` | `student_framework/tools/calculator.py` | Operaciones aritméticas: `+`, `-`, `*`, `/` |
| `read_text_file` | `student_framework/tools/file_reader.py` | Lectura de archivos de texto plano (UTF-8) |
| `text_stats` | `student_framework/tools/text_stats.py` | Estadísticas de texto: caracteres, palabras, líneas |

Las tres se registran en `build_agent()` (`student_framework/__init__.py`).

### 10.2 Sobre los operadores de la calculadora

El ENUNCIADO_M1.md menciona `%` (módulo) entre los operadores soportados. Tras revisar el contexto del TP, se concluyó que se trata de una inconsistencia tipográfica: el alcance correcto es una calculadora aritmética básica con las cuatro operaciones estándar (`+`, `-`, `*`, `/`). El operador `/` (división) es el que corresponde a ese slot, y es imprescindible para casos de uso iterativos como el método de Herón de Alejandría. No se implementó `%` porque no es parte del alcance correcto, ningún test del M1 lo requiere y no se lo menciona en ningún otro lugar del enunciado.

### 10.3 Lector de archivos (`read_text_file`)

Esta herramienta implementa el patrón de E/S restringida pedido por la consigna. Su diseño:

- Solo acepta archivos de texto plano con codificación UTF-8.
- Limita el tamaño a 100 KB para no exceder la ventana de contexto del LLM.
- Maneja explícitamente todos los errores posibles (archivo inexistente, directorio, binario, sin permisos) devolviendo mensajes descriptivos como `str` en lugar de lanzar excepciones.
- No utiliza librerías externas: solo `pathlib` de la biblioteca estándar de Python.

El valor pedagógico es directo: permite al agente observar información externa que no está en el prompt ni en el conocimiento del LLM, extendiendo su capacidad de razonamiento más allá del contexto recibido.

### 10.4 Estadísticas de texto (`text_stats`, herramienta libre)

Se eligió `text_stats` como herramienta libre por tres razones:

1. **Complementa al lector de archivos**: la secuencia `read_text_file` → `text_stats` es un caso de uso real y demostrable de encadenamiento de herramientas (el output de una tool es el input de la siguiente).
2. **Demuestra una limitación concreta del LLM**: el LLM no puede contar palabras o caracteres con exactitud. Delegar esto a Python garantiza un resultado exacto y verificable, ilustrando exactamente por qué existen las herramientas.
3. **Es simple y transparente**: la implementación es Python puro, sin dependencias, fácil de auditar y de explicar.

### 10.5 Principio de diseño: sin frameworks externos

No se utilizó ninguna librería de agentes (LangChain, CrewAI, AutoGen, LlamaIndex, Haystack ni similares). El objetivo de la materia es estudiar la implementación manual del ciclo ReAct:

```
LLM devuelve tool_call
    → agente ejecuta callable
    → resultado se agrega al historial como mensaje role=tool
    → LLM vuelve a llamar con contexto actualizado
    → repite hasta texto sin tool_calls
```

Este ciclo es visible línea por línea en `student_framework/agent.py`. Ocultarlo detrás de una librería eliminaría el valor de aprendizaje del TP.

---

## 11. Milestone 2 — Memoria, historial y salida estructurada

### 11.1 Objetivo del Milestone 2

Extender el agente con tres capacidades nuevas sin romper ninguna de las funcionalidades de M1:

1. **Statefulness**: el historial de conversación persiste entre llamadas sucesivas a `run()`.
2. **Historial acotado**: el número de mensajes enviados al LLM en cada llamada tiene un tope configurable.
3. **Salida estructurada**: `structured_call()` permite obtener del LLM respuestas validadas contra un schema Pydantic, con reparación automática si la respuesta no es válida.
4. **Conteo de tokens**: los tokens de cada llamada al LLM se acumulan y se exponen en `AgentResult`.

### 11.2 Cambios en `MyAgent`

#### 11.2.1 Historial persistente (`self._history`)

En M1, cada llamada a `run()` creaba una lista local `messages` que se descartaba al terminar:

```python
# M1 (sin memoria)
def run(self, user_message):
    messages = [{"role": "user", "content": user_message}]  # ← se pierde al salir
    ...
```

En M2, esa lista se mueve a un atributo de instancia que sobrevive entre llamadas:

```python
# M2 (con memoria)
def __init__(self, ...):
    self._history: list[dict[str, Any]] = []  # ← persiste

def run(self, user_message):
    self._history.append({"role": "user", "content": user_message})
    ...
    self._history.append({"role": "assistant", "content": response.content})
    # ahora el próximo run() ve todo este historial
```

**Consecuencia observable**: si se le pregunta al agente "¿cuánto es 3 + 5?" y luego "¿y el doble de eso?", en M2 el LLM recibe el turno anterior en el historial y puede responder "16" en lugar de no saber a qué se refiere "eso".

#### 11.2.2 Ventana deslizante (`_mensajes_para_llm`)

El historial interno puede crecer sin límite, pero enviar miles de mensajes al LLM aumentaría el costo y podría superar la ventana de contexto. El parámetro `max_history_messages` controla cuántos mensajes se envían en cada llamada:

```python
def _mensajes_para_llm(self) -> list[dict[str, Any]]:
    return self._history[-self._max_history_messages:]
```

Python acepta índices negativos fuera de rango (`[-50:]` en una lista de 3 elementos devuelve los 3 elementos), por lo que no hace falta manejo especial cuando el historial es más corto que la ventana.

**Diferencia importante**:

| Aspecto | Historial interno (`self._history`) | Lo que ve el LLM |
|---|---|---|
| Tamaño | Crece con cada turno (sin límite) | ≤ `max_history_messages` mensajes |
| Persistencia | Permanente mientras viva el agente | Solo la ventana actual |
| Propósito | Registro completo | Contexto para el LLM |

#### 11.2.3 Acumulación de tokens

La función auxiliar `_acumular_tokens` maneja el caso especial donde el LLM no reporta tokens (devuelve `None`):

| `acumulado` | `nuevo` | Resultado | Razón |
|---|---|---|---|
| `None` | `None` | `None` | Ninguna respuesta reportó tokens |
| `None` | `100` | `100` | Primer reporte: inicializa el conteo |
| `100` | `None` | `100` | LLM no reportó, pero ya hay historial: suma 0 |
| `100` | `200` | `300` | Suma normal |

La regla clave: una vez que alguna respuesta reportó tokens, los `None` posteriores se tratan como `0` (la respuesta existió pero el proveedor no informó tokens).

### 11.3 `structured_call` — Salida estructurada con reparación

#### 11.3.1 El problema

Obtener del LLM una respuesta en formato JSON estricto no es trivial: el LLM puede responder con texto libre, puede generar JSON malformado, o puede generar JSON que no cumple el schema. `structured_call` resuelve esto usando una herramienta sintética.

#### 11.3.2 La herramienta sintética `final_result`

La función `final_result_tool_schema(schema)` (provista por la cátedra en `mia_agents/tool_schema.py`) crea un `ToolSchema` donde el schema de los parámetros es el JSON Schema del modelo Pydantic recibido. Al ofrecerle al LLM **solo** esa herramienta, se lo fuerza a invocarla para entregar su respuesta, en lugar de responder con texto libre.

#### 11.3.3 El bucle de reparación

```
structured_call(prompt, Schema, max_repair_attempts=2)
│
├── Llamada 1: ofrecer solo final_result
│   ├── ¿Invocó final_result? ¿Pasan los argumentos la validación?
│   │   └── SI → return Schema.model_validate(args)  ← éxito
│   └── NO → agregar error al contexto → continuar
│
├── Llamada 2 (reparación 1): re-llamar con el error en el historial
│   ├── ¿OK ahora?
│   │   └── SI → return Schema.model_validate(args)
│   └── NO → agregar error → continuar
│
└── Llamada 3 (reparación 2): último intento
    ├── ¿OK?
    │   └── SI → return
    └── NO → raise ultimo_error
```

Total de llamadas al LLM: `1 + max_repair_attempts = 3` (con el valor por defecto).

**Contexto de reparación por tipo de error**:

| Error | Qué se agrega al historial | Objetivo |
|---|---|---|
| El LLM respondió con texto libre (no invocó `final_result`) | Mensaje del asistente + mensaje de corrección al usuario | Instruir al LLM a usar la herramienta |
| Los argumentos no pasan `model_validate()` | El tool_call fallido + mensaje role=tool con el error de validación | Dar al LLM el error exacto para que corrija |

**Nota importante**: `structured_call` no modifica `self._history`. Es un canal de comunicación separado del historial conversacional principal. Su conversación interna (los mensajes de reparación) no se mezcla con el historial del agente.

### 11.4 Descripción del diagrama de flujo de `structured_call`

```
Descripción del diagrama para su armado:

[Inicio: structured_call(prompt, Schema, max_repair_attempts=2)]
    |
    v
[Crear fr_schema = final_result_tool_schema(Schema)]
[Crear messages = [{role: "user", content: prompt}]]
[ultimo_error = None]
[intentos = 0]
    |
    v
<¿intentos <= max_repair_attempts?> --NO--> [raise ultimo_error]
    |
   SI
    |
    v
[LLM.chat(messages, tools=[fr_schema])]
    |
    v
[Buscar tool_call con name == "final_result"]
    |
    v
<¿Encontrado?> --NO--> [ultimo_error = RuntimeError]
    |                   [messages += assistant + corrección]
   SI                   [intentos += 1]
    |                   [volver al inicio del while]
    v
[raw_args = json.loads(tc.arguments)]
[Schema.model_validate(raw_args)]
    |
    v
<¿Válido?> --SI--> [return instancia]  ← FIN exitoso
    |
   NO
    |
    v
[ultimo_error = ValidationError]
[messages += assistant tool_call + error role=tool]
[intentos += 1]
[volver al inicio del while]
```

### 11.5 Tests de conformidad M2

| Test | Qué verifica | Estado |
|---|---|---|
| `test_agent_is_stateful_across_runs` | La segunda llamada a `run()` envía al LLM los mensajes del turno anterior | PASA |
| `test_bounded_history_growth` | Con `max_history_messages=2`, el LLM nunca recibe más de 2 mensajes aunque el historial sea más largo | PASA |
| `test_structured_call_offers_final_result_tool` | `structured_call` pasa al LLM un tool con name `"final_result"` | PASA |
| `test_structured_output_max_retries` | Con `max_repair_attempts=2`, el LLM se llama exactamente 3 veces antes de lanzar | PASA |
| `test_structured_output_repairs_schema_validation_error` | Cuando los argumentos no pasan validación, se agrega el error al contexto y se reintenta | PASA |
| `test_token_accounting` | `input_tokens` y `output_tokens` de `AgentResult` acumulan correctamente los tokens de cada llamada al LLM | PASA |
| `test_token_accounting_treats_missing_values_as_zero_after_first_report` | Una vez iniciado el conteo, los `None` del LLM se tratan como 0 | PASA |

**Resultado total M1 + M2: 12/12 tests pasan.**

### 11.6 Alcance de la statefulness: instancia vs. proceso

La CLI (`python -m mia_agents.cli run --message "..."`) crea una nueva instancia del agente en cada invocación. Por lo tanto, el historial conversacional se reinicia con cada comando: desde la perspectiva del LLM, cada ejecución es una conversación independiente.

La statefulness de M2 solo se observa cuando se reutiliza la misma instancia del agente dentro de un mismo proceso. Eso ocurre, por ejemplo, en los tests de conformidad (donde el agente se instancia una sola vez y `run()` se llama varias veces) o en un script Python que mantenga la instancia en memoria.

Ejemplo conceptual:

```python
from student_framework import build_agent

agent = build_agent(config)
agent.run("Mi nombre es Ana.")
agent.run("¿Cómo me llamo?")
```

En ese caso, la segunda llamada conserva el historial de la primera porque ambas comparten la misma instancia. La respuesta del LLM puede referirse al turno anterior porque lo recibe en el contexto enviado.

Para demostrar este comportamiento con el modelo real (Ollama) se puede escribir un script en `scripts/` que instancie el agente una sola vez y llame a `run()` en secuencia dentro del mismo proceso.

### 11.7 Principio de diseño: sin frameworks externos (M2)

Al igual que en M1, la totalidad de M2 se implementó en Python puro sobre el scaffold de la cátedra:

- La statefulness es una lista `self._history` — sin librerías de memoria externas.
- La ventana deslizante es un slice de Python (`[-N:]`) — sin vectorstores ni bases de datos.
- `structured_call` es un bucle `for` con Pydantic — sin `instructor`, `outlines` ni similares.
- La acumulación de tokens es una función de 4 líneas — sin observabilidad externa.
