# Especificación técnica para implementar un flujo tipo Unlimited OCR / R-SWA en un proyecto básico de agentes

**Versión:** 1.0  
**Idioma:** Español  
**Propósito:** describir cómo llevar la idea del paper *Unlimited OCR Works: Welcome the Era of One-shot Long-horizon Parsing* a un proyecto básico de agentes, con una arquitectura implementable, modular y justificable.  
**Audiencia:** equipos de desarrollo, investigación aplicada, ingeniería de datos, NLP/OCR y agentes LLM.  

---

## 1. Resumen ejecutivo

El trabajo *Unlimited OCR* propone una forma de resolver un cuello de botella típico de los modelos OCR generativos: cuando un modelo transcribe documentos largos, la memoria interna asociada a los tokens generados —la **KV cache**— crece linealmente con la longitud de la salida. Eso hace que la generación se vuelva cada vez más costosa en memoria y más lenta.

La solución central del paper es **Reference Sliding Window Attention** (**R-SWA**), una variante de atención que separa dos tipos de contexto:

1. **Referencia permanente:** el documento original, sus tokens visuales y el prompt. Esta parte siempre está disponible.
2. **Memoria operativa reciente:** solo los últimos `n` tokens generados, por ejemplo `n = 128`.

En vez de permitir que cada token generado atienda a todo el historial de salida, el modelo atiende a:

```text
referencia completa + últimos n tokens generados
```

Esto permite que el costo de memoria de la salida se mantenga constante:

```text
Atención estándar:       C(T) = Lm + T
R-SWA:                   C(T) = Lm + min(n, T)
Cuando T > n:            C(T) = Lm + n
```

Donde:

- `Lm` es la longitud de la referencia fija: tokens visuales + prompt;
- `T` es la cantidad de tokens generados;
- `n` es el tamaño de la ventana deslizante.

La intuición es parecida a copiar un libro: una persona no necesita recordar todo lo que ya copió desde la primera página. Necesita mirar la fuente original, conservar algo del contexto reciente y continuar.

Este documento traduce esa idea a un **proyecto básico de agentes**, con dos niveles de implementación:

1. **Implementación agente inspirada en R-SWA**, posible con LLMs estándar, APIs comerciales o modelos open-source sin tocar internals del Transformer.
2. **Implementación fiel a R-SWA a nivel modelo**, que requiere modificar la máscara de atención y el manejo de KV cache en el motor de inferencia.

La recomendación práctica es comenzar por el primer nivel, porque permite validar producto, flujo, métricas y errores sin necesidad de entrenar un modelo ni modificar kernels de atención. Luego, si el proyecto demuestra valor, se puede avanzar al segundo nivel.

---

## 2. Objetivo del proyecto

El objetivo del proyecto es construir un sistema de agentes capaz de procesar documentos largos, mantener una referencia estable del documento fuente y generar una salida estructurada sin arrastrar indefinidamente todo el historial textual generado.

Aunque el paper se enfoca en OCR, el patrón es general para tareas de **parsing basado en referencia**:

- OCR de documentos multipágina;
- extracción de información de PDFs;
- transcripción de audio largo;
- traducción documento-a-documento;
- resumen estructurado de documentos largos;
- conversión de documentos a Markdown/JSON/HTML;
- revisión de contratos o historias clínicas;
- lectura de expedientes;
- agentes que operan sobre una fuente fija y producen una salida extensa.

En un proyecto de agentes, la idea se puede reformular así:

> El sistema debe mantener una **fuente de verdad estable** y una **memoria de trabajo corta**, evitando que el agente dependa de todo el historial conversacional para continuar.

---

## 3. Principio de diseño: referencia fija + memoria de trabajo

### 3.1. Problema con agentes tradicionales

Un agente LLM básico suele funcionar acumulando historial:

```text
system prompt
+ instrucciones
+ documento o fragmentos recuperados
+ mensajes anteriores
+ outputs anteriores
+ nuevo pedido
```

Esto escala mal. A medida que el agente trabaja:

- crece el contexto;
- aumenta el costo por llamada;
- aparecen repeticiones;
- se diluye la instrucción principal;
- el modelo puede confundirse con outputs anteriores;
- se dificulta procesar documentos largos;
- se incrementa el riesgo de inconsistencias.

En OCR o parsing largo, este patrón es especialmente problemático porque el agente puede terminar “leyendo su propia salida” más que la fuente original.

### 3.2. Patrón R-SWA aplicado a agentes

En lugar de acumular todo, se separa el estado en tres componentes:

```text
1. Referencia fija
   Documento, páginas, imágenes, chunks, embeddings, metadatos.

2. Memoria de trabajo
   Últimos K tokens, párrafos, bloques o eventos generados.

3. Estado estructurado
   Progreso, página actual, bloque actual, offsets, validaciones, checkpoints.
```

El agente nunca debería depender de todo el texto generado previamente. Para continuar, debería recibir:

```text
- la parte relevante de la referencia original;
- el objetivo de la tarea;
- el estado estructurado de progreso;
- una ventana corta de salida reciente.
```

Esto permite construir un sistema más robusto, barato y controlable.

---

## 4. Qué significa “implementarlo” en un proyecto de agentes

Hay dos interpretaciones posibles.

### 4.1. Implementación pragmática, sin modificar el modelo

Esta es la opción recomendada para un MVP.

No se modifica la atención interna del LLM. En cambio, se implementa el patrón a nivel de orquestación:

- el documento fuente se guarda como **referencia externa**;
- se recuperan chunks relevantes por página, bloque o posición;
- la salida generada se guarda completa en almacenamiento externo;
- al agente se le pasa solo una ventana reciente de la salida;
- el estado de progreso se guarda en una estructura controlada;
- agentes validadores revisan consistencia, repeticiones y cobertura.

Esta versión no logra una KV cache constante real dentro del modelo si se usa una API externa, pero sí logra los beneficios de diseño:

- prompts más cortos;
- menor dependencia del historial;
- más control del flujo;
- menor repetición;
- mejor capacidad de reanudar;
- separación clara entre fuente, memoria y output.

### 4.2. Implementación fiel, modificando atención y KV cache

Esta opción requiere acceso al modelo y al motor de inferencia.

Implica:

- usar un modelo visión-lenguaje o texto-lenguaje open-source;
- modificar la máscara de atención;
- conservar siempre la KV cache del prefijo/referencia;
- conservar solo los últimos `n` tokens generados;
- desalojar de la KV cache los tokens antiguos de salida;
- adaptar el motor de inferencia para que soporte R-SWA.

Es más compleja, pero permite reproducir la contribución técnica real del paper.

---

## 5. Arquitectura propuesta para un proyecto básico de agentes

La arquitectura recomendada tiene seis módulos principales:

```text
┌──────────────────────────┐
│ 1. Ingestor de documento │
└────────────┬─────────────┘
             │
             v
┌──────────────────────────┐
│ 2. Constructor de        │
│    referencia            │
└────────────┬─────────────┘
             │
             v
┌──────────────────────────┐
│ 3. Orquestador /         │
│    Coordinator Agent     │
└────────────┬─────────────┘
             │
             v
┌──────────────────────────┐
│ 4. Parser Agent          │
└────────────┬─────────────┘
             │
             v
┌──────────────────────────┐
│ 5. Validator Agent       │
└────────────┬─────────────┘
             │
             v
┌──────────────────────────┐
│ 6. Ensamblador de salida │
└──────────────────────────┘
```

### 5.1. Ingestor de documento

Responsabilidad:

- recibir PDFs, imágenes, texto plano o audio;
- extraer páginas;
- generar representaciones intermedias;
- preservar metadatos.

Para OCR básico, puede usar:

- imágenes renderizadas por página;
- OCR tradicional inicial;
- layout detection;
- extracción de texto embebido si el PDF ya lo contiene;
- coordenadas de bloques.

Salida esperada:

```json
{
  "document_id": "doc_001",
  "pages": [
    {
      "page_id": "p001",
      "page_number": 1,
      "image_path": "data/pages/p001.png",
      "width": 1024,
      "height": 1024,
      "raw_text": "...",
      "blocks": []
    }
  ]
}
```

### 5.2. Constructor de referencia

Responsabilidad:

- transformar el documento en una referencia estable;
- dividirlo en chunks;
- asignar identificadores persistentes;
- generar embeddings si se usa recuperación semántica;
- guardar layout, coordenadas y orden de lectura estimado.

La referencia debe ser tratada como **inmutable** durante la generación. El agente puede consultarla, pero no sobrescribirla.

Ejemplo de chunk:

```json
{
  "chunk_id": "doc_001_p003_b007",
  "document_id": "doc_001",
  "page_number": 3,
  "block_number": 7,
  "type": "paragraph",
  "bbox": [72, 155, 901, 238],
  "text_hint": "...",
  "image_crop_path": "data/crops/doc_001_p003_b007.png",
  "embedding_id": "emb_abc123"
}
```

### 5.3. Coordinator Agent

Responsabilidad:

- decidir qué parte del documento se procesa;
- llamar al Parser Agent;
- actualizar el estado;
- invocar validaciones;
- reintentar si hay errores;
- controlar checkpoints;
- evitar acumulación innecesaria de contexto.

El coordinador es el equivalente agente del mecanismo de avance continuo.

No debería pasarle al Parser Agent todo el historial. Debería pasarle:

```text
- instrucción estable;
- referencia relevante;
- estado actual;
- memoria reciente;
- formato esperado de salida.
```

### 5.4. Parser Agent

Responsabilidad:

- leer la referencia relevante;
- producir la siguiente unidad de salida;
- mantener formato consistente;
- no inventar contenido;
- marcar incertidumbres.

La unidad de salida puede ser:

- un bloque;
- una página;
- un párrafo;
- una tabla;
- una sección Markdown;
- un objeto JSON;
- un segmento de transcripción.

Para un MVP, conviene que el Parser Agent trabaje por **bloques o páginas**, no token por token.

### 5.5. Validator Agent

Responsabilidad:

- detectar repeticiones;
- verificar que la salida corresponde a la referencia;
- revisar continuidad;
- identificar saltos de página omitidos;
- comprobar formato;
- detectar alucinaciones;
- medir diferencias contra OCR base si existe.

El validador no debe reescribir todo salvo que sea necesario. Debe devolver issues estructurados.

Ejemplo:

```json
{
  "status": "needs_revision",
  "issues": [
    {
      "type": "possible_omission",
      "severity": "medium",
      "page": 4,
      "block_id": "doc_001_p004_b002",
      "message": "El bloque contiene una fórmula que no aparece en la salida."
    }
  ]
}
```

### 5.6. Ensamblador de salida

Responsabilidad:

- guardar la salida completa;
- unir segmentos;
- resolver marcadores de página;
- exportar a Markdown, JSON, HTML o texto;
- generar logs y métricas.

La salida completa no vive dentro del prompt del agente. Vive en almacenamiento externo.

---

## 6. Mapeo entre Unlimited OCR y sistema de agentes

| Concepto del paper | Equivalente en proyecto de agentes |
|---|---|
| Tokens visuales | Páginas, crops, chunks, texto OCR base, embeddings |
| Prompt | Instrucción estable del sistema |
| Prefijo `P` | Referencia fija consultable |
| Decode region | Salida que el agente va generando |
| Sliding window `n` | Últimos K tokens/bloques/mensajes generados |
| KV cache constante | Contexto corto y controlado en cada llamada |
| Soft forgetting | No pasar todo el historial; solo memoria reciente |
| Long-horizon parsing | Procesamiento continuo de documentos largos |
| `<page>` token | Separador explícito entre páginas o secciones |

---

## 7. Diseño del estado del agente

El sistema debe guardar el estado en una estructura explícita, no depender de que el LLM “recuerde”.

### 7.1. Estado mínimo recomendado

```json
{
  "run_id": "run_2026_001",
  "document_id": "doc_001",
  "task": "convert_to_markdown",
  "current_page": 5,
  "current_block": 12,
  "last_completed_page": 4,
  "last_completed_block": 11,
  "output_path": "outputs/doc_001.md",
  "working_memory": [
    {
      "type": "generated_block",
      "page": 5,
      "block": 10,
      "text": "..."
    },
    {
      "type": "generated_block",
      "page": 5,
      "block": 11,
      "text": "..."
    }
  ],
  "validation_status": "ok",
  "errors": [],
  "retries": 0
}
```

### 7.2. Por qué el estado debe ser estructurado

El estado estructurado evita depender de memoria conversacional difusa. Esto permite:

- reanudar el procesamiento;
- auditar decisiones;
- evitar procesar dos veces el mismo bloque;
- detectar omisiones;
- paralelizar páginas;
- comparar salida contra referencia;
- reiniciar desde checkpoints.

---

## 8. Memoria de trabajo deslizante

La memoria de trabajo es el equivalente agente de la ventana `n` de R-SWA.

Puede medirse de varias maneras:

- últimos `K` bloques;
- últimos `K` párrafos;
- últimos `N` tokens;
- últimos `M` caracteres;
- últimas `P` páginas procesadas.

Para un proyecto básico, se recomienda usar bloques o caracteres, porque es más simple que controlar tokens exactos.

### 8.1. Configuración sugerida

```yaml
working_memory:
  mode: "blocks"
  max_blocks: 5
  max_chars: 6000
  include_last_page_summary: true
  include_last_output_fragment: true
```

### 8.2. Política de expulsión

Cuando la memoria supera el límite:

```text
1. Se elimina el bloque más antiguo.
2. Se conserva un resumen breve si es necesario.
3. Se mantiene el output completo fuera del prompt.
```

Pseudocódigo:

```python
class WorkingMemory:
    def __init__(self, max_blocks=5, max_chars=6000):
        self.max_blocks = max_blocks
        self.max_chars = max_chars
        self.items = []

    def add(self, item):
        self.items.append(item)
        self._evict()

    def _evict(self):
        while len(self.items) > self.max_blocks or self._char_count() > self.max_chars:
            self.items.pop(0)

    def _char_count(self):
        return sum(len(x.get("text", "")) for x in self.items)

    def render(self):
        return "\n\n".join(item["text"] for item in self.items)
```

---

## 9. Flujo completo del sistema

### 9.1. Etapa 1: ingesta

```text
Input: PDF/documento
Output: páginas, imágenes, texto base, metadatos
```

Pasos:

1. Crear `document_id`.
2. Renderizar páginas si el input es PDF.
3. Extraer texto embebido si existe.
4. Aplicar OCR base opcional.
5. Detectar bloques si se necesita layout.
6. Guardar imágenes, crops y JSON de referencia.

### 9.2. Etapa 2: construcción de referencia

```text
Input: páginas + bloques
Output: referencia consultable
```

Pasos:

1. Dividir el documento en chunks.
2. Normalizar coordenadas a `[0, 1000]` si se trabaja con layout.
3. Crear índices por página y bloque.
4. Generar embeddings si se usa retrieval.
5. Guardar la referencia como inmutable.

### 9.3. Etapa 3: planificación

```text
Input: referencia
Output: plan de parsing
```

Ejemplo de plan:

```json
{
  "strategy": "page_then_block",
  "pages": [1, 2, 3, 4, 5],
  "output_format": "markdown",
  "preserve_tables": true,
  "preserve_page_breaks": true
}
```

### 9.4. Etapa 4: generación incremental

Para cada unidad de trabajo:

```text
1. Recuperar referencia relevante.
2. Renderizar prompt corto.
3. Llamar al Parser Agent.
4. Validar salida.
5. Guardar salida en storage externo.
6. Actualizar memoria de trabajo.
7. Avanzar estado.
```

### 9.5. Etapa 5: validación global

Al finalizar:

- revisar cobertura de páginas;
- detectar páginas omitidas;
- detectar repeticiones;
- verificar formato Markdown/JSON;
- comparar longitud esperada vs generada;
- generar reporte de confianza.

---

## 10. Prompting recomendado

### 10.1. System prompt del Parser Agent

```text
Sos un agente de parsing documental. Tu tarea es convertir la referencia provista en una salida fiel, estructurada y verificable.

Reglas:
- Usá únicamente la referencia proporcionada.
- No inventes contenido.
- Conservá el orden de lectura.
- Si un fragmento es ilegible, marcá [ILEGIBLE] en vez de inventar.
- Si hay una tabla, mantené su estructura en Markdown.
- Si hay una fórmula, conservá la notación lo mejor posible.
- No repitas bloques ya generados salvo que la referencia los contenga repetidos.
- Continuá desde el estado indicado.
```

### 10.2. Prompt de llamada por bloque o página

```text
# Tarea
Convertir el siguiente fragmento de referencia a Markdown.

# Estado actual
Documento: {document_id}
Página actual: {page_number}
Bloque actual: {block_id}
Último bloque completado: {last_completed_block}

# Memoria reciente
{working_memory}

# Referencia fija relevante
{reference_chunk}

# Formato de salida
Devolver solo el Markdown del bloque actual. No agregar explicaciones.
```

### 10.3. Prompt del Validator Agent

```text
Sos un agente validador. Compará la salida generada contra la referencia.

Debés revisar:
- omisiones;
- repeticiones;
- alucinaciones;
- errores de orden de lectura;
- pérdida de tablas o fórmulas;
- problemas de formato.

Respondé en JSON con:
- status: ok | needs_revision | failed
- issues: lista de problemas
- corrected_output: solo si la corrección es local y segura
```

---

## 11. Pseudocódigo del orquestador agente

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class AgentState:
    run_id: str
    document_id: str
    current_page: int = 1
    current_block: int = 0
    output_path: str = "output.md"
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    completed_blocks: List[str] = field(default_factory=list)

class ReferenceStore:
    def get_next_block(self, state: AgentState):
        """Devuelve el próximo bloque según current_page/current_block."""
        raise NotImplementedError

    def get_block_reference(self, block_id: str):
        """Devuelve texto, layout, imagen/crop o metadatos del bloque."""
        raise NotImplementedError

class OutputStore:
    def append(self, path: str, text: str):
        with open(path, "a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n\n")

class Coordinator:
    def __init__(self, parser_agent, validator_agent, reference_store, output_store, memory):
        self.parser_agent = parser_agent
        self.validator_agent = validator_agent
        self.reference_store = reference_store
        self.output_store = output_store
        self.memory = memory

    def run(self, state: AgentState):
        while True:
            block = self.reference_store.get_next_block(state)
            if block is None:
                break

            reference = self.reference_store.get_block_reference(block["block_id"])

            parser_input = {
                "document_id": state.document_id,
                "page": block["page"],
                "block_id": block["block_id"],
                "working_memory": self.memory.render(),
                "reference": reference,
            }

            generated = self.parser_agent.parse(parser_input)

            validation = self.validator_agent.validate(
                reference=reference,
                generated=generated,
                state=state,
            )

            if validation["status"] == "ok":
                final_text = generated
            elif validation["status"] == "needs_revision" and validation.get("corrected_output"):
                final_text = validation["corrected_output"]
            else:
                final_text = self._retry_or_mark_uncertain(parser_input, validation)

            self.output_store.append(state.output_path, final_text)
            self.memory.add({"block_id": block["block_id"], "text": final_text})
            state.completed_blocks.append(block["block_id"])
            self._advance_state(state, block)

        return state

    def _retry_or_mark_uncertain(self, parser_input, validation):
        # En un MVP, conviene no entrar en loops infinitos.
        # Se puede hacer un único retry y luego marcar incertidumbre.
        parser_input["validation_feedback"] = validation
        retry_output = self.parser_agent.parse(parser_input)
        return retry_output

    def _advance_state(self, state, block):
        state.current_page = block["page"]
        state.current_block = block["block_number"] + 1
```

---

## 12. Versión mínima viable recomendada

Para un proyecto básico, no hace falta arrancar con OCR visual de punta a punta. La primera versión puede usar un OCR existente y aplicar el patrón R-SWA a nivel de agentes.

### 12.1. MVP funcional

Componentes:

1. **PDF loader**
   - convierte PDF a páginas;
   - extrae texto si existe;
   - opcionalmente usa OCR tradicional.

2. **Reference store**
   - guarda páginas y bloques;
   - permite recuperar por página/bloque.

3. **Parser Agent**
   - convierte bloques a Markdown estructurado.

4. **Working Memory Manager**
   - conserva solo últimos 3 a 5 bloques.

5. **Validator Agent**
   - revisa salida contra referencia.

6. **Output store**
   - guarda el Markdown final.

### 12.2. Stack sugerido

```text
Lenguaje: Python
Orquestación: LangGraph, CrewAI, AutoGen o implementación propia simple
Storage: filesystem + SQLite
Embeddings: FAISS o Chroma si hay recuperación semántica
OCR base: PaddleOCR, Tesseract, DocTR o proveedor externo
LLM: modelo local o API externa
Formato salida: Markdown y JSONL de trazabilidad
```

### 12.3. Estructura de carpetas

```text
project/
  README.md
  config.yaml
  data/
    raw/
    pages/
    crops/
    references/
  outputs/
    markdown/
    json/
    reports/
  src/
    ingest/
      pdf_loader.py
      ocr_adapter.py
      layout_detector.py
    reference/
      reference_store.py
      chunker.py
      embeddings.py
    agents/
      coordinator.py
      parser_agent.py
      validator_agent.py
    memory/
      working_memory.py
    evaluation/
      metrics.py
      report.py
    main.py
```

---

## 13. Configuración recomendada

```yaml
project:
  name: "r-swa-inspired-document-agent"
  output_format: "markdown"

input:
  type: "pdf"
  render_dpi: 160
  normalize_coordinates: true

reference:
  chunk_strategy: "page_block"
  immutable: true
  use_embeddings: false
  include_layout: true
  include_images: false

working_memory:
  mode: "blocks"
  max_blocks: 5
  max_chars: 6000
  summarize_evicted: true

parser_agent:
  model: "your-llm"
  temperature: 0.0
  max_output_tokens: 1500
  preserve_tables: true
  preserve_formulas: true
  mark_illegible: true

validator_agent:
  enabled: true
  model: "your-llm"
  temperature: 0.0
  max_retries: 1

output:
  markdown_path: "outputs/markdown/document.md"
  trace_path: "outputs/json/trace.jsonl"
  report_path: "outputs/reports/report.md"
```

---

## 14. Implementación más fiel a R-SWA a nivel modelo

Si el proyecto requiere reproducir el comportamiento técnico del paper, hay que modificar la atención del decodificador.

### 14.1. Idea formal

Sea:

```text
P = {1, ..., Lm}
```

el conjunto de tokens de referencia: imagen, prompt, instrucciones.

Sea:

```text
Dn(t) = últimos n tokens generados antes de t
```

Entonces el token actual atiende a:

```text
N(t) = P ∪ Dn(t)
```

No atiende a todos los tokens generados desde el inicio.

### 14.2. Máscara de atención conceptual

En atención estándar causal:

```text
Cada token atiende a todo lo anterior.
```

En R-SWA:

```text
Cada token generado atiende a:
- todos los tokens con índice <= Lm;
- tokens generados dentro de la ventana reciente n;
- no atiende a outputs antiguos fuera de la ventana.
```

### 14.3. Manejo de KV cache

La cache se divide en dos partes:

```text
prefix_cache: fija, nunca se elimina
output_cache: cola circular de tamaño n
```

Pseudocódigo:

```python
class RSWAKVCache:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.prefix_k = None
        self.prefix_v = None
        self.decode_k_queue = []
        self.decode_v_queue = []

    def set_prefix(self, k, v):
        self.prefix_k = k
        self.prefix_v = v

    def append_decode(self, k, v):
        self.decode_k_queue.append(k)
        self.decode_v_queue.append(v)

        if len(self.decode_k_queue) > self.window_size:
            self.decode_k_queue.pop(0)
            self.decode_v_queue.pop(0)

    def get_attention_cache(self):
        k = concat([self.prefix_k] + self.decode_k_queue, dim="seq")
        v = concat([self.prefix_v] + self.decode_v_queue, dim="seq")
        return k, v
```

### 14.4. Consideraciones importantes

Implementar esto de verdad no es solo cambiar un prompt. Hay que asegurarse de que:

- las posiciones relativas/absolutas sean consistentes;
- el motor de inferencia no asuma KV cache creciente;
- los kernels de atención acepten prefijo fijo + ventana móvil;
- la máscara impida atender a tokens expulsados;
- el modelo haya sido entrenado o adaptado para ese patrón;
- se evalúe si el modelo pierde coherencia por reducir historial.

### 14.5. Riesgo de aplicar R-SWA sin entrenamiento

Si se toma un modelo entrenado con atención completa y se le cambia la atención a R-SWA en inferencia, puede degradarse. El paper continúa el entrenamiento desde DeepSeek OCR usando R-SWA, lo cual ayuda a que el modelo aprenda a operar con esa memoria limitada.

Para una implementación seria se recomienda:

1. partir de un checkpoint VLM/OCR;
2. reemplazar la atención por R-SWA;
3. continuar entrenamiento con datos de parsing;
4. mezclar ejemplos cortos y largos;
5. evaluar en documentos de distinta longitud.

---

## 15. Estrategia de datos para entrenamiento o fine-tuning

Si el proyecto solo usa agentes con APIs, no hace falta entrenamiento. Pero si se quiere entrenar/adaptar un modelo, se necesita un dataset.

### 15.1. Tipos de datos

- páginas individuales con ground truth;
- documentos multipágina;
- tablas;
- fórmulas;
- documentos escaneados;
- documentos digitales;
- layouts complejos;
- documentos con ruido;
- separadores de página explícitos.

### 15.2. Formato recomendado

```json
{
  "document_id": "doc_001",
  "input_pages": [
    "pages/doc_001_001.png",
    "pages/doc_001_002.png"
  ],
  "target": "# Página 1\n...\n<page>\n# Página 2\n...",
  "metadata": {
    "language": "es",
    "domain": "medical",
    "num_pages": 2,
    "has_tables": true,
    "has_formulas": false
  }
}
```

### 15.3. Separador de páginas

Usar un token explícito como:

```text
<page>
```

O en Markdown:

```markdown
<!-- page: 3 -->
```

Esto ayuda al modelo y al sistema de agentes a preservar límites de página.

---

## 16. Métricas de evaluación

### 16.1. Métricas de calidad

1. **Edit distance / Character Error Rate**
   - mide diferencia textual contra ground truth.

2. **Word Error Rate**
   - útil en transcripción textual.

3. **Reading Order Accuracy**
   - mide si el orden de lectura se preserva.

4. **Table Structure Accuracy**
   - mide si tablas se mantienen como tablas.

5. **Formula Accuracy**
   - mide preservación de fórmulas.

6. **Coverage**
   - porcentaje de páginas/bloques procesados.

7. **Hallucination Rate**
   - contenido generado que no aparece en la referencia.

8. **Repetition Rate**
   - repeticiones no presentes en la fuente.

### 16.2. Métricas de eficiencia

1. **Tokens por segundo**
2. **Costo por documento**
3. **Memoria GPU**, si se controla el modelo
4. **Tamaño promedio de prompt**, si se usan APIs
5. **Cantidad de reintentos**
6. **Tiempo por página**
7. **Tiempo total end-to-end**

### 16.3. Métricas específicas para agentes

- cantidad de llamadas al LLM;
- tasa de validaciones fallidas;
- cantidad de bloques reprocesados;
- longitud media de memoria de trabajo;
- longitud media de referencia enviada;
- checkpoints exitosos;
- recuperaciones tras error.

---

## 17. Justificación técnica del enfoque

### 17.1. Por qué separar referencia y memoria

En tareas de parsing, la fuente original es más importante que el historial generado. Si el agente conserva todo el output anterior, puede confundirse y repetir. Si conserva la referencia original y solo una memoria reciente, se mantiene anclado a la fuente.

Esto reduce la probabilidad de:

- alucinación;
- repetición;
- deriva de formato;
- pérdida de foco;
- dependencia de errores previos.

### 17.2. Por qué no procesar simplemente página por página

Procesar página por página es simple, pero tiene limitaciones:

- pierde continuidad entre páginas;
- requiere lógica externa de unión;
- puede romper tablas o secciones que cruzan páginas;
- puede repetir headers/footers sin control;
- no mantiene una noción global del documento.

El patrón R-SWA-inspired permite mantener continuidad sin cargar todo el historial.

### 17.3. Por qué usar estado estructurado

El estado estructurado hace explícito lo que no debe quedar librado a la memoria del LLM:

- dónde está el agente;
- qué ya procesó;
- qué falta;
- qué errores hubo;
- qué salida se aceptó.

Esto es fundamental para agentes confiables.

---

## 18. Riesgos y mitigaciones

| Riesgo | Descripción | Mitigación |
|---|---|---|
| Omisiones | El agente salta bloques o páginas | Estado estructurado + coverage report |
| Repeticiones | El agente repite contenido anterior | Working memory corta + validator |
| Alucinaciones | El agente inventa texto | Prompt restrictivo + comparación contra referencia |
| Pérdida de tablas | Convierte tablas en texto desordenado | Parser especializado para tablas |
| Error acumulativo | Una mala salida afecta las siguientes | Validación local + checkpoints |
| Contexto insuficiente | La ventana reciente es muy chica | Aumentar `max_blocks` o agregar resumen local |
| Referencia demasiado grande | Prompts caros o largos | Chunking + retrieval + layout filtering |
| Baja calidad OCR base | El agente recibe una referencia mala | Usar crops/imágenes o OCR alternativo |

---

## 19. Recomendaciones de implementación incremental

### Fase 1: prototipo local

- Procesar PDFs con texto embebido.
- Dividir por página.
- Usar Parser Agent para convertir a Markdown.
- Mantener solo últimas 2 páginas como memoria.
- Guardar output incremental.

### Fase 2: soporte OCR

- Renderizar páginas.
- Ejecutar OCR base.
- Agregar coordenadas y bloques.
- Validar cobertura por bloque.

### Fase 3: agentes especializados

Agregar:

- Table Agent;
- Formula Agent;
- Layout Agent;
- Validator Agent más estricto;
- Repair Agent.

### Fase 4: optimización

- embeddings;
- retrieval por bloque;
- cache de referencias;
- paralelización por secciones;
- métricas automáticas.

### Fase 5: R-SWA real

- modelo open-source;
- modificación de atención;
- KV cache prefijo + cola;
- fine-tuning;
- benchmark propio.

---

## 20. Diseño de agentes especializados opcionales

### 20.1. Layout Agent

Detecta estructura:

- títulos;
- párrafos;
- tablas;
- figuras;
- encabezados;
- pies de página;
- notas al pie.

### 20.2. Table Agent

Procesa tablas y devuelve Markdown o JSON.

Contrato de salida:

```json
{
  "type": "table",
  "columns": ["A", "B", "C"],
  "rows": [
    ["...", "...", "..."]
  ],
  "confidence": 0.91
}
```

### 20.3. Repair Agent

Corrige errores locales señalados por el validador.

Regla clave: no debe reescribir el documento completo, solo el bloque afectado.

### 20.4. Report Agent

Genera un informe final:

- páginas procesadas;
- errores;
- calidad estimada;
- advertencias;
- tiempo total;
- costo;
- recomendaciones.

---

## 21. Ejemplo de salida esperada

```markdown
<!-- document_id: doc_001 -->
<!-- generated_by: r-swa-inspired-agent -->

# Título del documento

Texto introductorio...

<!-- page: 1 -->

## Sección 1

Contenido de la sección.

| Columna A | Columna B |
|---|---|
| Valor 1 | Valor 2 |

<!-- page: 2 -->

## Sección 2

Contenido de la segunda página.

[ILEGIBLE: fragmento pequeño en margen inferior]
```

---

## 22. Ejemplo de trace JSONL

Cada bloque aceptado debería tener una línea de trazabilidad:

```json
{"run_id":"run_001","block_id":"doc_001_p001_b001","status":"ok","chars":542,"validator_issues":0}
{"run_id":"run_001","block_id":"doc_001_p001_b002","status":"needs_revision","chars":311,"validator_issues":1}
```

Esto permite auditar y depurar.

---

## 23. Decisiones de diseño recomendadas

### 23.1. No guardar todo en el prompt

El prompt debe ser una ventana operativa, no una base de datos. El output completo debe guardarse en disco o base de datos.

### 23.2. No depender únicamente de embeddings

Para documentos con orden fuerte, como PDFs, el retrieval semántico puede romper la secuencia. Es mejor combinar:

- índice por página;
- índice por bloque;
- coordenadas;
- embeddings solo como apoyo.

### 23.3. Mantener separadores explícitos

Los separadores de página y sección reducen errores de continuidad.

### 23.4. Validar localmente

No conviene esperar al final para validar. Si una página se procesa mal, puede afectar el resto.

### 23.5. Diseñar para reanudación

Un sistema robusto debe poder continuar desde el último bloque exitoso.

---

## 24. Diferencias entre MVP y sistema de investigación

| Aspecto | MVP agente | R-SWA real |
|---|---|---|
| Modifica Transformer | No | Sí |
| KV cache constante real | No necesariamente | Sí |
| Requiere fine-tuning | No | Recomendado |
| Complejidad | Baja/media | Alta |
| Beneficio inmediato | Producto funcional | Eficiencia profunda |
| Riesgo técnico | Moderado | Alto |
| Ideal para | Validar flujo | Investigación/model serving |

---

## 25. Checklist de implementación

### Ingesta

- [ ] Cargar PDF/documento.
- [ ] Renderizar páginas.
- [ ] Extraer texto embebido.
- [ ] Aplicar OCR base si hace falta.
- [ ] Guardar metadatos.

### Referencia

- [ ] Dividir en páginas y bloques.
- [ ] Asignar IDs estables.
- [ ] Guardar coordenadas.
- [ ] Crear índice de recuperación.
- [ ] Marcar referencia como inmutable.

### Memoria

- [ ] Implementar ventana deslizante.
- [ ] Definir límite por bloques/caracteres.
- [ ] Guardar output completo fuera del prompt.
- [ ] Agregar resumen opcional de contexto expulsado.

### Agentes

- [ ] Parser Agent.
- [ ] Validator Agent.
- [ ] Coordinator Agent.
- [ ] Repair Agent opcional.
- [ ] Report Agent opcional.

### Salida

- [ ] Markdown final.
- [ ] JSONL de trazabilidad.
- [ ] Reporte de errores.
- [ ] Métricas.

### Evaluación

- [ ] Coverage por página.
- [ ] Repeticiones.
- [ ] Omisiones.
- [ ] Distancia de edición si hay ground truth.
- [ ] Tiempo por página.
- [ ] Costo por documento.

---

## 26. Conclusión

La contribución más valiosa de *Unlimited OCR* no es solamente “hacer OCR más rápido”, sino proponer una forma distinta de memoria para tareas de parsing largo:

```text
referencia permanente + memoria operativa corta
```

Ese patrón encaja muy bien con proyectos de agentes. En vez de construir agentes que acumulen todo el historial y se vuelvan progresivamente más caros y confusos, se puede diseñar un sistema que:

- mantiene una fuente de verdad estable;
- conserva solo el contexto reciente necesario;
- guarda estado estructurado fuera del modelo;
- valida localmente;
- produce salidas largas de forma incremental;
- permite reanudar, auditar y corregir.

Para un proyecto básico, la implementación recomendada es **R-SWA-inspired agentic parsing**: no modifica el Transformer, pero adopta su principio de diseño. Esta versión es suficiente para construir un sistema funcional y justificar la arquitectura. Si más adelante se necesita eficiencia a escala o investigación profunda, se puede avanzar hacia una implementación real de R-SWA modificando la atención y la KV cache del modelo.

En síntesis: el proyecto debe tratar al documento como referencia fija, al output previo como memoria desechable salvo una ventana reciente, y al estado de progreso como una estructura externa confiable. Esa separación es la clave para implementar agentes capaces de trabajar sobre documentos largos sin perder control, coherencia ni eficiencia.
