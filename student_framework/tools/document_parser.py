from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema


DEFAULT_LINES_PER_BLOCK = 50
DEFAULT_SOURCE_WINDOW_RADIUS = 0
DEFAULT_RECENT_WINDOW_CHARS = 3500
DEFAULT_PREVIEW_CHARS = 100
MAX_LINES_PER_BLOCK = 500
MAX_SOURCE_WINDOW_RADIUS = 3
MAX_RECENT_WINDOW_CHARS = 12000
MAX_PREVIEW_CHARS = 240


def _coerce_int(value: object, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    """Convierte valores de entrada del agente a int y aplica límites seguros."""
    try:
        value_int = int(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        value_int = default

    if minimum is not None:
        value_int = max(minimum, value_int)
    if maximum is not None:
        value_int = min(maximum, value_int)
    return value_int


def _coerce_bool(value: object, default: bool = True) -> bool:
    """Convierte strings comunes de agentes a bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "si", "sí", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _compact_text(text: str, max_chars: int) -> str:
    """Normaliza espacios y corta texto para previews compactas."""
    compact = " ".join(text.strip().split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def _fingerprint_document(path: Path, content: str) -> str:
    """Genera un identificador estable para saber si la referencia cambió."""
    hasher = hashlib.sha256()
    hasher.update(str(path.resolve()).encode("utf-8", errors="ignore"))
    hasher.update(b"\n---CONTENT---\n")
    hasher.update(content.encode("utf-8", errors="ignore"))
    return hasher.hexdigest()[:16]


def _build_blocks(lines: list[str], lines_per_block: int) -> list[list[str]]:
    return [lines[i : i + lines_per_block] for i in range(0, len(lines), lines_per_block)]


def _build_reference_index(
    blocks: list[list[str]],
    *,
    lines_per_block: int,
    preview_chars: int,
) -> str:
    """Construye una referencia global compacta del documento.

    Esta sección funciona como aproximación práctica del prefijo de referencia fija
    del paper: no contiene todo el documento, sino un mapa estable para que el
    agente pueda orientarse sin cargar todos los tokens fuente.
    """
    rows: list[str] = []
    for i, block in enumerate(blocks):
        start_line = i * lines_per_block + 1
        end_line = start_line + len(block) - 1
        first_line = _compact_text(block[0], preview_chars) if block else ""
        last_line = _compact_text(block[-1], preview_chars) if block else ""
        rows.append(
            f"[Bloque {i}] líneas {start_line}-{end_line} | "
            f"inicio: \"{first_line}\" | fin: \"{last_line}\""
        )
    return "\n".join(rows)


def _format_block(block: list[str], block_number: int, total_blocks: int, lines_per_block: int) -> str:
    start_line = block_number * lines_per_block + 1
    end_line = start_line + len(block) - 1
    body = "".join(block)
    return (
        f"### Bloque {block_number}/{total_blocks - 1}\n"
        f"Líneas: {start_line}-{end_line}\n"
        f"---\n"
        f"{body.rstrip()}"
    )


def _build_source_window(
    blocks: list[list[str]],
    *,
    block_index: int,
    lines_per_block: int,
    source_window_radius: int,
) -> str:
    """Devuelve el bloque objetivo y, opcionalmente, bloques vecinos.

    En un modelo con R-SWA real, la referencia visual completa permanece disponible.
    En una tool de agentes, una alternativa pragmática es exponer el bloque objetivo
    y una ventana local de fuente para mantener continuidad de lectura sin inflar el
    prompt con todo el documento.
    """
    total_blocks = len(blocks)
    start_block = max(0, block_index - source_window_radius)
    end_block = min(total_blocks - 1, block_index + source_window_radius)

    formatted_blocks: list[str] = []
    for current_block in range(start_block, end_block + 1):
        marker = " ← BLOQUE OBJETIVO" if current_block == block_index else ""
        formatted = _format_block(blocks[current_block], current_block, total_blocks, lines_per_block)
        formatted_blocks.append(formatted.replace(f"### Bloque {current_block}/", f"### Bloque {current_block}{marker}/", 1))
    return "\n\n".join(formatted_blocks)


def _build_recent_generation_window(recent_output: str, recent_window_chars: int) -> str:
    """Conserva solamente el final del texto generado por el agente.

    Esta es la analogía directa con la ventana deslizante de tokens generados:
    no se reinyecta todo el historial, solo una memoria operativa reciente.
    """
    if not recent_output or not recent_output.strip():
        return "Sin memoria reciente provista."

    recent_output = recent_output.strip()
    if len(recent_output) <= recent_window_chars:
        return recent_output

    return (
        f"[Se omitieron {len(recent_output) - recent_window_chars} caracteres antiguos. "
        f"Se conserva solo la ventana reciente.]\n"
        f"{recent_output[-recent_window_chars:]}"
    )


def document_parser(
    path: Annotated[str, Field(description="Ruta al archivo de texto a parsear.")],
    block_index: Annotated[
        int,
        Field(
            description=(
                "Índice del bloque objetivo a leer, empezando en 0. "
                "Usar -1 para obtener la referencia fija compacta del documento: "
                "cantidad de bloques, huella del documento y vista previa de cada bloque."
            ),
        ),
    ] = -1,
    lines_per_block: Annotated[
        int,
        Field(
            description=(
                "Cantidad de líneas por bloque. Por defecto 50. "
                "Se limita internamente para evitar respuestas demasiado grandes."
            ),
        ),
    ] = DEFAULT_LINES_PER_BLOCK,
    source_window_radius: Annotated[
        int,
        Field(
            description=(
                "Cantidad de bloques vecinos a incluir alrededor del bloque objetivo. "
                "0 devuelve solo el bloque objetivo; 1 devuelve anterior, actual y siguiente. "
                "Representa una ventana local de fuente, útil para continuidad de lectura."
            ),
        ),
    ] = DEFAULT_SOURCE_WINDOW_RADIUS,
    recent_output: Annotated[
        str,
        Field(
            description=(
                "Último texto generado por el agente en pasos previos. "
                "No pasar todo el historial: pasar solo la salida acumulada que se quiera resumir/continuar. "
                "La tool conservará únicamente el tramo final, simulando una ventana deslizante."
            ),
        ),
    ] = "",
    recent_window_chars: Annotated[
        int,
        Field(
            description=(
                "Máximo de caracteres de recent_output que se devuelven. "
                "Funciona como memoria operativa reciente, análoga a la ventana n de R-SWA."
            ),
        ),
    ] = DEFAULT_RECENT_WINDOW_CHARS,
    include_reference_index: Annotated[
        bool,
        Field(
            description=(
                "Si es True, incluye el índice global compacto junto con cada bloque. "
                "Esto aproxima el prefijo de referencia fija del artículo."
            ),
        ),
    ] = True,
    preview_chars: Annotated[
        int,
        Field(description="Cantidad máxima de caracteres por preview de bloque en el índice compacto."),
    ] = DEFAULT_PREVIEW_CHARS,
) -> str:
    """Parser de documentos orientado a agentes, inspirado en R-SWA / Unlimited OCR.

    La idea del artículo no puede implementarse completamente desde una tool de Python,
    porque R-SWA modifica el patrón de atención y la KV cache del decodificador del LLM.
    Esta función implementa una aproximación práctica para proyectos básicos de agentes:

    1. Referencia fija compacta:
       - huella del documento;
       - cantidad de líneas y bloques;
       - índice global con previews.

    2. Ventana local de fuente:
       - bloque objetivo;
       - opcionalmente bloques vecinos.

    3. Ventana deslizante de generación:
       - solo el tramo final de recent_output;
       - evita reinyectar todo el historial al agente.

    Protocolo recomendado:
    - llamar primero con block_index=-1 para obtener la referencia fija;
    - procesar bloques en orden;
    - en cada paso pasar en recent_output solo el resultado acumulado o resumen reciente;
    - mantener recent_window_chars acotado para evitar prompts crecientes.
    """
    block_index = _coerce_int(block_index, default=-1)
    lines_per_block = _coerce_int(
        lines_per_block,
        default=DEFAULT_LINES_PER_BLOCK,
        minimum=1,
        maximum=MAX_LINES_PER_BLOCK,
    )
    source_window_radius = _coerce_int(
        source_window_radius,
        default=DEFAULT_SOURCE_WINDOW_RADIUS,
        minimum=0,
        maximum=MAX_SOURCE_WINDOW_RADIUS,
    )
    recent_window_chars = _coerce_int(
        recent_window_chars,
        default=DEFAULT_RECENT_WINDOW_CHARS,
        minimum=0,
        maximum=MAX_RECENT_WINDOW_CHARS,
    )
    preview_chars = _coerce_int(
        preview_chars,
        default=DEFAULT_PREVIEW_CHARS,
        minimum=20,
        maximum=MAX_PREVIEW_CHARS,
    )
    include_reference_index = _coerce_bool(include_reference_index, default=True)

    file_path = Path(path).expanduser()

    try:
        content = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: no se encontró el archivo '{path}'."
    except UnicodeDecodeError:
        return f"Error: '{path}' no es un archivo de texto UTF-8 válido."
    except OSError as e:
        return f"Error al leer '{path}': {e}"

    if not content:
        return "El archivo está vacío."

    lines = content.splitlines(keepends=True)
    if not lines:
        return "El archivo no contiene líneas parseables."

    blocks = _build_blocks(lines, lines_per_block)
    total_blocks = len(blocks)
    doc_hash = _fingerprint_document(file_path, content)

    reference_header = (
        "# REFERENCIA FIJA DEL DOCUMENTO\n"
        f"Documento: {file_path}\n"
        f"Huella: {doc_hash}\n"
        f"Total: {len(lines)} líneas, {total_blocks} bloques de hasta {lines_per_block} líneas\n"
        "Criterio: mantener esta referencia como mapa estable del documento; "
        "no reemplaza al bloque fuente, pero orienta al agente sin cargar todo el archivo.\n"
    )

    reference_index = _build_reference_index(
        blocks,
        lines_per_block=lines_per_block,
        preview_chars=preview_chars,
    )

    if block_index == -1:
        return (
            f"{reference_header}"
            "\n## Índice compacto\n"
            f"{reference_index}\n"
            "\n\n# PROTOCOLO SUGERIDO PARA AGENTES\n"
            "1. Procesar los bloques secuencialmente, empezando por block_index=0.\n"
            "2. En cada llamada, usar source_window_radius=0 o 1 según necesidad de continuidad.\n"
            "3. Pasar en recent_output solo el resumen o salida reciente, no todo el historial.\n"
            "4. Mantener recent_window_chars acotado para simular una memoria deslizante.\n"
            "5. Fusionar resultados fuera de la tool mediante un estado estructurado del agente."
        )

    if block_index < 0 or block_index >= total_blocks:
        return (
            f"Error: block_index {block_index} fuera de rango. "
            f"El documento tiene {total_blocks} bloques: 0 a {total_blocks - 1}."
        )

    source_window = _build_source_window(
        blocks,
        block_index=block_index,
        lines_per_block=lines_per_block,
        source_window_radius=source_window_radius,
    )
    recent_generation_window = _build_recent_generation_window(recent_output, recent_window_chars)

    sections: list[str] = []
    sections.append("# PAQUETE DE LECTURA R-SWA PARA AGENTE")
    sections.append(
        "Este resultado separa referencia fija, fuente local y memoria reciente. "
        "La intención es evitar que el agente dependa de un historial conversacional creciente."
    )

    if include_reference_index:
        sections.append(reference_header)
        sections.append("## Índice compacto")
        sections.append(reference_index)
    else:
        sections.append(
            "# REFERENCIA FIJA DEL DOCUMENTO\n"
            f"Documento: {file_path}\n"
            f"Huella: {doc_hash}\n"
            f"Total: {len(lines)} líneas, {total_blocks} bloques de hasta {lines_per_block} líneas"
        )

    sections.append("# VENTANA LOCAL DE FUENTE")
    sections.append(
        f"Bloque objetivo: {block_index}\n"
        f"Radio de ventana fuente: {source_window_radius}\n"
        "Nota: esta ventana representa la porción fuente que el agente debe procesar ahora."
    )
    sections.append(source_window)

    sections.append("# VENTANA DESLIZANTE DE GENERACIÓN")
    sections.append(
        f"Máximo conservado: {recent_window_chars} caracteres\n"
        "Nota: usar esta sección solo como continuidad local; no debe reemplazar al estado estructurado."
    )
    sections.append(recent_generation_window)

    sections.append("# INSTRUCCIONES DE USO PARA EL AGENTE")
    sections.append(
        "- Procesar principalmente el BLOQUE OBJETIVO.\n"
        "- Usar la REFERENCIA FIJA solo para orientación global y control de ubicación.\n"
        "- Usar la VENTANA DESLIZANTE DE GENERACIÓN solo para continuidad inmediata.\n"
        "- No asumir que la memoria reciente contiene todo lo ya procesado.\n"
        "- Devolver una salida incremental y un estado/resumen compacto para el siguiente paso."
    )

    return "\n\n".join(sections)


document_parser_schema = ToolSchema.from_callable(document_parser)