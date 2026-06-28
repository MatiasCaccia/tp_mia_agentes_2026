"""Herramienta de estadísticas de texto.

Permite al agente obtener métricas exactas sobre un texto: caracteres,
palabras y líneas. El LLM no puede contar con exactitud por sí solo;
delegar esto a Python garantiza un resultado determinista y verificable.

Caso de uso típico: el agente lee un archivo con `read_text_file` y
luego usa `text_stats` para analizar su contenido, combinando las dos
herramientas en un flujo de razonamiento multistep.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema


def text_stats(
    text: Annotated[str, Field(description="Texto sobre el que se calcularán las estadísticas.")],
) -> str:
    """Devuelve estadísticas básicas de un texto: caracteres, palabras y líneas.

    Cuenta:
    - Caracteres totales (incluyendo espacios y saltos de línea).
    - Caracteres sin espacios (sin contar espacios ni tabulaciones).
    - Palabras (separadas por espacios o saltos de línea).
    - Líneas (incluyendo líneas vacías).
    """
    if not text:
        return "Caracteres: 0\nCaracteres sin espacios: 0\nPalabras: 0\nLíneas: 0"

    total_chars = len(text)
    chars_no_spaces = len(text.replace(" ", "").replace("\t", ""))
    word_count = len(text.split())
    line_count = len(text.splitlines()) or 1

    return (
        f"Caracteres: {total_chars}\n"
        f"Caracteres sin espacios: {chars_no_spaces}\n"
        f"Palabras: {word_count}\n"
        f"Líneas: {line_count}"
    )


text_stats_schema = ToolSchema.from_callable(text_stats)
