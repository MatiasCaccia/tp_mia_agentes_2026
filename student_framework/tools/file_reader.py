"""Herramienta de lectura de archivos de texto.

Permite al agente acceder a información externa que no está en el prompt
ni en el conocimiento interno del LLM. El agente recibe una ruta de
archivo y devuelve el contenido como string para incorporarlo al contexto.

Solo acepta archivos de texto plano con codificación UTF-8. No lee
directorios ni archivos binarios. Es la única forma en que el agente
puede observar el sistema de archivos local.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema

# Umbral de tamaño máximo aceptado (100 KB). Archivos más grandes
# probablemente excederían la ventana de contexto del LLM de todos modos,
# y devolverlos enteros no aportaría valor al razonamiento del agente.
_MAX_BYTES = 100_000


def read_text_file(
    path: Annotated[
        str,
        Field(description="Ruta al archivo de texto que se quiere leer (absoluta o relativa al CWD)."),
    ],
) -> str:
    """Lee el contenido de un archivo de texto y lo devuelve como string.

    Solo acepta archivos de texto plano con codificación UTF-8. No lee
    directorios ni archivos binarios. En cualquier condición de error
    (archivo inexistente, demasiado grande, binario, sin permisos) devuelve
    un mensaje descriptivo en lugar de lanzar una excepción, para que el
    agente pueda informar el problema al usuario sin interrumpir su bucle.

    Parameters
    ----------
    path : str
        Ruta al archivo. Puede ser absoluta o relativa al directorio de
        trabajo actual del proceso.

    Returns
    -------
    str
        Contenido completo del archivo, o un mensaje de error prefijado
        con "Error:" si la lectura no fue posible.

    Raises
    ------
    No lanza excepciones. Todos los errores se devuelven como strings.

    Examples
    --------
    read_text_file("/etc/hostname")         → "mi-pc\\n"
    read_text_file("/ruta/inexistente.txt") → "Error: el archivo '/ruta/inexistente.txt' no existe."
    read_text_file("/imagen.png")           → "Error: el archivo '/imagen.png' no es texto plano en UTF-8."
    """
    file_path = Path(path)

    if not file_path.exists():
        return f"Error: el archivo '{path}' no existe."

    if not file_path.is_file():
        return f"Error: '{path}' es un directorio, no un archivo."

    if file_path.stat().st_size > _MAX_BYTES:
        size_kb = file_path.stat().st_size // 1024
        return (
            f"Error: el archivo '{path}' es demasiado grande ({size_kb} KB). "
            f"El límite es {_MAX_BYTES // 1024} KB."
        )

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return (
            f"Error: el archivo '{path}' no es texto plano en UTF-8. "
            "Solo se pueden leer archivos de texto."
        )
    except PermissionError:
        return f"Error: no hay permisos para leer '{path}'."
    except OSError as e:
        return f"Error al leer '{path}': {e}"

    if not content:
        return f"(El archivo '{path}' existe pero está vacío.)"

    return content


read_text_file_schema = ToolSchema.from_callable(read_text_file)
