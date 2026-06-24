from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema


def file_reader(
    path: Annotated[str, Field(description="Ruta al archivo de texto a leer.")],
) -> str:
    """Lee y devuelve el contenido de un archivo de texto."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: no se encontró el archivo '{path}'."
    except IsADirectoryError:
        return f"Error: '{path}' es un directorio, no un archivo."
    except UnicodeDecodeError:
        return f"Error: '{path}' no es un archivo de texto válido (no se pudo decodificar como UTF-8)."
    except PermissionError:
        return f"Error: sin permisos para leer '{path}'."
    except OSError as e:
        return f"Error al leer '{path}': {e}"


file_reader_schema = ToolSchema.from_callable(file_reader)
