"""Paquete propio del grupo.

Implementen el agente en `agent.py` y registren sus herramientas a
continuación, en `build_agent`. Tanto el runner de la CLI como los tests
de conformidad llaman a `build_agent`, por lo que esta es la única puerta
de entrada pública de su entrega.

Herramientas registradas (M1):
  1. calculator     — operaciones aritméticas (+, -, *, /)
  2. read_text_file — lectura de archivos de texto plano (UTF-8)
  3. text_stats     — estadísticas de texto (caracteres, palabras, líneas)
"""

from __future__ import annotations

from typing import Any

from mia_agents.llm_client import LLMClient
from mia_agents.protocols import Agent

from .agent import MyAgent


def build_agent(config: dict[str, Any] | None = None) -> Agent:
    """Construye y configura el agente con todas las herramientas del M1.

    `config` es opcional. Si se proporciona `config["llm_client"]`, el
    agente usa ese cliente — así es como los tests de conformidad inyectan
    un MockLLMClient sin necesitar Ollama ni credenciales AWS. Si no se
    proporciona, se construye desde las variables de entorno (OLLAMA_HOST
    o BEDROCK_MODEL_ID).
    """

    config = config or {}  # NO CAMBIAR
    llm = config.get("llm_client") or LLMClient.from_env()  # NO CAMBIAR
    kwargs: dict[str, Any] = {"llm_client": llm}  # NO CAMBIAR

    if "max_history_messages" in config:
        kwargs["max_history_messages"] = config["max_history_messages"]

    agent = MyAgent(**kwargs)

    # --- Herramienta 1: Calculadora aritmética básica ---
    # Operadores soportados: +, -, *, /
    # Sin eval, sin librerías externas. Necesaria para cálculos exactos
    # que el LLM no puede realizar de forma confiable por sí solo.
    from student_framework.tools.calculator import calculator, calculator_schema
    agent.register_tool(calculator, calculator_schema)

    # --- Herramienta 2: Lector de archivos de texto ---
    # Permite al agente observar información externa (archivos en disco)
    # que no está en el prompt ni en el conocimiento del LLM.
    # Solo acepta texto plano UTF-8, con límite de 100 KB.
    from student_framework.tools.file_reader import read_text_file, read_text_file_schema
    agent.register_tool(read_text_file, read_text_file_schema)

    # --- Herramienta 3: Estadísticas de texto (herramienta libre) ---
    # Cuenta caracteres, palabras y líneas de forma exacta.
    # Demuestra que el agente puede delegar a Python tareas donde el LLM
    # falla (el LLM no cuenta con exactitud; Python sí).
    # Se combina naturalmente con read_text_file para análisis de archivos.
    from student_framework.tools.text_stats import text_stats, text_stats_schema
    agent.register_tool(text_stats, text_stats_schema)

    return agent
