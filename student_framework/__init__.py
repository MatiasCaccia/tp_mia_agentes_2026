"""Paquete del alumno — implementación del agente (Milestone 1).

Estructura:
  agent.py            MyAgent: bucle ReAct + register_tool + structured_call (stub M2)
  tools/calculator.py Herramienta 1: calculadora aritmética básica (+, -, *, /)
  tools/file_reader.py Herramienta 2: lector de archivos de texto plano (UTF-8)
  tools/thermo_converter.py Herramienta 3 (libre): conversor de unidades termodinámicas

El punto de entrada público es `build_agent`. Tanto el runner de la CLI
(`python -m mia_agents.cli run ...`) como los tests de conformidad llaman
a esta función para obtener un agente completamente configurado.
"""

from __future__ import annotations

from typing import Any

from mia_agents.llm_client import LLMClient
from mia_agents.protocols import Agent

from .agent import MyAgent


def build_agent(config: dict[str, Any] | None = None) -> Agent:
    """Construye y devuelve un MyAgent con las tres herramientas del M1.

    Si `config["llm_client"]` está presente, el agente usa ese cliente
    (así es como los tests de conformidad inyectan un MockLLMClient sin
    necesitar Ollama ni credenciales AWS). Si no se proporciona, el cliente
    se construye desde las variables de entorno (OLLAMA_HOST / BEDROCK_MODEL_ID).

    Si `config["max_history_messages"]` está presente, se pasa al constructor
    de MyAgent para preparar la instancia para M2 (en M1 el valor es aceptado
    pero ignorado durante run()).

    Parameters
    ----------
    config : dict | None
        Configuración opcional. Claves reconocidas:
          - "llm_client": instancia de LLMClient a usar (para tests).
          - "max_history_messages": tope de mensajes por llamada al LLM (M2).

    Returns
    -------
    Agent
        Instancia de MyAgent con calculator, read_text_file y thermo_converter
        ya registradas y listas para usar.
    """
    config = config or {}  # NO CAMBIAR
    llm = config.get("llm_client") or LLMClient.from_env()  # NO CAMBIAR
    kwargs: dict[str, Any] = {"llm_client": llm}  # NO CAMBIAR

    if "max_history_messages" in config:
        kwargs["max_history_messages"] = config["max_history_messages"]

    agent = MyAgent(**kwargs)

    # Herramienta 1: calculadora aritmética básica
    from student_framework.tools.calculator import calculator, calculator_schema
    agent.register_tool(calculator, calculator_schema)

    # Herramienta 2: lector de archivos de texto plano
    from student_framework.tools.file_reader import read_text_file, read_text_file_schema
    agent.register_tool(read_text_file, read_text_file_schema)

    # Herramienta 3 (libre): conversor de unidades termodinámicas
    from student_framework.tools.thermo_converter import thermo_converter, thermo_converter_schema
    agent.register_tool(thermo_converter, thermo_converter_schema)

    return agent
