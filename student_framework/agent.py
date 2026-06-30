"""Implementación del agente — Milestone 1.

MyAgent implementa el protocolo Agent de la cátedra. Su núcleo es un bucle
ReAct (Reason + Act) que alterna entre llamadas al LLM y ejecución de
herramientas hasta obtener una respuesta en texto plano.

Flujo de run():
  1. Armar messages = [{"role": "user", "content": user_message}]
  2. Llamar a LLM con el historial y los schemas de herramientas disponibles.
  3. Si la respuesta tiene tool_calls: ejecutar cada herramienta, agregar
     el resultado como mensaje role="tool" y volver al paso 2.
  4. Si la respuesta es texto puro (sin tool_calls): devolver AgentResult.
  5. Safety: si se llega a max_iterations sin respuesta final, cortar el
     bucle y devolver igualmente un AgentResult con lo que haya.

Pendiente (M2):
  - structured_call(): salida estructurada con herramienta sintética
    final_result y bucle de reparación Pydantic.
  - Historial persistente entre llamadas a run() y ventana deslizante
    max_history_messages.
  - Acumulación de tokens (input_tokens / output_tokens) por run().
"""

from __future__ import annotations

import json
from typing import Any, Callable

from mia_agents.protocols import LLMClient
from mia_agents.types import AgentResult, AgentStep, ToolSchema


class MyAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = "Eres un asistente útil.",
        max_iterations: int = 10,
        max_history_messages: int = 50,
    ) -> None:
        """Inicializa el agente con su cliente LLM y parámetros de control.

        Parameters
        ----------
        llm_client : LLMClient
            Cliente LLM (real o mock) que el agente utilizará para todas
            sus llamadas a chat(). Los tests inyectan un MockLLMClient aquí.
        system_prompt : str
            Instrucción de sistema enviada al LLM en cada llamada. Define
            el rol y el tono del asistente.
        max_iterations : int
            Número máximo de iteraciones del bucle ReAct por cada llamada
            a run(). Evita bucles infinitos si el LLM no converge.
        max_history_messages : int
            Máximo de mensajes que se envían al LLM en una sola llamada.
            En M1 este valor se almacena pero no se aplica; en M2 actúa
            como ventana deslizante sobre el historial persistente.
        """
        self._llm = llm_client
        self._system = system_prompt
        self._max_iterations = max_iterations
        self._max_history_messages = max_history_messages
        # Herramientas registradas: nombre → callable y nombre → schema.
        # Se mantienen en dos dicts separados para poder pasar solo los
        # schemas al LLM y luego despachar el callable por nombre.
        self._tools: dict[str, Callable[..., str]] = {}
        self._schemas: dict[str, ToolSchema] = {}

    def register_tool(
        self,
        tool: Callable[..., str],
        schema: ToolSchema,
    ) -> None:
        """Registra una herramienta callable junto a su esquema.

        El schema se genera con ToolSchema.from_callable(fn) y describe
        al LLM qué hace la herramienta y qué parámetros acepta. El callable
        se invoca con los kwargs parseados del JSON que devuelve el LLM.

        Parameters
        ----------
        tool : Callable[..., str]
            Función Python que implementa la herramienta. Debe aceptar
            los parámetros descritos en schema y devolver un str.
        schema : ToolSchema
            Descripción estructurada de la herramienta (nombre, descripción,
            parámetros). Se pasa al LLM para que sepa cómo invocarla.
        """
        self._tools[schema.name] = tool
        self._schemas[schema.name] = schema

    def run(self, user_message: str) -> AgentResult:
        """Ejecuta un turno del agente: razona, actúa con herramientas y responde.

        Implementa el bucle ReAct completo para el Milestone 1:
          1. Construye la lista de mensajes con el mensaje del usuario.
          2. Llama al LLM con los schemas de herramientas disponibles.
          3. Si el LLM responde con tool_calls: ejecuta cada herramienta,
             agrega su resultado al historial como role="tool" y repite.
          4. Si el LLM responde con texto puro: lo devuelve en AgentResult.
          5. Si se alcanza max_iterations: devuelve lo que haya acumulado.

        Las herramientas desconocidas y las excepciones durante la ejecución
        se registran como AgentStep con error y se devuelven como mensajes
        de error al LLM (que puede decidir reintentar o explicar el problema
        al usuario). En ningún caso se lanza una excepción al caller.

        Parameters
        ----------
        user_message : str
            Mensaje del usuario para este turno de conversación.

        Returns
        -------
        AgentResult
            Resultado del turno. Siempre incluye answer (str) y steps
            (lista de AgentStep con cada invocación de herramienta).
            En M1, input_tokens y output_tokens son None.
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        # Si no hay herramientas registradas, pasar None en lugar de lista vacía:
        # algunos proveedores rechazan tools=[] mientras que tools=None es válido.
        tools = list(self._schemas.values()) if self._schemas else None
        steps: list[AgentStep] = []

        for _ in range(self._max_iterations):
            response = self._llm.chat(
                messages=messages,
                tools=tools,
                system=self._system,
            )

            # Respuesta final: texto sin tool_calls → terminar el bucle.
            if not response.tool_calls:
                return AgentResult(
                    answer=response.content or "",
                    steps=steps,
                )

            # El LLM pidió herramientas: agregar su turno al historial.
            messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in response.tool_calls
                ],
            })

            # Ejecutar cada herramienta solicitada y agregar el resultado.
            for tc in response.tool_calls:
                if tc.name not in self._tools:
                    # Herramienta no registrada: informar al LLM del error.
                    steps.append(AgentStep(
                        tool_name=tc.name,
                        tool_input=tc.arguments,
                        tool_output=None,
                        error=f"Herramienta desconocida: {tc.name}",
                    ))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: herramienta '{tc.name}' no encontrada.",
                    })
                    continue

                try:
                    kwargs = json.loads(tc.arguments) if tc.arguments else {}
                    result = self._tools[tc.name](**kwargs)
                    steps.append(AgentStep(
                        tool_name=tc.name,
                        tool_input=tc.arguments,
                        tool_output=result,
                    ))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                except Exception as e:
                    # Error en la ejecución de la herramienta: registrar y
                    # devolver el error al LLM para que pueda reaccionar.
                    steps.append(AgentStep(
                        tool_name=tc.name,
                        tool_input=tc.arguments,
                        tool_output=None,
                        error=str(e),
                    ))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: {e}",
                    })

        # Se agotaron las iteraciones sin respuesta final del LLM.
        return AgentResult(
            answer=response.content or "",
            steps=steps,
        )

    def structured_call(
        self,
        prompt: str,
        schema: Any,
        max_repair_attempts: int = 2,
    ) -> Any:
        """Solicita al LLM una respuesta estructurada validada contra schema (M2).

        En M2 este método:
          - Ofrece al LLM únicamente la herramienta sintética `final_result`
            (creada con mia_agents.tool_schema.final_result_tool_schema).
          - Valida los argumentos del tool_call con schema.model_validate().
          - Si la validación falla o el LLM responde con texto libre, agrega
            el error al contexto y reintenta hasta max_repair_attempts veces.
          - Si tras todos los reintentos sigue fallando, lanza una excepción.

        En M1 es un stub: lanza NotImplementedError para que los tests de
        M2 puedan detectar que aún no está implementado.

        Parameters
        ----------
        prompt : str
            Instrucción al LLM describiendo qué debe devolver.
        schema : type[BaseModel]
            Clase Pydantic cuya estructura define la respuesta esperada.
        max_repair_attempts : int
            Número máximo de reintentos tras un fallo de validación.

        Returns
        -------
        Any
            Instancia validada de schema (en M2).

        Raises
        ------
        NotImplementedError
            Siempre en M1. En M2 se reemplaza por la implementación real.
        """
        raise NotImplementedError("M2: implementa salida estructurada con reparación")
