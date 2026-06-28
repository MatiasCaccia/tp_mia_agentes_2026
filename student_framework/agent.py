"""Implementación del agente.

Milestone 1 (conservado):
  - register_tool: registra un callable junto a su ToolSchema.
  - run: bucle ReAct — razonar, actuar con herramientas, observar resultado,
    repetir hasta respuesta final o hasta max_iterations.
  - Manejo de herramienta desconocida y errores de ejecución sin romper run.
  - AgentStep por cada herramienta invocada.

Milestone 2 (nuevo):
  - self._history: el historial de conversación persiste entre llamadas
    sucesivas a run(). Cada turno ve el contexto de los anteriores.
  - max_history_messages: ventana deslizante — el LLM recibe como máximo
    esa cantidad de mensajes en cada llamada, sin importar cuántos haya
    en el historial interno.
  - Acumulación de tokens: input_tokens y output_tokens de cada respuesta
    LLM se suman durante el run() y se exponen en AgentResult.
  - structured_call: solicita al LLM una respuesta estructurada usando la
    herramienta sintética final_result. Valida los argumentos con Pydantic
    y reintenta con contexto de reparación si fallan.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from mia_agents.protocols import LLMClient
from mia_agents.types import AgentResult, AgentStep, ToolSchema


def _acumular_tokens(acumulado: int | None, nuevo: int | None) -> int | None:
    """Suma los tokens de una nueva respuesta LLM al acumulado del run.

    Regla (del contrato de AgentResult):
    - Si ninguna respuesta reportó tokens todavía, devuelve None.
    - En cuanto alguna respuesta reporta un valor no-None, los None
      posteriores se tratan como 0 (la respuesta existió pero sin tokens).

    Ejemplos:
      _acumular_tokens(None, None)   → None   (nadie reportó nada)
      _acumular_tokens(None, 100)    → 100    (primer reporte)
      _acumular_tokens(100, None)    → 100    (None se suma como 0)
      _acumular_tokens(100, 200)     → 300    (suma normal)
    """
    if acumulado is None and nuevo is None:
        return None
    return (acumulado or 0) + (nuevo or 0)


class MyAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = "Eres un asistente útil.",
        max_iterations: int = 10,
        max_history_messages: int = 50,
    ) -> None:
        """Inicializa el agente.

        Parameters
        ----------
        llm_client : LLMClient
            Cliente LLM (real o mock) que el agente utilizará.
        system_prompt : str
            Prompt de sistema enviado al LLM en cada llamada.
        max_iterations : int
            Límite de iteraciones del bucle ReAct por cada run().
        max_history_messages : int
            Máximo de mensajes enviados al LLM en una sola llamada.
            El historial interno puede ser más largo; este parámetro
            controla cuánto ve el LLM (ventana deslizante).
        """
        self._llm = llm_client
        self._system = system_prompt
        self._max_iterations = max_iterations
        self._max_history_messages = max_history_messages
        self._tools: dict[str, Callable[..., str]] = {}
        self._schemas: dict[str, ToolSchema] = {}
        # Historial conversacional persistente entre llamadas a run().
        # Crece con cada turno: user → assistant → tool → assistant → ...
        self._history: list[dict[str, Any]] = []

    def register_tool(
        self,
        tool: Callable[..., str],
        schema: ToolSchema,
    ) -> None:
        """Registra una herramienta callable junto a su esquema.

        El esquema se pasa al LLM en cada run() para que sepa qué
        herramientas puede invocar. El callable se ejecuta cuando el
        LLM elige esa herramienta.
        """
        self._tools[schema.name] = tool
        self._schemas[schema.name] = schema

    def _mensajes_para_llm(self) -> list[dict[str, Any]]:
        """Aplica la ventana deslizante sobre el historial completo.

        El historial interno puede crecer indefinidamente, pero el LLM
        solo recibe los últimos max_history_messages mensajes. Esto
        evita superar la ventana de contexto y mantiene el costo acotado.
        """
        return self._history[-self._max_history_messages:]

    def run(self, user_message: str) -> AgentResult:
        """Ejecuta un turno de conversación.

        El mensaje del usuario se añade al historial persistente. Cada
        llamada al LLM usa la ventana deslizante del historial. Al
        terminar, la respuesta del asistente también queda en el historial
        para que el siguiente turno tenga contexto.

        El bucle ReAct (M1) se conserva intacto:
          1. Llamar al LLM con el historial limitado.
          2. Si devuelve tool_calls: ejecutar, agregar al historial, repetir.
          3. Si devuelve texto sin tool_calls: guardar y devolver.
          4. Si se agota max_iterations: devolver lo acumulado.
        """
        # Agregar el mensaje del usuario al historial persistente
        self._history.append({"role": "user", "content": user_message})

        tools = list(self._schemas.values()) if self._schemas else None
        steps: list[AgentStep] = []
        total_input: int | None = None
        total_output: int | None = None
        response = None

        for _ in range(self._max_iterations):
            # Enviar solo los últimos N mensajes al LLM
            response = self._llm.chat(
                messages=self._mensajes_para_llm(),
                tools=tools,
                system=self._system,
            )

            # Acumular tokens de esta llamada al LLM
            total_input = _acumular_tokens(total_input, response.input_tokens)
            total_output = _acumular_tokens(total_output, response.output_tokens)

            if not response.tool_calls:
                # El LLM respondió con texto: guardar en historial y terminar
                self._history.append({
                    "role": "assistant",
                    "content": response.content or "",
                })
                return AgentResult(
                    answer=response.content or "",
                    steps=steps,
                    input_tokens=total_input,
                    output_tokens=total_output,
                )

            # El LLM pidió herramientas: guardar su mensaje en el historial
            self._history.append({
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

            # Ejecutar cada herramienta solicitada
            for tc in response.tool_calls:
                if tc.name not in self._tools:
                    steps.append(AgentStep(
                        tool_name=tc.name,
                        tool_input=tc.arguments,
                        tool_output=None,
                        error=f"Herramienta desconocida: {tc.name}",
                    ))
                    self._history.append({
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
                    self._history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                except Exception as e:
                    steps.append(AgentStep(
                        tool_name=tc.name,
                        tool_input=tc.arguments,
                        tool_output=None,
                        error=str(e),
                    ))
                    self._history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: {e}",
                    })

        # Se agotaron las iteraciones: devolver lo que haya
        return AgentResult(
            answer=response.content if response is not None else "",
            steps=steps,
            input_tokens=total_input,
            output_tokens=total_output,
        )

    def structured_call(
        self,
        prompt: str,
        schema: Any,
        max_repair_attempts: int = 2,
    ) -> Any:
        """Solicita al LLM una respuesta estructurada validada contra schema.

        Implementación explícita del patrón de salida estructurada con
        herramienta sintética y reparación iterativa:

          1. Se crea el schema de la herramienta sintética final_result a
             partir del modelo Pydantic recibido.
          2. Se llama al LLM ofreciéndole solo esa herramienta. El LLM
             debe invocarla para entregar su respuesta.
          3. Si el LLM invoca final_result, se parsean y validan sus
             argumentos con schema.model_validate(). Si son válidos,
             se devuelve la instancia.
          4. Si el LLM responde con texto libre o los argumentos no pasan
             la validación, se agrega el error al contexto y se reintenta.
          5. El número total de llamadas al LLM es 1 + max_repair_attempts.
             Si se agotan, se levanta la última excepción capturada.

        No modifica self._history — es un canal de comunicación separado
        del historial conversacional principal.
        """
        from mia_agents.tool_schema import FINAL_RESULT_TOOL_NAME, final_result_tool_schema

        fr_schema = final_result_tool_schema(schema)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        ultimo_error: Exception | None = None

        for _ in range(1 + max_repair_attempts):
            response = self._llm.chat(
                messages=messages,
                tools=[fr_schema],
                system=self._system,
            )

            # Buscar si el LLM invocó final_result en esta respuesta
            fr_call = next(
                (tc for tc in response.tool_calls if tc.name == FINAL_RESULT_TOOL_NAME),
                None,
            )

            if fr_call is None:
                # El LLM respondió con texto libre en lugar de invocar final_result.
                # Agregar su respuesta al contexto y un mensaje de corrección.
                ultimo_error = RuntimeError(
                    f"El modelo no invocó '{FINAL_RESULT_TOOL_NAME}'. "
                    f"Respuesta recibida: {response.content!r}"
                )
                messages.append({"role": "assistant", "content": response.content or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tu respuesta no es válida. Debes invocar la herramienta "
                        f"'{FINAL_RESULT_TOOL_NAME}' para entregar tu respuesta estructurada. "
                        "No respondas con texto libre."
                    ),
                })
                continue

            # El LLM invocó final_result: intentar parsear y validar los argumentos
            try:
                raw_args = json.loads(fr_call.arguments) if fr_call.arguments else {}
                return schema.model_validate(raw_args)
            except Exception as e:
                # La validación falló: agregar el tool_call fallido y el error
                # al contexto para que el LLM pueda corregir en el siguiente intento.
                ultimo_error = e
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [{
                        "id": fr_call.id,
                        "function": {
                            "name": fr_call.name,
                            "arguments": fr_call.arguments,
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": fr_call.id,
                    "content": (
                        f"Error de validación: {e}. "
                        f"Corrige los argumentos y vuelve a invocar '{FINAL_RESULT_TOOL_NAME}'."
                    ),
                })

        # Se agotaron todos los intentos sin éxito
        if ultimo_error is not None:
            raise ultimo_error
        raise RuntimeError("structured_call agotó todos los intentos sin éxito.")
