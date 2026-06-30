"""Herramienta de calculadora aritmética básica.

Permite al agente realizar operaciones aritméticas exactas sobre dos
operandos numéricos. El LLM puede cometer errores de redondeo o alucinar
resultados en cálculos numéricos; delegarlos a esta herramienta garantiza
exactitud.

Operadores soportados: + (suma), - (resta), * (multiplicación), / (división).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema

# Mapa operador → función. Usar un dict en lugar de if/elif hace que agregar
# nuevos operadores en el futuro sea un cambio de una sola línea sin tocar
# la lógica de despacho.
_OPERATORS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
}


def calculator(
    left_operand: Annotated[
        float,
        Field(description="Primer operando numérico (lado izquierdo de la operación)."),
    ],
    right_operand: Annotated[
        float,
        Field(description="Segundo operando numérico (lado derecho de la operación)."),
    ],
    operator: Annotated[
        str,
        Field(description="Operador aritmético a aplicar. Valores válidos: +, -, * o /"),
    ],
) -> str:
    """Calcula el resultado de una operación aritmética entre dos números.

    Evalúa `left_operand operator right_operand` y devuelve el resultado
    como string. Si el resultado es un entero exacto (e.g., 6.0), lo
    devuelve sin decimales ("6"). En caso de error (operador inválido o
    división por cero), devuelve un mensaje de error descriptivo en lugar
    de lanzar una excepción, para no romper el bucle del agente.

    Parameters
    ----------
    left_operand : float
        Primer operando (el número de la izquierda).
    right_operand : float
        Segundo operando (el número de la derecha).
    operator : str
        Uno de "+", "-", "*", "/".

    Returns
    -------
    str
        El resultado numérico como string, o un mensaje de error si la
        operación no es válida.

    Examples
    --------
    calculator(15, 7, "*")  → "105"
    calculator(10, 3, "/")  → "3.3333333333333335"
    calculator(5,  0, "/")  → "Error: división por cero."
    calculator(1,  2, "^")  → "Error: operador '^' no soportado. Usa +, -, * o /"
    """
    if operator not in _OPERATORS:
        return f"Error: operador '{operator}' no soportado. Usa +, -, * o /"
    if operator == "/" and right_operand == 0:
        return "Error: división por cero."
    result = _OPERATORS[operator](left_operand, right_operand)
    # Mostrar enteros sin parte decimal para mejorar legibilidad.
    if result == int(result):
        return str(int(result))
    return str(result)


calculator_schema = ToolSchema.from_callable(calculator)
