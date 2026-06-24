from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema

_OPERATORS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
}


def calculator(
    left_operand: Annotated[float, Field(description="Primer operando numérico.")],
    right_operand: Annotated[float, Field(description="Segundo operando numérico.")],
    operator: Annotated[str, Field(description="Operador aritmético: +, -, * o /")],
) -> str:
    """Calcula el resultado de una operación aritmética entre dos números."""
    try:
        left_operand = float(left_operand)
        right_operand = float(right_operand)
    except (ValueError, TypeError):
        return "Error: los operandos deben ser numéricos."
    operator = str(operator).strip()
    if operator not in _OPERATORS:
        return f"Error: operador '{operator}' no soportado. Usa +, -, * o /"
    if operator == "/" and right_operand == 0:
        return "Error: división por cero."
    result = _OPERATORS[operator](left_operand, right_operand)
    if result == int(result):
        return str(int(result))
    return str(result)


calculator_schema = ToolSchema.from_callable(calculator)
