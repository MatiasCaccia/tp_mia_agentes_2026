"""Conversor de unidades termodinámicas.

Cubre las magnitudes relevantes para termodinámica general:

  Presión     : Pa, hPa, kPa, MPa, GPa, bar, mbar, atm, psi, mmHg, torr, inHg
  Volumen     : m3, L, mL, cm3, dm3, ft3, in3, gal
  Temperatura : K, C, F, R  (Kelvin, Celsius, Fahrenheit, Rankine)
  Energía     : J, kJ, MJ, cal, kcal, BTU, Wh, kWh, eV, erg
  Masa        : kg, g, mg, ug, lb, oz, t (tonelada métrica)
  Sustancia   : mol, mmol, umol, nmol, kmol
  Constante R : J_mol_K, kJ_mol_K, cal_mol_K, kcal_mol_K,
                Latm_mol_K, Lbar_mol_K, m3Pa_mol_K, BTU_lbmol_R

La constante de gas ideal R = 8.314 462 J/(mol·K) puede expresarse en
cualquier sistema de unidades usando los identificadores "R_…" de arriba.
Por ejemplo: thermo_converter(8.314462, "J_mol_K", "Latm_mol_K") → 0.082057.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema

# ---------------------------------------------------------------------------
# Tablas de conversión — factor para llevar 1 unidad a la unidad SI base
# ---------------------------------------------------------------------------

# Presión (base: Pa)
_PRESSURE: dict[str, float] = {
    "Pa":   1.0,
    "hPa":  1e2,
    "kPa":  1e3,
    "MPa":  1e6,
    "GPa":  1e9,
    "bar":  1e5,
    "mbar": 1e2,
    "atm":  101_325.0,
    "psi":  6_894.757,
    "mmHg": 133.322,
    "torr": 133.322,
    "inHg": 3_386.39,
}

# Volumen (base: m³)
_VOLUME: dict[str, float] = {
    "m3":  1.0,
    "L":   1e-3,
    "mL":  1e-6,
    "cm3": 1e-6,
    "dm3": 1e-3,
    "ft3": 0.028_316_8,
    "in3": 1.638_71e-5,
    "gal": 3.785_41e-3,
}

# Energía (base: J)
_ENERGY: dict[str, float] = {
    "J":   1.0,
    "kJ":  1e3,
    "MJ":  1e6,
    "cal": 4.184,
    "kcal":4_184.0,
    "BTU": 1_055.06,
    "Wh":  3_600.0,
    "kWh": 3_600_000.0,
    "eV":  1.602_18e-19,
    "erg": 1e-7,
}

# Masa (base: kg)
_MASS: dict[str, float] = {
    "kg": 1.0,
    "g":  1e-3,
    "mg": 1e-6,
    "ug": 1e-9,
    "lb": 0.453_592,
    "oz": 0.028_349_5,
    "t":  1_000.0,
}

# Cantidad de sustancia (base: mol)
_AMOUNT: dict[str, float] = {
    "mol":  1.0,
    "mmol": 1e-3,
    "umol": 1e-6,
    "nmol": 1e-9,
    "kmol": 1e3,
}

# Constante de gas ideal R en distintos sistemas (base: J/(mol·K))
# Factor = cuántos J/(mol·K) equivale 1 unidad de ese sistema.
# Derivados de: R_SI = 8.314 462 J/(mol·K)
#   1 L·atm/(mol·K) = 101.325 J/(mol·K)  → R = 8.314462/101.325 = 0.082057
#   1 L·bar/(mol·K) = 100.000 J/(mol·K)  → R = 8.314462/100.000 = 0.083145
#   1 BTU/(lbmol·R) = 4.18681 J/(mol·K)  → R = 8.314462/4.18681 = 1.98588
_R_UNITS: dict[str, float] = {
    "J_mol_K":     1.0,
    "kJ_mol_K":    1e3,
    "cal_mol_K":   4.184,
    "kcal_mol_K":  4_184.0,
    "Latm_mol_K":  101.325,
    "Lbar_mol_K":  100.0,
    "m3Pa_mol_K":  1.0,
    "BTU_lbmol_R": 4.186_81,
}

# Unidades de temperatura (conversión no lineal, tratadas aparte)
_TEMP_UNITS: frozenset[str] = frozenset({"K", "C", "F", "R"})

# Índice unidad → tabla de su categoría (para detección de errores mixtos)
_UNIT_TABLE: dict[str, dict[str, float]] = {}
for _tbl in (_PRESSURE, _VOLUME, _ENERGY, _MASS, _AMOUNT, _R_UNITS):
    for _u in _tbl:
        _UNIT_TABLE[_u] = _tbl

# ---------------------------------------------------------------------------
# Conversión de temperatura
# ---------------------------------------------------------------------------

def _to_kelvin(value: float, unit: str) -> float:
    match unit:
        case "K": return value
        case "C": return value + 273.15
        case "F": return (value + 459.67) * 5 / 9
        case "R": return value * 5 / 9
    raise ValueError(unit)


def _from_kelvin(kelvin: float, unit: str) -> float:
    match unit:
        case "K": return kelvin
        case "C": return kelvin - 273.15
        case "F": return kelvin * 9 / 5 - 459.67
        case "R": return kelvin * 9 / 5
    raise ValueError(unit)


# ---------------------------------------------------------------------------
# Herramienta principal
# ---------------------------------------------------------------------------

def thermo_converter(
    value: Annotated[
        float,
        Field(description="Valor numérico a convertir."),
    ],
    from_unit: Annotated[
        str,
        Field(description=(
            "Unidad de origen. "
            "Presión: Pa hPa kPa MPa GPa bar mbar atm psi mmHg torr inHg. "
            "Volumen: m3 L mL cm3 dm3 ft3 in3 gal. "
            "Temperatura: K C F R. "
            "Energía: J kJ MJ cal kcal BTU Wh kWh eV erg. "
            "Masa: kg g mg ug lb oz t. "
            "Sustancia: mol mmol umol nmol kmol. "
            "Constante R: J_mol_K kJ_mol_K cal_mol_K kcal_mol_K "
            "Latm_mol_K Lbar_mol_K m3Pa_mol_K BTU_lbmol_R."
        )),
    ],
    to_unit: Annotated[
        str,
        Field(description=(
            "Unidad de destino. Debe pertenecer a la misma magnitud "
            "física que from_unit (misma categoría)."
        )),
    ],
) -> str:
    """Convierte un valor entre unidades de la misma magnitud termodinámica.

    Soporta presión, volumen, temperatura (incluidas conversiones no
    lineales como Celsius ↔ Fahrenheit), energía, masa, cantidad de
    sustancia y la constante de gas ideal R en distintos sistemas.
    Devuelve un string con el resultado o un mensaje de error descriptivo.
    """
    # --- Temperatura (no lineal) ---
    if from_unit in _TEMP_UNITS or to_unit in _TEMP_UNITS:
        if from_unit not in _TEMP_UNITS or to_unit not in _TEMP_UNITS:
            return (
                f"Error: '{from_unit}' y '{to_unit}' no pertenecen ambas a "
                "temperatura. Unidades válidas: K, C, F, R."
            )
        result = _from_kelvin(_to_kelvin(value, from_unit), to_unit)
        return f"{value} {from_unit} = {result:.6g} {to_unit}"

    # --- Magnitudes lineales ---
    tbl_from = _UNIT_TABLE.get(from_unit)
    tbl_to   = _UNIT_TABLE.get(to_unit)

    all_units = ", ".join(sorted(_UNIT_TABLE) | _TEMP_UNITS)

    if tbl_from is None:
        return f"Error: unidad desconocida '{from_unit}'. Disponibles: {all_units}."
    if tbl_to is None:
        return f"Error: unidad desconocida '{to_unit}'. Disponibles: {all_units}."
    if tbl_from is not tbl_to:
        return (
            f"Error: '{from_unit}' y '{to_unit}' pertenecen a distintas "
            "magnitudes físicas. Solo se puede convertir dentro de la misma "
            "categoría (presión, volumen, energía, masa, sustancia o constante R)."
        )

    value_si = value * tbl_from[from_unit]
    result   = value_si / tbl_to[to_unit]

    if result != 0 and (abs(result) < 1e-4 or abs(result) >= 1e7):
        return f"{value} {from_unit} = {result:.6e} {to_unit}"
    return f"{value} {from_unit} = {result:.6g} {to_unit}"


thermo_converter_schema = ToolSchema.from_callable(thermo_converter)
