"""Conversor de unidades termodinámicas.

Cubre las siete magnitudes relevantes para termodinámica general:

  Presión     : Pa, hPa, kPa, MPa, GPa, bar, mbar, atm, psi, mmHg, torr, inHg
  Volumen     : m3, L, mL, cm3, dm3, ft3, in3, gal
  Temperatura : K, C, F, R  (Kelvin, Celsius, Fahrenheit, Rankine)
  Energía     : J, kJ, MJ, cal, kcal, BTU, Wh, kWh, eV, erg
  Masa        : kg, g, mg, ug, lb, oz, t (tonelada métrica)
  Sustancia   : mol, mmol, umol, nmol, kmol
  Constante R : J_mol_K, kJ_mol_K, cal_mol_K, kcal_mol_K,
                Latm_mol_K, Lbar_mol_K, m3Pa_mol_K, BTU_lbmol_R

Motivación: el LLM puede confundir factores de conversión o no recordar
valores exactos (e.g., 1 atm = 101 325 Pa, 1 BTU = 1 055.06 J). Delegar
la conversión a Python garantiza exactitud en problemas de termodinámica
(ley del gas ideal PV = nRT, ciclos de Carnot, cálculos de entalpía, etc.).

La constante de gas ideal R = 8.314 462 J/(mol·K) puede expresarse en
cualquier sistema de unidades usando los identificadores de _R_UNITS.
Ejemplo: thermo_converter(8.314462, "J_mol_K", "Latm_mol_K") → 0.082057.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema

# ---------------------------------------------------------------------------
# Tablas de conversión
# Cada tabla mapea: nombre_de_unidad → factor para convertir a la unidad SI
# base de esa categoría. La conversión general es:
#   valor_SI = valor_origen × factor_origen
#   valor_destino = valor_SI / factor_destino
# ---------------------------------------------------------------------------

# Presión (base: Pa = Pascal)
_PRESSURE: dict[str, float] = {
    "Pa":   1.0,
    "hPa":  1e2,        # hectopascal (meteorología)
    "kPa":  1e3,
    "MPa":  1e6,
    "GPa":  1e9,
    "bar":  1e5,
    "mbar": 1e2,
    "atm":  101_325.0,  # atmósfera estándar (definición exacta)
    "psi":  6_894.757,  # libra-fuerza por pulgada cuadrada
    "mmHg": 133.322,    # milímetro de mercurio (≈ torr)
    "torr": 133.322,    # torr (definición: 1/760 atm = 133.322... Pa)
    "inHg": 3_386.39,   # pulgada de mercurio
}

# Volumen (base: m³ = metro cúbico)
_VOLUME: dict[str, float] = {
    "m3":  1.0,
    "L":   1e-3,        # litro (dm³)
    "mL":  1e-6,        # mililitro (cm³)
    "cm3": 1e-6,
    "dm3": 1e-3,
    "ft3": 0.028_316_8, # pie cúbico
    "in3": 1.638_71e-5, # pulgada cúbica
    "gal": 3.785_41e-3, # galón estadounidense (US liquid gallon)
}

# Energía (base: J = Joule)
_ENERGY: dict[str, float] = {
    "J":   1.0,
    "kJ":  1e3,
    "MJ":  1e6,
    "cal": 4.184,         # caloría termoquímica (definición exacta)
    "kcal":4_184.0,       # kilocaloría
    "BTU": 1_055.06,      # British Thermal Unit (IT)
    "Wh":  3_600.0,       # vatio-hora
    "kWh": 3_600_000.0,
    "eV":  1.602_18e-19,  # electronvoltio (CODATA 2018)
    "erg": 1e-7,          # ergio (sistema CGS)
}

# Masa (base: kg = kilogramo)
_MASS: dict[str, float] = {
    "kg": 1.0,
    "g":  1e-3,
    "mg": 1e-6,
    "ug": 1e-9,         # microgramo
    "lb": 0.453_592,    # libra avoirdupois
    "oz": 0.028_349_5,  # onza avoirdupois
    "t":  1_000.0,      # tonelada métrica
}

# Cantidad de sustancia (base: mol)
_AMOUNT: dict[str, float] = {
    "mol":  1.0,
    "mmol": 1e-3,
    "umol": 1e-6,  # micromol
    "nmol": 1e-9,  # nanomol
    "kmol": 1e3,   # kilomol (usado en ingeniería de procesos)
}

# Constante universal de los gases ideales R en distintos sistemas de unidades.
# Factor = cuántos J/(mol·K) equivalen a 1 unidad de ese sistema.
# Todos los factores se derivan de R_SI = 8.314 462 J/(mol·K):
#
#   Sistema       | R en esas unidades  | Factor (= 1 unidad en J/(mol·K))
#   ------------- | ------------------- | --------------------------------
#   J/(mol·K)     | 8.314 462           | 1.0          (identidad)
#   kJ/(mol·K)    | 0.008 314 462       | 1 000.0      (1 kJ = 1 000 J)
#   cal/(mol·K)   | 1.985 88            | 4.184        (1 cal = 4.184 J)
#   kcal/(mol·K)  | 0.001 985 88        | 4 184.0
#   L·atm/(mol·K) | 0.082 057           | 101.325      (1 L·atm = 101.325 J)
#   L·bar/(mol·K) | 0.083 145           | 100.0        (1 L·bar = 100 J)
#   m³·Pa/(mol·K) | 8.314 462           | 1.0          (equiv. a J/(mol·K))
#   BTU/(lbmol·R) | 1.985 88            | 4.186 81     (1 BTU/lbmol·R ≈ 4.1868 J/mol·K)
#
# La conversión entre sistemas sigue la misma lógica que las demás magnitudes:
#   valor_destino = valor_origen × factor_origen / factor_destino
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

# Unidades de temperatura. Se tratan aparte porque la conversión entre ellas
# es no lineal (involucra sumas/restas de offsets, no solo factores).
_TEMP_UNITS: frozenset[str] = frozenset({"K", "C", "F", "R"})

# Índice global: unidad → tabla de su categoría.
# Se construye una sola vez al importar el módulo para que la búsqueda en
# thermo_converter sea O(1). La identidad del objeto dict (`is`) se usa
# después para detectar conversiones entre categorías distintas.
_UNIT_TABLE: dict[str, dict[str, float]] = {}
for _tbl in (_PRESSURE, _VOLUME, _ENERGY, _MASS, _AMOUNT, _R_UNITS):
    for _u in _tbl:
        _UNIT_TABLE[_u] = _tbl


# ---------------------------------------------------------------------------
# Auxiliares de conversión de temperatura
# ---------------------------------------------------------------------------

def _to_kelvin(value: float, unit: str) -> float:
    """Convierte un valor de temperatura a Kelvin.

    Kelvin actúa como unidad pivote para todas las conversiones de temperatura.
    Cada escala tiene un offset diferente respecto al cero absoluto (K = 0),
    por lo que la conversión no puede expresarse con un simple factor.

    Parameters
    ----------
    value : float
        Temperatura en la unidad indicada por `unit`.
    unit : str
        Escala de origen: "K" (Kelvin), "C" (Celsius), "F" (Fahrenheit)
        o "R" (Rankine).

    Returns
    -------
    float
        Temperatura equivalente en Kelvin.
    """
    match unit:
        case "K": return value                       # K = K
        case "C": return value + 273.15              # K = °C + 273.15
        case "F": return (value + 459.67) * 5 / 9   # K = (°F + 459.67) × 5/9
        case "R": return value * 5 / 9              # K = °R × 5/9  (Rankine es Fahrenheit absoluto)
    raise ValueError(f"Unidad de temperatura desconocida: '{unit}'")


def _from_kelvin(kelvin: float, unit: str) -> float:
    """Convierte una temperatura en Kelvin a la escala destino.

    Inversa de _to_kelvin. Se aplica tras convertir la unidad origen a
    Kelvin, completando la cadena origen → Kelvin → destino.

    Parameters
    ----------
    kelvin : float
        Temperatura en Kelvin.
    unit : str
        Escala de destino: "K", "C", "F" o "R".

    Returns
    -------
    float
        Temperatura equivalente en la escala indicada.
    """
    match unit:
        case "K": return kelvin
        case "C": return kelvin - 273.15
        case "F": return kelvin * 9 / 5 - 459.67
        case "R": return kelvin * 9 / 5
    raise ValueError(f"Unidad de temperatura desconocida: '{unit}'")


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

    El algoritmo de conversión depende de la categoría de las unidades:

    - **Temperatura** (K, C, F, R): conversión no lineal a través de Kelvin
      como pivote, usando _to_kelvin y _from_kelvin.
    - **Resto de magnitudes** (presión, volumen, energía, masa, sustancia,
      constante R): conversión lineal a través de la unidad SI base de cada
      categoría. El factor en cada tabla indica cuántas unidades SI base
      equivalen a 1 unidad de esa escala.

    En todos los casos de error (unidad desconocida, mezcla de categorías)
    devuelve un string con el mensaje descriptivo en lugar de lanzar una
    excepción, para no interrumpir el bucle del agente.

    Parameters
    ----------
    value : float
        Valor a convertir.
    from_unit : str
        Unidad de origen (ver Field.description para el listado completo).
    to_unit : str
        Unidad de destino de la misma categoría que from_unit.

    Returns
    -------
    str
        String con el resultado en formato "X from_unit = Y to_unit", o un
        mensaje de error prefijado con "Error:".

    Examples
    --------
    thermo_converter(1, "atm", "kPa")          → "1 atm = 101.325 kPa"
    thermo_converter(100, "C", "F")            → "100 C = 212 F"
    thermo_converter(0, "C", "K")             → "0 C = 273.15 K"
    thermo_converter(8.314462, "J_mol_K", "Latm_mol_K") → "8.314462 J_mol_K = 0.082057 Latm_mol_K"
    thermo_converter(1, "atm", "J")            → "Error: 'atm' y 'J' pertenecen a distintas..."
    """
    # --- Temperatura: conversión no lineal, requiere tratamiento especial ---
    if from_unit in _TEMP_UNITS or to_unit in _TEMP_UNITS:
        if from_unit not in _TEMP_UNITS or to_unit not in _TEMP_UNITS:
            return (
                f"Error: '{from_unit}' y '{to_unit}' no pertenecen ambas a "
                "temperatura. Unidades válidas: K, C, F, R."
            )
        result = _from_kelvin(_to_kelvin(value, from_unit), to_unit)
        return f"{value} {from_unit} = {result:.6g} {to_unit}"

    # --- Resto de magnitudes: conversión lineal via unidad SI base ---
    tbl_from = _UNIT_TABLE.get(from_unit)
    tbl_to   = _UNIT_TABLE.get(to_unit)

    all_units = ", ".join(sorted(_UNIT_TABLE) | _TEMP_UNITS)

    if tbl_from is None:
        return f"Error: unidad desconocida '{from_unit}'. Disponibles: {all_units}."
    if tbl_to is None:
        return f"Error: unidad desconocida '{to_unit}'. Disponibles: {all_units}."

    # Usar identidad de objeto (is) en lugar de igualdad (==) para verificar
    # que ambas unidades pertenecen exactamente a la misma tabla/categoría.
    # Dos dicts distintos con los mismos valores seguirían siendo objetos
    # diferentes, lo que impediría mezclar e.g. "atm" (presión) con "J" (energía).
    if tbl_from is not tbl_to:
        return (
            f"Error: '{from_unit}' y '{to_unit}' pertenecen a distintas "
            "magnitudes físicas. Solo se puede convertir dentro de la misma "
            "categoría (presión, volumen, energía, masa, sustancia o constante R)."
        )

    # Doble paso: origen → unidad SI base → destino
    value_si = value * tbl_from[from_unit]
    result   = value_si / tbl_to[to_unit]

    # Notación científica para valores muy pequeños o muy grandes.
    if result != 0 and (abs(result) < 1e-4 or abs(result) >= 1e7):
        return f"{value} {from_unit} = {result:.6e} {to_unit}"
    return f"{value} {from_unit} = {result:.6g} {to_unit}"


thermo_converter_schema = ToolSchema.from_callable(thermo_converter)
