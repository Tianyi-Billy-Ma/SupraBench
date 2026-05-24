"""Shared parsing utilities for the Suprabank cleaning pipeline."""

import re
import math

# ---------------------------------------------------------------------------
# Ka parsing
# ---------------------------------------------------------------------------
# Raw formats seen: "380.0", "7.76⋅104", "1.12⋅107M-1", "8709.64"
# Middle dot variants: ⋅ · • * x ×

_MIDDLE_DOT = r'[⋅·•×xX\*]'

def parse_ka(raw: str):
    """
    Parse a Ka string to float (M⁻¹).
    Returns None if not parseable.
    """
    if not raw:
        return None
    s = raw.strip()
    # Strip unit suffix (M-1, M⁻¹, /M, M^-1)
    s = re.sub(r'M[-⁻]1|M\^[-−]1|/M', '', s, flags=re.I).strip()
    # Normalise middle-dot notation: 7.76⋅104 → 7.76e4
    s = re.sub(_MIDDLE_DOT + r'(\d+)', r'e\1', s)
    # Try direct float conversion
    try:
        val = float(s)
        return val if val > 0 else None
    except ValueError:
        pass
    # Try scientific notation with explicit 10^ pattern
    m = re.match(r'([\d.]+)\s*[×xX\*]?\s*10\^?([\-−]?\d+)', s)
    if m:
        try:
            return float(m.group(1)) * 10 ** int(m.group(2).replace('−', '-'))
        except (ValueError, OverflowError):
            return None
    return None


def parse_temperature(raw: str):
    """'25.0°C' → 25.0. Returns None if not parseable."""
    if not raw:
        return None
    m = re.search(r'([\-\d.]+)\s*°?C', raw, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def parse_ph(raw: str):
    """'7.4' → 7.4. Returns None if not parseable."""
    if not raw:
        return None
    try:
        return float(raw.strip())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Solvent classification
# ---------------------------------------------------------------------------

ORGANIC_KEYWORDS = {
    'methanol', 'meoh', 'ethanol', 'etoh', 'acetonitrile', 'mecn', 'ch3cn',
    'dmso', 'dimethyl sulfoxide', 'chloroform', 'cdcl3', 'chcl3',
    'dichloromethane', 'dcm', 'ch2cl2', 'cd2cl2',
    'acetone', 'toluene', 'dioxane', 'thf', 'tetrahydrofuran',
    'dmf', 'dimethylformamide', 'diethyl ether', 'ethyl acetate',
    'hexane', 'benzene', 'pyridine', 'formic acid',
    'deuterated methanol', 'methanol-d', 'cd3od',
    'acetonitrile-d', 'toluene-d', 'dichloromethane-d',
}

AQUEOUS_KEYWORDS = {
    'water', 'buffer', 'aqueous', 'h2o', 'd2o', 'deuterium oxide',
    'heavy water', 'pbs', 'tris', 'hepes', 'mes', 'pipes', 'mops',
}


def classify_solvent(solvent_col: str, solvents_col: str) -> str:
    """
    Returns 'organic', 'aqueous', or 'complex' (mixed/unknown).
    Uses both the 'solvent' column and the 'solvents' column.
    """
    combined = ' '.join([solvent_col or '', solvents_col or '']).lower()

    is_organic = any(kw in combined for kw in ORGANIC_KEYWORDS)
    is_aqueous = any(kw in combined for kw in AQUEOUS_KEYWORDS)

    if is_organic and not is_aqueous:
        return 'organic'
    if is_aqueous:
        return 'aqueous'
    # 'complex' tag on the site often means mixed/buffered
    if 'complex' in combined:
        return 'aqueous'   # treat complex as aqueous (usually buffer mixtures)
    return 'unknown'


# ---------------------------------------------------------------------------
# Van't Hoff correction
# ---------------------------------------------------------------------------

R = 8.314  # J mol⁻¹ K⁻¹
T_STD = 298.15  # 25 °C in Kelvin


def vanthoff_correct_ka(ka: float, t_celsius: float, dh_j_mol: float) -> float:
    """
    Correct Ka measured at t_celsius to 25 °C using van't Hoff equation.
    ln(Ka2/Ka1) = −ΔH°/R × (1/T2 − 1/T1)
    Requires ΔH in J/mol (pass kJ/mol × 1000 if needed).
    """
    T1 = t_celsius + 273.15
    T2 = T_STD
    ratio = math.exp(-dh_j_mol / R * (1 / T2 - 1 / T1))
    return ka * ratio


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def iqr_outlier_mask(values: list, k: float = 1.5) -> list:
    """
    Returns a boolean mask (True = outlier) for a list of floats.
    Uses Tukey's IQR fence method. Requires ≥4 points; otherwise no outliers.
    """
    if len(values) < 4:
        return [False] * len(values)
    import statistics
    sorted_v = sorted(values)
    n = len(sorted_v)
    q1 = sorted_v[n // 4]
    q3 = sorted_v[(3 * n) // 4]
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return [v < lo or v > hi for v in values]
