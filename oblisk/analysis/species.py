import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from oblisk.analysis.element_data import _ELEMENT_MASS_AND_Z

A_BY_SYM: dict[str, int] = {
    sym: mass for sym, (mass, _) in _ELEMENT_MASS_AND_Z.items()
}

DEFAULT_CLASSIFICATION_ELEMENTS: tuple[str, ...] = ("H", "C", "O", "Si")


def parse_species(label: str) -> tuple[str | None, int | None]:
    """
    Parse labels like 'C^6+' -> ('C', 6).
    """
    match = re.match(r"^([A-Za-z]+)\^(\d+)\+$", label)
    if match is None:
        return None, None
    return match.group(1), int(match.group(2))


def is_known_element_symbol(symbol: str) -> bool:
    return symbol in _ELEMENT_MASS_AND_Z


def normalize_classification_elements(
    symbols: Sequence[str],
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols:
        s = raw.strip()
        if not s:
            continue
        if len(s) == 1:
            sym = s.upper()
        elif len(s) == 2:
            sym = s[0].upper() + s[1].lower()
        else:
            sym = s[0].upper() + s[1:].lower()
        if sym not in _ELEMENT_MASS_AND_Z:
            msg = f"Unknown element symbol: {raw!r}"
            raise ValueError(msg)
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    if not out:
        msg = "At least one element symbol is required"
        raise ValueError(msg)
    return out


def build_species_set(
    element_symbols: Sequence[str] | None = None,
) -> list[dict[str, float | str]]:
    """
    Species ladder used by classification code (mass numbers / charge states).
    Returns dicts of shape: {'name': str, 'm_over_q': float}.
    """
    elems_src = (
        element_symbols
        if element_symbols is not None
        else DEFAULT_CLASSIFICATION_ELEMENTS
    )
    elems = normalize_classification_elements(elems_src)

    species: list[dict[str, float | str]] = []

    m_u = 1.66053906660e-27  # kg
    m_p = 1.67262192369e-27  # kg

    for sym in elems:
        mass, zmax = _ELEMENT_MASS_AND_Z[sym]
        for q in range(1, zmax + 1):
            if sym == "H" and mass == 1:
                target = 1.0
            else:
                target = (mass * m_u) / (q * m_p)
            species.append({"name": f"{sym}^{q}+", "m_over_q": target})
    return species


class CandidateMatch(BaseModel):
    name: str
    mq_target: float
    rel_err: float


def candidate_from_mapping(raw_candidate: Mapping[str, Any]) -> CandidateMatch:
    return CandidateMatch(
        name=str(raw_candidate["name"]),
        mq_target=float(raw_candidate["mq_target"]),
        rel_err=float(raw_candidate["rel_err"]),
    )


def normalize_candidates(
    candidates: Sequence[CandidateMatch | Mapping[str, Any]],
) -> list[CandidateMatch]:
    output: list[CandidateMatch] = []
    for candidate in candidates:
        if isinstance(candidate, CandidateMatch):
            output.append(candidate)
        else:
            output.append(candidate_from_mapping(candidate))
    return output


def same_mq_names(
    candidates: Sequence[CandidateMatch],
    tol: float = 1e-12,
) -> list[str]:
    if not candidates:
        return []

    target = float(candidates[0].mq_target)
    output: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        mq_ok = abs(float(candidate.mq_target) - target) <= tol
        if mq_ok and candidate.name not in seen:
            seen.add(candidate.name)
            output.append(candidate.name)
    return output


def nearby_names(
    candidates: Sequence[CandidateMatch],
    max_rel: float,
) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        ok_err = float(candidate.rel_err) <= max_rel
        if ok_err and candidate.name not in seen:
            seen.add(candidate.name)
            output.append(candidate.name)
    return output
