"""Sensitivity-gated PLOG proposals and lossless coordinate initialization.

Eligibility is a caller-owned declaration of artificial reactions. YAML rate
forms cannot establish whether a reaction was borrowed from another model.
These helpers never select reactions outside that declaration or alter rates.
"""
from numbers import Integral
import math
import numpy as np
from .uncertainty import _load, _entries


def rank_plog_candidates(screen, scheme, eligible, top=1, min_relative_score=.1):
    """Return eligible one-based reactions ordered by resolved sensitivity.

    Scores must be positive, fully resolved and have reliable coordinates.
    The cutoff is relative to the strongest resolved score in the whole
    screen. Only plain positive-A Arrhenius reactions qualify; existing PLOG,
    third-body, falloff and explicitly marked borrowed reactions are excluded.
    ``eligible`` must contain only caller-verified artificial reactions.
    """
    if isinstance(top, bool) or not isinstance(top, Integral) or top < 1:
        raise ValueError('top must be a positive integer.')
    if not np.isfinite(min_relative_score) or not 0 <= min_relative_score <= 1:
        raise ValueError('min_relative_score must be between zero and one.')
    data = _load(scheme)
    eligible = list(eligible)
    if any(isinstance(n, bool) or not isinstance(n, Integral) or not 1 <= n <= len(data['reactions'])
           for n in eligible) or len(set(eligible)) != len(eligible):
        raise ValueError('eligible must contain distinct valid one-based reaction integers.')
    allowed = set()
    for n in eligible:
        reaction = data['reactions'][n-1]
        if (reaction.get('type', 'elementary') != 'elementary'
                or 'rate-constant' not in reaction
                or any(k in reaction for k in ('rate-constants', 'low-P-rate-constant', 'high-P-rate-constant', 'efficiencies'))
                or reaction.get('borrowed', False) or reaction.get('ffcm2_R') is not None
                or '(+M)' in reaction.get('equation', '')
                or ' + M' in reaction.get('equation', '')):
            continue
        try:
            _entries(reaction)
        except (ValueError, TypeError, OverflowError):
            continue
        allowed.add(int(n))
    resolved = []
    for row in screen.get('ranking', []):
        score = row.get('score', 0.)
        if (row.get('all_resolved') and row.get('reliable_coordinates')
                and isinstance(score, (int, float)) and np.isfinite(score) and score > 0):
            resolved.append(row)
    if not resolved:
        return []
    cutoff = max(row['score'] for row in resolved) * min_relative_score
    ranked = sorted((row for row in resolved if row.get('group') in allowed and row['score'] >= cutoff),
                    key=lambda row: (-row['score'], int(row['group'])))
    # A valid screen contains one row per reaction; avoid accidental duplicate proposals.
    return list(dict.fromkeys(int(row['group']) for row in ranked))[:top]


def transfer_coordinates(old_space, new_space, x):
    """Copy registry coordinates; newly split nodes inherit the shared value.

    Matches reaction, pressure (Pa), and parameter. Ea/b scales and pivot
    temperature must be unchanged. New nodes can inherit only an unsplit
    reaction coordinate; ambiguous interpolation is rejected. Nominal anchors
    must be preserved by the caller and checked with check_rate_equivalence.
    Bounds of the new stage are not checked here: use the mapped vector as its
    center, then construct/validate the centered stage space.
    """
    x = np.asarray(x, dtype=float)
    old_labels, new_labels = old_space.coordinate_labels, new_space.coordinate_labels
    if x.shape != (len(old_labels),) or not np.isfinite(x).all():
        raise ValueError('Coordinates must be finite and match the old registry.')
    if hasattr(old_space, 'pivot') or hasattr(new_space, 'pivot'):
        if not math.isclose(old_space.pivot, new_space.pivot, rel_tol=1e-12):
            raise ValueError('Pivot temperatures must remain unchanged.')
    def match_pressure(a, b):
        return a is b if a is None or b is None else math.isclose(float(a), float(b), rel_tol=1e-12)
    values = []
    for new in new_labels:
        candidates = [(i, old) for i, old in enumerate(old_labels)
                      if old['reaction'] == new['reaction'] and old.get('parameter') == new.get('parameter')
                      and old.get('species') == new.get('species')]
        matches = [(i, old) for i, old in candidates
                   if match_pressure(old.get('pressure_pa'), new.get('pressure_pa'))]
        if not matches and new.get('pressure_pa') is not None:
            matches = [(i, old) for i, old in candidates if old.get('pressure_pa') is None]
        if len(matches) != 1:
            raise ValueError(f'Cannot unambiguously transfer coordinate {new}.')
        index, old = matches[0]
        if not math.isclose(float(old.get('scale', 1.)), float(new.get('scale', 1.)), rel_tol=1e-12):
            raise ValueError('Coordinate scales must remain unchanged.')
        values.append(x[index])
    return np.asarray(values)


def check_rate_equivalence(old_space, new_space, old_x, new_x,
                           temperatures=(300., 1000., 1800., 3000.),
                           pressures=(1e5, 5e6, 1e7), rtol=1e-10):
    """Numerically check all forward rate constants on a T/P grid (not a proof).

    Equal composition is used for both gases, including concentration-dependent
    rates. No integration is performed. Returns valid and max_relative_error.
    """
    temperatures, pressures = np.asarray(temperatures, float), np.asarray(pressures, float)
    if any(v.ndim != 1 or not len(v) or not np.isfinite(v).all() or np.any(v <= 0)
           for v in (temperatures, pressures)) or not np.isfinite(rtol) or rtol < 0:
        raise ValueError('Use positive finite T/P grids and nonnegative finite rtol.')
    old, new = old_space.make_gas(old_x), new_space.make_gas(new_x)
    if old.species_names != new.species_names or old.reaction_equations() != new.reaction_equations():
        raise ValueError('Species and reaction ordering must be preserved.')
    maximum = 0.
    valid = True
    for temperature in temperatures:
        for pressure in pressures:
            old.TPX = float(temperature), float(pressure), np.ones(old.n_species)
            new.TPX = float(temperature), float(pressure), old.X
            a, b = old.forward_rate_constants, new.forward_rate_constants
            finite = np.isfinite(a).all() and np.isfinite(b).all()
            relative = np.abs(a-b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), np.finfo(float).tiny)
            maximum = max(maximum, float(relative.max())) if finite else float('inf')
            valid &= bool(finite and np.all(relative <= rtol))
    return dict(valid=bool(valid), max_relative_error=maximum,
                temperatures=temperatures.tolist(), pressures_pa=pressures.tolist(), rtol=rtol)
