"""Arrhenius rates, explicit reaction orders, and balanced stoichiometry."""
import re
import numpy as np


def rates(scheme):
    # Keep YAML order; returned entries are references for in-place rate edits.
    """Return editable Arrhenius-entry dictionaries in reaction and stored pressure order."""
    found = []
    for i, reaction in enumerate(scheme['reactions'], 1):
        entries = []
        for key in ('rate-constant', 'rate-constants', 'high-P-rate-constant', 'low-P-rate-constant'):
            if key in reaction:
                entries.extend(reaction[key] if key == 'rate-constants' else [reaction[key]])
        if not entries or any(not {'A', 'b', 'Ea'} <= set(r) for r in entries):
            raise ValueError(f'Reaction {i} has an unsupported rate form.')
        found.extend(entries)
    return found


def parameter_dict(scheme):
    """Collect all rate entries into aligned A, b, and Ea lists."""
    entries = rates(scheme)
    return {k: [float(r[k]) for r in entries] for k in ('A', 'b', 'Ea')}


def order_dict(scheme):
    """Collect explicit reaction orders by one-based reaction number and species."""
    return {i: {name: float(value) for name, value in reaction['orders'].items()}
            for i, reaction in enumerate(scheme['reactions'], 1) if reaction.get('orders')}


def modify_orders(scheme, orders):
    """Update existing explicit orders in place from {reaction: {species: value}}."""
    updates = []
    for number, values in orders.items():
        if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= len(scheme['reactions']):
            raise ValueError('Reaction numbers are one-based.')
        stored = scheme['reactions'][number - 1].get('orders', {})
        if not values or not set(values) <= set(stored):
            raise ValueError(f'Reaction {number}: only existing explicit orders can be modified.')
        for species, value in values.items():
            value = float(value)
            if not np.isfinite(value):
                raise ValueError('Reaction orders must be finite.')
            updates.append((stored, species, value))
    # Validate every value before changing the mechanism.
    for stored, species, value in updates:
        stored[species] = value
    for number in orders:
        reaction = scheme['reactions'][number - 1]
        if any(float(value) < 0 for value in reaction['orders'].values()):
            reaction['negative-orders'] = True
    return scheme


class Balance:
    """Derive dependent stoichiometric coefficients from elemental conservation."""
    def __init__(self, scheme, specification):
        """Build the signed element matrix and reject ambiguous or constrained parameterizations."""
        self.spec = specification
        index = specification.reaction
        if not isinstance(index, int) or not 1 <= index <= len(scheme['reactions']):
            raise ValueError('Reaction numbers are one-based.')
        fixed = specification.fixed
        independent = specification.independent
        dependent = specification.dependent
        if not independent or not dependent:
            raise ValueError('Declare independent and dependent coefficients.')
        locations = list(fixed) + [c.location for c in independent.values()] + list(dependent)
        if len(set(locations)) != len(locations):
            raise ValueError('Coefficient locations must be unique.')
        species = {s['name']: s['composition'] for s in scheme['species']}
        elements = sorted({e for s in species.values() for e in s})
        self.columns = {}
        phase_species = set(scheme['phases'][0]['species'])
        for location in locations:
            side, name = location.split('.', 1)
            if side not in ('reactants', 'products') or name not in phase_species:
                raise ValueError(f'Invalid coefficient location {location}.')
            # Signed atom counts make a balanced reaction sum to zero.
            sign = -1 if side == 'reactants' else 1
            self.columns[location] = sign * np.array([species[name].get(e, 0) for e in elements], dtype=float)
        if not any(k.startswith('reactants.') and v == 1 for k, v in fixed.items()):
            raise ValueError('Fix a reactant (normally fuel) coefficient to 1.')
        if any(not np.isfinite(v) or v < 0 for v in fixed.values()):
            raise ValueError('Fixed coefficients must be finite and nonnegative.')
        for c in independent.values():
            lo, _ = c.effective_bounds()
            if lo < 0:
                raise ValueError('Stoichiometric coefficients cannot be negative.')
        # Each dependent coefficient needs a uniquely determined solution.
        self.matrix = np.column_stack([self.columns[k] for k in dependent])
        if np.linalg.matrix_rank(self.matrix) != len(dependent):
            raise ValueError('Dependent coefficients are underdetermined by elemental balance.')
        # Independent coefficients must truly be free, not hide extra constraints.
        for rhs in [sum((self.columns[k] * v for k, v in fixed.items()), np.zeros(len(elements)))] + [self.columns[c.location] for c in independent.values()]:
            solution = np.linalg.lstsq(self.matrix, -rhs, rcond=None)[0]
            if not np.allclose(self.matrix @ solution, -rhs, rtol=1e-10, atol=1e-10):
                raise ValueError('Stoichiometric definition is overconstrained.')
        equation = scheme['reactions'][index - 1]['equation']
        self.arrow = re.search(r'<=>|=>|=', equation).group()
        self.locations = locations

    def apply(self, scheme, values):
        """Validate independent values and replace the selected reaction equation in place."""
        if set(values) != set(self.spec.independent):
            raise ValueError('Missing or unexpected stoichiometric parameters.')
        coefficients = dict(self.spec.fixed)
        for name, definition in self.spec.independent.items():
            value = float(values[name])
            lo, hi = definition.effective_bounds()
            if not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{name} is outside its bounds.')
            coefficients[definition.location] = value
        rhs = -sum((self.columns[k] * v for k, v in coefficients.items()), np.zeros(self.matrix.shape[0]))
        # Solve for the remaining coefficients after fixing the tunable values.
        derived = np.linalg.lstsq(self.matrix, rhs, rcond=None)[0]
        if np.any(derived < -1e-10):
            raise ValueError('Elemental balance requires negative coefficients.')
        # Clamp only tiny negative roundoff; physical negatives were rejected.
        derived = np.maximum(derived, 0)
        if not np.allclose(self.matrix @ derived, rhs, atol=1e-10, rtol=1e-10):
            raise ValueError('Could not satisfy elemental balance.')
        coefficients.update(zip(self.spec.dependent, derived))
        # Rebuild the equation, omitting zero terms and preserving its arrow.
        sides = []
        for side in ('reactants', 'products'):
            terms = [f'{float(coefficients[k])!r} {k.split(".", 1)[1]}' for k in self.locations
                     if k.startswith(side + '.') and coefficients[k] > 0]
            if not terms:
                raise ValueError('Reaction sides cannot be empty.')
            sides.append(' + '.join(terms))
        scheme['reactions'][self.spec.reaction - 1]['equation'] = f' {self.arrow} '.join(sides)
