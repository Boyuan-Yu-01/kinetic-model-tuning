"""Immutable nominal Arrhenius uncertainty coordinates and physical-rate audits.

Published uncertainty factors are re-centered on the supplied nominal mechanism;
this is an explicit convention, not a claim about the original published bounds.
"""
from copy import deepcopy
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
import math
import numpy as np
import yaml


def _load(value):
    if isinstance(value, dict):
        return deepcopy(value)
    with open(value) as stream:
        return yaml.safe_load(stream)


def _entries(reaction):
    result = []
    for key in ('rate-constant', 'low-P-rate-constant', 'high-P-rate-constant', 'rate-constants'):
        if key in reaction:
            result.extend(reaction[key] if key == 'rate-constants' else [reaction[key]])
    if not result or any(not {'A', 'b', 'Ea'} <= set(e) for e in result):
        raise ValueError('Unsupported Arrhenius rate form.')
    for entry in result:
        if not all(np.isfinite(float(entry[k])) for k in ('A', 'b', 'Ea')) or float(entry['A']) <= 0:
            raise ValueError('All Arrhenius A values must be finite and positive; b and Ea finite.')
    return result


def _identity(reaction):
    import re
    eq = reaction['equation']
    sides = re.split(r'(<=>|=>|=)', eq)
    if len(sides) != 3:
        raise ValueError(f'Unsupported reaction equation: {eq}')
    # Preserve coefficients and third bodies, allowing term reordering only.
    return tuple(tuple(sorted(''.join(t.split()) for t in side.replace('(+M)', '').split('+')))
                 if i != 1 else side for i, side in enumerate(sides)), '(+M)' in eq


def _shape(reaction):
    value = deepcopy(reaction)
    value.pop('equation', None)
    value.pop('efficiencies', None)  # Checked after projection onto retained species.
    for entry in _entries(value):
        entry.pop('A')
    return value


def _near(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_near(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_near(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def _validate_expected_factors(indexed, expected_factors):
    """Check a complete matched-factor declaration without changing any bounds."""
    if expected_factors is None:
        return
    if not isinstance(expected_factors, Mapping):
        raise ValueError('expected_factors must be a mapping of matched reaction numbers to factors.')
    declared = {}
    for key, value in expected_factors.items():
        # JSON transports integer dictionary keys as canonical decimal strings.
        if isinstance(key, Integral) and not isinstance(key, (bool, np.bool_)):
            number = int(key)
        elif (isinstance(key, str) and key.isascii() and key.isdigit()
              and str(int(key)) == key):
            number = int(key)
        else:
            raise ValueError('expected_factors reaction numbers must be positive one-based integers.')
        if number < 1:
            raise ValueError('expected_factors reaction numbers must be positive one-based integers.')
        if number in declared:
            raise ValueError(f'expected_factors repeats reaction R{number}.')
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f'R{number}: expected uncertainty factor must be a finite number greater than one.')
        try:
            factor = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f'R{number}: expected uncertainty factor must be a finite number greater than one.') from exc
        if not math.isfinite(factor) or factor <= 1:
            raise ValueError(f'R{number}: expected uncertainty factor must be a finite number greater than one.')
        declared[number] = factor
    matched = {number: row for number, row in indexed.items() if row.get('ffcm2_R') is not None}
    missing, extra = sorted(set(matched) - set(declared)), sorted(set(declared) - set(matched))
    if missing or extra:
        raise ValueError('expected_factors must identify every matched reaction exactly once; '
                         f'missing={missing}, unexpected={extra}.')
    mismatches = [f'R{number}: declared {declared[number]!r}, manifest {float(row["uncertainty_factor"])!r}'
                  for number, row in matched.items()
                  if declared[number] != float(row['uncertainty_factor'])]
    if mismatches:
        raise ValueError('Expected uncertainty factors do not match the immutable manifest: '
                         + '; '.join(mismatches))


def _manifest_from_factors(source, nominal, factors):
    """Match unchanged rate shapes to reference kinetics without an external file.

    A values are deliberately excluded from matching; duplicate channels must
    still have unique b/Ea/rate-form matches. Ambiguity requires an explicit
    manifest instead of guessing a nominal reaction.
    """
    if not isinstance(factors, Mapping):
        raise ValueError('Without a manifest, supply uncertainty_factors as a reaction-to-factor mapping.')
    values = {str(key): value for key, value in factors.items()}
    retained = {species['name'] for species in source['species']}
    reference_by_identity = {}
    for ref_number, candidate in enumerate(nominal['reactions'], 1):
        reference_by_identity.setdefault(_identity(candidate), []).append((ref_number, candidate))
    rows = []
    for number, reaction in enumerate(source['reactions'], 1):
        matches = []
        candidates = reference_by_identity.get(_identity(reaction), [])
        shape = _shape(reaction) if candidates else None
        for ref_number, candidate in candidates:
            if not _near(shape, _shape(candidate)):
                continue
            a, b = reaction.get('efficiencies', {}), candidate.get('efficiencies', {})
            if all(_near(a.get(species, 1.), b.get(species, 1.)) for species in retained):
                matches.append((ref_number, candidate))
        if len(matches) > 1:
            raise ValueError(f'R{number}: ambiguous reference match; supply an explicit manifest.')
        ref_number, candidate = matches[0] if matches else (None, None)
        rows.append(dict(reduced_R=number, equation=reaction['equation'],
            ffcm2_R=ref_number, uncertainty_factor=values.get(str(number)),
            nominal_optimized_ffcm2=deepcopy(candidate)))
    _validate_expected_factors({row['reduced_R']: row for row in rows}, factors)
    return dict(reactions=rows)


class RateUncertaintySpace:
    """One log multiplier per reaction; falloff limits share the multiplier.

    Without a manifest, ``uncertainty_factors`` supplies all matched factors;
    nominal reactions are matched by equation, rate shape and efficiencies.
    ``expected_factors`` optionally declares all matched factors and must equal
    the manifest exactly; it is an assertion, never a bound override.
    ``custom_factors`` maps otherwise unmatched one-based reaction numbers to
    factors greater than one. These priors are explicitly source-centered.
    ``unbounded_reactions`` explicitly selects unmatched reactions with no
    uncertainty factor: source-centered log coordinates span (-inf, inf),
    subject only to positive finite representable A values and multipliers.
    PLOG reactions listed in ``plog_free`` use one multiplier per distinct pressure;
    other reactions share one multiplier. b and Ea stay fixed.
    ``x0`` can differ from ``source_coordinates`` only by a reported tiny
    inward adjustment at a strict bound. The original source is never changed.
    """
    def __init__(self, model, reference, manifest=None, strict_margin=1e-9, custom_factors=None,
                 unbounded_reactions=(), expected_factors=None, uncertainty_factors=None, plog_free=None):
        self._source = _load(model)
        nominal = _load(reference)
        if manifest is None:
            manifest = _manifest_from_factors(self._source, nominal, uncertainty_factors)
        else:
            if uncertainty_factors is not None:
                raise ValueError('Use either a manifest or uncertainty_factors, not both.')
            manifest = _load(manifest)
        self.audit = []
        self.strict_margin = float(strict_margin)
        if not np.isfinite(self.strict_margin) or not 0 < self.strict_margin < 0.01:
            raise ValueError('strict_margin must be positive and less than 0.01.')
        # Cantera defaults time to seconds when the YAML omits it.
        source_units = {'time': 's', **self._source.get('units', {})}
        nominal_units = {'time': 's', **nominal.get('units', {})}
        if source_units != nominal_units:
            raise ValueError('Source and nominal mechanisms must use identical YAML units.')
        rows = manifest['reactions']
        indexed = {int(r['reduced_R']): r for r in rows}
        if len(indexed) != len(rows) or set(indexed) != set(range(1, len(self._source['reactions']) + 1)):
            raise ValueError('Manifest must identify every source reaction exactly once.')
        _validate_expected_factors(indexed, expected_factors)
        custom = {int(k): float(v) for k, v in (custom_factors or {}).items()}
        if not set(custom) <= set(indexed):
            raise ValueError('Custom factor identifies an unknown source reaction.')
        unbounded = set(unbounded_reactions)
        if any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) for k in unbounded):
            raise ValueError('Unbounded reaction numbers must be one-based integers.')
        if not unbounded <= set(indexed):
            raise ValueError('Unbounded selection identifies an unknown source reaction.')
        if unbounded & set(custom):
            raise ValueError('A reaction cannot have both an uncertainty factor and unbounded A.')
        self.plog_free = tuple(() if plog_free is None else plog_free)
        if (any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) for k in self.plog_free)
                or len(set(self.plog_free)) != len(self.plog_free)
                or any(k not in indexed or self._source['reactions'][k-1].get('type') != 'pressure-dependent-Arrhenius'
                       for k in self.plog_free)):
            raise ValueError('plog_free must contain distinct one-based PLOG reaction numbers, or None.')
        import cantera as ct
        self._source_gas = ct.Solution(yaml=yaml.safe_dump(self._source, sort_keys=False))
        self._plog_rates = {n: tuple((p, r.pre_exponential_factor, r.temperature_exponent, r.activation_energy)
            for p, r in self._source_gas.reaction(n-1).rate.rates) for n in self.plog_free}
        numbers, factors, coords, origins = [], [], [], []
        entry_indices, labels = [], []
        used_reference = set()
        retained = {s['name'] for s in self._source['species']}
        for number, source in enumerate(self._source['reactions'], 1):
            source_entries = _entries(source)
            row = indexed[number]
            if _identity(source) != _identity(row):
                raise ValueError(f'R{number}: manifest equation does not match source.')
            ref_number = row.get('ffcm2_R')
            if ref_number is None:
                if number not in custom and number not in unbounded:
                    if number in self.plog_free:
                        raise ValueError(f'R{number}: plog_free requires an uncertainty factor or explicit unbounded selection.')
                    self.audit.append({'reaction': number, 'status': 'fixed', 'reason': 'no exact nominal match'})
                    continue
                origin = source
                factor = None if number in unbounded else custom[number]
                label = ('source-centered A with no uncertainty factor' if factor is None
                         else 'explicit source-centered custom prior')
            else:
                if number in custom or number in unbounded:
                    raise ValueError('Custom or unbounded choices may only specify unmatched reactions.')
                if not isinstance(ref_number, int) or not 1 <= ref_number <= len(nominal['reactions']) or ref_number in used_reference:
                    raise ValueError('Invalid or repeated nominal reaction number.')
                used_reference.add(ref_number)
                origin = nominal['reactions'][ref_number - 1]
                if _identity(source) != _identity(origin) or not _near(_shape(source), _shape(origin)):
                    raise ValueError(f'R{number}: source and exact nominal rate shapes differ.')
                se, ne = source.get('efficiencies', {}), origin.get('efficiencies', {})
                if any(not _near(se.get(s, 1.0), ne.get(s, 1.0)) for s in retained):
                    raise ValueError(f'R{number}: third-body efficiencies differ.')
                declared = row.get('nominal_optimized_ffcm2')
                if declared is None or not _near(_entries(declared), _entries(origin)):
                    raise ValueError(f'R{number}: manifest nominal rate tuples differ from reference.')
                factor = float(row['uncertainty_factor'])
                label = ('run-configured factor centered on supplied nominal' if uncertainty_factors is not None
                         else 'published factor re-centered on supplied optimized nominal')
            if factor is not None and (not np.isfinite(factor) or factor <= 1):
                raise ValueError(f'R{number}: uncertainty factor must exceed one.')
            orig = tuple(float(e['A']) for e in _entries(origin))
            ratios = [math.log(float(e['A']) / a) for e, a in zip(source_entries, orig)]
            if number in self.plog_free:
                unit = self._source.get('units', {}).get('pressure', 'Pa')
                pressures = [ct.UnitSystem().convert_to(e['P'] if isinstance(e['P'], str)
                    else f"{e['P']} {unit}", 'Pa') for e in source_entries]
                groups = [(p, [i for i, q in enumerate(pressures) if q == p]) for p in sorted(set(pressures))]
            else:
                groups = [(None, list(range(len(orig))))]
            for pressure, indices in groups:
                values = [ratios[i] for i in indices]
                if not np.allclose(values, values[0], rtol=1e-12, atol=1e-12):
                    raise ValueError(f'R{number}: tied entries do not share one nominal multiplier.')
                numbers.append(number); factors.append(factor); coords.append(values[0])
                origins.append(tuple(orig[i] for i in indices)); entry_indices.append(tuple(indices))
                labels.append(dict(reaction=number, pressure_pa=pressure, entries=[i+1 for i in indices],
                    label=f'R{number}' if pressure is None else f'R{number} @ {pressure/1e5:g} bar'))
            self.audit.append({'reaction': number, 'nominal_reaction': ref_number, 'factor': factor, 'kind': 'unbounded' if factor is None else 'bounded', 'origin': label})
        self.reaction_numbers = tuple(numbers)
        self._origins = tuple(origins)
        self._entry_indices = tuple(entry_indices)
        self.coordinate_labels = labels
        self.source_coordinates = np.array(coords)
        widths = np.array([math.inf if factor is None else math.log(factor) for factor in factors])
        self.lower = -widths + self.strict_margin
        self.upper = widths - self.strict_margin
        if np.any(self.lower >= self.upper):
            raise ValueError('Strict margin consumes the uncertainty interval.')
        self.x0 = np.clip(self.source_coordinates, self.lower, self.upper)
        for i, number in enumerate(numbers):
            delta = float(self.x0[i] - self.source_coordinates[i])
            if abs(self.source_coordinates[i]) > widths[i] + 1e-10:
                raise ValueError(f'R{number}: source lies outside nominal uncertainty bounds.')
            if delta:
                self.audit.append({'reaction': number, 'status': 'start adjusted inward', 'source_coordinate': float(coords[i]), 'start_coordinate': float(self.x0[i]), 'delta': delta})
        for a in (self.source_coordinates, self.lower, self.upper, self.x0):
            a.flags.writeable = False

    def validate(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != self.x0.shape or not np.all(np.isfinite(x)) or np.any(x < self.lower) or np.any(x > self.upper):
            raise ValueError('Coordinates must be finite and inside the strict nominal uncertainty bounds.')
        # Infinite mathematical bounds do not license machine overflow or zero A.
        # Check all physical entries, including both falloff limits and PLOG nodes,
        # before any gas or YAML mutation. These are representation limits only.
        try:
            for number, origin, value, source in zip(self.reaction_numbers, self._origins, x, self.source_coordinates):
                nominal_multiplier = math.exp(float(value))
                source_multiplier = math.exp(float(value - source))
                physical = [a * nominal_multiplier for a in origin]
                if (not math.isfinite(source_multiplier) or source_multiplier <= 0 or
                    any(not math.isfinite(a) or a <= 0 for a in physical)):
                    raise ValueError(f'R{number}: A values and source multipliers must be positive finite floats.')
        except OverflowError as exc:
            raise ValueError('Candidate A values or rate multipliers overflow floating-point representation.') from exc
        return x

    def scheme(self, x):
        x = self.validate(x)
        result = deepcopy(self._source)
        for number, indices, origin, value in zip(self.reaction_numbers, self._entry_indices, self._origins, x):
            entries = _entries(result['reactions'][number - 1])
            for index, a in zip(indices, origin):
                entries[index]['A'] = a * math.exp(float(value))
        return result

    def physical_parameters(self, x):
        """Return all actual A values, including frozen entries, for change audits."""
        return np.array([float(entry['A']) for reaction in self.scheme(x)['reactions']
                         for entry in _entries(reaction)])

    def export(self, path, x):
        result = self.scheme(x)
        Path(path).write_text(yaml.safe_dump(result, sort_keys=False))
        return Path(path)

    def make_gas(self, x=None):
        import cantera as ct
        gas = ct.Solution(yaml=yaml.safe_dump(self._source, sort_keys=False))
        return self.apply(gas, self.x0 if x is None else x)

    def apply(self, gas, x):
        """Set absolute multipliers on a gas created from this space's source."""
        x = self.validate(x)
        if gas.n_reactions != len(self._source['reactions']):
            raise ValueError('Gas must be created from the original source mechanism.')
        # Always reconstruct from the immutable source: cached gases must not compound updates.
        import cantera as ct
        for number, value, source in zip(self.reaction_numbers, x, self.source_coordinates):
            if number not in self.plog_free:
                gas.set_multiplier(math.exp(float(value - source)), number - 1)
        for number in self.plog_free:
            reaction = gas.reaction(number-1)
            rates = []
            for pressure, a, b, ea in self._plog_rates[number]:
                j = next(i for i, label in enumerate(self.coordinate_labels)
                         if label['reaction'] == number and math.isclose(label['pressure_pa'], pressure, rel_tol=1e-12))
                rates.append((pressure, ct.ArrheniusRate(a * math.exp(float(x[j]-self.source_coordinates[j])), b, ea)))
            reaction.rate = ct.PlogRate(rates)
            gas.modify_reaction(number-1, reaction)
            gas.set_multiplier(1., number-1)
        return gas

    def linear_monotonicity(self, t_min, t_max, direction='either'):
        """Return branches C @ x >= b, exact over T for single-term PLOG nodes.

        Each branch selects one pressure direction per free PLOG reaction.
        Derivative probes may leave these sets; accepted updates may not.
        """
        from itertools import product
        import cantera as ct
        if direction not in (None, 'either', 'inc', 'dec'):
            raise ValueError('direction must be None, either, inc, or dec.')
        if direction is None or not self.plog_free:
            return [(np.empty((0, len(self.x0))), np.empty(0))]
        if not 0 < t_min <= t_max or not np.isfinite(t_max):
            raise ValueError('Invalid monotonicity temperature interval.')
        choices = []
        for number in self.plog_free:
            rates = sorted(self._source_gas.reaction(number-1).rate.rates, key=lambda pair: pair[0])
            if len({p for p, _ in rates}) != len(rates):
                raise ValueError(f'R{number}: monotonicity currently requires one Arrhenius expression per pressure.')
            inc, dec = [], []
            for (pa, a), (pb, b) in zip(rates, rates[1:]):
                ia, ib = [next(i for i, label in enumerate(self.coordinate_labels)
                    if label['reaction'] == number and math.isclose(label['pressure_pa'], p, rel_tol=1e-12)) for p in (pa, pb)]
                row = np.zeros(len(self.x0)); row[ib] = 1.; row[ia] = -1.
                da = math.log(b.pre_exponential_factor/a.pre_exponential_factor)
                db = b.temperature_exponent-a.temperature_exponent
                de = (b.activation_energy-a.activation_energy)/ct.gas_constant
                ts = [t_min, t_max]
                if db and t_min < -de/db < t_max: ts.append(-de/db)
                base = [da+db*math.log(t)-de/t - self.source_coordinates[ib]+self.source_coordinates[ia] for t in ts]
                inc.append((row, -min(base))); dec.append((-row, max(base)))
            choices.append([inc, dec] if direction == 'either' else [inc if direction == 'inc' else dec])
        branches = []
        for combination in product(*choices):
            rows = [row for group in combination for row in group]
            branches.append((np.array([a for a, b in rows]).reshape(-1, len(self.x0)), np.array([b for a, b in rows])))
        return branches


def parameter_change(before, after, parameters=('A',)):
    """Maximum physical relative change; zero-to-zero is 0, zero-to-nonzero inf."""
    if not parameters or not set(parameters) <= {'A', 'b', 'Ea'}:
        raise ValueError('Select A and/or b and/or Ea.')
    before, after = _load(before), _load(after)
    if len(before['reactions']) != len(after['reactions']):
        raise ValueError('Mechanisms must have matching reactions.')
    changes = []
    for old, new in zip(before['reactions'], after['reactions']):
        a, b = _entries(old), _entries(new)
        if _identity(old) != _identity(new) or len(a) != len(b):
            raise ValueError('Mechanisms must have aligned reaction rate entries.')
        for x, y in zip(a, b):
            for key in parameters:
                u, v = float(x[key]), float(y[key])
                changes.append(abs(v - u) / abs(u) if u else (0.0 if not v else math.inf))
    return max(changes, default=0.0)


def plog_monotonicity(model, t_min, t_max, direction='either'):
    """Exact pressure monotonicity over a closed temperature interval.

    Tests adjacent pressure-node log-rate differences at endpoints and their
    interior stationary temperature. One Arrhenius expression per pressure is
    required. For ``either``, each reaction may independently increase or
    decrease, but its direction must be consistent across temperatures.
    """
    if direction not in (None, 'either', 'inc', 'dec'):
        raise ValueError('direction must be None, either, inc, or dec.')
    t_min, t_max = float(t_min), float(t_max)
    if not 0 < t_min <= t_max or not np.isfinite(t_max):
        raise ValueError('Temperature interval must be finite, positive, and ordered.')
    import cantera as ct
    data = _load(model)
    gas = ct.Solution(yaml=yaml.safe_dump(data, sort_keys=False))
    reports = []
    reaction_reports = []
    increasing = decreasing = True
    for number, reaction in enumerate(gas.reactions(), 1):
        if not isinstance(reaction.rate, ct.PlogRate):
            continue
        reaction_increasing = reaction_decreasing = True
        entries = sorted(reaction.rate.rates, key=lambda pair: pair[0])
        if len({p for p, _ in entries}) != len(entries):
            raise ValueError(f'R{number}: multiple expressions at one PLOG pressure are unsupported.')
        for (p1, a), (p2, b) in zip(entries, entries[1:]):
            if a.pre_exponential_factor <= 0 or b.pre_exponential_factor <= 0:
                raise ValueError('PLOG monotonicity requires positive A values.')
            da = math.log(b.pre_exponential_factor / a.pre_exponential_factor)
            db = b.temperature_exponent - a.temperature_exponent
            de = (b.activation_energy - a.activation_energy) / ct.gas_constant
            temperatures = [t_min, t_max]
            if db and t_min < -de / db < t_max:
                temperatures.append(-de / db)
            differences = [da + db * math.log(t) - de / t for t in temperatures]
            lo, hi = min(differences), max(differences)
            reaction_increasing &= lo >= -1e-12
            reaction_decreasing &= hi <= 1e-12
            reports.append({'reaction': number, 'pressures_pa': [p1, p2], 'temperatures': temperatures, 'log_rate_ratios': differences, 'minimum': lo, 'maximum': hi})
        reaction_valid = True if direction is None else (reaction_increasing if direction == 'inc' else reaction_decreasing if direction == 'dec' else reaction_increasing or reaction_decreasing)
        reaction_reports.append({'reaction': number, 'increasing': bool(reaction_increasing), 'decreasing': bool(reaction_decreasing), 'valid': bool(reaction_valid)})
        increasing &= reaction_increasing
        decreasing &= reaction_decreasing
    valid = all(row['valid'] for row in reaction_reports)
    return {'valid': bool(valid), 'reactions': reaction_reports, 'increasing': bool(increasing), 'decreasing': bool(decreasing), 'direction': direction, 'pairs': reports, 'temperature_interval': [t_min, t_max]}
