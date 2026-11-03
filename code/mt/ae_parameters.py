"""Joint rate-magnitude and activation-energy coordinates with fixed b."""
from copy import deepcopy
from collections.abc import Mapping
from pathlib import Path
import math
import numpy as np
import cantera as ct
import yaml

from .mechanism_yaml import dump_mechanism
from .uncertainty import RateUncertaintySpace, _entries


def _plain(value):
    """Detach Cantera AnyMap rate data into ordinary Python containers."""
    if isinstance(value, Mapping): return {k:_plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_plain(v) for v in value]
    return value


class AEParameters:
    """Independent PLOG magnitudes with optional pressure-node Ea shifts.

    Magnitude coordinates are log rate ratios at a fixed pivot temperature.
    Physical A bounds and full-rate uncertainty envelopes are coupled linear
    constraints. Derivative probes may leave those sets, but accepted models may not.
    """
    physical_fields = ('A', 'Ea')

    def __init__(self, *args, a_ratio=(.5, 2.), ea_delta=60000., pivot_temperature=1800., plog_shape_free=False, **kwargs):
        self.base = RateUncertaintySpace(*args, **kwargs)
        units = self.base._source.get("units", {})
        self.energy_unit = units.get("activation-energy", "J/kmol")
        self.source_units = ct.UnitSystem(units)
        # The editable limit always uses the mechanism's activation-energy units.
        self.energy_yaml = float(ea_delta)
        self.ea_scale = self.source_units.convert_activation_energy_to(self.energy_yaml, "kJ/mol")
        self.energy_si = self.source_units.convert_activation_energy_to(self.energy_yaml, "J/kmol")
        self.pivot = float(pivot_temperature)
        ratio = np.asarray(a_ratio, dtype=float)
        if (ratio.shape != (2,) or not np.isfinite(ratio).all() or not 0 < ratio[0] < 1 < ratio[1]
                or not np.isfinite(self.ea_scale) or self.ea_scale <= 0
                or not np.isfinite(self.pivot) or self.pivot <= 0):
            raise ValueError('Require positive A ratio bounds bracketing 1, ea_delta > 0, and pivot_temperature > 0.')
        self._source_si = [_plain(r.input_data) for r in self.base._source_gas.reactions()]
        self.na = len(self.base.x0)
        self.numbers = list(dict.fromkeys(self.base.reaction_numbers))
        if not isinstance(plog_shape_free, (bool, np.bool_)):
            raise ValueError('plog_shape_free must be boolean.')
        self.plog_shape_free = bool(plog_shape_free)
        self.shape_keys = list(dict.fromkeys(
            (n, label['pressure_pa']) if self.plog_shape_free and n in self.base.plog_free else n
            for n, label in zip(self.base.reaction_numbers, self.base.coordinate_labels)))
        self.shape_for_a = [(n, label['pressure_pa']) if self.plog_shape_free and n in self.base.plog_free else n
                            for n, label in zip(self.base.reaction_numbers, self.base.coordinate_labels)]
        self.ne = len(self.shape_keys)
        self.ea_index = {key:self.na+i for i,key in enumerate(self.shape_keys)}
        self.reaction_numbers = self.base.reaction_numbers + tuple(k[0] if isinstance(k, tuple) else k for k in self.shape_keys)
        self.plog_free = self.base.plog_free
        self.audit = self.base.audit
        self.source_coordinates = np.r_[self.base.source_coordinates, np.zeros(self.ne)]
        self.x0 = np.r_[self.base.x0, np.zeros(self.ne)]
        self.a_low = np.maximum(self.base.lower, self.base.source_coordinates+np.log(ratio[0]))
        self.a_high = np.minimum(self.base.upper, self.base.source_coordinates+np.log(ratio[1]))
        shift = self.energy_si/(ct.gas_constant*self.pivot)
        self.lower = np.r_[self.a_low-shift, -np.ones(self.ne)]
        self.upper = np.r_[self.a_high+shift, np.ones(self.ne)]
        self.coordinate_labels = [dict(row, parameter='log_k_pivot',
            label=row['label']+' ln k(Tref)') for row in self.base.coordinate_labels]
        self.coordinate_labels += [dict(reaction=k[0] if isinstance(k,tuple) else k, pressure_pa=k[1] if isinstance(k,tuple) else None, parameter='scaled_ea',
            label=f'R{k} ΔEa/{self.ea_scale:g} kJ/mol at fixed k(Tref)', scale=self.ea_scale) for k in self.shape_keys]
        # Each row maps pivot coordinates to the original nominal log-A coordinate.
        self.a_map = np.zeros((self.na, len(self.x0)))
        for i,n in enumerate(self.base.reaction_numbers):
            self.a_map[i,i] = 1.
            self.a_map[i,self.ea_index[self.shape_for_a[i]]] = self.energy_si/(ct.gas_constant*self.pivot)
        self.parameter_scales = np.array([v for r in self.base._source['reactions']
            for _ in _entries(r) for v in (np.nan,self.energy_yaml)])

    def validate(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != self.x0.shape or not np.isfinite(x).all() or np.any(x < self.lower) or np.any(x > self.upper):
            raise ValueError('Joint coordinates must be finite and within their coordinate bounds.')
        for a, origin in zip(self.a_map@x, self.base._origins):
            values = np.asarray(origin)*math.exp(float(a))
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError('A must remain positive and finite.')
        return x

    def scheme(self, x):
        x = self.validate(x)
        result = deepcopy(self.base._source)
        for n, indices, origin, a in zip(self.base.reaction_numbers, self.base._entry_indices,
                                         self.base._origins, self.a_map@x):
            entries = _entries(result['reactions'][n-1])
            for i,value in zip(indices,origin): entries[i]['A'] = value*math.exp(float(a))
        for i, n in enumerate(self.base.reaction_numbers):
            entries = _entries(result['reactions'][n-1])
            for j in self.base._entry_indices[i]:
                e = entries[j]
                e['Ea'] = self.source_units.convert_activation_energy_to(e['Ea'], self.energy_unit)+float(self.energy_yaml*x[self.ea_index[self.shape_for_a[i]]])
        return result

    def apply(self, gas, x):
        # Cantera input_data uses SI values; from_dict does not inherit the YAML
        # file's unit system. Modify immutable SI rates, never raw YAML numbers.
        x = self.validate(x)
        native = self.a_map@x
        gas.set_multiplier(1.)
        for n in self.numbers:
            entry = deepcopy(self._source_si[n-1])
            for rate in _entries(entry):
                if n in self.plog_free:
                    i = next(i for i, label in enumerate(self.base.coordinate_labels)
                        if label['reaction']==n and math.isclose(label['pressure_pa'],rate['P'],rel_tol=1e-12))
                else:
                    i = self.base.reaction_numbers.index(n)
                rate['A'] *= math.exp(float(native[i]-self.base.source_coordinates[i]))
                rate['Ea'] += float(self.energy_si*x[self.ea_index[self.shape_for_a[i]]])
            reaction = ct.Reaction.from_dict(entry, gas)
            gas.modify_reaction(n-1, reaction)
        return gas

    def make_gas(self, x=None):
        return ct.Solution(yaml=yaml.safe_dump(self.scheme(self.x0 if x is None else x), sort_keys=False))

    def physical_parameters(self, x):
        return np.array([(self.source_units.convert_activation_energy_to(e[k], self.energy_unit) if k == 'Ea' else float(e[k])) for r in self.scheme(x)['reactions'] for e in _entries(r) for k in self.physical_fields])

    def export(self, path, x):
        Path(path).write_text(dump_mechanism(self.scheme(x)))
        return Path(path)

    def linear_monotonicity(self, t_min, t_max, direction='either'):
        """Full-T rate envelopes, A bounds, and pressure proposal constraints.

        Shared Ea shifts cancel in adjacent PLOG node comparisons. Independent
        shape uses sampled pressure cuts; exact acceptance must use
        uncertainty.plog_monotonicity. For matched
        rates b is unchanged and delta log(k) is affine in 1/T, so endpoint
        checks suffice. Shared falloff shifts also leave reduced pressure fixed.
        """
        branches = self._pressure_branches(t_min,t_max,direction)
        rows = [*self.a_map, *(-self.a_map)]
        bounds = [*self.a_low, *(-self.a_high)]
        for i,n in enumerate(self.base.reaction_numbers):
            if not np.isfinite(self.base.lower[i]): continue
            for t in (t_min,t_max):
                row = self.a_map[i].copy();row[self.ea_index[self.shape_for_a[i]]] -= self.energy_si/(ct.gas_constant*t)
                rows.extend((row,-row));bounds.extend((self.base.lower[i],-self.base.upper[i]))
        c,d = np.array(rows),np.array(bounds)
        return [(np.vstack([old,c]),np.r_[rhs,d]) for old,rhs in branches]

    def _rate_row(self, i, n, t):
        row = self.a_map[i].copy()
        row[self.ea_index[self.shape_for_a[i]]] -= self.energy_si/(ct.gas_constant*t)
        return row

    def _pressure_branches(self, t_min, t_max, direction):
        if not self.plog_shape_free:
            return [(c@self.a_map,d) for c,d in self.base.linear_monotonicity(t_min,t_max,direction)]
        from itertools import product
        if direction not in (None, 'either', 'inc', 'dec'):
            raise ValueError('Invalid pressure monotonicity direction.')
        if not 0 < t_min <= t_max or not np.isfinite(t_max):
            raise ValueError('Invalid temperature interval.')
        if direction is None or not self.plog_free:
            return [(np.empty((0,len(self.x0))),np.empty(0))]
        choices=[]
        current=self.make_gas(self.x0)
        for n in self.plog_free:
            source=sorted(self.base._source_gas.reaction(n-1).rate.rates)
            rates=sorted(current.reaction(n-1).rate.rates)
            if len({p for p,_ in source}) != len(source):
                raise ValueError('Monotonicity requires one expression per pressure.')
            inc=[];dec=[]
            for j, ((pa,a),(pb,b)) in enumerate(zip(source,source[1:])):
                ia,ib=[next(i for i,l in enumerate(self.base.coordinate_labels)
                    if l['reaction']==n and math.isclose(l['pressure_pa'],p,rel_tol=1e-12)) for p in (pa,pb)]
                ca,cb=rates[j][1],rates[j+1][1]
                db=cb.temperature_exponent-ca.temperature_exponent
                de=(cb.activation_energy-ca.activation_energy)/ct.gas_constant
                ts=list(np.geomspace(t_min,t_max,33))
                if db and t_min < -de/db < t_max: ts.append(-de/db)
                for t in ts:
                    row=self._rate_row(ib,n,t)-self._rate_row(ia,n,t)
                    baseline=(math.log(b.pre_exponential_factor/a.pre_exponential_factor)
                        +(b.temperature_exponent-a.temperature_exponent)*math.log(t)
                        -(b.activation_energy-a.activation_energy)/(ct.gas_constant*t)
                        -self.base.source_coordinates[ib]+self.base.source_coordinates[ia])
                    inc.append((row,-baseline));dec.append((-row,baseline))
            choices.append([inc,dec] if direction=='either' else [inc if direction=='inc' else dec])
        return [(np.array([r for group in combo for r,_ in group]).reshape(-1,len(self.x0)),
                 np.array([b for group in combo for _,b in group])) for combo in product(*choices)]

    def rate_envelope(self, x, t_min, t_max):
        x = self.validate(x)
        a = self.a_map@x
        valid = bool(np.all(a>=self.a_low-1e-12) and np.all(a<=self.a_high+1e-12))
        for i,n in enumerate(self.base.reaction_numbers):
            for t in (t_min,t_max):
                v = a[i]-self.energy_si*x[self.ea_index[self.shape_for_a[i]]]/(ct.gas_constant*t)
                valid &= bool(self.base.lower[i]-1e-12 <= v <= self.base.upper[i]+1e-12)
        return dict(valid=valid,temperature_interval=[t_min,t_max])

    def reported_multipliers(self, x):
        return [*np.exp(self.a_map@x-self.base.source_coordinates), *([None]*self.ne)]
