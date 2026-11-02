"""Feasible search coordinates for sums of selected actual reaction orders."""
import numpy as np


class OrderSpace:
    """Map independent coordinates to orders while satisfying optional sum limits."""

    def __init__(self, spec, original):
        """Convert multiplier bounds to actual orders and reject impossible sums."""
        self.spec = spec
        self.names = list(spec.ratios)
        self.original = {name: original[name] for name in self.names}
        self.limits = np.array([sorted(original[name] * np.asarray(spec.ratios[name], dtype=float))
                                for name in self.names])
        self.sum_limits = None
        self.fixed = False
        if spec.sum_fixed is not None or spec.sum_bounds is not None:
            requested = (spec.sum_fixed, spec.sum_fixed) if spec.sum_fixed is not None else spec.sum_bounds
            lo = max(float(self.limits[:, 0].sum()), requested[0])
            hi = min(float(self.limits[:, 1].sum()), requested[1])
            if lo > hi:
                if not np.isclose(lo, hi, rtol=1e-12, atol=1e-12):
                    raise ValueError(f'Reaction {spec.reaction}: order sum is incompatible with individual bounds.')
                hi = lo
            self.sum_limits = (lo, hi)
            self.fixed = lo == hi

    @property
    def bounds(self):
        """Use unit coordinates for hard sums; preserve legacy coordinates otherwise."""
        if self.sum_limits is not None:
            return [(0., 1.)] * (len(self.names) - int(self.fixed))
        return [self.spec.coordinate_bounds(name) for name in self.names]

    def decode(self, coordinates):
        """Choose each order from the interval leaving a feasible remainder."""
        if self.sum_limits is None:
            return {name: float(self.original[name] * self.spec.multiplier(name, value))
                    for name, value in zip(self.names, coordinates)}
        values = {}
        prefix = 0.
        for i, name in enumerate(self.names):
            remaining = self.limits[i + 1:]
            lo = max(self.limits[i, 0], self.sum_limits[0] - prefix - remaining[:, 1].sum())
            hi = min(self.limits[i, 1], self.sum_limits[1] - prefix - remaining[:, 0].sum())
            # Roundoff can invert an interval that mathematically has zero width.
            if lo > hi and not np.isclose(lo, hi, rtol=1e-12, atol=1e-12):
                raise ValueError('No feasible remaining reaction order.')
            hi = max(lo, hi)
            unit = 0. if self.fixed and i == len(self.names) - 1 else coordinates[i]
            value = float(np.clip(lo + unit * (hi - lo), *self.limits[i]))
            values[name] = value
            prefix += value
        self.validate(values)
        return values

    def validate(self, values):
        """Check externally supplied orders as well as decoded candidates."""
        if set(values) != set(self.names):
            raise ValueError(f'Reaction {self.spec.reaction}: order species do not match configuration.')
        actual = np.asarray([values[name] for name in self.names], dtype=float)
        tolerance = 1e-10 * np.maximum(1., np.max(np.abs(self.limits), axis=1))
        if (not np.all(np.isfinite(actual)) or np.any(actual < self.limits[:, 0] - tolerance)
                or np.any(actual > self.limits[:, 1] + tolerance)):
            raise ValueError('Reaction orders are outside individual bounds.')
        total = float(actual.sum())
        if self.sum_limits is not None:
            tolerance = 1e-10 * max(1., float(np.abs(actual).sum()), *map(abs, self.sum_limits))
            if not self.sum_limits[0] - tolerance <= total <= self.sum_limits[1] + tolerance:
                raise ValueError('Reaction orders violate the configured sum constraint.')
        return total

    def penalty(self, values):
        """Return the actual order sum and its weighted soft penalty."""
        total = self.validate(values)
        spec = self.spec
        penalty = 0. if spec.sum_weight == 0 else spec.sum_weight * ((total - spec.sum_target) / spec.sum_scale)**2
        return total, float(penalty)
