"""User-facing configuration; all physical quantities use SI except rate Ea."""
from dataclasses import dataclass, field
import numpy as np


def positive(value, name):
    """Reject a scalar unless it is finite and strictly positive."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive.')


def _condition_arrays(conditions, keys, label=''):
    """Copy paired positive arrays and normalize their case weights once."""
    for key in keys:
        values = np.asarray(getattr(conditions, key), dtype=float)
        if (values.ndim != 1 or not len(values) or not np.all(np.isfinite(values))
                or np.any(values <= 0)):
            raise ValueError(f'{key} must be a nonempty positive finite array.')
        setattr(conditions, key, values.copy())
    if len({len(getattr(conditions, key)) for key in keys}) != 1:
        raise ValueError(f'{label}condition arrays must have equal lengths.'.capitalize())
    weights = (np.ones(len(conditions.P)) if conditions.weights is None
               else np.asarray(conditions.weights, dtype=float))
    if (weights.shape != conditions.P.shape or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.any(weights > 0)):
        raise ValueError(f'{label}case weights must be aligned, finite, nonnegative, and not all zero.'.capitalize())
    # Rescale before summing to avoid overflow for large finite weights.
    conditions.weights = weights / weights.max()
    conditions.weights /= conditions.weights.sum()


@dataclass
class Conditions:
    """Paired pressure (Pa), temperature (K), phi, and duration (s) arrays with case weights."""
    P: object
    T: object
    phi: object
    t_end: object
    weights: object = None

    def __post_init__(self):
        """Validate paired cases and normalize their weights to sum to one."""
        _condition_arrays(self, ('P', 'T', 'phi', 't_end'))


@dataclass(frozen=True)
class SquaredError:
    """Scaled additive residual with a relative objective weight; score squares the residual."""
    scale: float
    weight: float = 1.0

    def __post_init__(self):
        """Require a positive scale and a finite nonnegative objective weight."""
        positive(self.scale, 'scale')
        validate_weight(self.weight)

    def residual(self, actual, reference):
        # Squaring and objective weighting happen in TuningProblem.score.
        """Return (actual - reference) / scale before squaring."""
        return (actual - reference) / self.scale


@dataclass(frozen=True)
class RelativeSquaredError:
    """Relative residual (model-reference)/reference; score squares it."""
    weight: float = 1.0

    def __post_init__(self):
        """Require a finite nonnegative objective weight."""
        validate_weight(self.weight)

    def residual(self, actual, reference):
        """Normalize each case by its positive reference value."""
        return (actual - reference) / reference


@dataclass(frozen=True)
class LogSquaredError:
    """Natural-log ratio residual with a relative objective weight."""
    weight: float = 1.0

    def __post_init__(self):
        """Validate the objective weight."""
        validate_weight(self.weight)

    def residual(self, actual, reference):
        # Log ratio penalizes equal multiplicative errors symmetrically.
        """Return ln(actual/reference) for positive predictions and references."""
        return np.log(actual) - np.log(reference)


def validate_weight(weight):
    """Reject nonfinite or negative weights; zero disables a contribution."""
    if not np.isfinite(weight) or weight < 0:
        raise ValueError('weight must be finite and nonnegative.')


@dataclass(frozen=True)
class TrendPenalty:
    """Weight times the mean squared residual slope over normalized log pressure."""
    weight: float = 1.0
    objectives: tuple = ('laminar_flame_speed', 'flame_thickness')

    def __post_init__(self):
        """Require a nonnegative weight and distinct selected objective names."""
        validate_weight(self.weight)
        if (not isinstance(self.objectives, (tuple, list)) or not self.objectives
                or any(not isinstance(name, str) or not name for name in self.objectives)
                or len(set(self.objectives)) != len(self.objectives)):
            raise ValueError('Trend objectives must be a nonempty sequence of distinct names.')
        object.__setattr__(self, 'objectives', tuple(self.objectives))

    def coordinates(self, pressure, weights):
        """Center log-pressure coordinates spanning zero to one over weighted cases."""
        pressure, weights = np.asarray(pressure, dtype=float), np.asarray(weights, dtype=float)
        if (pressure.ndim != 1 or weights.shape != pressure.shape
                or not np.all(np.isfinite(pressure)) or np.any(pressure <= 0)
                or not np.all(np.isfinite(weights)) or np.any(weights < 0)
                or not np.any(weights > 0)):
            raise ValueError('Trend cases require positive pressures and valid aligned weights.')
        weights = weights / weights.max()
        weights /= weights.sum()
        log_p = np.log(pressure)
        active = log_p[weights > 0]
        span = np.ptp(active)
        if not np.isfinite(span) or span <= 0:
            raise ValueError('Trend penalty needs at least two distinct pressures with positive case weights.')
        z = (log_p - active.min()) / span
        x = z - np.sum(weights * z)
        variance = float(np.sum(weights * x**2))
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError('Trend coordinate has no weighted variation.')
        return x, weights, variance


@dataclass(frozen=True)
class RateRegularization:
    """Weight and scales for penalties on log A, b, and Ea departures from the template."""
    weight: float = 0.01
    scale_A: float = float(np.log(10))
    scale_b: float = 0.5
    scale_Ea: float = 2000.0

    def __post_init__(self):
        """Validate the penalty weight and positive normalization scales."""
        validate_weight(self.weight)
        for key in ('scale_A', 'scale_b', 'scale_Ea'):
            positive(getattr(self, key), key)


def interval(bounds, name, positive_only=False):
    """Validate a lower/upper pair and return it as a float array."""
    a = np.asarray(bounds, dtype=float)
    if a.shape != (2,) or not np.all(np.isfinite(a)) or a[0] >= a[1]:
        raise ValueError(f'{name}: expected finite lower < upper.')
    if positive_only and a[0] <= 0:
        raise ValueError(f'{name}: ratio bounds must be positive.')
    return a


@dataclass(frozen=True)
class RateParameters:
    # A and Ea use baseline multipliers; b uses absolute bounds.
    """Bounds for all rate entries: A and Ea multipliers, absolute b, and regularization."""
    A_ratio: tuple = (1e-5, 1e8)
    b: tuple = (0.5, 10)
    Ea_ratio: tuple = (1e-3, 1e5)
    regularization: RateRegularization = field(default_factory=RateRegularization)

    def __post_init__(self):
        """Validate rate bounds, requiring positive multiplier intervals."""
        interval(self.A_ratio, 'A_ratio', True)
        interval(self.b, 'b')
        interval(self.Ea_ratio, 'Ea_ratio', True)


@dataclass(frozen=True)
class OrderParameters:
    """Order multiplier bounds with optional penalties or hard sums of actual orders."""
    reaction: int
    ratios: dict
    sum_target: float | None = None
    sum_scale: float = 1.0
    sum_weight: float = 0.0
    sum_bounds: tuple | None = None
    sum_fixed: float | None = None

    def __post_init__(self):
        """Validate the reaction, multiplier bounds, and optional sum settings."""
        if isinstance(self.reaction, bool) or not isinstance(self.reaction, int) or self.reaction < 1:
            raise ValueError('Reaction numbers are one-based positive integers.')
        if not isinstance(self.ratios, dict) or not self.ratios:
            raise ValueError('Declare order multiplier bounds by species.')
        for species, bounds in self.ratios.items():
            if not isinstance(species, str) or not species:
                raise ValueError('Order species names must be nonempty strings.')
            interval(bounds, f'Order ratio for {species}')
        positive(self.sum_scale, 'sum_scale')
        validate_weight(self.sum_weight)
        for name in ('sum_target', 'sum_fixed'):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f'{name} must be finite.')
        if self.sum_weight > 0 and self.sum_target is None:
            raise ValueError('A positive sum_weight requires sum_target.')
        if self.sum_bounds is not None:
            interval(self.sum_bounds, 'sum_bounds')
        if self.sum_bounds is not None and self.sum_fixed is not None:
            raise ValueError('Use sum_bounds or sum_fixed, not both.')

    def coordinate_bounds(self, species):
        """Keep signed/zero-inclusive ranges linear; preserve positive log searches."""
        bounds = interval(self.ratios[species], f'Order ratio for {species}')
        return tuple(np.log10(bounds) if bounds[0] > 0 else bounds)

    def multiplier(self, species, coordinate):
        """Decode one search coordinate to a signed or positive order multiplier."""
        return float(10. ** coordinate if self.ratios[species][0] > 0 else coordinate)


@dataclass(frozen=True)
class Coefficient:
    """One independent stoichiometric coefficient identified by side.species and bounds."""
    location: str
    bounds: tuple
    open: bool = False

    def effective_bounds(self):
        """Return finite bounds, moving open endpoints inward by a numerical margin."""
        lo, hi = interval(self.bounds, self.location)
        if self.open:
            # Practical finite-precision approximation; exact endpoints excluded.
            margin = max((hi - lo) * 1e-12, np.spacing(max(abs(lo), abs(hi))))
            lo, hi = lo + margin, hi - margin
        return float(lo), float(hi)


@dataclass
class StoichiometricParameters:
    # Optimize independent coefficients; derive dependent ones by atom balance.
    """Declare fixed, independently tuned, and balance-derived coefficients for one reaction."""
    reaction: int  # One-based reaction number
    fixed: dict
    independent: dict
    dependent: list


@dataclass(frozen=True)
class SolverSettings:
    """ZND integration controls: maximum step (s), tolerances, and integration method."""
    max_step: float = 1e-6
    rel_tol: float = 1e-6
    abs_tol: float = 1e-9
    method: str = 'BDF'

    def __post_init__(self):
        """Validate positive step/tolerances and a supported integration method."""
        for key in ('max_step', 'rel_tol', 'abs_tol'):
            positive(getattr(self, key), key)
        if self.method not in ('BDF', 'Radau', 'LSODA', 'RK45', 'RK23', 'DOP853'):
            raise ValueError('Unsupported integration method.')


@dataclass
class FlameConditions:
    """Paired unburned pressure (Pa), temperature (K), and phi arrays for flames."""
    P: object
    T: object
    phi: object
    weights: object = None

    def __post_init__(self):
        """Validate paired flame cases and normalize case weights."""
        _condition_arrays(self, ('P', 'T', 'phi'), label='Flame ')


@dataclass(frozen=True)
class FlameSettings:
    """Free-flame domain width (m), transport model, grid refinement, and verbosity."""
    width: float = 0.01
    transport_model: str = 'multicomponent'
    refine_ratio: float = 3.0
    refine_slope: float = 0.06
    refine_curve: float = 0.12
    loglevel: int = 0

    def __post_init__(self):
        """Check domain size, refinement ranges, and solver verbosity."""
        positive(self.width, 'width')
        if not np.isfinite(self.refine_ratio) or self.refine_ratio < 2:
            raise ValueError('refine_ratio must be finite and at least 2.')
        for name in ('refine_slope', 'refine_curve'):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be between 0 and 1.')
        if not isinstance(self.transport_model, str) or not self.transport_model:
            raise ValueError('transport_model must be a nonempty string.')
        if isinstance(self.loglevel, bool) or not isinstance(self.loglevel, int) or not 0 <= self.loglevel <= 5:
            raise ValueError('loglevel must be an integer between 0 and 5.')


# Retain Conditions for existing ZND callers while exposing an explicit family name.
ZNDConditions = Conditions


@dataclass
class IgnitionConditions(Conditions):
    """Paired ignition pressure (Pa), initial temperature (K), phi, and end time (s)."""


@dataclass(frozen=True)
class IgnitionSettings:
    """Ignition reactor and marker; temperature works without radical species."""
    marker_species: str = 'temperature'
    reactor_type: str = 'constant-volume'
    rtol: float = 1e-9
    atol: float = 1e-15
    max_time_step: float | None = None

    def __post_init__(self):
        """Validate reactor type, tolerances, and optional maximum time step."""
        if self.reactor_type not in ('constant-volume', 'constant-pressure'):
            raise ValueError('Unsupported ignition reactor type.')
        if not isinstance(self.marker_species, str) or not self.marker_species:
            raise ValueError('marker_species must name a species or temperature.')
        positive(self.rtol, 'rtol')
        positive(self.atol, 'atol')
        if self.max_time_step is not None:
            positive(self.max_time_step, 'max_time_step')
