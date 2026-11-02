"""Pure objective aggregation for independently sampled combustion branches.

Use ``TuningProblem`` for existing simulations and optimization. These helpers
also accept externally measured reference rows and do not launch any solver.
They deliberately leave regularization and trend penalties to the caller.
"""
import numpy as np

from .config import SquaredError, RelativeSquaredError, LogSquaredError
from .problem import ZND_KEYS, FLAME_KEYS, IGNITION_KEYS, validate_outputs

BRANCH_OBJECTIVES = {'znd': ZND_KEYS, 'flame': FLAME_KEYS, 'ignition': IGNITION_KEYS}


def _weights(values, label):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.all(np.isfinite(values))
            or np.any(values < 0) or not np.any(values > 0)):
        raise ValueError(f'{label} must be finite, nonnegative, and not all zero.')
    values = values / values.max()
    return values / values.sum()


def normalized_residuals(results, references, *, conditions, objectives, branch_weights=None):
    """Return a residual vector whose squared norm is the normalized fit loss.

    ``results`` and ``references`` map ``znd``, ``flame``, or ``ignition`` to
    lists of output dictionaries, as returned by TuningProblem.evaluate.
    ``conditions`` maps those names to their own condition objects; branch case
    counts may differ. ``objectives`` maps supported output names to existing
    SquaredError, RelativeSquaredError, or LogSquaredError instances.

    Branch weights default to equal weights among branches with active
    objectives. An explicit mapping must name exactly those branches; zero
    disables a branch. Within each branch, objective weights and case weights
    independently sum to one. Each entry is the objective residual multiplied
    by sqrt(branch_weight * objective_weight * case_weight). Ordering is znd,
    flame, ignition, then the declared branch objective order, then case order.
    This hierarchical weighting differs from TuningProblem.score's global
    objective-weight normalization. Disabled objectives need no output values.
    """
    allowed = {key for keys in BRANCH_OBJECTIVES.values() for key in keys}
    if not objectives or not set(objectives) <= allowed:
        raise ValueError('Specify supported combustion objectives.')
    if any(not isinstance(o, (SquaredError, RelativeSquaredError, LogSquaredError))
           for o in objectives.values()):
        raise TypeError('Use the existing squared-error objective classes.')
    groups = {branch: [key for key in keys if key in objectives and objectives[key].weight > 0]
              for branch, keys in BRANCH_OBJECTIVES.items()}
    groups = {branch: keys for branch, keys in groups.items() if keys}
    if not groups:
        raise ValueError('At least one objective must have positive weight.')
    if branch_weights is None:
        branch_weights = dict.fromkeys(groups, 1.)
    if set(branch_weights) != set(groups):
        raise ValueError('Branch weights must name exactly the branches with active objectives.')
    weights = _weights([branch_weights[b] for b in groups], 'Branch weights')
    parts = []
    for (branch, keys), branch_weight in zip(groups.items(), weights):
        if branch_weight == 0:
            continue
        if branch not in conditions or branch not in results or branch not in references:
            raise ValueError(f'Missing {branch} conditions, results, or references.')
        cases = conditions[branch]
        count = len(cases.P)
        case_weights = _weights(cases.weights, f'{branch} case weights')
        if case_weights.shape != (count,):
            raise ValueError(f'{branch} case weights must align with its cases.')
        actual = validate_outputs(results[branch], count, keys)
        reference = validate_outputs(references[branch], count, keys)
        objective_weights = _weights([objectives[key].weight for key in keys], 'Objective weights')
        for column, (key, objective_weight) in enumerate(zip(keys, objective_weights)):
            residual = objectives[key].residual(actual[:, column], reference[:, column])
            weighted = residual * np.sqrt(branch_weight * objective_weight * case_weights)
            if not np.all(np.isfinite(weighted)):
                raise ValueError(f'Nonfinite residual for {key}.')
            parts.append(weighted)
    return np.concatenate(parts)


def objective_score(results, references, *, conditions, objectives, branch_weights=None):
    """Return the squared norm of :func:`normalized_residuals` (fit only)."""
    residuals = normalized_residuals(results, references, conditions=conditions,
                                     objectives=objectives, branch_weights=branch_weights)
    score = float(residuals @ residuals)
    if not np.isfinite(score):
        raise ValueError('Objective score overflowed.')
    return score
