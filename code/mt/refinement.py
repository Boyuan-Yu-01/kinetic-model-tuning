"""One bounded Powell search starting from a saved rate-only best mechanism."""
from copy import deepcopy
from pathlib import Path
import shutil
from time import monotonic
import warnings

import numpy as np
from scipy.optimize import minimize
import yaml

from .history import HistoryWriter
from .mechanism import rates, parameter_dict
from .problem import CandidateObjective, TuningResult, validate_outputs
from .restart import atomic_yaml, load_problem
from .search import limit_threads


class RefinementTimeLimit(Exception):
    """Stop between evaluations when the optional elapsed-time budget is reached."""


def starting_vector(problem, scheme):
    """Encode the saved best rates relative to the original template and bounds."""
    if problem._balances or problem._orders:
        raise ValueError('This refinement entry point currently supports rate-only tuning.')
    original, candidate = deepcopy(problem.scheme), deepcopy(scheme)
    original_entries, candidate_entries = rates(original), rates(candidate)
    if len(original_entries) != len(candidate_entries):
        raise ValueError('The best mechanism has a different number of rate entries.')
    for entry in original_entries + candidate_entries:
        for key in ('A', 'b', 'Ea'):
            entry.pop(key)
    if original != candidate:
        raise ValueError('The best mechanism changes fixed data or reaction structure.')

    initial, base = parameter_dict(scheme), problem.original_rates
    a, a0 = np.asarray(initial['A']), np.asarray(base['A'])
    ea, ea0 = np.asarray(initial['Ea']), np.asarray(base['Ea'])
    if np.any(a <= 0) or np.any(a0 <= 0):
        raise ValueError('Rate-multiplier tuning requires positive A values.')
    ratio = np.ones(len(ea))
    nonzero = ea0 != 0
    if np.any(ea[~nonzero] != 0):
        raise ValueError('A nonzero Ea cannot be represented as a multiplier of zero.')
    ratio[nonzero] = ea[nonzero] / ea0[nonzero]
    if np.any(ratio <= 0):
        raise ValueError('The best Ea values have signs incompatible with their baseline.')
    bounds = problem.bounds
    ea_coordinates = np.log10(ratio)
    # Zero baseline Ea is unchanged by every multiplier; use an allowed neutral coordinate.
    ea_coordinates[~nonzero] = np.clip(0., bounds[-len(ea):, 0][~nonzero],
                                         bounds[-len(ea):, 1][~nonzero])
    vector = np.r_[np.log10(a / a0), initial['b'], ea_coordinates]
    if not np.all(np.isfinite(vector)):
        raise ValueError('The best mechanism contains unrepresentable rate parameters.')
    tolerance = 1e-12 * np.maximum(1., bounds[:, 1] - bounds[:, 0])
    if np.any(vector < bounds[:, 0] - tolerance) or np.any(vector > bounds[:, 1] + tolerance):
        raise ValueError('The best mechanism is outside the saved parameter bounds.')
    return np.clip(vector, bounds[:, 0], bounds[:, 1])


def refine_best(source_dir, output_dir, *, max_evaluations=1000,
                max_seconds=None, progress_every=10):
    """Refine one saved best; preserve its original baseline, reference, bounds, and loss.

    This runs sequentially. Output goes to a new directory, with every improvement
    saved immediately. max_seconds is checked between evaluations; it cannot
    interrupt an in-progress solver call. The starting evaluation counts toward
    max_evaluations. No random screening is performed.
    """
    for name, value in (('max_evaluations', max_evaluations), ('progress_every', progress_every)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if max_seconds is not None and (isinstance(max_seconds, bool)
                                    or not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('max_seconds must be positive and finite, or None.')
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    problem, _ = load_problem(source)
    scheme = yaml.safe_load((source / 'best_model.yaml').read_text())
    vector = starting_vector(problem, scheme)
    reference = yaml.safe_load((source / 'reference_results.yaml').read_text())
    for family, cases, keys in problem._groups():
        validate_outputs(reference[family], len(cases.P), keys)
    limit_threads()
    output.mkdir(parents=True, exist_ok=False)
    # Preserve the original reference/baseline, not the tuned model as a new baseline.
    for name in ('settings.yaml', 'original_model.yaml', 'reference_model.yaml',
                 'mechanism_hashes.yaml', 'reference_results.yaml'):
        shutil.copy2(source / name, output / name)
    shutil.copy2(source / 'best_model.yaml', output / 'starting_model.yaml')
    atomic_yaml(output / 'refinement_settings.yaml', dict(
        method='Powell', source_dir=str(source), max_evaluations=max_evaluations,
        max_seconds=max_seconds, progress_every=progress_every, workers=1,
        bounds=problem.bounds.tolist(), starting_vector=vector.tolist(),
        note='Original source settings define the objective; no random sampling is performed.'))
    lower = problem.bounds[:, 0]
    width = problem.bounds[:, 1] - lower
    unit_start = (vector - lower) / width
    objective = CandidateObjective(problem, reference)
    best, initial = None, None
    nfev = failed = 0
    begin = monotonic()
    result = None
    termination = 'running'
    with HistoryWriter(output) as history:
        def evaluate(unit):
            """Score one bounded point and preserve improvements before the next solver call."""
            nonlocal best, initial, nfev, failed
            if nfev and max_seconds is not None and monotonic() - begin >= max_seconds:
                raise RefinementTimeLimit()
            physical = np.clip(lower + np.asarray(unit) * width,
                               problem.bounds[:, 0], problem.bounds[:, 1])
            record = objective(physical)
            record.update(stage='local', sample_index=0)
            nfev += 1
            finite = np.isfinite(record['loss'])
            failed += int(not finite)
            history.record(record, nfev)
            if initial is None:
                initial = {'loss': record['loss']}
                atomic_yaml(output / 'starting_loss.yaml', {
                    k: v for k, v in record.items()
                    if k not in ('vector', 'results', 'diagnostics', 'stage', 'sample_index')})
                if not finite:
                    raise RuntimeError('The saved best failed re-evaluation: ' + record.get('error', 'unknown error'))
                atomic_yaml(output / 'starting_results.yaml', record['results'])
                print(f'Starting loss: {record["loss"]:.8g}', flush=True)
            if finite and (best is None or record['loss'] < best['loss']):
                best = deepcopy({k: v for k, v in record.items() if k not in ('diagnostics', 'error')})
                # Save the complete record before exporting its derived YAML files.
                atomic_yaml(output / 'best_record.yaml', best)
                parameters = problem.decode(best['vector'])
                atomic_yaml(output / 'best_model.yaml', problem.mechanism(parameters))
                atomic_yaml(output / 'best_parameters.yaml', parameters)
                atomic_yaml(output / 'best_results.yaml', best['results'])
                atomic_yaml(output / 'best_loss.yaml', {
                    k: v for k, v in best.items()
                    if k not in ('vector', 'results', 'stage', 'sample_index')})
            if nfev % progress_every == 0:
                print(f'Powell {nfev} | best {best["loss"]:.8g} | failed {failed}', flush=True)
            return record['loss']

        try:
            # SciPy evaluates the starting point first; retain it even if later steps fail.
            with warnings.catch_warnings(record=True) as caught:
                result = minimize(evaluate, unit_start, method='Powell',
                    bounds=[(0., 1.)] * len(vector), options={'maxfev': max_evaluations})
            termination = str(result.message)
            if caught:
                atomic_yaml(output / 'optimizer_warnings.yaml', sorted({str(w.message) for w in caught}))
        except RefinementTimeLimit:
            termination = 'Elapsed-time budget reached between evaluations.'
        except BaseException as error:
            termination = f'{type(error).__name__}: {error}'
            raise
        finally:
            summary = dict(method='Powell', source_dir=str(source),
                nfev=nfev, failed=failed, elapsed_seconds=monotonic()-begin,
                initial_loss=None if initial is None else float(initial['loss']),
                best_loss=None if best is None else float(best['loss']),
                improved=bool(best is not None and best['loss'] < initial['loss']),
                optimizer_success=bool(result is not None and result.success),
                iterations=None if result is None else int(result.nit),
                termination=termination)
            atomic_yaml(output / 'refinement_summary.yaml', summary)
    print(f'Finished | best {best["loss"]:.8g} | results: {output}', flush=True)
    return TuningResult(problem, best, reference, output, summary)
