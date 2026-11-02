"""Restore optimization runs from saved configuration and durable progress."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import socket

import numpy as np
import yaml

from .history import error_category, json_value


def atomic_yaml(path, value):
    """Replace one YAML artifact only after its complete contents reach disk."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        yaml.safe_dump(value, stream, sort_keys=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def run_lock(output):
    """Prevent two coordinators from modifying the same run on macOS or Linux."""
    import fcntl
    path = Path(output) / 'restart.lock'
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process is using this run: {output}') from error
        stream.seek(0)
        stream.truncate()
        stream.write(f'{socket.gethostname()} pid={os.getpid()}\n')
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def scientific_settings(settings):
    """Exclude file locations and scheduling choices from restart compatibility."""
    result = dict(settings)
    for key in ('model', 'reference', 'backend', 'history', 'progress_every'):
        result.pop(key, None)
    result['search'] = {k: v for k, v in result['search'].items()
                        if k not in ('workers', 'backend', 'sample_batch_size',
                                     'max_in_memory_valid', 'worker_batch_size')}
    # Dict equality ignores order, but these ordered mappings define vector coordinates.
    result['coordinate_layout'] = {
        'stoichiometry': [[item['reaction'], list(item['independent'])]
                          for item in settings.get('stoichiometry', [])],
        'orders': [[item['reaction'], list(item['ratios'])]
                   for item in settings.get('orders', [])],
        'objectives': list(settings['objectives']),
    }
    return json_value(result)


def check_compatible(saved, current):
    """Reject changes that would mix different scientific problems or searches."""
    if scientific_settings(saved) != scientific_settings(current):
        raise ValueError('Restart settings differ from the saved problem/search. '
                         'Use the generated restart.py, or start a new output directory.')


def verify_mechanisms(output, problem=None):
    """Verify saved mechanism snapshots and optionally the configured problem."""
    output = Path(output)
    hashes = yaml.safe_load((output / 'mechanism_hashes.yaml').read_text())
    for name, attr in (('original_model.yaml', 'scheme'),
                       ('reference_model.yaml', 'reference_scheme')):
        path = output / name
        contents = path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != hashes[name]:
            raise ValueError(f'Saved mechanism was modified: {name}')
        if problem is not None and yaml.safe_load(contents) != getattr(problem, attr):
            raise ValueError(f'Configured mechanism differs from {name}; use a new run.')


def load_problem(output):
    """Rebuild every solver family and tuning option from portable YAML snapshots."""
    from .problem import TuningProblem
    from .config import (ZNDConditions, FlameConditions, IgnitionConditions,
        SolverSettings, FlameSettings, IgnitionSettings, RateParameters,
        RateRegularization, OrderParameters, StoichiometricParameters, Coefficient,
        TrendPenalty, SquaredError, RelativeSquaredError, LogSquaredError)
    output = Path(output).resolve()
    verify_mechanisms(output)
    saved = yaml.safe_load((output / 'settings.yaml').read_text())
    def cases(name, cls):
        """Preserve already-normalized saved weights without a second rounding."""
        values = saved.get(name)
        if values is None:
            return None
        result = cls(**values)
        result.weights = np.asarray(values['weights'], dtype=float).copy()
        return result
    problem = TuningProblem(
        output / 'original_model.yaml', output / 'reference_model.yaml',
        saved['fuel'], saved['oxidizer'],
        znd_conditions=cases('znd_conditions', ZNDConditions),
        flame_conditions=cases('flame_conditions', FlameConditions),
        ignition_conditions=cases('ignition_conditions', IgnitionConditions),
        znd_solver=SolverSettings(**saved['znd_solver']),
        flame_solver=FlameSettings(**saved['flame_solver']),
        ignition_solver=IgnitionSettings(**saved['ignition_solver']))
    rates = dict(saved['rates'])
    rates['regularization'] = RateRegularization(**rates['regularization'])
    stoichiometry = []
    for raw in saved.get('stoichiometry', []):
        spec = dict(raw)
        spec['independent'] = {k: Coefficient(**v) for k, v in spec['independent'].items()}
        stoichiometry.append(StoichiometricParameters(**spec))
    problem.parameters(rates=RateParameters(**rates), stoichiometry=stoichiometry,
                       orders=[OrderParameters(**v) for v in saved.get('orders', [])])
    classes = {cls.__name__: cls for cls in (SquaredError, RelativeSquaredError, LogSquaredError)}
    objectives = {}
    for name, raw in saved['objectives'].items():
        spec = dict(raw)
        objectives[name] = classes[spec.pop('type')](**spec)
    problem.objectives(**objectives)
    problem.trend(TrendPenalty(**saved['trend']) if saved.get('trend') is not None else None)
    return problem, saved


def write_restart_script(output):
    """Create a plain Python entry point for this run without changing its settings."""
    output = Path(output).resolve()
    root = str(Path(__file__).resolve().parent.parent)
    source = (
        '"""Resume this run; optional command-line argument: --workers 8."""\n'
        'from pathlib import Path\n'
        'import sys\n\n'
        '# Use the codebase that created this file, or mt from the active environment.\n'
        f'codebase = Path({root!r})\n'
        'if codebase.is_dir():\n'
        '    sys.path.insert(0, str(codebase))\n'
        'from mt.restart import main\n\n'
        'main(Path(__file__).resolve().parent)\n'
    )
    path = output / 'restart.py'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(source)
    os.replace(temporary, path)
    return path


def import_legacy(output, store, bounds, search_settings):
    """Stream pre-checkpoint JSONL into SQLite; never retain all sample IDs or vectors."""
    output = Path(output)
    path = output / 'history.jsonl'
    if not path.exists():
        return None
    if search_settings['seed'] is None:
        raise ValueError('Legacy runs with seed=None cannot recover their original RNG stream.')
    nfev = failed = offset = 0
    failures, best = {}, None
    lower, width = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    print('Indexing existing history for restart (streaming from disk)...', flush=True)
    with path.open('rb') as stream:
        while line := stream.readline():
            # An abrupt stop may leave an incomplete final line. Do not accept its contents.
            if not line.endswith(b'\n'):
                break
            record = json.loads(line)
            vector = np.asarray(record['vector'], dtype=float)
            if vector.shape != lower.shape or not np.all(np.isfinite(vector)):
                raise ValueError('Invalid vector in existing history.')
            loss = float('inf') if record['loss'] is None else float(record['loss'])
            index = int(record['sample_index'])
            if not 0 <= index < search_settings['samples']:
                raise ValueError('Existing history has a sample index outside the saved search.')
            if record['stage'] == 'sampling':
                unit = None
                if math.isfinite(loss):
                    # Recover the original normalized draw, not a rounded inverse of its vector.
                    rng = np.random.Generator(np.random.PCG64(search_settings['seed']))
                    rng.bit_generator.advance(index * len(bounds))
                    unit = rng.random(len(bounds))
                    expected = np.clip(lower + unit * width, bounds[:, 0], bounds[:, 1])
                    if not np.array_equal(vector, expected):
                        raise ValueError('Legacy sample coordinates disagree with its saved seed/bounds.')
                store.import_sample(index, unit, loss)
            elif record['stage'] != 'local':
                raise ValueError('Unknown search stage in existing history.')
            nfev += 1
            if not math.isfinite(loss):
                failed += 1
                category = record.get('error_code') or error_category(record.get('error'))
                failures[category] = failures.get(category, 0) + 1
            elif best is None or loss < best['loss']:
                best = dict(record, loss=loss)
            offset = stream.tell()
            if nfev % 1000000 == 0:
                print(f'Indexed {nfev} evaluations.', flush=True)
    # A completed historical search supplies its per-seed completion information.
    summary_path = output / 'search_summary.yaml'
    if summary_path.exists():
        summary = yaml.safe_load(summary_path.read_text())
        for run in summary.get('local_runs', []):
            store.save_local_run(run)
    if best is not None:
        saved_loss = yaml.safe_load((output / 'best_loss.yaml').read_text())
        if not math.isclose(saved_loss['loss'], best['loss'], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Legacy best artifacts disagree with complete history. '
                             'Preserve the run and inspect it before restarting.')
        best['results'] = yaml.safe_load((output / 'best_results.yaml').read_text())
    old_summary = output / 'history_summary.yaml'
    old = yaml.safe_load(old_summary.read_text()) if old_summary.exists() else {}
    # Detailed example counts remain bounded even after an old run's hard kill.
    history = dict(evaluations=nfev, failed=failed, failure_counts=failures,
                   failure_examples=old.get('failure_examples', {k: 3 for k in failures}),
                   diagnostic_examples=old.get('successful_diagnostic_examples', 3),
                   history_offset=offset,
                   solver_offset=(output / 'solver.log').stat().st_size)
    return dict(nfev=nfev, failed=failed, best=best, history=history)


def resume(output_dir, *, workers='auto', progress_every=None):
    """Continue saved sampling/refinement with unchanged mechanisms and scientific settings."""
    problem, saved = load_problem(output_dir)
    settings = dict(saved['search'])
    settings['workers'] = workers
    settings['progress_every'] = (saved.get('progress_every', 1000)
                                  if progress_every is None else progress_every)
    return problem.optimize(output_dir=Path(output_dir).resolve(), resume=True,
                            history_mode=saved.get('history', {}).get('mode', 'compact'),
                            **settings)


def main(output_dir=None):
    """Run a generated restart file or python -m mt.restart RUN_DIRECTORY."""
    parser = argparse.ArgumentParser(description=__doc__)
    if output_dir is None:
        parser.add_argument('output_dir')
    parser.add_argument('--workers', default='auto')
    parser.add_argument('--progress-every', type=int)
    args = parser.parse_args()
    workers = args.workers if args.workers == 'auto' else int(args.workers)
    return resume(output_dir if output_dir is not None else args.output_dir,
                  workers=workers, progress_every=args.progress_every)


if __name__ == '__main__':
    main()
