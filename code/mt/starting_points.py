"""A-only ZND starting-point search with multiple candidate outputs."""
from contextlib import redirect_stdout, redirect_stderr, closing
from copy import deepcopy
from pathlib import Path
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import warnings
import sqlite3
from tempfile import TemporaryDirectory
from time import monotonic

import cantera as ct
from sdtoolbox.postshock import CJspeed, PostShock_fr

import numpy as np
import yaml
from scipy.optimize import minimize

from .znd_functions import calculate_znd, _integrate_znd


def _case_outputs(output):
    """Extract valid temperature and resolved lengths without retaining a trajectory."""
    values = [float(output['T'][-1]), float(output['ind_len_ZND']),
              float(output['exo_len_ZND'])]
    if not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError('Invalid final temperature or unresolved ZND lengths.')
    return dict(zip(('T_ZND', 'ind_len_ZND', 'exo_len_ZND'), values))


def _case(job):
    """Run one independent case; each process owns its Cantera/LSODA state."""
    mechanism, pressure, temperature, phi, end, fuel, oxidizer = job
    p, temperature, phi, end = map(float, (pressure / 1e5, temperature, phi, end))
    record = dict(P_bar=p, T0=temperature, phi=phi, t_end=end)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = calculate_znd(
                P0=p * 1e5, T0=temperature, fuel=fuel, oxidizer=oxidizer,
                phi=phi, mechanism=str(mechanism), file_name=None,
                method='LSODA', t_end=end, max_step=end / 1000,
                rel_tol=1e-8, abs_tol=1e-12,
            )
        record.update(_case_outputs(out))
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    return record


def _simulate(mechanism, conditions, fuel, oxidizer, pool=None, log_dir=None):
    """Evaluate the fixed reference mechanism once, preserving case order."""
    jobs = [(str(mechanism), p, t, phi, end, fuel, oxidizer)
            for p, t, phi, end in zip(conditions.P, conditions.T, conditions.phi, conditions.t_end)]
    if pool is None:
        return [_case(job) for job in jobs]
    return list(pool.map(_logged_task, [('reference', job, str(log_dir)) for job in jobs]))


class _GasPair:
    """Own two gas objects and immutable original rates in a single process."""
    def __init__(self, source, indices):
        """Load the fixed mechanism and save original Arrhenius constants."""
        self.gas = ct.Solution(source)
        self.upstream = ct.Solution(source)
        self.indices = indices
        self.rates = []
        for index in indices:
            reaction = self.gas.reaction(index)
            if not isinstance(reaction.rate, ct.ArrheniusRate):
                raise ValueError('A-only starting-point search requires ordinary Arrhenius rates.')
            self.rates.append((reaction.rate.pre_exponential_factor,
                               reaction.rate.temperature_exponent, reaction.rate.activation_energy))

    def solve(self, state, vector):
        """Restore exact upstream/frozen states and replace A from original values."""
        gas = self.gas
        gas.set_multiplier(1.0)
        self.upstream.set_multiplier(1.0)
        self.upstream.state = state['upstream']
        gas.state = state['frozen']
        for index, (a, b, ea), ratio in zip(self.indices, self.rates, 10.0 ** np.asarray(vector)):
            reaction = gas.reaction(index)
            reaction.rate = ct.ArrheniusRate(a * float(ratio), b, ea)
            gas.modify_reaction(index, reaction)
        row = state['case']
        end = row['t_end']
        return _integrate_znd(
            gas, self.upstream, state['cj_speed'], P0=row['P_bar'] * 1e5,
            T0=row['T0'], phi=row['phi'], t_end=end, max_step=end / 1000,
            rel_tol=1e-8, abs_tol=1e-12, method='LSODA',
        )


# At most one fixed model per run/worker; no per-candidate YAML or gas cache.
_GAS_PAIR = None
_GAS_KEY = None


def _prepare_case(job):
    """Compute thermodynamic shock states once; A changes cannot alter them."""
    source, pressure, temperature, phi, end, fuel, oxidizer = job
    pressure, temperature, phi, end = map(float, (pressure, temperature, phi, end))
    state = {'case': dict(P_bar=pressure / 1e5, T0=temperature, phi=phi, t_end=end)}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            upstream = ct.Solution(source)
            upstream.TP = temperature, pressure
            upstream.set_equivalence_ratio(phi, fuel, oxidizer, basis='mole')
            x = upstream.X.copy()
            speed = CJspeed(pressure, temperature, x, source)
            frozen = PostShock_fr(speed, pressure, temperature, x, source)
            state.update(upstream=upstream.state.copy(), frozen=frozen.state.copy(), cj_speed=speed)
    except Exception as exc:
        state['error'] = f'{type(exc).__name__}: {exc}'
    return state


def _cached_case(job):
    """Return compact results, restoring all modified state even after failures."""
    global _GAS_PAIR, _GAS_KEY
    source, indices, state, vector = job
    record = state['case'].copy()
    if 'error' in state:
        return dict(record, error=state['error'])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            key = (source, tuple(indices))
            if key != _GAS_KEY:
                _GAS_PAIR = _GasPair(source, indices)
                _GAS_KEY = key
            out = _GAS_PAIR.solve(state, vector)
            record.update(_case_outputs(out))
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    return record


def _logged_task(job):
    """Write setup/solver diagnostics to one log per process, not to the terminal."""
    import os
    kind, payload, directory = job
    function = {'prepare': _prepare_case, 'solve': _cached_case, 'reference': _case}[kind]
    with (Path(directory) / f'solver_{os.getpid()}.log').open('a') as log:
        with redirect_stdout(log), redirect_stderr(log):
            return function(payload)


class _CaseEvaluator:
    """Share prepared states and keep a bounded pool of independent case tasks busy."""
    def __init__(self, source, indices, conditions, fuel, oxidizer, pool, output, verbose=True):
        """Prepare one reusable shock state for each requested condition."""
        self.source, self.indices = str(source), indices
        self.pool, self.output = pool, str(output)
        self.verbose = verbose
        jobs = [('prepare', (self.source, p, t, phi, end, fuel, oxidizer), self.output)
                for p, t, phi, end in zip(conditions.P, conditions.T, conditions.phi, conditions.t_end)]
        self.states = list(map(_logged_task, jobs) if pool is None else pool.map(_logged_task, jobs))

    def jobs(self, vector):
        """Send only rates and small state arrays; profile arrays never leave workers."""
        return [('solve', (self.source, self.indices, state, vector), self.output) for state in self.states]

    def evaluate(self, vector):
        """Evaluate all cases for a Powell step, preserving condition order."""
        jobs = self.jobs(vector)
        return list(map(_logged_task, jobs) if self.pool is None else self.pool.map(_logged_task, jobs))

    def sample(self, points):
        """Refill on any completed case; spool results to disk for ordered scoring.

        Only 2*workers case futures are outstanding. A slow early case does not
        block dispatch of later candidates. Completed records wait in temporary
        SQLite storage rather than accumulating full result dictionaries in RAM.
        Scoring still receives the original sample and condition order.
        """
        if self.pool is None:
            for point in points:
                yield point, self.evaluate(point)
            return
        ncases, nsamples = len(self.states), len(points)
        counts = np.zeros(nsamples, dtype=np.int64)
        jobs = iter((sample, case, job) for sample, point in enumerate(points)
                    for case, job in enumerate(self.jobs(point)))
        pending = {}
        submitted = completed = finished = next_sample = 0
        began, last_status = monotonic(), -float('inf')
        status_path = Path(self.output) / 'sampling_status.json'

        def refill():
            """Submit another case as soon as any worker slot becomes available."""
            nonlocal submitted
            while len(pending) < 2 * self.pool._max_workers:
                item = next(jobs, None)
                if item is None:
                    break
                sample, case, job = item
                pending[self.pool.submit(_logged_task, job)] = sample, case
                submitted += 1

        def status(force=False):
            """Show actual completed work even when ordered scoring awaits a slow case."""
            nonlocal last_status
            now = monotonic()
            if not force and now - last_status < 10:
                return
            last_status = now
            value = dict(total_candidates=nsamples, total_cases=nsamples * ncases,
                         submitted_cases=submitted, completed_cases=completed,
                         completed_candidates=finished, scored_candidates=next_sample,
                         pending_cases=len(pending), workers=self.pool._max_workers,
                         elapsed_seconds=now - began)
            temporary = status_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(value, indent=2))
            temporary.replace(status_path)
            if self.verbose:
                print(f'Sampling cases {completed}/{nsamples * ncases} | '
                      f'complete candidates {finished}/{nsamples} | '
                      f'scored {next_sample} | pending cases {len(pending)}', flush=True)

        with TemporaryDirectory(prefix='.sampling_', dir=self.output) as directory:
            with closing(sqlite3.connect(str(Path(directory) / 'results.sqlite'))) as db:
                # Scratch storage is not a restart checkpoint; persistent results are
                # still written by the coordinator's normal evaluation/archive path.
                db.execute('PRAGMA journal_mode=OFF')
                db.execute('PRAGMA synchronous=OFF')
                db.execute('PRAGMA cache_size=-4096')
                db.execute('CREATE TABLE results (sample INTEGER, condition INTEGER, '
                           'record TEXT, PRIMARY KEY(sample, condition)) WITHOUT ROWID')
                try:
                    refill()
                    status(force=True)
                    while pending:
                        done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                        for future in done:
                            sample, case = pending.pop(future)
                            record = future.result()
                            db.execute('INSERT INTO results VALUES (?, ?, ?)',
                                       (sample, case, json.dumps(record, allow_nan=False)))
                            counts[sample] += 1
                            completed += 1
                            if counts[sample] == ncases:
                                finished += 1
                        db.commit()
                        # Refill BEFORE ordered delivery, independent of the slowest sample.
                        refill()
                        while next_sample < nsamples and counts[next_sample] == ncases:
                            records = [json.loads(row[0]) for row in db.execute(
                                'SELECT record FROM results WHERE sample=? ORDER BY condition',
                                (next_sample,))]
                            db.execute('DELETE FROM results WHERE sample=?', (next_sample,))
                            db.commit()
                            yield points[next_sample], records
                            next_sample += 1
                        status()
                    assert next_sample == nsamples, 'Not all sampling results were delivered.'
                    status(force=True)
                finally:
                    for future in pending:
                        future.cancel()


class _StartingPointRun:
    """Use the shared guarded launcher so plain user scripts remain safe with spawn."""
    _reference_results = None
    _reference_signature = None

    def _starting_points(self, workers, **settings):
        """Reuse one pool for reference, random screening and all Powell evaluations."""
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            return _find_znd_starting_points(**settings, _pool=pool)


def find_znd_starting_points(model, reference_model, conditions, *, workers=1, verbose=True, **settings):
    """Find candidates with serial cases or independent process workers.

    workers=1 runs directly; workers>1 uses the shared launcher. Both modes
    cache shock states and update A in memory. Parallel sampling refills on
    any completed case, with bounded futures and disk-backed ordered results;
    scoring stays in sample order. verbose=False hides routine terminal output;
    diagnostic logs and results are still saved. Remaining settings are documented by
    _find_znd_starting_points.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer.')
    if not isinstance(verbose, bool):
        raise ValueError('verbose must be True or False.')
    settings.update(model=model, reference_model=reference_model, conditions=conditions, verbose=verbose)
    if workers == 1:
        global _GAS_PAIR, _GAS_KEY
        try:
            return _find_znd_starting_points(**settings)
        finally:
            _GAS_PAIR = _GAS_KEY = None
    from .launcher import launch
    return launch(_StartingPointRun(), '_starting_points', dict(workers=workers, **settings))


def _a_multiplier_bounds(a_ratio, count):
    """Expand a shared pair or validate one multiplier pair per selected reaction."""
    try:
        ratios = np.asarray(a_ratio, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('a_ratio must be a shared (low, high) pair or one pair per reaction.') from exc
    if ratios.shape == (2,):
        ratios = np.tile(ratios, (count, 1))
    elif ratios.shape != (count, 2):
        raise ValueError(f'a_ratio must have shape (2,) or ({count}, 2), in reactions order.')
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0) or np.any(ratios[:, 0] >= ratios[:, 1]):
        raise ValueError('A multiplier bounds must be positive, finite and increasing.')
    return np.log10(ratios)


def _find_znd_starting_points(model, reference_model, conditions, *, output_dir,
                             reactions=(1, 2, 3), a_ratio=(1e-5, 1e5),
                             samples=3000, starts=5, max_evaluations=300,
                             keep=10, min_distance=0.10, max_temperature_error=0.05,
                             seed=42, fuel='C2H4:1', oxidizer='O2:1, N2:3.76', verbose=True, _pool=None):
    """Tune selected A values and retain separated, warning-free temperature fits.

    Reaction numbers are one-based. Distances are in log10 multiplier space.
    a_ratio accepts a shared (low, high) pair or a list of pairs in reactions order.
    Cached A-only screening plus Powell refinement; lengths are validity
    checks only. Each case must satisfy max_temperature_error (relative).
    Uses LSODA, rtol=1e-8, atol=1e-12 and max_step=t_end/1000.
    output_dir is the exact destination; previous generated run files are replaced.
    Returns the output directory and up to keep qualifying candidates.
    """
    for name, value in [('samples', samples), ('starts', starts),
                        ('max_evaluations', max_evaluations), ('keep', keep)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if not (np.isfinite(min_distance) and min_distance > 0
            and np.isfinite(max_temperature_error) and max_temperature_error > 0):
        raise ValueError('Distance and temperature tolerance must be finite and positive.')
    if not reactions or len(set(reactions)) != len(reactions) or any(
            isinstance(i, bool) or not isinstance(i, int) or i < 1 for i in reactions):
        raise ValueError('Reaction numbers must be distinct positive integers.')
    indices = [i - 1 for i in reactions]
    bounds = _a_multiplier_bounds(a_ratio, len(indices))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(Path(model).read_text())
    original = np.array([float(r['rate-constant']['A']) for r in [base['reactions'][i] for i in indices]])
    if np.any(original <= 0) or not np.all(np.isfinite(original)):
        raise ValueError('Selected reactions must have positive Arrhenius A values.')
    # Start a fresh run so old candidates and appended logs cannot mix with new results.
    for pattern in ('candidate_*.yaml', 'candidates.yaml', 'evaluations.jsonl',
                    'solver*.log', 'settings.yaml', 'base_model.yaml',
                    'FFCM2_results.yaml', 'trial.yaml', 'sampling_status.json', 'sampling_status.json.tmp'):
        for path in output.glob(pattern):
            if path.is_file():
                path.unlink()
    archive, cache = [], {}
    count = 0
    failed_sets_in_block = 0

    def report(message):
        """Keep progress optional without changing computation or saved diagnostics."""
        if verbose:
            print(message, flush=True)

    def write_yaml(path, value):
        """Write ordinary Python data without NumPy-specific YAML tags."""
        path.write_text(yaml.safe_dump(value, sort_keys=False))

    def mechanism(vector):
        """Preserve all model fields except the selected pre-exponential factors."""
        model = deepcopy(base)
        for reaction, value in zip([model['reactions'][i] for i in indices], original * 10.0 ** vector):
            reaction['rate-constant']['A'] = float(value)
        return model

    settings = dict(workers=_pool._max_workers if _pool else 1, verbose=verbose, reactions=list(reactions), A_ratio=np.asarray(a_ratio, dtype=float).tolist(), samples=samples, starts=starts,
                    max_evaluations=max_evaluations, keep=keep, seed=seed,
                    min_distance_decades=min_distance,
                    max_temperature_relative_error=max_temperature_error,
                    method='LSODA', rel_tol=1e-8, abs_tol=1e-12,
                    max_step='t_end / 1000')
    write_yaml(output / 'settings.yaml', settings)
    write_yaml(output / 'base_model.yaml', base)
    report('Calculating FFCM2 reference cases...')
    with (output / 'solver.log').open('a') as log, redirect_stdout(log), redirect_stderr(log):
        reference = _simulate(reference_model, conditions, fuel, oxidizer, _pool, output)
    write_yaml(output / 'FFCM2_results.yaml', reference)
    if any('error' in row for row in reference):
        raise RuntimeError(f'Reference cases failed; inspect {output / "FFCM2_results.yaml"}')
    target = np.array([row['T_ZND'] for row in reference])
    report('Preparing reusable model shock states...')
    evaluator = _CaseEvaluator(output / 'base_model.yaml', indices, conditions, fuel, oxidizer, _pool, output, verbose=verbose)

    def evaluate(vector, records=None):
        """Rank feasibility first; fit temperature only once every case succeeds."""
        nonlocal count, archive, failed_sets_in_block
        vector = np.asarray(vector)
        key = tuple(vector)
        if key in cache:
            return cache[key]
        count += 1
        if records is None:
            records = evaluator.evaluate(vector)
        errors = [None if 'error' in row else float(row['T_ZND'] / ref - 1)
                  for row, ref in zip(records, target)]
        failed = sum(e is None for e in errors)
        failed_sets_in_block += int(failed > 0)
        fit = float(np.mean([e * e for e in errors if e is not None])) if failed < len(target) else None
        # A partial result can guide the feasibility search, but cannot enter the archive.
        score = float(failed + (fit / (1 + fit) if fit is not None else 0))
        maximum = max((abs(e) for e in errors if e is not None), default=None)
        item = dict(evaluation=count, log10_A_ratios=vector.tolist(),
                    A_ratios=(10.0 ** vector).tolist(), A=(original * 10.0 ** vector).tolist(),
                    failed_cases=failed, temperature_mse=fit, max_relative_error=maximum,
                    score=score, cases=records)
        with (output / 'evaluations.jsonl').open('a') as log:
            log.write(json.dumps(item, allow_nan=False) + '\n')
        if failed == 0 and maximum <= max_temperature_error:
            # Greedy ranking retains separated candidates instead of near-identical Powell steps.
            chosen = []
            for entry in sorted(archive + [item], key=lambda r: r['temperature_mse']):
                if all(np.linalg.norm(np.array(entry['log10_A_ratios']) - r['log10_A_ratios'])
                       >= min_distance for r in chosen):
                    chosen.append(entry)
                if len(chosen) == keep:
                    break
            if [r['evaluation'] for r in chosen] != [r['evaluation'] for r in archive]:
                previous = archive
                archive = chosen
                for rank, entry in enumerate(archive, 1):
                    if rank > len(previous) or entry['evaluation'] != previous[rank - 1]['evaluation']:
                        write_yaml(output / f'candidate_{rank:02d}.yaml',
                                   mechanism(np.array(entry['log10_A_ratios'])))
                for rank in range(len(archive) + 1, len(previous) + 1):
                    (output / f'candidate_{rank:02d}.yaml').unlink(missing_ok=True)
                write_yaml(output / 'candidates.yaml', archive)
        if count % 10 == 0:
            report(f'Eval {count} | failed sets {failed_sets_in_block}/10 | saved candidates {len(archive)}/{keep}')
            failed_sets_in_block = 0
        cache[key] = score
        return score

    rng = np.random.default_rng(seed)
    points = np.vstack([np.clip(np.zeros(len(indices)), bounds[:, 0], bounds[:, 1]),
                        rng.uniform(bounds[:, 0], bounds[:, 1], (samples, len(indices)))])
    scores = [evaluate(point, records) for point, records in evaluator.sample(points)]
    selected = []
    for i in np.argsort(scores):
        if all(np.linalg.norm(points[i] - point) >= min_distance for point in selected):
            selected.append(points[i])
        if len(selected) == starts:
            break
    for i, point in enumerate(selected, 1):
        report(f'Powell start {i}/{len(selected)}')
        minimize(evaluate, point, method='Powell', bounds=bounds,
                 options={'maxfev': max_evaluations, 'xtol': 1e-3, 'ftol': 1e-5})
    write_yaml(output / 'candidates.yaml', archive)
    report(f'Saved {len(archive)} qualifying candidates in {output}')
    if len(archive) < keep:
        report('Fewer candidates met the temperature, validity, and separation requirements.')

    return {"output_dir": output, "candidates": archive}
