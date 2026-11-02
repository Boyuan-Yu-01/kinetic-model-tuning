"""Portable process-based random screening and independent local searches."""
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, as_completed
from contextlib import nullcontext, contextmanager
import warnings
from uuid import uuid4
import numpy as np
from scipy.optimize import minimize
from tempfile import TemporaryDirectory
from pathlib import Path
from .sample_store import SampleStore


_MPI_POOL = None
_MPI_WORKERS = 0
_WORKER_STATE = None
_WORKER_FAILURES = {}


@contextmanager
def mpi_context(pool, workers):
    """Expose the MPI coordinator's executor without changing local execution."""
    global _MPI_POOL, _MPI_WORKERS
    previous = _MPI_POOL, _MPI_WORKERS
    _MPI_POOL, _MPI_WORKERS = pool, workers
    try:
        yield
    finally:
        _MPI_POOL, _MPI_WORKERS = previous


def mpi_active():
    """Return whether this process is coordinating an MPI executor."""
    return _MPI_POOL is not None


def execution_backend(backend="process"):
    """Name the executor backend for saved run metadata."""
    if backend not in ('process', 'openmp'):
        raise ValueError('backend must be process or openmp.')
    if mpi_active() and backend == 'openmp':
        raise ValueError('OpenMP cannot be nested in the MPI executor.')
    return 'mpi' if mpi_active() else backend


def pool_context(count, backend="process"):
    """Borrow the MPI executor or own a local spawned process pool."""
    execution_backend(backend)
    if mpi_active():
        return nullcontext(_MPI_POOL)
    if backend == "openmp":
        from .executors import create_executor
        return create_executor(backend, count)
    return (ProcessPoolExecutor(max_workers=count, mp_context=mp.get_context('spawn'))
            if count > 1 else nullcontext(None))


def initialize_worker(key, objective, bounds, compact_errors=False):
    """Install one evaluation context per worker instead of resending it per candidate."""
    global _WORKER_STATE, _WORKER_FAILURES
    _WORKER_STATE = (key, objective, np.asarray(bounds, dtype=float), compact_errors)
    _WORKER_FAILURES = {}


def initialize_mpi_worker(*args):
    """Initialize every MPI worker; the barrier keeps one setup task on each rank."""
    from mpi4py.futures import get_comm_workers
    initialize_worker(*args)
    get_comm_workers().Barrier()


@contextmanager
def evaluation_pool(count, objective, bounds, compact_errors=False, backend="process"):
    """Own a process pool or initialize the existing MPI pool for one search."""
    global _WORKER_STATE, _WORKER_FAILURES
    execution_backend(backend)
    key = uuid4().hex
    args = (key, objective, bounds, compact_errors)
    if mpi_active():
        # All ranks participate even when the caller caps the number of pending tasks.
        setup = [_MPI_POOL.submit(initialize_mpi_worker, *args) for _ in range(_MPI_WORKERS)]
        for future in as_completed(setup):
            future.result()
        yield _MPI_POOL, key
    elif backend == "openmp":
        from .executors import create_executor
        with create_executor(backend, count) as pool:
            yield _InitializedExecutor(pool, args), key
    elif count > 1:
        with ProcessPoolExecutor(max_workers=count, mp_context=mp.get_context('spawn'),
                                 initializer=initialize_worker, initargs=args) as pool:
            yield pool, key
    else:
        previous = _WORKER_STATE, _WORKER_FAILURES
        initialize_worker(*args)
        try:
            yield None, key
        finally:
            _WORKER_STATE, _WORKER_FAILURES = previous


def _initialized_task(args, function, task_args):
    """Install search state lazily in an isolated OpenMP worker process."""
    if _WORKER_STATE is None or _WORKER_STATE[0] != args[0]:
        initialize_worker(*args)
    return function(*task_args)


class _InitializedExecutor:
    """Adapt submit without assuming native support for pool initializers."""
    def __init__(self, pool, args):
        self.pool, self.args = pool, args

    def submit(self, function, *args):
        return self.pool.submit(_initialized_task, self.args, function, args)


def worker_state(key):
    """Reject stale tasks if an executor has been reused for another search."""
    if _WORKER_STATE is None or _WORKER_STATE[0] != key:
        raise RuntimeError('Worker evaluation context is missing or belongs to another search.')
    return _WORKER_STATE[1:]


def worker_record(record, compact_errors):
    """Send only a few verbose examples per failure category in compact mode."""
    if not compact_errors or np.isfinite(record['loss']):
        return record
    from .history import ERROR_CATEGORIES, error_category
    record = dict(record)
    category = record.get('error_code')
    if not isinstance(category, str) or category not in ERROR_CATEGORIES:
        category = error_category(record.get('error', ''))
    record['error_code'] = category
    seen = _WORKER_FAILURES.get(category, 0)
    _WORKER_FAILURES[category] = seen + 1
    if seen >= 3:
        record.pop('error', None)
        record.pop('diagnostics', None)
    return record


def candidate_batches(key, bounds, samples, seed, sample_batch_size, worker_batch_size):
    """Yield small dispatch batches while preserving the original RNG stream and indices."""
    rng = np.random.default_rng(seed)
    for offset in range(0, samples, sample_batch_size):
        batch = rng.random((min(sample_batch_size, samples - offset), len(bounds)))
        for start in range(0, len(batch), worker_batch_size):
            yield key, offset + start, batch[start:start + worker_batch_size].copy()


def restart_batches(key, bounds, samples, state, sample_batch_size, worker_batch_size, checkpoint):
    """Regenerate only unfinished samples, keeping each index's original PCG64 draw."""
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = state
    position = 0
    for first, stop in checkpoint.missing_ranges(samples):
        # PCG64 emits one raw 64-bit draw for each Generator.random float64.
        rng.bit_generator.advance((first - position) * len(bounds))
        for offset in range(first, stop, sample_batch_size):
            batch = rng.random((min(sample_batch_size, stop - offset), len(bounds)))
            for start in range(0, len(batch), worker_batch_size):
                yield key, offset + start, batch[start:start + worker_batch_size].copy()
        position = stop


def sample_batch_task(job):
    """Evaluate a batch with cached worker context and retain every candidate's result."""
    key, offset, units = job
    objective, bounds, compact_errors = worker_state(key)
    lo, hi = bounds[:, 0], bounds[:, 1]
    records = []
    for j, unit in enumerate(units):
        vector = np.clip(lo + unit * (hi - lo), lo, hi)
        record = worker_record(objective(vector), compact_errors)
        records.append((offset + j, unit, record))
    return records


def local_worker_task(job):
    """Run a local search using the same worker context as random sampling."""
    key, index, start, budget = job
    objective, bounds, compact_errors = worker_state(key)
    def evaluate(vector):
        """Preserve the exact objective while compacting repeated failure diagnostics."""
        return worker_record(objective(vector), compact_errors)
    return local_task((evaluate, index, start, bounds, budget))


def worker_count(workers='auto'):
    """Resolve auto or explicit workers within visible CPUs and any Slurm allocation."""
    if mpi_active():
        if workers == 'auto':
            return _MPI_WORKERS
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= _MPI_WORKERS:
            raise ValueError(f'MPI workers must be auto or between 1 and {_MPI_WORKERS}.')
        return workers
    ranks = max(int(os.environ.get(key, '1')) for key in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE'))
    if ranks > 1 or ('SLURM_PROCID' in os.environ and int(os.environ.get('SLURM_NTASKS', '1')) > 1):
        raise RuntimeError('Launch multi-rank runs with python -m mt.mpi optimizer.py, not python optimizer.py.')
    slurm = os.environ.get('SLURM_CPUS_PER_TASK')
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    allocated = min(available, int(slurm)) if slurm else available
    if workers == 'auto':
        return allocated if slurm else min(4, allocated)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be auto or a positive integer.')
    if workers > allocated:
        raise ValueError(f'Requested {workers} workers but only {allocated} CPUs available.')
    return workers


def limit_threads():
    """Limit numerical-library threads inherited by workers to avoid CPU oversubscription."""
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'


def completed(pool, function, jobs, count):
    """Bound pending tasks to avoid queueing 100,000 full model copies."""
    if pool is None:
        for job in jobs:
            yield function(job)
        return
    jobs = iter(jobs)
    pending = set()
    for _ in range(count if mpi_active() else 2 * count):
        try:
            pending.add(pool.submit(function, next(jobs)))
        except StopIteration:
            break
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            yield future.result()
            try:
                pending.add(pool.submit(function, next(jobs)))
            except StopIteration:
                pass


def sample_task(job):
    """Evaluate an indexed candidate and return its index and result record."""
    objective, index, vector = job
    return index, objective(vector)


def local_task(job):
    """Run bounded Powell refinement for one seed, returning all records and a summary."""
    objective, index, start, bounds, budget = job
    # Powell works in unit coordinates so parameter scales are comparable.
    lo, width = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    records = []

    def function(unit):
        """Map unit coordinates to parameter values, recording each evaluated loss."""
        # Affine mapping can round an endpoint just outside its physical bounds.
        record = objective(np.clip(lo + unit * width, bounds[:, 0], bounds[:, 1]))
        # Retain intermediate improvements even if the final point fails.
        records.append(record)
        return record['loss']

    with warnings.catch_warnings(record=True) as messages:
        result = minimize(function, (start - lo) / width, method='Powell',
                          bounds=[(0, 1)] * len(start), options={'maxfev': budget})
    return index, records, {
        'seed_index': index, 'success': bool(result.success and np.isfinite(result.fun)),
        'message': str(result.message), 'loss': float(result.fun),
        'nfev': int(result.nfev), 'vector': np.clip(lo + result.x * width, bounds[:, 0], bounds[:, 1]).tolist(),
        'warnings': [str(m.message) for m in messages],
    }


def sample_jobs(objective, bounds, samples, seed, batch_size):
    """Yield reproducible candidates in bounded batches; RNG order matches a dense draw."""
    rng = np.random.default_rng(seed)
    for offset in range(0, samples, batch_size):
        batch = rng.random((min(batch_size, samples - offset), len(bounds)))
        for j, unit in enumerate(batch):
            # Copy so a queued row cannot retain the entire batch allocation.
            yield objective, offset + j, unit.copy(), bounds


def streamed_sample_task(job):
    """Map one normalized sample to physical parameters and return its coordinates."""
    objective, index, unit, bounds = job
    vector = np.clip(bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0]), bounds[:, 0], bounds[:, 1])
    return index, unit, objective(vector)


def search(objective, bounds, *, samples, starts, neighbors=5, min_distance=.2,
           max_evaluations=1000, seed=42, workers='auto', on_record=None,
           sample_batch_size=1024, max_in_memory_valid=100000, sample_directory=None,
           worker_batch_size=16, compact_errors=False, checkpoint=None, on_checkpoint=None, backend="process"):
    """Stream samples to disk, select exact separated minima, and refine their rates."""
    for name, value in [('samples', samples), ('starts', starts), ('neighbors', neighbors),
                        ('max_evaluations', max_evaluations), ('sample_batch_size', sample_batch_size),
                        ('worker_batch_size', worker_batch_size),
                        ('max_in_memory_valid', max_in_memory_valid)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if starts > samples:
        raise ValueError('starts cannot exceed samples.')
    if not np.isfinite(min_distance) or min_distance < 0:
        raise ValueError('min_distance must be finite and nonnegative.')
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not len(bounds) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError('Invalid bounds.')

    def persist(force=False):
        """Checkpoint only complete record deliveries; never retry a failed callback."""
        if on_checkpoint:
            try:
                on_checkpoint(force=force)
            except BaseException:
                if checkpoint is not None:
                    checkpoint.safe_to_commit = False
                raise

    if checkpoint is not None:
        if on_checkpoint is None:
            raise ValueError('A restart checkpoint requires an on_checkpoint callback to persist runner state.')
        config = dict(bounds=bounds.tolist(), samples=samples, starts=starts, neighbors=neighbors,
                      min_distance=min_distance, max_evaluations=max_evaluations, seed=repr(seed))
        saved = checkpoint.get('search_config')
        if saved is not None and saved != config:
            raise ValueError('Restart search settings or bounds differ from the saved checkpoint.')
        checkpoint.set('search_config', config)
        rng_state = checkpoint.get('initial_rng_state')
        if rng_state is None:
            # Explicit PCG64 keeps saved index-to-vector mapping independent of worker scheduling.
            rng_state = np.random.Generator(np.random.PCG64(seed)).bit_generator.state
            checkpoint.set('initial_rng_state', rng_state)
        persist(force=True)
    count = worker_count(workers)
    limit_threads()
    remaining = samples - checkpoint.completed_count if checkpoint is not None else samples
    print(f'Sampling {remaining} remaining candidates using {count} workers '
          f'({worker_batch_size} candidates/task).', flush=True)
    directory_context = (nullcontext(None) if checkpoint is not None else
                         TemporaryDirectory(prefix='.mt_samples_', dir=sample_directory))
    with directory_context as directory:
        store = checkpoint if checkpoint is not None else SampleStore(
            Path(directory) / 'valid.sqlite', len(bounds), sample_batch_size)
        try:
            with evaluation_pool(count, objective, bounds, compact_errors, backend) as (pool, key):
                try:
                    jobs = (candidate_batches(key, bounds, samples, seed, sample_batch_size, worker_batch_size)
                            if checkpoint is None else restart_batches(
                                key, bounds, samples, rng_state, sample_batch_size, worker_batch_size, checkpoint))
                    for batch in completed(pool, sample_batch_task, jobs, count):
                        if checkpoint is not None:
                            checkpoint.safe_to_commit = False
                        for i, unit, record in batch:
                            if checkpoint is not None:
                                checkpoint.import_sample(i, unit, record['loss'])
                            elif np.isfinite(record['loss']):
                                store.add(i, record['loss'], unit)
                            record.update(stage='sampling', sample_index=i)
                            if on_record:
                                on_record(record)
                        if checkpoint is not None:
                            checkpoint.safe_to_commit = True
                        persist()
                    store.finish()
                    selection = checkpoint.get_selection() if checkpoint is not None else None
                    if selection is None:
                        selected, selection_method = store.select(
                            starts, neighbors, min_distance, max_in_memory_valid)
                        if checkpoint is not None:
                            checkpoint.set_selection(selected, selection_method)
                            checkpoint.set('phase', 'local')
                            persist(force=True)
                    else:
                        selected, selection_method = selection
                    valid_samples = store.count
                    actual_starts = len(selected)
                    if actual_starts < starts:
                        print(f'Only {actual_starts} of {starts} requested seeds qualify; continuing with {actual_starts}.', flush=True)
                    runs = checkpoint.local_runs() if checkpoint is not None else []
                    finished = {run['seed_index'] for run in runs}
                    print(f'Refining {actual_starts - len(runs)} remaining seeds using up to '
                          f'{min(count, actual_starts - len(runs))} workers.', flush=True)
                    jobs = ((key, i, np.clip(bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0]),
                                                   bounds[:, 0], bounds[:, 1]), max_evaluations)
                            for i, unit in selected if i not in finished)
                    for i, records, result in completed(pool, local_worker_task, jobs, count):
                        if checkpoint is not None:
                            checkpoint.safe_to_commit = False
                        for record in records:
                            record.update(stage='local', sample_index=int(i))
                            if on_record:
                                on_record(record)
                        runs.append(result)
                        if checkpoint is not None:
                            checkpoint.save_local_run(result)
                            checkpoint.safe_to_commit = True
                        persist()
                        print(f'Local searches completed: {len(runs)}/{actual_starts}', flush=True)
                    if checkpoint is not None:
                        checkpoint.set('phase', 'complete')
                finally:
                    # Persist received work before executor teardown waits for unfinished tasks.
                    if checkpoint is None or checkpoint.safe_to_commit:
                        persist(force=True)
        finally:
            if checkpoint is None:
                store.close()
    return {'requested_starts': starts, 'actual_starts': actual_starts,
            'selected_indices': [i for i, _ in selected], 'local_runs': sorted(runs, key=lambda r: r['seed_index']),
            'workers': count, 'backend': execution_backend(backend), 'valid_samples': valid_samples,
            'sample_batch_size': sample_batch_size, 'max_in_memory_valid': max_in_memory_valid,
            'worker_batch_size': worker_batch_size,
            'seed_selection': selection_method}
