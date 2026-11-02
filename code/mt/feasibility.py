"""Explicit, source-anchored A-only initialization search; never redesign reactions."""
from collections import Counter
from pathlib import Path
import time
import numpy as np


def find_feasible(batch, space, refs, cfg, root):
    """Screen bounded log-uniform A draws, then verify finalists at run accuracy.

    ``initial_search`` requires ``a_ratio`` mapping reaction numbers to positive
    source-relative [low, high] multipliers. Optional keys: samples, batch_size,
    seed, finalists. Only the stage-A magnitude coordinates are changed.
    Every selected PLOG node receives the same source-relative A multiplier.
    """
    from .iterative_znd import _score, _candidate_monotonicity, _interval, _save
    options = cfg.get('initial_search')
    if not isinstance(options, dict) or not options.get('a_ratio'):
        raise ValueError('initial_search requires an explicit a_ratio reaction mapping.')
    if cfg.get('stage', 'A') != 'A':
        raise ValueError('Initial feasibility search is allowed only in stage A.')
    def integer(name, default):
        value = options.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('initial_search '+name+' must be a positive integer.')
        return value
    samples = integer('samples', 2000)
    size = integer('batch_size', 32)
    finalists = integer('finalists', 20)
    seed = options.get('seed', 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('initial_search seed must be a nonnegative integer.')
    rng = np.random.default_rng(seed)
    base = getattr(space, 'base', space)
    initial = np.asarray(space.x0, dtype=float).copy()
    # This search is initialization from the original human-provided mechanism.
    if hasattr(space, 'na') and np.any(initial[space.na:] != 0):
        raise ValueError('A-only initialization requires zero Ea and b shifts.')
    selected, low, high = [], [], []
    seen = set()
    for key, limits in options['a_ratio'].items():
        if isinstance(key, bool):
            raise ValueError('Search reaction numbers must be positive integers.')
        number = int(key)
        if str(number) != str(key) or number in seen or number < 1:
            raise ValueError('Search reaction numbers must be distinct positive integers.')
        seen.add(number)
        ratio = np.asarray(limits, dtype=float)
        if ratio.shape != (2,) or not np.isfinite(ratio).all() or not 0 < ratio[0] < ratio[1]:
            raise ValueError(f'R{number}: search A ratios require finite 0 < low < high.')
        indices = [i for i,n in enumerate(base.reaction_numbers) if n == number]
        if not indices:
            raise ValueError(f'R{number} has no active A coordinate.')
        origins = np.asarray(base.source_coordinates)[indices]
        if not np.allclose(initial[indices], origins, rtol=0., atol=1e-8):
            raise ValueError(f'R{number}: initialization must use the original source center.')
        # One shared log multiplier per reaction preserves the supplied PLOG
        # pressure trend during initialization. Later stages retain free nodes.
        lo, hi = np.log(ratio)
        for i, origin in zip(indices, origins):
            lo = max(lo, space.lower[i]-origin)
            hi = min(hi, space.upper[i]-origin)
            if hasattr(space, 'a_low'):
                lo = max(lo, space.a_low[i]-origin)
                hi = min(hi, space.a_high[i]-origin)
        if not np.isfinite([lo,hi]).all() or lo >= hi:
            raise ValueError(f'R{number}: shared search interval has no width inside original bounds.')
        selected.append(indices); low.append(lo); high.append(hi)
    coarse = dict(cfg, steps_per_end=1000, rel_tol=1e-8, abs_tol=1e-12, max_rhs_calls=60000)
    interval = _interval(refs)
    started = time.monotonic()
    candidates = []
    failures = Counter()
    valid_count = 0
    root = Path(root)
    destination = root/'initial_search.json'
    def report(status, checked, **extra):
        record = dict(status=status, samples=checked, requested_samples=samples,
                      valid_coarse=valid_count, elapsed_seconds=time.monotonic()-started,
                      seed=seed, selected_reactions=sorted(seen), a_ratio=options['a_ratio'],
                      coarse_settings={k:coarse[k] for k in ('steps_per_end','rel_tol','abs_tol','max_rhs_calls')},
                      failure_counts=dict(failures),
                      source_anchor='Original input mechanism; no rebasing or rate-form changes', **extra)
        _save(destination,record)
    def score(rows, vector, settings):
        if len(rows) != len(refs):
            failures['incomplete_case_batch'] += 1
            return None
        residual, _ = _score(rows, refs, settings)
        if residual is None:
            for row in rows:
                if 'error' in row:
                    reason = str(row['error'])[:300]
                    if reason not in failures and len(failures) >= 32:
                        reason = 'other_solver_errors'
                    failures[reason] += 1
            return None
        check = _candidate_monotonicity(space,vector,rows,interval,cfg['plog_direction'])
        if not check['valid']:
            failures['kinetic_constraints'] += 1
            return None
        return float(residual@residual)
    for offset in range(0,samples,size):
        count = min(size,samples-offset)
        vectors = np.repeat(initial[None,:],count,axis=0)
        shifts = rng.uniform(low,high,size=(count,len(selected)))
        for column, indices in enumerate(selected):
            vectors[:,indices] = np.asarray(base.source_coordinates)[indices]+shifts[:,column,None]
        for vector in vectors: space.validate(vector)
        rows_batch = batch(vectors, full=False, settings=coarse)
        if len(rows_batch) != count:
            raise RuntimeError('Feasibility batch returned an incomplete candidate set.')
        for vector,rows in zip(vectors,rows_batch):
            loss = score(rows,vector,coarse)
            if loss is not None:
                valid_count += 1
                candidates.append((loss,vector.copy()))
        candidates.sort(key=lambda item:item[0])
        del candidates[finalists:]
        checked = offset+count
        report('screening',checked,best_coarse_coordinates=None if not candidates else candidates[0][1],
               best_coarse_loss=None if not candidates else candidates[0][0])
        print(f'Initial A search: {checked}/{samples} candidates, {valid_count} coarse-valid, '
              f'{time.monotonic()-started:.1f} s elapsed',flush=True)
    best = None
    verified_candidates = []
    # Recheck one at a time: full trajectories never accumulate in memory.
    for index,(_,vector) in enumerate(candidates,1):
        rows = batch([vector],full=True,settings=cfg)[0]
        loss = score(rows,vector,cfg)
        if loss is not None:
            verified_candidates.append(dict(loss=loss, coordinates=vector.tolist()))
        if loss is not None and (best is None or loss < best[0]):
            best = (loss,vector.copy(),rows)
        report('verifying',samples,verified=index,finalists=len(candidates),
               best_coordinates=None if best is None else best[1],
               best_loss=None if best is None else best[0])
        print(f'Initial A search verification: {index}/{len(candidates)}, '
              f'{time.monotonic()-started:.1f} s elapsed',flush=True)
    if best is None:
        report('no_feasible_candidate',samples,verified=len(candidates))
        raise RuntimeError(f'No feasible A-only initialization found; inspect {destination}. '
                           'Adjust the supplied mechanism or explicit search bounds.')
    verified_candidates.sort(key=lambda item: item['loss'])
    _save(root/'verified_candidates.json', verified_candidates)
    report('feasible',samples,verified=len(candidates),coordinates=best[1],loss=best[0])
    return best[1],best[2]
