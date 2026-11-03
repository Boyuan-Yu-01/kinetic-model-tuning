"""Optional order coordinates and fixed reference-state anchors for staged tuning."""
from collections import defaultdict
from copy import deepcopy
import numpy as np


def order_specs(orders):
    """Normalize one-based reaction keys; bounds are actual orders, not multipliers."""
    result = {}
    for key, spec in (orders or {}).items():
        number = int(key)
        if isinstance(key, bool) or str(number) != str(key) or number < 1:
            raise ValueError('Order reaction numbers must be positive integers.')
        spec = deepcopy(spec)
        if set(spec) - {'bounds', 'sum_bounds', 'scale', 'step'} or not spec.get('bounds'):
            raise ValueError('Orders require bounds, with optional sum_bounds, scale and step.')
        for species, bounds in spec['bounds'].items():
            interval = np.asarray(bounds, float)
            if not isinstance(species, str) or interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] >= interval[1]:
                raise ValueError('Each species needs finite, increasing actual order bounds.')
        if spec.get('sum_bounds') is not None:
            lo, hi = spec['sum_bounds']
            if not np.isfinite([lo, hi]).all() or lo > hi:
                raise ValueError('Order sum bounds must be finite and ordered.')
        # A trust step of 1 means a 0.2 change in an order by default.
        spec.setdefault('scale', .2)
        spec.setdefault('step', .02)
        if not np.isfinite([spec['scale'], spec['step']]).all() or min(spec['scale'], spec['step']) <= 0:
            raise ValueError('Order coordinate scale and derivative step must be positive.')
        result[number] = spec
    return result


def anchor_species(orders):
    """Collect only the reactants whose orders are explicitly tunable."""
    return sorted({s for spec in order_specs(orders).values() for s in spec['bounds']})


def state_anchor(gas, species):
    """Snapshot a thermochemical state with concentrations in Cantera SI units."""
    values = {s: float(gas.concentrations[gas.species_index(s)]) for s in species}
    if not np.isfinite([gas.T, *values.values()]).all() or min(gas.T, *values.values()) <= 0:
        raise ValueError('Order anchors require positive temperature and reactant concentrations.')
    return dict(temperature_K=float(gas.T), concentrations_kmol_m3=values)


def combine_anchors(rows, cases):
    """Average reference anchors equally by case within equally weighted families."""
    groups = defaultdict(list)
    for row, case in zip(rows, cases):
        groups[case.get('kind', 'flame')].append(row['order_anchor'])
    if not groups or len(rows) != len(cases):
        raise ValueError('Reference anchors must align with all cases.')
    species = list(rows[0]['order_anchor']['concentrations_kmol_m3'])
    inverse_t, logs = [], []
    for anchors in groups.values():
        ts = np.array([r['temperature_K'] for r in anchors])
        cs = np.array([[r['concentrations_kmol_m3'][s] for s in species] for r in anchors])
        if not np.isfinite(ts).all() or not np.isfinite(cs).all() or np.any(ts <= 0) or np.any(cs <= 0):
            raise ValueError('Cannot construct an anchor from depleted or invalid reference states.')
        inverse_t.append(np.mean(1 / ts))
        logs.append(np.mean(np.log(cs), axis=0))
    return dict(temperature_K=float(1 / np.mean(inverse_t)),
                concentrations_kmol_m3=dict(zip(species, np.exp(np.mean(logs, axis=0)).tolist())),
                definition='Reference LFS max dT/dx; IDT max dT/dt; ZND induction length. Fixed for all stages.',
                case_count=len(rows))


def order_report(space, x):
    """Return actual fitted orders without confusing them with rate multipliers."""
    scheme = space.scheme(x)
    return {n: dict(orders={s: scheme['reactions'][n-1]['orders'][s] for s in spec['bounds']},
                    sum=float(sum(scheme['reactions'][n-1]['orders'][s] for s in spec['bounds'])))
            for n, spec in getattr(space, 'orders', {}).items()}


def prepare_znd_anchor(config, destination):
    """Resolve one reference-only anchor before constructing any ZND stage space."""
    if not config.get('orders') or config.get('order_anchor'):
        return config
    import json
    import tempfile
    from pathlib import Path
    from .iterative_znd import _configuration, _prepare, _solve, _save
    from .run_diagnostics import reference_key
    from .executors import create_executor
    cfg = _configuration(config)
    fingerprint = reference_key(cfg) + json.dumps(config['orders'], sort_keys=True)
    path = Path(destination)
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved.get('fingerprint') == fingerprint:
            return dict(config, order_anchor=saved['anchor'])
    with tempfile.TemporaryDirectory(prefix='mt-znd-anchor-') as temp:
        cfg['output_dir'] = temp
        (Path(temp)/'solver_logs').mkdir()
        with create_executor(cfg['backend'], cfg['workers']) as pool:
            jobs = [(cfg['reference'], p, phi, end, t, cfg)
                    for p,phi,end,t in zip(cfg['P_bar'],cfg['phi'],cfg['t_end'],cfg['T0'])]
            print(f'Preparing fixed order anchor from {len(jobs)} reference ZND cases.',flush=True)
            states = list(pool.map(_prepare,jobs))
            if any('error' in state for state in states):
                raise RuntimeError('Reference ZND anchor preparation failed: '+str(states))
            rows = list(pool.map(_solve, [(cfg['reference'], state, None, cfg, None, False) for state in states]))
            if any('error' in row for row in rows):
                raise RuntimeError('Reference ZND anchor integration failed: '+str(rows))
    anchor = combine_anchors(rows, [{'kind':'znd'}]*len(rows))
    if path.parent.exists():
        _save(path,dict(fingerprint=fingerprint,anchor=anchor))
    return dict(config, order_anchor=anchor)


def rate_penalty_residual(space, x, settings):
    """Optional historical Kyuho penalty, measured in the original YAML units."""
    if not settings or not settings.get('weight', 0.):
        return np.empty(0)
    from .uncertainty import _entries
    weight=float(settings['weight'])
    scales=np.array([settings.get('scale_A',np.log(10)),settings.get('scale_b',.5),settings.get('scale_Ea',2000.)])
    if not np.isfinite([weight,*scales]).all() or weight<0 or np.any(scales<=0):
        raise ValueError('Rate regularization requires a nonnegative weight and positive scales.')
    old=[e for r in space.base._source['reactions'] for e in _entries(r)]
    new=[e for r in space.scheme(x)['reactions'] for e in _entries(r)]
    values=[[np.log(b['A']/a['A']),b['b']-a['b'],b['Ea']-a['Ea']] for a,b in zip(old,new)]
    return (np.sqrt(weight/len(old))*np.asarray(values)/scales).ravel()
