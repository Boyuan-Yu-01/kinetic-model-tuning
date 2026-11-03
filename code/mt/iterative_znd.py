"""Repeated sensitivity-selected A tuning with fixed uncertainty bounds and ZND data.

Use ``launch_iterative_znd(config)`` in an unguarded user script. The lower-level
``run_iterative_znd(config)`` requires the usual multiprocessing main guard.
"""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
import warnings
import numpy as np
import cantera as ct
from ._execution import launcher_environment, wait_for_process
from .executors import create_executor
from .plot_settings import yscale
from .history import json_value
from .starting_points import _prepare_case
from .staged_znd import _work_limit
from .znd_functions import _sensitivity_lengths
from .uncertainty import RateUncertaintySpace, plog_monotonicity
from .iterative_sensitivity import iterative_sensitivity_optimize

_GASES = OrderedDict()
DEFAULTS = dict(stagnation_rounds=3, stagnation_relative_tolerance=1e-5, fit_targets=None, plog_shape_free=False, backend='process', max_tasks_per_worker=None, workers=8, max_rounds=20, inner_iterations=15, top_reactions=5,
                threshold=.01, steps=.01, profile_points=100, profile_fraction=.05,
                temperature_scale=25., length_factor=1.2, steps_per_end=4000,
                rel_tol=1e-10, abs_tol=1e-14, interpolation='cubic', max_rhs_calls=100000,
                strict_margin=1e-9, tune_ea=False, validate=True, plog_direction='either', plog_free=None, unbounded_reactions=[],
                sensitivity_retries=3, shrink_factor=.5, minimum_step=1e-5, compact_output=True, diagnostics=False, save_sensitivity_plots=False)


def _save(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(json_value(data), indent=2, allow_nan=False))
    temp.replace(path)


def _export(space, path, x):
    """Validate a complete YAML before atomically replacing an accepted model."""
    path = Path(path); temporary = path.with_suffix('.tmp.yaml')
    space.export(temporary,x)
    ct.Solution(str(temporary))
    temporary.replace(path)


def _bound_values(values):
    """Distinguish intentionally unbounded coordinates from unresolved diagnostics."""
    return ['-Infinity' if v == -np.inf else 'Infinity' if v == np.inf else float(v)
            for v in values]


def _registry(cfg):
    if cfg.get('staged_parameters', False):
        from .staged_parameters import StagedParameters
        factory = lambda *a, **k: StagedParameters(*a, **k,
            a_ratio=cfg['a_ratio'], ea_delta=cfg['ea_delta'], b_delta=cfg['b_delta'],
            pivot_temperature=cfg.get('pivot_temperature',1800.), plog_shape_free=cfg.get('plog_shape_free',False),
            stage=cfg['stage'], center=cfg.get('stage_center'),
            stage_a_ratio=cfg['stage_a_ratio'], stage_ea_delta=cfg['stage_ea_delta'],
            orders=cfg.get('orders'),order_anchor=cfg.get('order_anchor'),rate_bounds=cfg.get('rate_bounds'))
    elif cfg.get('tune_ea', False):
        from .ae_parameters import AEParameters
        factory = lambda *a, **k: AEParameters(*a, **k, a_ratio=cfg.get('a_ratio', (.5,2.)),
            ea_delta=cfg.get('ea_delta', 60000.), pivot_temperature=cfg.get('pivot_temperature',1800.), plog_shape_free=cfg.get('plog_shape_free',False))
    else:
        factory = RateUncertaintySpace
    return factory(cfg['model'], cfg['reference'], cfg.get('manifest'),
        strict_margin=cfg['strict_margin'], custom_factors=cfg.get('custom_factors'),
        unbounded_reactions=cfg.get('unbounded_reactions',[]),
        expected_factors=cfg.get('expected_factors'),
        uncertainty_factors=cfg.get('uncertainty_factors'), plog_free=cfg.get('plog_free'))


def _prepare(job):
    source, p, phi, end, temperature, cfg = job
    with (Path(cfg['output_dir'])/'solver_logs'/f'worker_{os.getpid()}.log').open('a') as log:
        with redirect_stdout(log), redirect_stderr(log):
            return _prepare_case((source, p*1e5, temperature, phi, end, cfg['fuel'], cfg['oxidizer']))


def _solve(job):
    """Restore exact states and reset every multiplier before each candidate."""
    source, state, vector, cfg, reference, full = job
    from sdtoolbox.znd import zndsolve
    with (Path(cfg['output_dir'])/'solver_logs'/f'worker_{os.getpid()}.log').open('a') as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                key = (source, cfg['model'], cfg['reference'], cfg.get('manifest'),
                       json.dumps(cfg.get('uncertainty_factors'), sort_keys=True),
                       json.dumps(cfg.get('custom_factors'), sort_keys=True), cfg['strict_margin'],
                       tuple(cfg.get('unbounded_reactions',[])), tuple(cfg.get('plog_free') or []), cfg.get('tune_ea',False),
                       tuple(cfg.get('a_ratio',(.5,2.))), cfg.get('ea_delta',60000.),cfg.get('pivot_temperature',1800.),
                       cfg.get('staged_parameters',False),cfg.get('plog_shape_free',False),cfg.get('stage'),cfg.get('b_delta'),
                       tuple(cfg.get('stage_center') or []),tuple(cfg.get('stage_a_ratio',[])),cfg.get('stage_ea_delta'),
                       json.dumps(cfg.get('orders'),sort_keys=True),json.dumps(cfg.get('order_anchor'),sort_keys=True),json.dumps(cfg.get('rate_bounds'),sort_keys=True))
                if key not in _GASES:
                    _GASES[key] = (ct.Solution(source), ct.Solution(source),
                                   _registry(cfg) if vector is not None else None)
                    if len(_GASES)>4: _GASES.popitem(last=False)
                _GASES.move_to_end(key)
                gas, upstream, registry = _GASES[key]
                gas.state, upstream.state = state['frozen'], state['upstream']
                gas.set_multiplier(1.); upstream.set_multiplier(1.)
                if vector is not None:
                    if registry is None:
                        registry = _registry(cfg)
                        _GASES[key] = gas, upstream, registry
                    gas = registry.apply(gas, vector)
                end = state['case']['t_end']
                with warnings.catch_warnings(), _work_limit(cfg['max_rhs_calls']):
                    warnings.simplefilter('error')
                    out = zndsolve(gas, upstream, state['cj_speed'], t_end=end,
                        max_step=end/cfg['steps_per_end'], relTol=cfg['rel_tol'],
                        absTol=cfg['abs_tol'], Method='LSODA', advanced_output=False)
                    lengths = _sensitivity_lengths(out, end, interpolation=cfg['interpolation'])
                distance, temperature = np.asarray(out['distance']), np.asarray(out['T'])
                if (len(distance)<2 or not np.isfinite(distance).all() or
                    not np.isfinite(temperature).all() or np.any(np.diff(distance)<0) or
                    distance[0]>0 or np.min(temperature)<=0):
                    raise ValueError('Invalid trajectory or distance coverage.')
                row = dict(lengths, T_final=float(temperature[-1]),
                           T_min_K=float(temperature.min()), T_max_K=float(temperature.max()))
                if not all(np.isfinite(row[k]) and row[k]>0 for k in
                           ('T_final','ind_len_ZND','exo_len_ZND')):
                    raise ValueError('Unresolved final temperature or characteristic lengths.')
                if reference is not None:
                    if distance[-1]<reference['grid'][-1]:
                        raise ValueError('Profile does not cover the fixed reference grid.')
                    row['grid_T'] = _profile_values(distance, temperature, reference['grid'], cfg['interpolation'])
                if full:
                    for name in ('distance','T','P','time','thermicity'):
                        value = np.asarray(out[name])
                        if not np.isfinite(value).all(): raise ValueError('Nonfinite '+name)
                        row[name] = value
                if vector is None and cfg.get('orders'):
                    from .order_tuning import anchor_species, state_anchor
                    position=lengths['ind_len_ZND']
                    y=np.array([np.interp(position,distance,v) for v in out['species']])
                    gas.TPY=np.interp(position,distance,temperature),np.interp(position,distance,out['P']),y
                    row['order_anchor']=state_anchor(gas,anchor_species(cfg['orders']))
                return row
            except Exception as exc:
                return dict(error=f'{type(exc).__name__}: {exc}')


def _profile_values(distance, temperature, grid, interpolation):
    """Sample profiles smoothly without extrapolating beyond solved distances."""
    if interpolation == 'linear':
        return np.interp(grid, distance, temperature)
    from scipy.interpolate import CubicSpline
    return CubicSpline(distance, temperature, extrapolate=False)(grid)


def _score(rows, refs, cfg):
    if any('error' in row for row in rows): return None, []
    residual, metrics = [], []
    names = ('T_profile', 'T_ZND', 'ind_len_ZND', 'exo_len_ZND')
    selected = cfg.get('objective_names', names)
    mask = np.array([name in selected for name in names], dtype=float)
    denominator = len(selected)*len(rows)
    for row, ref in zip(rows, refs):
        diff = np.asarray(row['grid_T'])-ref['grid_T']
        profile = diff/ref['profile_scale']
        temperature = (row['T_final']-ref['T_final'])/cfg['temperature_scale']
        length = np.array([np.log(row[k]/ref[k])/(1. if cfg['length_factor'] is None else np.log(cfg['length_factor']))
                           for k in ('ind_len_ZND','exo_len_ZND')])
        residual.extend(mask[0]*profile/np.sqrt(denominator*len(profile)))
        residual.extend(mask[1:]*np.r_[temperature,length]/np.sqrt(denominator))
        metrics.append(dict(T_final=row['T_final'], T_error_K=row['T_final']-ref['T_final'],
            profile_rmse_K=float(np.sqrt(np.mean(diff**2))),
            ind_len_ZND=row['ind_len_ZND'], exo_len_ZND=row['exo_len_ZND'],
            components=(mask*np.r_[np.mean(profile**2),temperature**2,length**2]*4/len(selected)).tolist()))
    residual = np.asarray(residual)
    return (residual,metrics) if np.isfinite(residual).all() else (None,[])


def _interval(rows, fallback=()):
    values = list(fallback)
    for row in rows:
        if 'error' not in row: values.extend([row['T_min_K'],row['T_max_K']])
    if not values or not np.isfinite(values).all() or min(values)<=0:
        raise ValueError('No valid temperature interval.')
    return [float(min(values)),float(max(values))]


def _candidate_monotonicity(space, coordinates, rows, interval, direction):
    """Reject numerically unrepresentable trial kinetics without ending a run."""
    try:
        temperatures = _interval(rows,interval)
        result = plog_monotonicity(space.scheme(coordinates),*temperatures,direction=direction)
        if hasattr(space, 'rate_envelope'):
            result['pressure_monotonicity_valid'] = result['valid']
            result['rate_envelope'] = space.rate_envelope(coordinates,*temperatures)
            result['valid'] &= result['rate_envelope']['valid']
        return result
    except (ValueError, OverflowError, ct.CanteraError, FloatingPointError) as exc:
        return dict(valid=False,error=f'{type(exc).__name__}: {exc}',
                    status='candidate_rate_validation_failed')


def _plots(root, refs, baseline, best, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def draw(ax, i, title):
        extent=0.
        for row, label in ((refs[i], 'FFCM2'), (baseline[i], 'Stage entry'), (best[i], 'Tuned')):
            xx,tt=np.asarray(row['distance']),np.asarray(row['T'])
            hits=np.flatnonzero(tt>=tt[0]+.95*(tt.max()-tt[0]))
            extent=max(extent,1.4*xx[hits[0]] if len(hits) else xx[-1])
            ax.plot(xx * 1000, tt, label=label)
        ax.set_yscale(yscale(cfg.get('plot_yscales'), 'T_profile'))
        ax.set(xlabel='Distance (mm)', ylabel='Temperature (K)', title=title,
               xlim=(0, extent * 1000))
        ax.legend()
        ax.grid(alpha=.2)

    n = len(refs)
    columns = min(3, n)
    titles = [f"{cfg['P_bar'][i]:g} bar, phi={cfg['phi'][i]:g}" for i in range(n)]
    fig, axes = plt.subplots((n+columns-1)//columns, columns, squeeze=False,
                            figsize=(5*columns, 3.6*((n+columns-1)//columns)))
    for i, ax in enumerate(axes.flat):
        if i < n:
            draw(ax, i, titles[i])
        else:
            ax.set_visible(False)
    fig.tight_layout(); fig.savefig(root/'profiles.png', dpi=180); plt.close(fig)
    folder = root/'temperature_profiles'; folder.mkdir(exist_ok=True)
    for i, title in enumerate(titles):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        draw(ax, i, f'Case {i+1}: {title}')
        fig.tight_layout()
        fig.savefig(folder/f'case_{i+1:02d}.png', dpi=180); plt.close(fig)


def _round_text(info, previous_loss, max_rounds):
    """Describe accepted progress, sensitivity reliability, and stopping separately."""
    changes = info.get('relative_changes', [])
    maximum = max((v for v in changes if v is not None), default=0.)
    change = 'unavailable' if any(v is None for v in changes) else f'{100*maximum:.2f}%'
    improvement = (f'{100*(previous_loss-info["loss"])/previous_loss:.4g}%'
                   if previous_loss > 0 else 'n/a')
    ranking = info['screen']['ranking']
    unresolved = [f'R{row["group"]}' for row in ranking if not row['all_resolved']]
    selected = ', '.join(f'R{v}' for v in info['selected_groups']) or 'none'
    outcomes = {
        'iterating': 'Continuing to the next round.',
        'maximum_rounds': 'Stopped: round limit reached; NOT CONVERGED.',
        'unresolved_sensitivities': 'Stopped: unresolved sensitivities prevent further reliable progress; NOT CONVERGED.',
        'parameter_change': 'CONVERGED: parameter-change and sensitivity checks passed.',
        'loss_plateau': 'Stopped: loss plateau detected across consecutive rounds; NOT CONVERGED.',
        'stalled': 'Stopped: no accepted improvement after the full audit; NOT CONVERGED.',
    }
    lines = [f'\nRound {info["round"]}/{max_rounds}',
             f'  Selected reactions: {selected}',
             f'  Loss: {previous_loss:.8g} -> {info["loss"]:.8g} ({improvement} reduction)',
             f'  Largest normalized parameter change: {change} (target: < {100*info["threshold"]:g}%)',
             f'  Sensitivities: {len(ranking)-len(unresolved)}/{len(ranking)} reactions resolved']
    if unresolved:
        lines.append('  Unresolved: ' + ', '.join(unresolved) +
                     ('; retrying next round.' if info['status'] == 'iterating' else '.'))
    if info.get('all_free_audit'):
        lines.append('  Additional optimization audit: all eligible reactions.')
    if info.get('inner_status'):
        lines.append('  Inner optimizer: '+str(info['inner_status']))
    for blocker in info.get('convergence_blockers',[]):
        lines.append('  Convergence blocked: '+blocker['reason'].replace('_',' ')+((': '+str(blocker['status'])) if 'status' in blocker else ''))
    for row in info.get('update_audit',{}).get('largest_A_updates',[]):
        lines.append(f"  R{row['reaction']} entry {row['entry']} ({row['pressure'] or 'Arrhenius'}): A {row['old_A']:.4e} -> {row['new_A']:.4e} ({row['multiplier']:.4g}x); sampled {row.get('rate_comparison','rate_constant')} ratio {row['sampled_k_ratio_min']:.4g}–{row['sampled_k_ratio_max']:.4g}")
    if 'stagnation_count' in info:
        lines.append(f"  Consecutive negligible improvements: {info['stagnation_count']}")
    lines.append('  ' + outcomes.get(info['status'], f'Stopped: {info["status"]}; NOT CONVERGED.'))
    return '\n'.join(lines)


def _configuration(config):
    cfg = deepcopy(DEFAULTS); cfg.update(deepcopy(config))
    for key in ('model','reference','output_dir'):
        cfg[key] = str(Path(cfg[key]).resolve())
    cfg['reference_cache'] = str(Path(cfg.get('reference_cache') or (Path(cfg['output_dir']).parent/'reference_cache')).resolve())
    cfg['manifest'] = str(Path(cfg['manifest']).resolve()) if cfg.get('manifest') is not None else None
    if cfg['interpolation'] not in ('linear', 'cubic'):
        raise ValueError('interpolation must be linear or cubic.')
    for key in ('compact_output', 'diagnostics', 'save_sensitivity_plots', 'tune_ea', 'plog_shape_free'):
        if not isinstance(cfg[key], bool):
            raise ValueError(key + ' must be True or False.')
    if cfg['backend'] not in ('process', 'openmp'):
        raise ValueError('backend must be process or openmp.')
    n = len(cfg['P_bar'])
    if not n or len(cfg['phi'])!=n or len(cfg['t_end'])!=n:
        raise ValueError('P_bar, phi and t_end must be nonempty equal-length arrays.')
    cfg['T0'] = np.broadcast_to(np.asarray(cfg['T0'],dtype=float),(n,)).tolist()
    for key in ('P_bar','phi','t_end','T0'):
        if not np.isfinite(cfg[key]).all() or np.any(np.asarray(cfg[key])<=0):
            raise ValueError(key+' must contain positive finite values.')
    for key in ('workers','profile_points','steps_per_end','max_rhs_calls'):
        if isinstance(cfg[key],bool) or not isinstance(cfg[key],int) or cfg[key]<1:
            raise ValueError(key+' must be a positive integer.')
    if cfg['profile_points']<2: raise ValueError('At least two profile points required.')
    for key in ('profile_fraction','temperature_scale','rel_tol','abs_tol'):
        if not np.isfinite(cfg[key]) or cfg[key]<=0: raise ValueError('Invalid '+key)
    if cfg['length_factor'] is not None and (not np.isfinite(cfg['length_factor']) or cfg['length_factor']<=1):
        raise ValueError('length_factor must exceed one.')
    return cfg


def run_iterative_znd(config):
    """Run optimization with compact public results and private working records."""
    cfg = _configuration(config)
    public = Path(cfg['output_dir'])
    public.mkdir(parents=True, exist_ok=True)
    if any((public/name).exists() for name in
           ('settings.json', 'summary.json', 'checkpoint.json', 'results.yaml', '.work')):
        raise FileExistsError('Choose a fresh output directory; existing results are retained.')
    root = public / '.work' if cfg['compact_output'] else public
    root.mkdir(exist_ok=True)
    if cfg['compact_output']:
        cfg['_public_output_dir'] = str(public)
        cfg['output_dir'] = str(root)
    (root/'solver_logs').mkdir(exist_ok=True)
    _save(root/'settings.json', cfg)
    begin = time.monotonic()
    print(f'Preparing iterative ZND run: {public}', flush=True)
    try:
        report = _run(cfg, root, begin)
    except BaseException as exc:
        _save(root/'failure.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                                      elapsed_seconds=time.monotonic()-begin))
        if cfg['compact_output']:
            # Preserve all working evidence if the optimization did not finish.
            _publish_results(root, public)
        raise
    if cfg['compact_output'] and not cfg.get('exploration_only', False):
        from .iterative_output import finalize_iterative_output
        finalize_iterative_output(root, public, diagnostics=cfg['diagnostics'], cleanup=True)
    return report


def _publish_results(work, public):
    """Keep a readable accepted-state snapshot without disturbing solver files."""
    from .iterative_output import write_iterative_results
    try:
        write_iterative_results(work, public)
    except Exception as exc:
        # A report failure must not discard an already saved numerical checkpoint.
        print(f'Could not refresh results.yaml: {exc}; working records are retained.', flush=True)


def _run(cfg, root, begin):
    space = _registry(cfg)
    _save(root/'uncertainty_audit.json',space.audit)
    _save(root/'bounds.json',dict(reaction_numbers=space.reaction_numbers,
          coordinates=('Pivot log-rate and scaled temperature-dependence coordinates; see labels' if cfg['tune_ea'] else 'Natural-log A multipliers, nominal-centered when matched and source-centered otherwise.'),
          coordinate_labels=space.coordinate_labels, plog_free=cfg.get('plog_free'),
          source_coordinates=space.source_coordinates, initial_coordinates=space.x0,
          lower=_bound_values(space.lower), upper=_bound_values(space.upper),
          bound_encoding='Finite bounds are numbers; -Infinity/Infinity strings denote intentional mathematical bounds, not missing or failed values. A and simulation multipliers must remain positive finite machine values.',
          unbounded_reactions=cfg.get('unbounded_reactions',[]),strict_margin=cfg['strict_margin']))
    _save(root/'provenance.json',dict(python=sys.version,cantera=ct.__version__,
        inputs={k:dict(path=cfg[k],sha256=hashlib.sha256(Path(cfg[k]).read_bytes()).hexdigest())
                for k in ('model','reference','manifest') if cfg.get(k) is not None}))
    from .run_diagnostics import reference_key, update_audit, active_bounds, fit_quality
    n = len(cfg['P_bar']); cache = OrderedDict(); evaluations = 0
    cache_path=Path(cfg['reference_cache'])/(reference_key(cfg)+'.json')
    cached=json.loads(cache_path.read_text()) if cache_path.is_file() else None
    with create_executor(cfg['backend'], cfg['workers'], max_tasks_per_worker=cfg['max_tasks_per_worker']) as pool:
        jobs = [(source,p,phi,end,T,cfg) for source in (cfg['reference'],cfg['model'])
                for p,phi,end,T in zip(cfg['P_bar'],cfg['phi'],cfg['t_end'],cfg['T0'])]
        states = (cached['states'] + list(pool.map(_prepare,jobs[n:])) if cached is not None
                  else list(pool.map(_prepare,jobs)))
        _save(root/'prepared_states.json',states)
        if any('error' in state for state in states): raise RuntimeError('Thermodynamic preparation failed; see prepared_states.json.')
        ref_states, source_states = states[:n], states[n:]
        def batch(vectors, *, full=False, source=None, settings=None, reference=None, prepared=None):
            rows = [[None]*n for _ in vectors]
            futures = {pool.submit(_solve,(source or cfg['model'],state,vector,settings or cfg,
                                            None if ref is None else {'grid':ref['grid']},full)):(j,i)
                for j,vector in enumerate(vectors)
                for i,(state,ref) in enumerate(zip(prepared or source_states,
                    reference if reference is not None else refs))}
            for future in as_completed(futures):
                j,i = futures[future]; rows[j][i] = future.result()
            return rows
        print(f"{'Reusing cached' if cached is not None else 'Computing'} {n} reference profiles...",flush=True)
        refs = cached['profiles'] if cached is not None else batch([None],full=True,source=cfg['reference'],prepared=ref_states,reference=[None]*n)[0]
        if any('error' in row for row in refs):
            _save(root/'reference_profiles.json',refs); raise RuntimeError('Reference integration failed.')
        if cached is None:
            cache_path.parent.mkdir(parents=True,exist_ok=True)
            _save(cache_path,dict(states=ref_states,profiles=refs))
        for ref in refs:
            temperature, distance = np.asarray(ref['T']), np.asarray(ref['distance'])
            rise = ref['T_final']-temperature[0]
            hits = np.flatnonzero(temperature>=temperature[0]+.95*rise)
            if rise<=0 or not len(hits): raise ValueError('Reference temperature rise is unresolved.')
            xmax = 1.2*float(distance[hits[0]])
            if xmax<=0 or xmax>distance[-1]: raise ValueError('Reference must cover 1.2 times the 95% rise distance.')
            ref.update(grid=np.linspace(0,xmax,cfg['profile_points']),profile_scale=cfg.get('profile_scale_K', cfg['profile_fraction']*rise))
            ref['grid_T'] = _profile_values(distance, temperature, ref['grid'], cfg['interpolation'])
        print(f'Checking {n} stage-entry cases...',flush=True)
        baseline = batch([space.x0],full=True)[0]
        initial_r,initial_metrics = _score(baseline,refs,cfg)
        _save(root/'reference_profiles.json',refs); _save(root/'baseline_profiles.json',baseline)
        if initial_r is None or cfg.get("exploration_only", False):
            if not cfg.get('initial_search'):
                raise RuntimeError('Initial mechanism failed ZND evaluation; supply a feasible model or enable initial_search.')
            _save(root/'original_baseline_failures.json',baseline)
            from .feasibility import find_feasible
            start, baseline = find_feasible(batch,space,refs,cfg,root)
            space.x0 = np.asarray(start,dtype=float)
            initial_r,initial_metrics = _score(baseline,refs,cfg)
            _save(root/'baseline_profiles.json',baseline)
            _save(root/'feasible_start.json',dict(coordinates=start,loss=float(initial_r@initial_r)))
            _export(space,root/'feasible_start.yaml',space.x0)
        if cfg.get('exploration_only', False):
            return dict(status='explored')
        interval = _interval(refs+baseline,cfg.get('temperature_interval',()))
        check = _candidate_monotonicity(space,space.x0,baseline,interval,cfg['plog_direction'])
        _save(root/'initial_monotonicity.json',dict(temperature_range_K=interval,
            range_source='Full reference and source-baseline trajectories',
            model=cfg['model'],monotonicity=check,
            note='Constraint applies to the reduced candidate; reference pressure dependence is not constrained.'))
        if not check['valid']:
            raise ValueError('Initial kinetic constraints failed; see initial_monotonicity.json.')
        print(f"Baseline loss {float(initial_r@initial_r):.8g}; {np.count_nonzero(space.upper > space.lower)} active / {len(space.x0)} total parameter coordinates, {n} cases, {cfg['workers']} workers.",flush=True)
        def evaluate(vectors):
            nonlocal evaluations
            keys = [tuple(map(float,vector)) for vector in vectors]
            missing = list(dict.fromkeys(key for key in keys if key not in cache))
            valid = []
            for key in missing:
                try: space.validate(key); valid.append(key)
                except (ValueError, OverflowError, ct.CanteraError, FloatingPointError) as exc:
                    cache[key] = (None,[],dict(valid=False,error=f'{type(exc).__name__}: {exc}',status='candidate_coordinate_validation_failed'))
            for key,rows in zip(valid,batch(valid) if valid else []):
                residual,metrics = _score(rows,refs,cfg)
                check = _candidate_monotonicity(space,key,rows,interval,cfg['plog_direction'])
                # Independent-coordinate probes may leave the monotonic set.
                # Only accepted updates/checkpoints must satisfy monotonicity.
                if not check['valid'] and not (space.plog_free or cfg['tune_ea']): residual = None
                cache[key] = residual,metrics,check; evaluations += 1
                record = dict(evaluation=evaluations,coordinates=key,
                    loss=None if residual is None else float(residual@residual),
                    monotonicity=check,errors=[row.get('error') for row in rows],
                    candidate_error=check.get('error'))
                if not cfg.get('compact_output', False) or cfg.get('diagnostics', False):
                    with (root/'evaluations.jsonl').open('a') as stream:
                        stream.write(json.dumps(json_value(record),allow_nan=False)+'\n')
            result = [cache[key][0] for key in keys]
            for key in keys: cache.move_to_end(key)
            while len(cache)>256: cache.popitem(last=False)
            return result
        branches = space.linear_monotonicity(*interval, direction=cfg['plog_direction']) if (space.plog_free or cfg['tune_ea']) else None
        def feasible(x):
            key = tuple(map(float, x))
            if key not in cache: evaluate([x])
            return bool(cache[key][0] is not None and cache[key][2]['valid'])

        def checkpoint(x, number, info=None):
            residual = evaluate([x])[0]
            if residual is None or not feasible(x): raise RuntimeError('Accepted state failed feasibility evaluation.')
            key = tuple(map(float,x)); _,metrics,check = cache[key]
            _export(space,root/'best_model.yaml',x)
            record = dict(round=number,loss=float(residual@residual),coordinates=x,
                reaction_numbers=space.reaction_numbers, coordinate_labels=space.coordinate_labels,
                rate_multipliers_from_source=(space.reported_multipliers(x) if cfg['tune_ea'] else np.exp(np.asarray(x)-space.source_coordinates)),
                relative_change_from_source=(None if cfg['tune_ea'] else np.expm1(np.asarray(x)-space.source_coordinates)),
                physical_parameter_fields=list(getattr(space,'physical_fields',('A',))),
                physical_parameters=space.physical_parameters(x),
                order_parameter_keys=[dict(reaction=n,species=s) for n,s in getattr(space,'order_indices',{})],
                physical_parameter_meaning='Physical entries in reaction/entry/field order; adjustable fields declared above.',
                cases=metrics,monotonicity=check,evaluations=evaluations)
            _save(root/'checkpoint.json',record)
            if info is not None:
                folder = root/f'round_{number:03d}'; folder.mkdir(exist_ok=True)
                _export(space,folder/'model.yaml',x)
                _save(folder/'accepted.json',record); _save(folder/'sensitivity.json',info)
            if cfg.get('_public_output_dir'):
                _publish_results(root, Path(cfg['_public_output_dir']))
        checkpoint(space.x0,0)
        accepted_residual = evaluate([space.x0])[0]
        previous_loss = float(accepted_residual @ accepted_residual)
        def on_round(info):
            nonlocal previous_loss, branches
            space.x0=np.asarray(info['coordinates']).copy()
            if branches is not None:
                branches=space.linear_monotonicity(*interval,direction=cfg['plog_direction'])
            info['update_audit']=update_audit(space,np.asarray(info['before_coordinates']),np.asarray(info['coordinates']),interval)
            info['active_bounds']=active_bounds(space,np.asarray(info['coordinates']))
            print(_round_text(info, previous_loss, cfg['max_rounds']), flush=True)
            print(f"  Active parameter bounds: {len(info['active_bounds'])} (details in rounds.yaml)",flush=True)
            previous_loss = info['loss']
            info = dict(info,nonfinite_diagnostic_encoding='null means nonfinite/unresolved, not zero; consult column status and reliability.')
            checkpoint(np.asarray(info['coordinates']),info['round'],info)
        if cfg['save_sensitivity_plots']:
            from .iterative_output import plot_round_sensitivity
            destination = Path(cfg.get('_public_output_dir', root)) / 'sensitivity'
            equations = {i: row['equation'] for i, row in
                         enumerate(space.scheme(space.x0)['reactions'], 1)}

        def on_round_start(event):
            plot_round_sensitivity(event, destination, equations)
            if space.plog_free or cfg['tune_ea']:
                from .iterative_output import plot_coordinate_sensitivity
                plot_coordinate_sensitivity(event['screen'], space.coordinate_labels, cfg,
                    destination / f"round_{event['round']:03d}")

        def on_sensitivity_retry(event):
            names = ', '.join(f'R{number}' for number in event['unresolved_groups'])
            before, after = max(event['previous_steps']), max(event['next_steps'])
            print(f"  Sensitivity retry {event['attempt']}/{event['limit']}: {names} unresolved; "
                  f"scaling all base perturbations by {event['shrink_factor']:g} "
                  f"(largest {before:g} -> {after:g}).", flush=True)
        groups = list(space.reaction_numbers)
        optimization = iterative_sensitivity_optimize(evaluate,space.x0,space.lower,space.upper,groups,
            physical_parameters=space.physical_parameters, parameter_scales=getattr(space,'parameter_scales',None), max_rounds=cfg['max_rounds'],
            stagnation_rounds=cfg['stagnation_rounds'],stagnation_relative_tolerance=cfg['stagnation_relative_tolerance'],
            inner_iterations=cfg['inner_iterations'],top_reactions=cfg['top_reactions'],
            threshold=cfg['threshold'],steps=cfg['steps'],options=cfg.get('options'),on_round=on_round,
            sensitivity_retries=cfg['sensitivity_retries'],shrink_factor=cfg['shrink_factor'],
            minimum_step=cfg['minimum_step'],on_sensitivity_retry=on_sensitivity_retry,
            on_round_start=on_round_start if cfg['save_sensitivity_plots'] else None,
            constraints=(lambda: branches) if branches is not None else None,
            feasible=feasible if (space.plog_free or cfg['tune_ea']) else None)
        x = np.asarray(optimization['bestx'])
        if cfg['save_sensitivity_plots'] and (space.plog_free or cfg['tune_ea']):
            from .iterative_output import plot_coordinate_sensitivity
            plot_coordinate_sensitivity(optimization['screen'], space.coordinate_labels, cfg, destination / 'final')
        _save(root/'initial_sensitivity.json',dict(screen=optimization.get('initial_screen'),nonfinite_diagnostic_encoding='null means nonfinite/unresolved, not zero; consult column status and reliability.'))
        checkpoint(x,len(optimization.get('rounds',[])))
        profiles = batch([x],full=True)[0]; final_r,metrics = _score(profiles,refs,cfg)
        _save(root/'best_profiles.json',profiles)
        if final_r is None: raise RuntimeError('Final accepted mechanism failed integration.')
        validation = dict(performed=False,passed=None)
        validated = []
        if cfg['validate']:
            tight = dict(cfg,steps_per_end=4*cfg['steps_per_end'],
                         rel_tol=cfg['rel_tol']/10,abs_tol=cfg['abs_tol']/10)
            validated = batch([None],full=True,source=str(root/'best_model.yaml'),settings=tight)[0]
            _save(root/'validation_profiles.json',validated)
            tight_r,_ = _score(validated,refs,cfg)
            validation = dict(performed=True,passed=False,errors=[r.get('error') for r in validated])
            if tight_r is not None:
                temperature_error = max(abs(a['T_final']-b['T_final']) for a,b in zip(profiles,validated))
                length_error = max(abs(b[k]/a[k]-1)*100 for a,b in zip(profiles,validated)
                                   for k in ('ind_len_ZND','exo_len_ZND'))
                validation.update(passed=bool(temperature_error<.1 and length_error<1.),
                    max_temperature_change_K=temperature_error,max_length_change_percent=length_error,
                    tighter_loss=float(tight_r@tight_r),reference_recomputed=False)
        space.validate(x)
        exported, expected = ct.Solution(str(root/'best_model.yaml')), space.make_gas(x)
        export_error = 0.
        for temperature in np.linspace(*_interval(profiles,interval),3):
            for pressure in (1e5,1e6,1e8):
                exported.TP = expected.TP = temperature,pressure
                a,b = exported.forward_rate_constants,expected.forward_rate_constants
                export_error = max(export_error,float(np.max(abs(a-b)/np.maximum(abs(b),1e-300))))
        if not np.isfinite(export_error) or export_error>1e-10:
            raise RuntimeError('Exported mechanism does not reproduce the accepted rate multipliers.')
        check = plog_monotonicity(str(root/'best_model.yaml'),*_interval(profiles+validated,interval),direction=cfg['plog_direction'])
        if hasattr(space,'rate_envelope'):
            check['pressure_monotonicity_valid'] = check['valid']
            check['rate_envelope'] = space.rate_envelope(x,*_interval(profiles+validated,interval))
            check['valid'] &= check['rate_envelope']['valid']
        has_plog=any(isinstance(r.rate,ct.PlogRate) for r in exported.reactions())
        check['applicable']=has_plog
        quality=fit_quality(metrics,refs,cfg.get('fit_targets'))
        _plots(root,refs,baseline,profiles,cfg)
        report = dict(status=('converged' if optimization['converged'] else 'not_converged')
                      if check['valid'] and validation['passed'] is not False else 'validation_failed',
            fit_quality=quality,active_bounds=active_bounds(space,x),
            baseline_loss=float(initial_r@initial_r),best_loss=float(final_r@final_r),
            baseline_cases=initial_metrics,cases=metrics,coordinates=x,optimization=optimization,
            validation=validation,monotonicity=check,export_max_relative_rate_error=export_error,evaluations=evaluations,
            elapsed_seconds=time.monotonic()-begin,
            note='Accepted local optimizer state; sensitivity probes never become the saved best implicitly.')
        _save(root/'summary.json',report)
        total_reduction = (100*(report['baseline_loss']-report['best_loss'])/report['baseline_loss']
                           if report['baseline_loss'] > 0 else 0.)
        validation_text = ('PASSED' if validation['passed'] is True else
                           'FAILED' if validation['passed'] is False else 'not requested')
        destination = Path(cfg.get('_public_output_dir', root))
        summary_name = 'results.yaml' if cfg.get('_public_output_dir') else 'summary.json'
        print(f"\nOptimization result: {report['status'].upper()}\n"
              f"  Rounds: {len(optimization['rounds'])}/{cfg['max_rounds']}\n"
              f"  Stop reason: {optimization['status'].replace('_', ' ')}\n"
              f"  Loss: {report['baseline_loss']:.8g} -> {report['best_loss']:.8g} "
              f"({total_reduction:.2f}% reduction)\n"
              f"  Solver validation: {validation_text}\n"
              f"  PLOG monotonicity: {('PASSED' if check.get('pressure_monotonicity_valid',check['valid']) else 'FAILED') if has_plog else 'not applicable'}\n"
              f"  Best model: {destination/'best_model.yaml'}\n"
              f"  Summary: {destination/summary_name}", flush=True)
        print('  Fit errors: '+', '.join(f'{k}={v:.4g}' for k,v in quality['mean_absolute_errors'].items()),flush=True)
        print('  Objective loss contributions: '+', '.join(f'{k}={v:.5g}' for k,v in quality['loss_contributions'].items()),flush=True)
        print('  Fit targets: '+('not specified' if quality['passed'] is None else 'PASSED' if quality['passed'] else 'NOT MET'),flush=True)
        return report


def launch_iterative_znd(config):
    """Run safely from a plain script without a user-written __main__ guard."""
    cfg = _configuration(config)
    with tempfile.TemporaryDirectory(prefix='iterative_znd_') as folder:
        request = Path(folder)/'request.json'; _save(request,cfg)
        process = subprocess.Popen([sys.executable,'-u','-m','mt.iterative_znd',str(request)],
                                   env=launcher_environment(),start_new_session=os.name=='posix')
        code = wait_for_process(process)
        if code:
            raise RuntimeError(f"Iterative ZND exited with status {code}; inspect {cfg['output_dir']} for results and working diagnostics.")
    if cfg['compact_output']:
        import yaml
        return yaml.safe_load((Path(cfg['output_dir'])/'results.yaml').read_text())
    return json.loads((Path(cfg['output_dir'])/'summary.json').read_text())


if __name__ == '__main__':
    run_iterative_znd(json.loads(Path(sys.argv[1]).read_text()))
