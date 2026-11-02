"""Joint flame/ignition objectives using separate case grids and shared kinetics."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import tempfile
import numpy as np
from .combustion_calculator import CombustionCalculator


def solve_ignition_case(job):
    """Evaluate OH-gradient delay and endpoint temperature at a fixed case horizon."""
    source,case,settings,_=job
    try:
        with tempfile.TemporaryDirectory(prefix='mt-idt-') as folder:
            if source.startswith('description:'):
                path=Path(folder)/'model.yaml';path.write_text(source);source=str(path)
            calc=CombustionCalculator(mechanism=source,fuel=case.get('fuel','C2H4'),oxidizer=case.get('oxidizer','O2:1, N2:3.76'),P=case['P_atm'],T=case['T'],phi=case['phi'])
            options={k:v for k,v in settings.items() if k!='timeout_seconds'}
            delay=calc.ignition_delay(**options)
            temperature=calc.final_temperature
            if not np.isfinite([delay,temperature]).all() or delay<=0:raise ValueError('Invalid ignition result.')
            return dict(ignition_delay_s=float(delay),burnt_temperature_K=float(temperature))
    except Exception as exc:return dict(error=f'{type(exc).__name__}: {exc}')


def combined_residual(rows,refs,cfg):
    """Average each branch over its own cases, then apply normalized branch weights."""
    from .flame_tuning import flame_residual
    if len(rows)!=len(refs) or any('error' in r for r in rows):return None
    ids=[i for i,c in enumerate(cfg['cases']) if c.get('kind')=='ignition']
    flames=[i for i,c in enumerate(cfg['cases']) if c.get('kind')!='ignition']
    if not ids:return flame_residual(rows,refs,**cfg.get('objectives',{}))
    opts=cfg['ignition_objectives'];w=np.array([opts['delay_weight'],opts['temperature_weight']],float)
    weights=np.array([cfg['branch_weights']['flame'],cfg['branch_weights']['ignition']],float)
    if not np.isfinite(np.r_[w,weights]).all() or min(*w,*weights)<0 or w.sum()<=0 or weights.sum()<=0:raise ValueError('Invalid joint objective weights.')
    try:
        delays=np.array([rows[i]['ignition_delay_s']/refs[i]['ignition_delay_s'] for i in ids])
        if np.any(delays<=0):return None
        r_idt=np.r_[np.sqrt(w[0])*np.log(delays),np.sqrt(w[1])*np.array([rows[i]['burnt_temperature_K']-refs[i]['burnt_temperature_K'] for i in ids])/opts['temperature_scale_K']]/np.sqrt(len(ids)*w.sum())
        r_flame=flame_residual([rows[i] for i in flames],[refs[i] for i in flames],**cfg['objectives'])
        if r_flame is None:return None
        r=np.r_[np.sqrt(weights[0])*r_flame,np.sqrt(weights[1])*r_idt]/np.sqrt(weights.sum())
        return r if np.isfinite(r).all() else None
    except (ValueError,ZeroDivisionError):return None


def resolve_combined_settings(problem,tuning,execution):
    """Keep all flame settings; expand IDT cases with one fixed integration end time."""
    from .flame_settings import resolve_flame_settings
    cfg=resolve_flame_settings(problem,tuning,execution)
    c=problem['conditions']['ignition']
    cases=list(product(c['pressure_atm'],c['temperature_K'],c['phi']))
    end_time=float(problem['ignition_solver']['end_time'])
    if not np.isfinite(end_time) or end_time<=0:raise ValueError('Ignition end_time must be positive and finite.')
    for p,t,phi in cases:
        cfg['cases'].append(dict(kind='ignition',P_atm=p,T=t,phi=phi,end_time_s=end_time,**cfg.get('mixture',{'fuel':'C2H4','oxidizer':'O2:1, N2:3.76'})))
    obj=problem['objectives']
    if obj['ignition_delay_time']['loss']!='log_squared' or obj['T_ignition']['loss']!='squared':raise ValueError('Use log_squared IDT and squared ignition temperature.')
    scale=obj['T_ignition']['scale']
    if not np.isfinite(scale) or scale<=0:raise ValueError('Ignition temperature scale must be positive.')
    cfg.update(ignition_solver=deepcopy(problem['ignition_solver']),
        ignition_objectives=dict(delay_weight=obj['ignition_delay_time'].get('weight',1.),temperature_weight=obj['T_ignition'].get('weight',1.),temperature_scale_K=scale),
        branch_weights=deepcopy(problem.get('branch_weights',{'flame':1.,'ignition':1.})))
    return cfg


def run_flame_ignition_case(problem,tuning,execution):
    """Launch the joint objective with the existing staged/PLOG machinery."""
    from .flame_tuning import launch_flame_tuning
    return launch_flame_tuning(resolve_combined_settings(problem,tuning,execution))


def ignition_comparison(cases,refs,rows):
    """Report matched IDT and endpoint temperature for each ignition case."""
    return [dict(**case,reference_delay_us=ref['ignition_delay_s']*1e6,model_delay_us=row['ignition_delay_s']*1e6,
        delay_error_percent=100*(row['ignition_delay_s']/ref['ignition_delay_s']-1),
        reference_temperature_K=ref['burnt_temperature_K'],model_temperature_K=row['burnt_temperature_K'],
        temperature_error_K=row['burnt_temperature_K']-ref['burnt_temperature_K'])
        for case,ref,row in zip(cases,refs,rows) if case.get('kind')=='ignition']


def plot_ignition_comparison(root, comparison, scales, *, phis=None,
                             inverse_temperature=False, separate_pressures=False):
    """Plot selected equivalence ratios, optionally with 1000/T and one figure per pressure."""
    from .plot_settings import yscale
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pressures = sorted({r['P_atm'] for r in comparison})
    selected = sorted({r['phi'] for r in comparison}) if phis is None else list(phis)
    groups = [[p] for p in pressures] if separate_pressures else [pressures]
    xvalue = (lambda r: 1000. / r['T']) if inverse_temperature else (lambda r: r['T'])
    xlabel = r'$1000/T_0$ (K$^{-1}$)' if inverse_temperature else 'Initial temperature (K)'
    paths = []
    for group in groups:
        fig, axes = plt.subplots(2, len(group), squeeze=False, figsize=(6*len(group), 9))
        for j, pressure in enumerate(group):
            for phi in selected:
                rows = sorted([r for r in comparison if r['P_atm'] == pressure
                               and np.isclose(r['phi'], phi)], key=xvalue)
                if not rows:
                    continue
                for ax, ref, model in [(axes[0,j], 'reference_delay_us', 'model_delay_us'),
                                       (axes[1,j], 'reference_temperature_K', 'model_temperature_K')]:
                    line, = ax.plot([xvalue(r) for r in rows], [r[ref] for r in rows],
                                    '-', linewidth=2, label=f'FFCM2 φ={phi:g}')
                    ax.plot([xvalue(r) for r in rows], [r[model] for r in rows],
                            'o--', color=line.get_color(), markerfacecolor='none',
                            linewidth=2, label=f'best_model φ={phi:g}')
                    baseline_key = 'baseline_delay_us' if ref == 'reference_delay_us' else 'baseline_temperature_K'
                    if baseline_key in rows[0]:
                        ax.plot([xvalue(r) for r in rows], [r[baseline_key] for r in rows],
                                ':', color=line.get_color(), linewidth=2, label=f'Untuned φ={phi:g}')
                    ax.set(xlabel=xlabel, title=f'{pressure:g} atm')
                    ax.legend(fontsize=9); ax.grid(alpha=.3)
            axes[0,j].set(ylabel='Ignition delay (µs)', yscale=yscale(scales, 'ignition_delay_time', 'log'))
            axes[1,j].set(ylabel='Final temperature (K)', yscale=yscale(scales, 'T_ignition'))
        name = f'ignition_comparison_{group[0]:g}atm.png' if separate_pressures else 'ignition_comparison.png'
        path = Path(root) / name
        fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
        paths.append(path)
    return paths
