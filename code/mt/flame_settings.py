from .plog_settings import resolve_plog_settings
"""Translate public LFS settings without changing numerical defaults."""
from copy import deepcopy
from itertools import product
from .flame_tuning import launch_flame_tuning


def resolve_flame_settings(problem, tuning, execution):
    """Return the same engine configuration formerly assembled in run.py."""
    cases=problem['conditions']['flame']
    objectives=problem['objectives']
    if objectives['laminar_flame_speed']['loss']!='log_squared' or objectives['T_flame']['loss']!='squared':
        raise ValueError('Use log_squared flame speed and squared burned temperature.')
    mode=cases.get('mode','product')
    if mode not in ('product','paired'): raise ValueError('Flame condition mode must be product or paired.')
    if mode=='paired' and len({len(cases[k]) for k in ('pressure_atm','temperature_K','phi')})!=1:
        raise ValueError('Paired flame arrays must have equal lengths.')
    triples=(zip if mode=='paired' else product)(cases['pressure_atm'],cases['temperature_K'],cases['phi'])
    cfg=dict(model=problem['model'],reference=problem['reference'],output_dir=problem['output'],
        cases=[dict(P_atm=p,T=t,phi=phi) for p,t,phi in triples],
        objectives=dict(temperature_scale_K=objectives['T_flame']['scale'],
                        speed_weight=objectives['laminar_flame_speed'].get('weight',1.),
                        temperature_weight=objectives['T_flame'].get('weight',1.)),
        solver=deepcopy(problem['solver']),artificial=problem['reactions']['artificial'],
        uncertainty_factors=deepcopy(problem['reactions']['uncertainty_factors']),
        stages=deepcopy(tuning['stages']),samples=tuning['exploration']['samples'],
        sample_batch=tuning['exploration']['batch_size'],seed=tuning['exploration']['seed'])
    if 'flame_thickness' in objectives:
        if objectives['flame_thickness']['loss'] != 'log_squared':
            raise ValueError('Use log_squared flame thickness.')
        cfg['objectives']['thickness_weight'] = objectives['flame_thickness'].get('weight',1.)
    from .order_tuning import order_specs
    if problem.get('initial_model'):cfg['initial_model']=problem['initial_model']
    cfg['rate_regularization'] = deepcopy(problem['reactions'].get('rate_regularization',{}))
    cfg['orders'] = order_specs(problem['reactions'].get('orders'))
    cfg['top_reactions'] = tuning.get('selection',{}).get('top_reactions',5)
    if 'T_profile' in objectives:
        profile=deepcopy(objectives['T_profile'])
        if profile.pop('loss')!='squared': raise ValueError('Use squared profile loss.')
        cfg['objectives']['profile']=profile
    cfg.update(deepcopy(tuning['limits']))
    cfg.update(deepcopy(tuning['stopping']))
    cfg.update(resolve_plog_settings(tuning['plog'], problem['model']))
    cfg.update(deepcopy(execution))
    mixture=problem.get('mixture', {'fuel':'C2H4', 'oxidizer':'O2:1, N2:3.76'})
    for case in cfg['cases']:
        case.update(fuel=mixture['fuel'], oxidizer=mixture['oxidizer'])
    cfg['mixture']=deepcopy(mixture)
    return cfg


def run_flame_tuning_case(problem, tuning, execution):
    """Resolve grouped settings and launch the protected flame optimizer."""
    return launch_flame_tuning(resolve_flame_settings(problem, tuning, execution))
