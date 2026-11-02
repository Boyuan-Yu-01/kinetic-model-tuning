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
    cfg=dict(model=problem['model'],reference=problem['reference'],output_dir=problem['output'],
        cases=[dict(P_atm=p,T=t,phi=phi) for p,t,phi in product(cases['pressure_atm'],cases['temperature_K'],cases['phi'])],
        objectives=dict(temperature_scale_K=objectives['T_flame']['scale'],
                        speed_weight=objectives['laminar_flame_speed'].get('weight',1.),
                        temperature_weight=objectives['T_flame'].get('weight',1.)),
        solver=deepcopy(problem['solver']),artificial=problem['reactions']['artificial'],
        uncertainty_factors=deepcopy(problem['reactions']['uncertainty_factors']),
        stages=deepcopy(tuning['stages']),samples=tuning['exploration']['samples'],
        sample_batch=tuning['exploration']['batch_size'],seed=tuning['exploration']['seed'])
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
