"""Public PLOG controls translated into the registry's reaction-index lists."""
from copy import deepcopy
from .uncertainty import _load


def resolve_plog_settings(settings, model):
    """Resolve Boolean freedom for all existing PLOG reactions; keep legacy inputs readable."""
    settings=deepcopy(settings)
    if not any(k in settings for k in ('enabled','reactions','pressures_bar','direction')) and not isinstance(settings.get('plog_free'),bool):
        return settings
    allowed={'enabled','reactions','pressures_bar','direction','plog_free','advanced'}
    if set(settings)-allowed:raise ValueError('Use the simplified PLOG keys; internal overrides belong in advanced.')
    enabled=settings.get('enabled',False);free=settings.get('plog_free',True)
    if not isinstance(enabled,bool) or not isinstance(free,bool):raise ValueError('enabled and plog_free must be True or False.')
    if enabled and not free:raise ValueError('PLOG insertion requires plog_free=True so new entries can develop pressure dependence.')
    overrides=settings.get('advanced',{})
    defaults=dict(plog_max_trials=1,plog_min_relative_score=.1,plog_trial_rounds=5,plog_min_improvement=.001)
    if set(overrides)-set(defaults):raise ValueError('Unknown advanced PLOG setting.')
    defaults.update(overrides)
    entries=[i+1 for i,r in enumerate(_load(model)['reactions']) if r.get('type')=='pressure-dependent-Arrhenius'] if free else []
    defaults.update(enable_plog=enabled,plog_candidates=settings.get('reactions',[]),
        plog_pressures_bar=settings.get('pressures_bar',[1.,50.,100.]),
        plog_direction=settings.get('direction','either'),plog_free=entries or None)
    return defaults
