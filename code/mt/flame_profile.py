"""Reference-fixed, midpoint-aligned flame temperature comparisons (metres/K)."""
import numpy as np
from scipy.interpolate import PchipInterpolator


def profile_settings(options):
    """Validate editable rise thresholds, width margins, point count and scale."""
    cfg=dict(points=100, lower_fraction=.01, upper_fraction=.99,
             margin_fraction=.1, scale=25., weight=1.)
    unknown=set(options)-set(cfg)
    if unknown: raise ValueError(f'Unknown flame profile settings: {sorted(unknown)}')
    cfg.update(options)
    if isinstance(cfg['points'],bool) or not isinstance(cfg['points'],int) or cfg['points']<2:
        raise ValueError('Profile points must be an integer >= 2.')
    if not 0<cfg['lower_fraction']<.5<cfg['upper_fraction']<1:
        raise ValueError('Require 0 < lower_fraction < 0.5 < upper_fraction < 1.')
    for key in ('margin_fraction','weight'):
        if not np.isfinite(cfg[key]) or cfg[key]<0: raise ValueError(f'{key} must be finite and nonnegative.')
    scale=cfg['scale']
    if isinstance(scale,dict):
        if set(scale)!={'reference_temperature_rise_fraction'}: raise ValueError('Unknown profile scale.')
        scale=scale['reference_temperature_rise_fraction']
    if not np.isfinite(scale) or scale<=0: raise ValueError('Profile scale must be positive and finite.')
    return cfg


def aligned_profile(row):
    """Align at the first upward half-rise crossing without stretching distance."""
    x=np.asarray(row['distance_m'],float);t=np.asarray(row['temperature_K'],float)
    if x.ndim!=1 or t.shape!=x.shape or len(x)<2 or not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(np.diff(x)<=0):
        raise ValueError('Flame profile requires finite temperatures and increasing positions.')
    if t[-1]<=t[0]: raise ValueError('Flame temperature rise must be positive.')
    def crossing(fraction):
        target=t[0]+fraction*(t[-1]-t[0])
        hits=np.flatnonzero((t[:-1]<target)&(t[1:]>=target))
        if not len(hits): raise ValueError('Temperature rise crossing is unresolved.')
        j=hits[0]
        return float(x[j]+(target-t[j])*(x[j+1]-x[j])/(t[j+1]-t[j]))
    return x-crossing(.5),t,crossing


def sample_profile(row, grid):
    """Interpolate monotonically where appropriate; never extrapolate."""
    x,t,_=aligned_profile(row)
    if grid[0]<x[0] or grid[-1]>x[-1]: raise ValueError('Flame does not cover the fixed reference interval.')
    return PchipInterpolator(x,t,extrapolate=False)(grid)


def prepare_profile_reference(row, options):
    """Construct one immutable reference grid and temperature-error scale."""
    cfg=profile_settings(options)
    x,t,crossing=aligned_profile(row)
    midpoint=crossing(.5)
    lo=crossing(cfg['lower_fraction'])-midpoint
    hi=crossing(cfg['upper_fraction'])-midpoint
    margin=cfg['margin_fraction']*(hi-lo)
    grid=np.linspace(lo-margin,hi+margin,cfg['points'])
    scale=cfg['scale']
    if isinstance(scale,dict): scale=scale['reference_temperature_rise_fraction']*(t[-1]-t[0])
    return dict(grid_m=grid.tolist(),temperature_K=sample_profile(row,grid).tolist(),scale_K=float(scale))


def profile_residual(row, reference):
    """Return pointwise normalized errors on the stored reference grid."""
    grid=np.asarray(reference['grid_m'])
    return (sample_profile(row,grid)-np.asarray(reference['temperature_K']))/reference['scale_K']
