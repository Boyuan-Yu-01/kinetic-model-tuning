"""Reference-cache fingerprints and transparent numerical tuning diagnostics."""
from pathlib import Path
import hashlib
import inspect
import json
import numpy as np
import cantera as ct
import scipy
from .uncertainty import _entries


def reference_key(cfg):
    """Invalidate reuse on changes to mechanism, cases, solver, or solver code."""
    import sdtoolbox.znd
    import sdtoolbox.postshock
    keys = ('P_bar','T0','phi','t_end','fuel','oxidizer','steps_per_end',
            'rel_tol','abs_tol','max_rhs_calls','interpolation')
    record = {k:cfg[k] for k in keys}
    record.update(mechanism_sha256=hashlib.sha256(Path(cfg['reference']).read_bytes()).hexdigest(),
                  cantera=ct.__version__, scipy=scipy.__version__, schema=1)
    record['code_sha256'] = {str(i):hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for i,p in enumerate([inspect.getfile(sdtoolbox.znd), inspect.getfile(sdtoolbox.postshock),
                              Path(__file__).with_name('znd_functions.py'),
                              Path(__file__).with_name('starting_points.py'),
                              Path(__file__).with_name('iterative_znd.py')])}
    return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()


def update_audit(space, before, after, interval):
    """Report physical A changes and sampled complete forward-rate changes."""
    old, new = space.scheme(before), space.scheme(after)
    changes=[]
    for n,(a,b) in enumerate(zip(old['reactions'],new['reactions']),1):
        for i,(u,v) in enumerate(zip(_entries(a),_entries(b))):
            ratio=float(v['A'])/float(u['A'])
            changes.append(dict(reaction=n,entry=i+1,pressure=v.get('P'),
                                old_A=float(u['A']),new_A=float(v['A']),multiplier=ratio))
    changes.sort(key=lambda r:abs(np.log(r['multiplier'])),reverse=True)
    a,b=space.make_gas(before),space.make_gas(after)
    pressures={1e5,1e6,1e7}
    for rxn in b.reactions():
        if isinstance(rxn.rate,ct.PlogRate):pressures.update(p for p,_ in rxn.rate.rates)
    ratios=[]
    for t in np.geomspace(*interval,9):
        for p in sorted(pressures):
            a.TPX=b.TPX=t,p,np.ones(a.n_species)
            ratio=b.forward_rate_constants/a.forward_rate_constants
            for n in getattr(space,'orders',{}):
                ratio[n-1]=b.forward_rates_of_progress[n-1]/a.forward_rates_of_progress[n-1]
            ratios.append(ratio)
    ratios=np.asarray(ratios)
    for row in changes:
        j=row['reaction']-1
        row['rate_comparison']='progress_rate_at_equal_composition' if row['reaction'] in getattr(space,'orders',{}) else 'rate_constant'
        row.update(sampled_k_ratio_min=float(ratios[:,j].min()),sampled_k_ratio_max=float(ratios[:,j].max()))
    return dict(largest_A_updates=changes[:3],temperature_interval_K=list(interval),
                temperatures_sampled=9,pressures_sampled_Pa=sorted(pressures))


def active_bounds(space,x):
    """Name tight coordinate and physical-A bounds, including stage limits."""
    found=[]
    for i,v in enumerate(x):
        if space.lower[i]==space.upper[i]:continue
        for side,limit in [('lower',space.lower[i]),('upper',space.upper[i])]:
            if np.isfinite(limit) and abs(v-limit)<=1e-6*max(1,abs(limit)):
                found.append(dict(parameter=space.coordinate_labels[i]['label'],side=side,value=float(v),bound=float(limit)))
    if hasattr(space,'a_map'):
        for i,v in enumerate(space.a_map@x):
            for side,limit in [('lower',space.a_low[i]),('upper',space.a_high[i])]:
                if np.isfinite(limit) and abs(v-limit)<=1e-6*max(1,abs(limit)):
                    found.append(dict(parameter=space.base.coordinate_labels[i]['label']+' physical ln A',side=side,value=float(v),bound=float(limit)))
    return found


def fit_quality(metrics,refs,targets=None):
    """Separate average fit errors/targets from optimizer stopping status."""
    errors=dict(final_temperature_percent=float(np.mean([abs(m['T_final']/r['T_final']-1)*100 for m,r in zip(metrics,refs)])),
                induction_percent=float(np.mean([abs(m['ind_len_ZND']/r['ind_len_ZND']-1)*100 for m,r in zip(metrics,refs)])),
                exothermic_percent=float(np.mean([abs(m['exo_len_ZND']/r['exo_len_ZND']-1)*100 for m,r in zip(metrics,refs)])),
                profile_rmse_K=float(np.mean([m['profile_rmse_K'] for m in metrics])))
    contributions=dict(zip(('temperature_profile','final_temperature','induction_length','exothermic_length'),
                           map(float,np.mean([m['components'] for m in metrics],axis=0)/4)))
    targets=targets or {}
    if set(targets)-set(errors) or any(not np.isfinite(v) or v<=0 for v in targets.values()):
        raise ValueError('Fit targets must use known error keys and positive finite limits.')
    return dict(mean_absolute_errors=errors,loss_contributions=contributions,targets=targets,
                passed=all(errors[k]<=v for k,v in targets.items()) if targets else None)
