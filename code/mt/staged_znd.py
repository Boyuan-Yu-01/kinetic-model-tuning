"""Three-stage, sensitivity-guided refinement of selected ZND reactions."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import OrderedDict
from numbers import Integral
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy
from pathlib import Path
from threading import BoundedSemaphore
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import warnings
import cantera as ct
import numpy as np
import yaml
from .starting_points import _prepare_case
from .znd_functions import _sensitivity_lengths
from .sensitivity_search import DEFAULTS, sensitivity_trust_region
from ._execution import launcher_environment, wait_for_process

_GASES = OrderedDict()


@contextmanager
def _work_limit(limit):
    """Bound stored trajectories by limiting RHS work within one worker process."""
    import sdtoolbox.znd as znd
    original = znd.solve_ivp
    def bounded(fun, *args, **kwargs):
        calls = 0
        def rhs(*values):
            nonlocal calls
            calls += 1
            if calls > limit:
                raise RuntimeError(f'ZND work limit exceeded: {limit} RHS calls')
            return fun(*values)
        return original(rhs, *args, **kwargs)
    znd.solve_ivp = bounded
    try:
        yield
    finally:
        znd.solve_ivp = original


def _case(job):
    """Apply selected A/Ea/b changes using a bounded process-local gas cache."""
    source, state, vector, config, logs = job
    from sdtoolbox.znd import zndsolve
    with (Path(logs)/f'worker_{os.getpid()}.log').open('a') as log, redirect_stdout(log), redirect_stderr(log):
        try:
            indices = tuple(i-1 for i in config['reactions']) if vector is not None else ()
            n = len(indices)
            key = (source, indices)
            if key not in _GASES:
                gas, upstream = ct.Solution(source), ct.Solution(source)
                rates = [(gas.reaction(i).rate.pre_exponential_factor,
                          gas.reaction(i).rate.temperature_exponent,
                          gas.reaction(i).rate.activation_energy) for i in indices]
                _GASES[key] = gas, upstream, rates
                if len(_GASES) > 4:
                    _GASES.popitem(last=False)
            _GASES.move_to_end(key)
            gas, upstream, rates = _GASES[key]
            gas.state, upstream.state = state['frozen'], state['upstream']
            gas.set_multiplier(1.)
            upstream.set_multiplier(1.)
            for j, (index, (a,b,ea)) in enumerate(zip(indices, rates)):
                reaction = gas.reaction(index)
                reaction.rate = ct.ArrheniusRate(a*np.exp(vector[j]), b+.1*vector[2*n+j], ea+5e6*vector[n+j])
                gas.modify_reaction(index,reaction)
            end = state['case']['t_end']
            with warnings.catch_warnings(), _work_limit(config['max_rhs_calls']):
                warnings.simplefilter('error')
                out = zndsolve(gas,upstream,state['cj_speed'],t_end=end,max_step=end/1000,
                               relTol=1e-8,absTol=1e-12,Method='LSODA',advanced_output=False)
                lengths = _sensitivity_lengths(out,end)
            return dict(T_ZND=float(out['T'][-1]),**lengths)
        except Exception as exc:
            return {'error':f'{type(exc).__name__}: {exc}'}


def _prepare(job):
    """Cache thermodynamic shock states independently of kinetic parameters."""
    payload, logs = job
    with (Path(logs)/f'worker_{os.getpid()}.log').open('a') as log, redirect_stdout(log), redirect_stderr(log):
        return _prepare_case(payload)


def _save(path,data):
    """Atomically preserve the latest accepted result or diagnostic record."""
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)


def _residual(records,target,cfg):
    """Normalized temperature and log-length residuals; invalid cases are rejected."""
    if any('error' in row for row in records):return None
    values=np.array([[row[k] for k in ('T_ZND','ind_len_ZND','exo_len_ZND')] for row in records])
    if not np.all(np.isfinite(values)) or np.any(values<=0):return None
    result=np.log(values/target)
    result[:,0]=(values[:,0]-target[:,0])/cfg['temperature_scale_K']
    weights=np.asarray(cfg['weights'])
    return (result*np.sqrt(weights/(len(records)*weights.sum()))).ravel()


def _export(source,x,path,reactions):
    """Change only selected rate entries, preserving all other YAML contents."""
    data=yaml.safe_load(Path(source).read_text());gas=ct.Solution(source)
    n=len(reactions)
    for j, number in enumerate(reactions):
        rate=gas.reaction(number-1).rate; entry=data['reactions'][number-1]['rate-constant']
        entry['A']=float(entry['A'])*float(np.exp(x[j]))
        entry['b']=float(rate.temperature_exponent+.1*x[2*n+j])
        entry['Ea']=f'{(rate.activation_energy+5e6*x[n+j])/1000:.17g} J/mol'
    temp=Path(str(path)+'.tmp');temp.write_text(yaml.safe_dump(data,sort_keys=False));temp.replace(path)


def _bounds(stage,center,cfg):
    """Stage limits are anchored to the preceding accepted mechanism, not each step."""
    n=len(cfg['reactions'])
    low,high=center.copy(),center.copy()
    low[:n]+=np.log(cfg['a_ratios'][stage][0]);high[:n]+=np.log(cfg['a_ratios'][stage][1])
    active=list(range(n))
    if stage>=1:
        d=cfg['ea_delta_kj_mol'][stage-1]/5
        low[n:2*n]-=d;high[n:2*n]+=d;active+=list(range(n,2*n))
    if stage==2:
        d=cfg['b_delta']/.1
        low[2*n:]-=d;high[2*n:]+=d;active+=list(range(2*n,3*n))
    return low,high,active


def _refine(source,states,target,batch,output,cfg):
    """Use refreshed finite-difference Jacobians and adaptive trust-region steps."""
    folder=output/Path(source).stem;folder.mkdir()
    n=len(cfg['reactions'])
    x=np.zeros(3*n);evaluations=0
    cache=OrderedDict()
    def evaluate(vectors):
        nonlocal evaluations
        # Deduplicate boundary probes and reuse recent, exactly identical vectors.
        keys=[tuple(float(v) for v in vector) for vector in vectors]
        missing=list(dict.fromkeys(key for key in keys if key not in cache))
        new_rows=batch(source,states,[np.asarray(key) for key in missing]) if missing else []
        for key,records in zip(missing,new_rows):
            r=_residual(records,target,cfg);evaluations+=1
            row=dict(evaluation=evaluations,coordinates=list(key),loss=None if r is None else float(r@r),cases=records)
            with (folder/'evaluations.jsonl').open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
            cache[key]=(r,records)
        results=[cache[key] for key in keys]
        for key in keys:cache.move_to_end(key)
        while len(cache)>512:cache.popitem(last=False)
        return results
    r,records=evaluate([x])[0]
    if r is None:
        result=dict(candidate=Path(source).stem,status='invalid_start',cases=records)
        _save(folder/'result.json',result);return result
    initial=float(r@r);stages=[]
    def save_best():
        _export(source,x,folder/'best_model.yaml',cfg['reactions'])
        _save(folder/'best_results.json',dict(loss=float(r@r),coordinates=x.tolist(),cases=records))
    save_best()
    for stage,name in enumerate(('A','Ea','b')):
        low,high,active=_bounds(stage,x,cfg)
        options={key:cfg.get(key,value) for key,value in DEFAULTS.items()}
        def residual_batch(vectors):
            """Adapt the physics evaluation to the general numerical search interface."""
            return [result[0] for result in evaluate(vectors)]
        def on_iteration(info):
            """Persist complete derivative diagnostics and the latest accepted mechanism."""
            nonlocal x,r,records
            diagnostic=dict(info,reaction_numbers=cfg['reactions'],
                coordinate_units=['ln(A/A_start)']*n+['Ea offset / (5 kJ/mol)']*n+['b offset / 0.1']*n,
                residual_order='case-major: scaled temperature, log induction length, log exothermic length')
            _save(folder/f'sensitivity_{name}_{info["iteration"]-1:03d}.json',diagnostic)
            if info['accepted']:
                x=np.asarray(info['accepted_coordinates'])
                r,records=evaluate([x])[0]
                save_best()
            _save(folder/'progress.json',dict(stage=name,iteration=info['iteration'],
                loss=info['loss'],evaluations=evaluations,radius=info['radius'],
                status=info['status'],projected_gradient=info['projected_gradient']))
            print(f'{Path(source).stem} | {name} {info["iteration"]} | loss {info["loss"]:.6g} | eval {evaluations} | {info["status"]}',flush=True)
        result=sensitivity_trust_region(residual_batch,x,low,high,active,
            np.repeat(cfg['fd_steps'],n),max_iterations=cfg['max_iterations'][stage],
            options=options,callback=on_iteration)
        x,r=result.x,result.residual
        _,records=evaluate([x])[0]
        _export(source,x,folder/f'after_{name}.yaml',cfg['reactions'])
        stages.append(dict(stage=name,loss=float(r@r),iterations=result.iterations,
                           status=result.status,converged=result.converged,
                           lower_bounds=low.tolist(),upper_bounds=high.tolist(),
                           active_bounds=[int(i) for i in active if abs(x[i]-low[i])<1e-8 or abs(x[i]-high[i])<1e-8]))
        _save(folder/'stages.json',stages)
        _save(folder/'progress.json',dict(stage=name,iteration=result.iterations,loss=float(r@r),
            evaluations=evaluations,status=result.status,converged=result.converged,
            radius=result.diagnostics.get('radius')))
    result=dict(candidate=Path(source).stem,status='finished',converged=stages[-1]['converged'],
                initial_loss=initial,final_loss=float(r@r),evaluations=evaluations,stages=stages)
    _save(folder/'result.json',result);return result


def _run(cfg):
    """Share a bounded process pool across independent candidate optimizations."""
    sources=sorted(Path(cfg['candidates_dir']).glob('candidate_*.yaml'))
    if not sources:raise ValueError('No candidate mechanisms found.')
    _validate_reactions(cfg,sources)
    output=Path(cfg['output_dir']);output.mkdir(parents=True,exist_ok=False)
    logs=output/'solver_logs';logs.mkdir();_save(output/'settings.json',cfg)
    workers=cfg['workers']
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        gate=BoundedSemaphore(2*workers)
        def submit(fn,job):
            gate.acquire()
            try:f=pool.submit(fn,job)
            except BaseException:gate.release();raise
            f.add_done_callback(lambda _:gate.release());return f
        def batch(source,prepared,vectors):
            futures=[submit(_case,(str(source),state,x,cfg,str(logs))) for x in vectors for state in prepared]
            # Collect as tasks finish, then restore the input vector/case order.
            rows=[None]*len(futures)
            positions={future:i for i,future in enumerate(futures)}
            for future in as_completed(futures):rows[positions[future]]=future.result()
            n=len(prepared)
            return [rows[i:i+n] for i in range(0,len(rows),n)]
        reference=cfg['reference_model']
        # Kinetic changes cannot alter CJ/frozen states. Share preparation only
        # when species, phase definitions, and their unit systems match exactly.
        groups={}; source_groups={}
        for source in [reference,*map(str,sources)]:
            data=yaml.safe_load(Path(source).read_text())
            identity=json.dumps({k:data.get(k) for k in ('species','phases','units')},sort_keys=True)
            source_groups[source]=identity
            groups.setdefault(identity,source)
        preparations={}
        for identity,source in groups.items():
            jobs=[((source,p*1e5,298,phi,end,'C2H4','O2:1, N2:3.76'),str(logs))
                  for p,phi,end in zip(cfg['P_bar'],cfg['phi'],cfg['t_end'])]
            preparations[identity]=[submit(_prepare,job) for job in jobs]
        states={key:[future.result() for future in futures] for key,futures in preparations.items()}
        if any('error' in row for rows in states.values() for row in rows):
            raise RuntimeError('Shock preparation failed; inspect worker logs.')
        records=batch(reference,states[source_groups[reference]],[None])[0]
        _save(output/'FFCM2_results.json',records)
        if any('error' in r for r in records):raise RuntimeError('Reference failed; inspect FFCM2_results.json.')
        target=np.array([[r[k] for k in ('T_ZND','ind_len_ZND','exo_len_ZND')] for r in records])
        prepared={str(source):states[source_groups[str(source)]] for source in sources}
        with ThreadPoolExecutor(max_workers=min(workers,len(sources))) as coordinators:
            futures=[coordinators.submit(_refine,str(source),prepared[str(source)],target,batch,output,cfg) for source in sources]
            results=[f.result() for f in futures]
        _save(output/'summary.json',results)
        print(f'Completed {len(results)} starting points: {output}',flush=True)


def _reaction_numbers(reactions):
    """Normalize the one-based selection before serializing launcher settings."""
    if not isinstance(reactions,(list,tuple)) or not reactions or any(
            isinstance(i,bool) or not isinstance(i,Integral) or i<1 for i in reactions):
        raise ValueError('reactions must be a nonempty list of positive one-based integers.')
    if len(set(reactions))!=len(reactions):raise ValueError('Reaction numbers must be unique.')
    return [int(i) for i in reactions]


def _validate_reactions(cfg,sources):
    """Check selected Arrhenius rates once, before the worker pool or output creation."""
    cfg['reactions']=_reaction_numbers(cfg.get('reactions',[1,2,3]))
    for source in sources:
        gas=ct.Solution(str(source));data=yaml.safe_load(Path(source).read_text())
        if (len(data.get('phases',[]))!=1 or data['phases'][0].get('reactions','all')!='all'
                or len(data.get('reactions',[]))!=gas.n_reactions
                or not all(isinstance(species,dict) and 'name' in species for species in data.get('species',[]))):
            raise ValueError('Candidates must use one self-contained phase with all reactions in a single list.')
        for number in cfg['reactions']:
            if number>gas.n_reactions:raise ValueError(f'{source}: reaction {number} does not exist.')
            if not isinstance(gas.reaction(number-1).rate,ct.ArrheniusRate) or 'rate-constant' not in data['reactions'][number-1]:
                raise ValueError(f'{source}: reaction {number} must have an ordinary Arrhenius rate (including three-body). Falloff/PLOG are not supported here.')
            if gas.reaction(number-1).rate.pre_exponential_factor<=0:raise ValueError('Selected A values must be positive.')


def optimize_staged_znd(**cfg):
    """Launch all stages safely from a plain Python run script, locally or on Slurm."""
    cfg=deepcopy(cfg)
    for key,value in DEFAULTS.items():cfg.setdefault(key,value)
    cfg['reactions']=_reaction_numbers(cfg.get('reactions',[1,2,3]))
    for key in ('candidates_dir','reference_model','output_dir'):cfg[key]=str(Path(cfg[key]).resolve())
    if not isinstance(cfg['workers'],int) or cfg['workers']<1:raise ValueError('workers must be positive.')
    if len(cfg['weights'])!=3 or min(cfg['weights'])<=0 or cfg['temperature_scale_K']<=0:raise ValueError('Positive objective weights and temperature scale required.')
    if len(cfg['max_iterations'])!=3 or min(cfg['max_iterations'])<1:raise ValueError('Provide positive iteration counts for all three stages.')
    if not (len(cfg['P_bar'])==len(cfg['phi'])==len(cfg['t_end'])):raise ValueError('Condition arrays must match.')
    if len(cfg['a_ratios'])!=3:raise ValueError('Provide A bounds for each stage.')
    for low,high in cfg['a_ratios']:
        if not 0<low<1<high:raise ValueError('Stage A ratios must bracket one.')
    for key in ('ea_delta_kj_mol','fd_steps'):
        if not np.all(np.isfinite(cfg[key])) or min(cfg[key])<=0:raise ValueError(f'{key} must be positive and finite.')
    for key in ('b_delta','initial_radius','maximum_radius','minimum_radius','gradient_tolerance','fd_relative_tolerance','max_rhs_calls'):
        if not np.isfinite(cfg[key]) or cfg[key]<=0:raise ValueError(f'{key} must be positive and finite.')
    if len(cfg['ea_delta_kj_mol'])!=2 or len(cfg['fd_steps'])!=3:raise ValueError('Provide two Ea ranges and three finite-difference steps.')
    if not 0<cfg['minimum_radius']<=cfg['initial_radius']<=cfg['maximum_radius']:raise ValueError('Invalid trust-region radii.')
    for key in ('fd_attempts','max_trial_steps','stagnation_iterations'):
        if isinstance(cfg[key],bool) or not isinstance(cfg[key],int) or cfg[key]<1:raise ValueError(f'{key} must be a positive integer.')
    for key in ('fd_absolute_tolerance','weak_response_tolerance','loss_tolerance'):
        if not np.isfinite(cfg[key]) or cfg[key]<=0:raise ValueError(f'{key} must be positive and finite.')

    with tempfile.TemporaryDirectory(prefix='staged_znd_') as temp:
        request=Path(temp)/'request.json';request.write_text(json.dumps(cfg))
        process=subprocess.Popen([sys.executable,'-u','-m','mt.staged_znd',str(request)],
                                 env=launcher_environment(),start_new_session=(os.name=='posix'))
        code=wait_for_process(process)
        if code:raise RuntimeError(f'Staged ZND optimization exited with status {code}.')
    return Path(cfg['output_dir'])


if __name__=='__main__':
    _run(json.loads(Path(sys.argv[1]).read_text()))
