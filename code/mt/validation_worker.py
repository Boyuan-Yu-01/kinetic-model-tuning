"""Guarded process launcher for mt.validation; not a public run script."""
from concurrent.futures import as_completed
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import json
import math
import os
import sys
import warnings
from .executors import create_executor


def _solve(job):
    """Run an existing solver and return JSON-safe predictions, never Cantera objects."""
    cfg, name, i = job
    case = {**cfg['cases'][i], 'fuel': cfg['fuel'], 'oxidizer': cfg['oxidizer']}
    settings = dict(cfg['solver'])
    source = cfg['models'][name]
    logdir = Path(cfg['output_dir'])/'solver_logs' if cfg['save_logs'] else None
    if logdir is not None:
        logdir.mkdir(parents=True, exist_ok=True)
    filename = logdir/f'{cfg["kind"]}_model_{list(cfg["models"]).index(name)}_case_{i+1:03d}.log' if logdir else os.devnull
    try:
        with open(filename, 'w') as log, redirect_stdout(log), redirect_stderr(log), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if cfg['kind'] == 'znd':
                from .znd_functions import calculate_znd
                steps = settings.pop('steps')
                settings.setdefault('max_step', case['end_time_s']/steps)
                out = calculate_znd(P0=case['P_atm']*101325., T0=case['T'], phi=case['phi'],
                                    fuel=cfg['fuel'], oxidizer=cfg['oxidizer'], mechanism=source,
                                    file_name=None, t_end=case['end_time_s'], **settings)
                row = dict(T_ZND=float(out['T'][-1]), ind_len_ZND=float(out['ind_len_ZND']),
                           exo_len_ZND=float(out['exo_len_ZND']),
                           distance_m=list(map(float,out['distance'])), temperature_K=list(map(float,out['T'])))
            elif cfg['kind'] == 'flame':
                from .flame_tuning import solve_flame_case
                settings.update(flame_speed_sensitivities=False, full_flame_adjoints=False)
                out = solve_flame_case((source,case,settings,str(logdir) if logdir else None))
                row = out if 'error' in out else dict(laminar_flame_speed=out['speed_m_s'], T_flame=out['burnt_temperature_K'])
            else:
                from .flame_ignition import solve_ignition_case
                out = solve_ignition_case((source,case,settings,None))
                row = out if 'error' in out else dict(ignition_delay_time=out['ignition_delay_s'], T_ignition=out['burnt_temperature_K'])
            notices = list(dict.fromkeys(str(w.message) for w in caught))
            if notices:
                row['warnings'] = notices
                for message in notices: print('WARNING:', message)
        return _clean(row)
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {exc}'}


def _clean(value):
    """Represent unavailable nonfinite metrics as JSON null, retaining other outputs."""
    if isinstance(value,dict): return {k:_clean(v) for k,v in value.items()}
    if isinstance(value,list): return [_clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def _run(cfg):
    """Schedule individual model/case solves and retain successful partial results."""
    rows = {name:[None]*len(cfg['cases']) for name in cfg['models']}
    jobs = [(cfg,name,i) for name in cfg['models'] for i in range(len(cfg['cases']))]
    print(f'{cfg["kind"].upper()} validation: {len(cfg["cases"])} cases × '
          f'{len(cfg["models"])} models | {cfg["workers"]} workers',flush=True)
    def receive(job,row,done):
        _,name,i = job
        rows[name][i] = row
        status = 'FAILED' if 'error' in row else 'OK (warnings)' if row.get('warnings') else 'OK'
        print(f'  {done}/{len(jobs)} | case {i+1} | {name}: {status}',flush=True)
    if cfg['workers']==1:
        for done,job in enumerate(jobs,1): receive(job,_solve(job),done)
    else:
        with create_executor('process',min(cfg['workers'],len(jobs))) as pool:
            pending = {pool.submit(_solve,job):job for job in jobs}
            for done,future in enumerate(as_completed(pending),1):
                job = pending[future]
                try: row = future.result()
                except Exception as exc: row = {'error': f'{type(exc).__name__}: {exc}'}
                receive(job,row,done)
    return rows


if __name__ == '__main__':
    request,response = map(Path,sys.argv[1:])
    response.write_text(json.dumps(_run(json.loads(request.read_text())),allow_nan=False))
