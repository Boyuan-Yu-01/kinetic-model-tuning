"""Sensitivity-selected staged flame-speed fitting with immutable rate anchors."""
from pathlib import Path
from concurrent.futures import as_completed
from contextlib import redirect_stdout, redirect_stderr
import json
import os
import signal
import subprocess
import sys
import tempfile
import numpy as np
import cantera as ct
import yaml
from .flame_profile import profile_settings, prepare_profile_reference, profile_residual
from .uncertainty import plog_monotonicity
from .plot_settings import yscale
from .flame_ignition import solve_ignition_case, combined_residual, ignition_comparison, plot_ignition_comparison
from .staged_parameters import StagedParameters
from .iterative_sensitivity import iterative_sensitivity_optimize
from .flame_reporting import report_round, report_stage
from .executors import create_executor
from .iterative_znd import _save
from ._execution import launcher_environment, wait_for_process


def solve_flame_case(job):
    """Solve one isolated flame and report SI speed and burned temperature."""
    source, case, settings, logdir = job
    def timeout(*_):
        raise TimeoutError('Flame case exceeded configured time limit.')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(settings.get('timeout_seconds', 900))
    try:
        with open(Path(logdir)/f'worker_{os.getpid()}.log', 'a') as log, redirect_stdout(log), redirect_stderr(log):
            gas = ct.Solution(yaml=source) if source.startswith('description:') else ct.Solution(source)
            gas.TP = case['T'], case['P_atm']*ct.one_atm
            gas.set_equivalence_ratio(case['phi'], case.get('fuel','C2H4'), case.get('oxidizer','O2:1, N2:3.76'))
            flame = ct.FreeFlame(gas, width=settings['width'])
            flame.transport_model = settings['transport_model']
            flame.soret_enabled = settings['soret']
            flame.set_refine_criteria(ratio=3., slope=settings['slope'], curve=settings['curve'])
            flame.solve(loglevel=0, auto=True)
            if settings.get('flame_speed_sensitivities', False):
                # Resolve the state Jacobian accurately for dilute intermediates.
                flame.set_max_jac_age(0, 0)
                flame.set_jacobian_perturbation(1e-8, 1e-24, 0.)
                flame.flame.set_steady_tolerances(default=(1e-10, 1e-14))
                flame.solve(loglevel=0, refine_grid=False, auto=False)
            speed=float(flame.velocity[0])
            if not np.isfinite(speed) or speed<=0 or flame.T[-1]-case['T']<200:
                raise ValueError('Flame is not a positive-speed burning solution.')
            row=dict(speed_m_s=speed, burnt_temperature_K=float(flame.T[-1]))
            if settings.get('flame_speed_sensitivities', False):
                sensitivities = flame.get_flame_speed_reaction_sensitivities()
                if not np.isfinite(sensitivities).all():
                    raise ValueError('Nonfinite flame-speed adjoint sensitivities.')
                row['speed_log_rate_sensitivities'] = sensitivities.tolist()
            if settings.get('full_flame_adjoints', False) and all(type(r.rate) is ct.ArrheniusRate for r in gas.reactions()):
                from .flame_adjoint import flame_parameter_adjoints
                row['flame_parameter_adjoints'] = flame_parameter_adjoints(flame)
            if settings.get('save_profile', False):
                row.update(distance_m=flame.grid.tolist(), temperature_K=flame.T.tolist())
            return row
    except Exception as exc:
        return dict(error=f'{type(exc).__name__}: {exc}')
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def flame_residual(rows, references, temperature_scale_K=None, speed_weight=1., temperature_weight=1., profile=None):
    """Weighted log-speed and optional scaled burned-temperature residuals."""
    if len(rows)!=len(references) or not rows or any('error' in r for r in rows): return None
    weights=[speed_weight] if temperature_scale_K is None else [speed_weight,temperature_weight]
    if profile is not None:
        profile=profile_settings(profile)
        weights.append(profile['weight'])
    if not np.isfinite(weights).all() or min(weights)<0 or sum(weights)<=0:
        raise ValueError('Objective weights must be finite, nonnegative, with a positive sum.')
    speed=np.log([r['speed_m_s']/ref['speed_m_s'] for r,ref in zip(rows,references)])
    parts=[np.sqrt(speed_weight)*speed]
    if temperature_scale_K is not None:
        if not np.isfinite(temperature_scale_K) or temperature_scale_K<=0:
            raise ValueError('temperature_scale_K must be positive.')
        temperature=np.array([r['burnt_temperature_K']-ref['burnt_temperature_K'] for r,ref in zip(rows,references)])
        parts.append(np.sqrt(temperature_weight)*temperature/temperature_scale_K)
    if profile is not None:
        try:
            for row,ref in zip(rows,references):
                target=ref.get('profile_reference')
                if target is None: target=prepare_profile_reference(ref,profile)
                errors=profile_residual(row,target)
                parts.append(np.sqrt(profile['weight']/len(errors))*errors)
        except ValueError:
            return None
    values=np.concatenate(parts)/np.sqrt(len(rows)*sum(weights))
    return values if np.isfinite(values).all() else None


def save_flame_comparison(root, cases, comparison, enabled=True, scales=None):
    """Optionally save the flame-speed and burned-temperature comparison figure."""
    if not enabled:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for column,(ax,p) in enumerate(zip(axes[0],sorted({c['P_atm'] for c in cases}))):
        group=[c for c in comparison if c['P_atm']==p]
        ax.plot([c['phi'] for c in group],[c['reference_cm_s'] for c in group],'o-',label='FFCM2')
        ax.plot([c['phi'] for c in group],[c['model_cm_s'] for c in group],'s--',label='Tuned')
        ax.set_yscale(yscale(scales, 'laminar_flame_speed'))
        ax.set(xlabel='Equivalence ratio',ylabel='Flame speed (cm/s)',title=f'{p} atm, 298 K');ax.legend();ax.grid(alpha=.3)
        temp_ax=axes[1,column]
        temp_ax.plot([c['phi'] for c in group],[c['reference_temperature_K'] for c in group],'o-',label='FFCM2')
        temp_ax.plot([c['phi'] for c in group],[c['model_temperature_K'] for c in group],'s--',label='Tuned')
        temp_ax.set_yscale(yscale(scales, 'T_flame'))
        temp_ax.set(xlabel='Equivalence ratio',ylabel='Burned temperature (K)',title=f'{p} atm, 298 K')
        temp_ax.legend();temp_ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)


def run_flame_tuning(cfg):
    """Fit all stages; keep stage reports, exported models, and reference comparisons."""
    if cfg.get('enable_plog', False):
        from .adaptive_flame import run_adaptive_flame
        return run_adaptive_flame(cfg)
    profile=cfg.get('objectives',{}).get('profile')
    if profile is not None: profile=profile_settings(profile)
    root=Path(cfg['output_dir']);root.mkdir(parents=True,exist_ok=False)
    logs=root/'solver_logs';logs.mkdir()
    _save(root/'settings.json',cfg)
    settings=dict(cfg['solver'],save_profile=profile is not None);cases=cfg['cases'];history=[];center=cfg.get('initial_coordinates')
    def registry(stage):
        return StagedParameters(cfg['model'],cfg['reference'], uncertainty_factors=cfg['uncertainty_factors'],
            unbounded_reactions=cfg['artificial'],a_ratio=tuple(cfg['a_ratio']),ea_delta=cfg['ea_delta'],
            b_delta=cfg['b_delta'], plog_free=cfg.get('plog_free'), stage=stage['parameters'],center=center,
            stage_a_ratio=stage['a_ratio'],stage_ea_delta=stage.get('ea_delta',cfg['ea_delta']))
    with create_executor(cfg['backend'],cfg['workers'],max_tasks_per_worker=50) as pool:
        def simulate(sources, solver=settings, *, refine_ignition=False):
            rows=[[None]*len(cases) for _ in sources]
            idt_solver=dict(cfg.get('ignition_solver',{}))
            if refine_ignition and idt_solver:
                idt_solver['rtol']/=10;idt_solver['atol']/=10
            futures={pool.submit(solve_ignition_case if case.get('kind')=='ignition' else solve_flame_case,
                (source,case,idt_solver if case.get('kind')=='ignition' else (dict(solver,flame_speed_sensitivities=False,full_flame_adjoints=False) if source==cfg['reference'] else solver),str(logs))):(j,i)
                for j,source in enumerate(sources) for i,case in enumerate(cases)}
            for future in as_completed(futures):
                j,i=futures[future];rows[j][i]=future.result()
            return rows
        print(f'Computing FFCM2 reference cases: {len(cases)} cases, {cfg["workers"]} workers.',flush=True)
        refs=simulate([cfg['reference']])[0]
        _save(root/'FFCM2_results.json',refs)
        if any('error' in r for r in refs):raise RuntimeError('Reference flames failed; inspect FFCM2_results.json.')
        if profile is not None:
            for case,ref in zip(cases,refs):
                if case.get('kind')!='ignition':ref['profile_reference']=prepare_profile_reference(ref,profile)
            _save(root/'FFCM2_results.json',refs)
        console=sys.stdout
        for stage in cfg['stages']:
            space=registry(stage);folder=root/stage['parameters'];folder.mkdir()
            print(f'\nStage {stage["parameters"]} | sensitivity-selected reactions', flush=True)
            cache={};evaluations=0
            def source(x):
                scheme=space.scheme(x);scheme['description']='Flame tuning candidate'
                return yaml.safe_dump({'description': scheme.pop('description'), **scheme},sort_keys=False)
            def evaluate(points):
                nonlocal evaluations
                keys=[tuple(map(float,p)) for p in points]
                missing=list(dict.fromkeys(k for k in keys if k not in cache))
                valid=[]
                for k in missing:
                    try: space.validate(k);valid.append(k)
                    except (ValueError,OverflowError):cache[k]=(None,None)
                for k,rows in zip(valid,simulate([source(k) for k in valid])):
                    cache[k]=(combined_residual(rows,refs,cfg),rows);evaluations+=1
                return [cache[k][0] for k in keys]
            def jacobian_entries(x):
                """Supply weighted log-speed/A derivatives; other entries remain numerical."""
                from .flame_adjoint import speed_jacobian_entries, full_flame_entries
                evaluate([x])
                residual, rows = cache[tuple(map(float,x))]
                return (full_flame_entries if settings.get('full_flame_adjoints',False) else speed_jacobian_entries)(rows, cfg, space, len(residual))
            initial=evaluate([space.x0])[0]
            if initial is None and center is None:
                rng=np.random.default_rng(cfg['seed']);best=None
                for offset in range(0,cfg['samples'],cfg['sample_batch']):
                    points=np.repeat(space.x0[None,:],min(cfg['sample_batch'],cfg['samples']-offset),axis=0)
                    for j,number in enumerate(space.base.reaction_numbers):
                        if number in cfg['artificial']:
                            points[:,j]=rng.uniform(space.lower[j],space.upper[j],len(points))
                    for x,r in zip(points,evaluate(points)):
                        if r is not None and (best is None or r@r<best[0]):best=(float(r@r),x.copy())
                if best is None:raise RuntimeError('No feasible initial flame candidate. Inspect solver_logs and adjust initialization.')
                space.x0=best[1];initial=evaluate([space.x0])[0]
            elif initial is None:raise RuntimeError('Stage-entry flame calculation failed.')
            interval=(min(c['T'] for c in cases),max(r['burnt_temperature_K'] for r in refs)+300.)
            branches=space.linear_monotonicity(*interval, direction=cfg.get('plog_direction', 'either'))
            def feasible(x):
                try:return space.rate_envelope(x,*interval)['valid']
                except ValueError:return False
            def on_round(info):
                nonlocal branches
                space.x0=np.asarray(info['coordinates'])
                branches=space.linear_monotonicity(*interval, direction=cfg.get('plog_direction', 'either'))
                space.export(folder/'best_model.yaml',space.x0)
                _save(folder/f'round_{info["round"]:02d}.json',info)
                report_round(info,cfg['max_rounds'],console)
            steps=[.02]*space.na+[60./space.energy_yaml]*space.ne+[.005/space.b_scale]*space.ne
            print(f'Baseline loss {float(initial@initial):.8g}; {len(cases)} cases, {cfg["workers"]} workers.',flush=True)
            print(f'  Optimizer diagnostics: {folder / "optimizer.log"}',flush=True)
            with (folder/'optimizer.log').open('w',buffering=1) as diagnostic, redirect_stdout(diagnostic):
                result=iterative_sensitivity_optimize(evaluate,space.x0,space.lower,space.upper,list(space.reaction_numbers),
                    physical_parameters=space.physical_parameters,parameter_scales=space.parameter_scales,
                    max_rounds=cfg['max_rounds'],inner_iterations=cfg['inner_iterations'],top_reactions=5,
                    steps=steps,minimum_step=np.asarray(steps)/2,on_round=on_round,constraints=lambda:branches,feasible=feasible,
                    options=dict(fd_extrapolate=True, fd_min_steps=(np.asarray(steps)/2).tolist(), **({'jacobian_entries':jacobian_entries} if settings.get('flame_speed_sensitivities',False) else {})))
            center=result['bestx'];space.export(folder/'best_model.yaml',center)
            _save(folder/'results.json',result)
            history.append(dict(stage=stage['parameters'],initial_loss=float(initial@initial),loss=result['loss'],status=result['status']))
            # Bound trajectory cache lifetime to one stage.
            cache.clear()
            prediction_settings=dict(settings,flame_speed_sensitivities=False,full_flame_adjoints=False)
            rows=simulate([str(folder/'best_model.yaml')],prediction_settings)[0]
            tighter=dict(prediction_settings,slope=settings['slope']/2,curve=settings['curve']/2)
            refined=simulate([str(folder/'best_model.yaml')],tighter,refine_ignition=True)[0]
            failures=[dict(case=i+1,conditions=case,settings=mode,error=row['error'])
                      for mode,values in [('normal',rows),('tighter',refined)]
                      for i,(case,row) in enumerate(zip(cases,values)) if 'error' in row]
            _save(folder/'validation_cases.json',dict(normal=rows,tighter=refined,failures=failures))
            if failures:raise RuntimeError('Final prediction validation failed: '+json.dumps(failures))
            if combined_residual(rows,refs,cfg) is None or combined_residual(refined,refs,cfg) is None:
                raise RuntimeError('Final objective/profile coverage validation failed.')
            drift=max(abs(a['speed_m_s']/b['speed_m_s']-1) for case,a,b in zip(cases,refined,rows) if case.get('kind')!='ignition')
            idt_pairs=[(a,b) for case,a,b in zip(cases,refined,rows) if case.get('kind')=='ignition']
            idt_drift=max((abs(a['ignition_delay_s']/b['ignition_delay_s']-1) for a,b in idt_pairs),default=0.)
            idt_temperature_drift=max((abs(a['burnt_temperature_K']-b['burnt_temperature_K']) for a,b in idt_pairs),default=0.)
            _save(folder/'validation.json',dict(max_speed_change_fraction=drift,max_idt_change_fraction=idt_drift,
                max_idt_temperature_change_K=idt_temperature_drift,passed=drift<=.02 and idt_drift<=.02 and idt_temperature_drift<=1.))
            if idt_drift>.02 or idt_temperature_drift>1.:raise RuntimeError('Ignition tighter-tolerance validation failed.')
            if drift>.02:raise RuntimeError('Flame refinement changes speed by more than 2%; use a finer grid.')
            interval=(interval[0],max(interval[1],max(r['burnt_temperature_K'] for r in rows+refined)))
            if not space.rate_envelope(center,*interval)['valid']:raise RuntimeError('Final rate uncertainty envelope failed.')
            monotonicity=plog_monotonicity(str(folder/'best_model.yaml'),*interval,direction=cfg.get('plog_direction','either'))
            _save(folder/'monotonicity.json',monotonicity)
            if not monotonicity['valid']:raise RuntimeError('Final PLOG monotonicity validation failed.')
            report_stage(result,float(initial@initial),folder,console)
            space.export(root/'best_model.yaml',center)
            if profile is not None:
                _save(folder/'best_profiles.json',rows)
                _save(folder/'validation_profiles.json',refined)
            comparison=[dict(**case,reference_cm_s=100*ref['speed_m_s'],model_cm_s=100*row['speed_m_s'],
                error_percent=100*(row['speed_m_s']/ref['speed_m_s']-1),
                reference_temperature_K=ref['burnt_temperature_K'],model_temperature_K=row['burnt_temperature_K'],
                temperature_error_K=row['burnt_temperature_K']-ref['burnt_temperature_K']) for case,ref,row in zip(cases,refs,rows) if case.get('kind')!='ignition']
            idt_comparison=ignition_comparison(cases,refs,rows)
            _save(root/'results.json',dict(stages=history,cases=comparison,ignition_cases=idt_comparison,validation_passed=True))
            if idt_comparison and cfg.get('comparison_plots',True):plot_ignition_comparison(root,idt_comparison,cfg.get('plot_yscales'))
            save_flame_comparison(root, [c for c in cases if c.get('kind')!='ignition'], comparison, cfg.get('comparison_plots', True), cfg.get('plot_yscales'))
    print(f'\nTuning finished.\n  Best model: {root / "best_model.yaml"}\n  Summary: {root / "results.json"}',flush=True)
    return dict(space=space, coordinates=center, optimization=result, loss=result['loss'], validation_passed=True)


def launch_flame_tuning(config):
    """Launch from an unguarded example using a protected child interpreter."""
    with tempfile.TemporaryDirectory(prefix='mt-flame-') as folder:
        path=Path(folder)/'settings.json';_save(path,config)
        p=subprocess.Popen([sys.executable,'-u','-m','mt.flame_tuning',str(path)],env=launcher_environment(),start_new_session=True)
        if wait_for_process(p):raise RuntimeError('Flame tuning failed; inspect the output and solver logs.')

if __name__ == '__main__':
    # Execute through the importable module so worker callables are pickled as
    # mt.flame_tuning.solve_flame_case, never as __main__.solve_flame_case.
    from mt.flame_tuning import run_flame_tuning as run
    run(json.loads(Path(sys.argv[1]).read_text()))
