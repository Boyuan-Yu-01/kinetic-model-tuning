"""Staged ZND tuning with optional, evidence-gated PLOG insertion."""
from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
import yaml
from .history import json_value
from .iterative_znd import _save, _registry, run_iterative_znd
from .staged_iterative_znd import stage_config
from .initialization import add_plog
from .plog_adaptation import rank_plog_candidates, transfer_coordinates, check_rate_equivalence


def run_adaptive_znd(config):
    """Fit the supplied structure first; retain a PLOG trial only after validation."""
    if not config.get('compact_output',True):
        raise ValueError('The adaptive workflow requires compact_output=True.')
    if config.get('resume',False):
        raise ValueError('Adaptive structural runs require a fresh output directory.')
    root=Path(config['output_dir']).resolve()
    root.mkdir(parents=True,exist_ok=False)
    _save(root/'settings.json',config)
    stages=config['stages']
    if [s['parameters'] for s in stages]!=['A','AE','AEB']:
        raise ValueError('Provide stages A, AE, AEB in order.')
    active=deepcopy(config)
    active['plog_free']=config.get('plog_free')
    history=[]
    center=None
    interval=[]
    def stage(stage_settings,folder_name,coordinates,*,trial=False):
        coordinates=None if coordinates is None else np.asarray(coordinates,dtype=float).tolist()
        cfg=stage_config(active,stage_settings,coordinates,root/folder_name)
        if coordinates is not None:cfg.pop('initial_search',None)
        if interval:cfg['temperature_interval']=interval
        print(f'\n{folder_name}: {stage_settings["parameters"]} | sensitivity-selected reactions',flush=True)
        result=run_iterative_znd(cfg)
        history.append(dict(stage=folder_name,trial=trial,accepted=not trial,
            status=result['status'],converged=result['optimization']['converged'],
            stop_reason=result['optimization']['status'],rounds=len(result['optimization']['rounds']),
            baseline_loss=result['baseline_loss'],best_loss=result['best_loss']))
        if result['status']=='validation_failed' and not trial:
            _save(root/'failure.json',dict(stage=folder_name,error='Solver or rate constraint validation failed'))
            raise RuntimeError(f'{folder_name} validation failed; next stage not started.')
        return result,cfg
    def publish(result,folder,*,finished=False):
        source=root/folder
        shutil.copyfile(source/'best_model.yaml',root/'best_model.yaml')
        for name in ('profiles.png','temperature_profiles'):
            item=source/name
            if item.is_dir():shutil.copytree(item,root/name,dirs_exist_ok=True)
            elif item.is_file():shutil.copyfile(item,root/name)
        detail=yaml.safe_load((source/'results.yaml').read_text())
        public=dict(status='completed' if finished else 'running',
            converged=bool(result['optimization']['converged']) if finished else False,
            stages=history,validation=detail['validation'],cases=detail['cases'],
            fit_quality=detail.get('fit_quality'),active_bounds=detail.get('active_bounds'),
            best_model='best_model.yaml',comparison_plot='profiles.png',
            active_model=str(active['model']))
        text=yaml.safe_dump(json_value(public),sort_keys=False)
        temporary=root/'results.tmp.yaml';temporary.write_text(text);temporary.replace(root/'results.yaml')
        _save(root/'results.json',dict(public,coordinates=result['coordinates']))
    first,first_cfg=stage(stages[0],'01_A',config.get('initial_coordinates'))
    center=np.asarray(first['coordinates'])
    publish(first,'01_A')
    constraints_path=root/'01_A/run_info/initial_monotonicity.json'
    if not constraints_path.exists():constraints_path=root/'01_A/initial_monotonicity.json'
    initial_constraints=yaml.safe_load(constraints_path.read_text())
    interval=initial_constraints['temperature_range_K']
    proposal=dict(enabled=True,accepted=False,pressure_entries_bar=config.get('plog_pressures_bar',[1.,50.,100.]))
    # Only explicitly artificial reactions may acquire a new rate form automatically.
    old_space=_registry(first_cfg)
    artificial=set(config.get('unbounded_reactions',[]))
    eligible=config.get('plog_candidates',sorted(artificial))
    if not set(eligible)<=artificial:
        raise ValueError('Automatic PLOG candidates must be explicitly artificial reactions.')
    candidates=rank_plog_candidates(first['optimization']['screen'],old_space.scheme(center),eligible,
        top=config.get('plog_max_trials',1),min_relative_score=config.get('plog_min_relative_score',.1))
    proposal['eligible_reactions']=eligible
    proposal['selected_reactions']=candidates
    proposal['ranking']=first['optimization']['screen']['ranking']
    _save(root/'plog_decision.json',proposal)
    accepted=first
    if not candidates:
        print('PLOG: no eligible reaction passed the resolved-sensitivity importance threshold.',flush=True)
    for number in candidates:
        print(f'PLOG trial: R{number}; fixed entries {proposal["pressure_entries_bar"]} bar.',flush=True)
        trial_model=add_plog(active['model'],root/f'inputs/plog_R{number}.yaml',
                             {number:proposal['pressure_entries_bar']})
        original_config=deepcopy(active)
        active['model']=trial_model
        active['plog_free']=list(dict.fromkeys([*(active.get('plog_free') or []),number]))
        trial_settings=dict(stages[0],max_rounds=config.get('plog_trial_rounds',5))
        trial_cfg=stage_config(active,trial_settings,None,root/f'plog_trial_R{number}')
        new_space=_registry(trial_cfg)
        mapped=transfer_coordinates(old_space,new_space,center)
        parity=check_rate_equivalence(old_space,new_space,center,mapped)
        _save(root/f'plog_R{number}_rate_equivalence.json',parity)
        if not parity['valid']:
            raise RuntimeError('PLOG insertion changed the accepted rate before tuning.')
        trial,cfg=stage(trial_settings,f'plog_trial_R{number}',mapped,trial=True)
        gain=(accepted['best_loss']-trial['best_loss'])/max(accepted['best_loss'],1e-30)
        minimum=config.get('plog_min_improvement',.001)
        keep=bool(trial['status']!='validation_failed' and trial['validation'].get('passed') is True and gain>minimum)
        history[-1]['accepted']=keep
        history[-1]['relative_improvement']=gain
        proposal.setdefault('trials',[]).append(dict(reaction=number,accepted=keep,
             relative_improvement=gain,required_improvement=minimum,rate_equivalence=parity))
        if keep:
            proposal['accepted']=True;proposal['reaction']=number
            accepted=trial;center=np.asarray(trial['coordinates']);old_space=_registry(cfg)
            publish(accepted,f'plog_trial_R{number}')
            print(f'PLOG accepted: R{number}; loss improved {100*gain:.3f}%.',flush=True)
            break
        active=original_config
        print(f'PLOG rejected: R{number}; validated improvement criterion not met ({100*gain:.3f}%).',flush=True)
    _save(root/'plog_decision.json',proposal)
    for index,settings in enumerate(stages[1:],2):
        folder=f'{index:02d}_{settings["parameters"]}'
        result,cfg=stage(settings,folder,center)
        center=np.asarray(result['coordinates'])
        publish(result,folder,finished=index==len(stages))
    return yaml.safe_load((root/'results.yaml').read_text())
