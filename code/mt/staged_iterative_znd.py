"""A → A/Ea → A/Ea/b using one immutable mechanism and uncertainty anchor."""
from copy import deepcopy
from pathlib import Path
import json
import hashlib
from datetime import datetime
import os
import shutil
import subprocess
import sys
import tempfile
import yaml
from .iterative_znd import _configuration, _registry, _save, run_iterative_znd
from ._execution import launcher_environment, wait_for_process


def stage_config(config, stage, center, output):
    """Build one stage without rebasing the original Arrhenius shapes or limits."""
    cfg=deepcopy(config)
    cfg.update(staged_parameters=True,tune_ea=True,stage=stage['parameters'],
               stage_center=center,stage_a_ratio=stage['a_ratio'],
               stage_ea_delta=stage.get('ea_delta',cfg['ea_delta']),
               max_rounds=stage.get('max_rounds',cfg.get('max_rounds',20)),output_dir=str(output))
    cfg=_configuration(cfg)
    space=_registry(cfg)
    cfg['steps']=space.sensitivity_steps(cfg.get('a_step',.02),cfg.get('ea_step',60.),cfg.get('b_step',.005))
    cfg['options']=dict(cfg.get('options') or {},step_scales=space.trust_step_scales().tolist())
    return cfg


def run_staged_iterative_znd(config, *, runner=run_iterative_znd):
    """Carry accepted coordinates through stages and publish stage summaries."""
    from .order_tuning import prepare_znd_anchor
    config=prepare_znd_anchor(config,Path(config['output_dir'])/'order_anchor.json')
    if config.get('enable_plog',False):
        from .adaptive_znd import run_adaptive_znd
        return run_adaptive_znd(config)
    stages=config['stages']
    if [s['parameters'] for s in stages]!=['A','AE','AEB']:
        raise ValueError('Provide stages A, AE, AEB in order.')
    root=Path(config['output_dir']).resolve()
    # Preflight all settings before creating an output folder or running solvers.
    for i,stage in enumerate(stages):stage_config(config,stage,None,root/f'{i+1:02d}_{stage["parameters"]}')
    center=config.get('initial_coordinates');history=[];completed=False
    if config.get('resume',False) and root.exists():
        saved=json.loads((root/'settings.json').read_text())
        clean=lambda value: {k:v for k,v in value.items() if k!='resume'}
        from .history import json_value
        if clean(json_value(config)) != clean(saved):
            raise ValueError('Resume requires the original settings; only resume may change.')
        if (root/'results.json').exists():
            previous=json.loads((root/'results.json').read_text())
            if previous.get('status')=='validation_failed':raise ValueError('Cannot resume a validation-failed stage.')
            history=previous['stages'];center=previous['coordinates']
            if [row['stage'] for row in history]!=[row['parameters'] for row in stages[:len(history)]]:
                raise ValueError('Saved stage order differs from the requested stages.')
            if history:
                folder=root/f"{len(history):02d}_{history[-1]['stage']}"
                result=yaml.safe_load((folder/'results.yaml').read_text())
                if result.get('validation',{}).get('passed') is False:
                    raise ValueError('Last completed stage did not pass validation.')
                digest=hashlib.sha256((folder/'best_model.yaml').read_bytes()).hexdigest()
                if digest!=result['model_sha256']:raise ValueError('Saved stage model hash changed.')
                provenance=json.loads((folder/'run_info/provenance.json').read_text())
                for key,record in provenance['inputs'].items():
                    if hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()!=record['sha256']:
                        raise ValueError('Resume input changed: '+key)
            if previous.get('status')=='completed':return previous
        print(f'Resuming after {len(history)} completed stage(s); earlier results preserved.',flush=True)
    else:
        root.mkdir(parents=True,exist_ok=False)
        _save(root/'settings.json',config)
    for i,stage in enumerate(stages):
        if i<len(history):continue
        folder=root/f'{i+1:02d}_{stage["parameters"]}'
        if folder.exists():
            archive=root/'failed_attempts';archive.mkdir(exist_ok=True)
            folder.rename(archive/(folder.name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
        cfg=stage_config(config,stage,center,folder)
        if i: cfg.pop('initial_search',None)
        print(f'\nStage {i+1}/3: {stage["parameters"]} | sensitivity-selected reactions',flush=True)
        result=runner(cfg)
        history.append(dict(stage=stage['parameters'],status=result['status'],
                            converged=result['optimization']['converged'],
                            stop_reason=result['optimization']['status'],
                            baseline_loss=result['baseline_loss'],best_loss=result['best_loss'],
                            rounds=len(result['optimization']['rounds'])))
        if result['status']=='validation_failed':
            _save(root/'results.json',dict(status='validation_failed',stages=history))
            raise RuntimeError('Stage validation failed; next stage was not started.')
        center=list(result['coordinates'])
        temp=root/'best_model.tmp.yaml';shutil.copyfile(folder/'best_model.yaml',temp);temp.replace(root/'best_model.yaml')
        completed=i==len(stages)-1
        # Publish final comparison artifacts at the top level for easy inspection.
        for name in ('profiles.png','temperature_profiles'):
            item=folder/name
            if item.is_dir():shutil.copytree(item,root/name,dirs_exist_ok=True)
            elif item.is_file():shutil.copyfile(item,root/name)
        last=yaml.safe_load((folder/'results.yaml').read_text())
        public=dict(status='completed' if completed else 'running',
            converged=result['optimization']['converged'] if completed else False,
            stages=history, validation=last['validation'], cases=last['cases'],
            fit_quality=last.get('fit_quality'),active_bounds=last.get('active_bounds'),
            best_model='best_model.yaml', comparison_plot='profiles.png')
        (root/'results.yaml').write_text(yaml.safe_dump(public,sort_keys=False))
        _save(root/'results.json',dict(status='completed' if completed else 'running',
            converged=result['optimization']['converged'] if completed else False,
            stages=history,coordinates=center,best_model='best_model.yaml'))
    return json.loads((root/'results.json').read_text())


def launch_staged_iterative_znd(config):
    """Run from a plain script; multiprocessing stays in a guarded subprocess."""
    with tempfile.TemporaryDirectory(prefix='staged_iterative_znd_') as folder:
        request=Path(folder)/'request.json';_save(request,config)
        process=subprocess.Popen([sys.executable,'-u','-m','mt.staged_iterative_znd',str(request)],
                                 env=launcher_environment(),start_new_session=os.name=='posix')
        code=wait_for_process(process)
        if code:raise RuntimeError(f'Staged ZND exited with status {code}; inspect {config["output_dir"]}.')
    return json.loads((Path(config['output_dir'])/'results.json').read_text())


if __name__=='__main__':
    run_staged_iterative_znd(json.loads(Path(sys.argv[1]).read_text()))
