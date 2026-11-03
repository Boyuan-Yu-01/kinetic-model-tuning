"""Sensitivity-gated flame PLOG trials using immutable mechanism anchors."""
from copy import deepcopy
from pathlib import Path
import shutil
import json
from .initialization import add_plog
from .plog_adaptation import rank_plog_candidates, transfer_coordinates, check_rate_equivalence
from .staged_parameters import StagedParameters
from .iterative_znd import _save


def run_adaptive_flame(config, runner=None):
    """Fit A, validate a structural trial, then continue AE/AEB from accepted rates."""
    if runner is None:
        from .flame_tuning import run_flame_tuning
        runner=run_flame_tuning
    cfg=deepcopy(config)
    eligible=cfg.get('plog_candidates',cfg['artificial'])
    if not set(eligible)<=set(cfg['artificial']):
        raise ValueError('PLOG candidates must be explicitly artificial reactions.')
    root=Path(cfg['output_dir']);root.mkdir(parents=True,exist_ok=False)
    _save(root/'settings.json',cfg)
    cfg['enable_plog']=False
    stages=cfg['stages'];history=[]
    def run(active, stage, folder, center=None):
        settings=deepcopy(active)
        settings.update(stages=[stage],output_dir=str(root/folder),initial_coordinates=center)
        settings['max_rounds']=stage.get('max_rounds',cfg['max_rounds'])
        return runner(settings)
    def publish(result, folder):
        source=root/folder
        for name in ('best_model.yaml','comparison.png','ignition_comparison.png'):
            if (source/name).exists():shutil.copy2(source/name,root/name)
        report=json.loads((source/'results.json').read_text()) if (source/'results.json').exists() else {}
        report.update(stages=history,best_loss=result['loss'],best_folder=folder)
        _save(root/'results.json',report)
    accepted=run(cfg,stages[0],'01_A',cfg.get('initial_coordinates'))
    center=accepted['coordinates'];old_space=accepted['space']
    if cfg.get('orders'):cfg['order_anchor']=old_space.order_anchor
    history.append(dict(stage='A',loss=accepted['loss'],status=accepted['optimization']['status']))
    publish(accepted,'01_A')
    candidates=rank_plog_candidates(accepted['optimization']['screen'],old_space.scheme(center),eligible,
        top=cfg.get('plog_max_trials',1),min_relative_score=cfg.get('plog_min_relative_score',.1))
    decision=dict(accepted=False,selected_reactions=candidates,pressure_entries_bar=cfg.get('plog_pressures_bar',[1.,50.,100.]),trials=[])
    _save(root/'plog_decision.json',decision)
    for number in candidates:
        trial=deepcopy(cfg)
        trial['model']=str(add_plog(cfg['model'],root/f'inputs/plog_R{number}.yaml',{number:decision['pressure_entries_bar']}))
        trial['plog_free']=list(dict.fromkeys([*(cfg.get('plog_free') or []),number]))
        space=StagedParameters(trial['model'],trial['reference'],uncertainty_factors=trial['uncertainty_factors'],
            unbounded_reactions=trial['artificial'],a_ratio=trial['a_ratio'],ea_delta=trial['ea_delta'],b_delta=trial['b_delta'],
            plog_free=trial['plog_free'],stage='A',stage_a_ratio=stages[0]['a_ratio'],
            orders=trial.get('orders'),order_anchor=trial.get('order_anchor'),rate_bounds=trial.get('rate_bounds'))
        mapped=transfer_coordinates(old_space,space,center)
        parity=check_rate_equivalence(old_space,space,center,mapped)
        if not parity['valid']:raise RuntimeError('PLOG insertion changed rates before tuning.')
        folder=f'plog_trial_R{number}'
        print(f'PLOG trial: R{number}, {decision["pressure_entries_bar"]} bar.',flush=True)
        try:
            result=run(trial,dict(stages[0],max_rounds=cfg.get('plog_trial_rounds',5)),folder,mapped)
        except RuntimeError as exc:
            decision['trials'].append(dict(reaction=number,accepted=False,error=str(exc)))
            _save(root/'plog_decision.json',decision)
            continue
        gain=(accepted['loss']-result['loss'])/max(accepted['loss'],1e-30)
        keep=bool(result['validation_passed'] and gain>cfg.get('plog_min_improvement',.001))
        decision['trials'].append(dict(reaction=number,accepted=keep,relative_improvement=gain,rate_equivalence=parity))
        if keep:
            cfg=trial;accepted=result;center=result['coordinates'];decision['accepted']=True
            history.append(dict(stage=folder,loss=result['loss'],status=result['optimization']['status']))
            publish(result,folder)
        _save(root/'plog_decision.json',decision)
        print(f'PLOG {"accepted" if keep else "rejected"}: improvement {gain:.3%}.',flush=True)
        if keep:break
    for index,stage in enumerate(stages[1:],2):
        folder=f'{index:02d}_{stage["parameters"]}'
        accepted=run(cfg,stage,folder,center);center=accepted['coordinates']
        history.append(dict(stage=stage['parameters'],loss=accepted['loss'],status=accepted['optimization']['status']))
        publish(accepted,folder)
    return accepted
