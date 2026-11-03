from .plog_settings import resolve_plog_settings
"""Organized ZND settings and sequential multistart with one shared worker budget."""
from copy import deepcopy
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import yaml
import numpy as np
from .iterative_znd import _save, run_iterative_znd
from .staged_iterative_znd import stage_config, run_staged_iterative_znd
from ._execution import launcher_environment, wait_for_process


def resolve_settings(problem, tuning, execution):
    """Translate the public groups into the existing numerical configuration."""
    from .workflow_defaults import NUMERICS
    cfg = deepcopy(NUMERICS)
    cfg.update(tuning.get('advanced', {}))
    cfg.update(model=problem['model'], reference=problem['reference'],
               output_dir=problem['output'], **problem['mixture'])
    cases = problem['conditions']['znd']
    cfg.update(P_bar=cases['pressure_bar'], T0=cases['temperature_K'],
               phi=cases['phi'], t_end=cases['end_time_s'])
    from .order_tuning import order_specs
    cfg['orders']=order_specs(problem['reactions'].get('orders'))
    cfg.update(uncertainty_factors=problem['reactions']['uncertainty_factors'],
               unbounded_reactions=problem['reactions']['artificial'])
    objectives = problem['objectives']
    if any(o.get('weight', 1) != 1 for o in objectives.values()):
        raise ValueError('This staged workflow currently requires equal objective weights.')
    if any('factor' in objectives.get(k, {}) for k in ('ind_len_ZND', 'exo_len_ZND')):
        raise ValueError('Length objectives use unscaled natural-log errors; omit factor.')
    allowed = ('T_profile', 'T_ZND', 'ind_len_ZND', 'exo_len_ZND')
    if not objectives or set(objectives)-set(allowed):
        raise ValueError('Select at least one supported ZND objective.')
    cfg.update(objective_names=list(objectives), length_factor=None,
               temperature_scale=objectives.get('T_ZND', {}).get('scale', 25.))
    scale = objectives.get('T_profile', {}).get('scale', 25.)
    if isinstance(scale, dict):
        cfg['profile_fraction'] = scale['reference_temperature_rise_fraction']
    else:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Profile temperature scale must be finite and positive.')
        cfg['profile_scale_K'] = float(scale)
    cfg.update(tuning['limits'])
    cfg.update(tuning['stopping'])
    cfg.update(resolve_plog_settings(tuning['plog'], problem['model']))
    cfg.update(execution)
    cfg['stages'] = deepcopy(tuning['stages'])
    cfg['top_reactions'] = tuning['selection']['top_reactions']
    search = deepcopy(tuning['exploration'])
    search['finalists'] = search.pop('keep_candidates')
    cfg['initial_search'] = search
    cfg['multistart'] = deepcopy(tuning['multistart'])
    starts = cfg['multistart']['starts']
    if isinstance(starts, bool) or not isinstance(starts, int) or not 1 <= starts <= search['finalists']:
        raise ValueError('starts must be an integer between 1 and keep_candidates.')
    if cfg['multistart'].get('selection', 'best') != 'best':
        raise ValueError('Supported multistart selection: best.')
    cfg.setdefault('reference_cache', str(Path(cfg['output_dir'])/'reference_cache'))
    return cfg


def run_multistart(cfg):
    """Explore once, optimize verified seeds separately, and retain every result."""
    root = Path(cfg['output_dir']).resolve()
    from .history import json_value
    from datetime import datetime
    resuming = cfg.get('resume', False)
    if resuming:
        saved = json.loads((root/'settings.json').read_text())
        clean = lambda c: {k:v for k,v in json_value(c).items() if k != 'resume'}
        if clean(saved) != clean(cfg):
            raise ValueError('Resume settings differ; only resume may change.')
    else:
        root.mkdir(parents=True, exist_ok=False)
        _save(root/'settings.json', cfg)
    def archive(folder):
        if folder.exists():
            target=root/'diagnostics'; target.mkdir(exist_ok=True)
            folder.rename(target/(folder.name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))

    from .order_tuning import prepare_znd_anchor
    cfg=prepare_znd_anchor(cfg,root/'order_anchor.json')
    exploration = stage_config(cfg, cfg['stages'][0], None, root/'exploration')
    exploration.update(exploration_only=True, compact_output=False)
    if not (root/'exploration/verified_candidates.json').exists():
        archive(root/'exploration')
        run_iterative_znd(exploration)
    candidates = json.loads((root/'exploration/verified_candidates.json').read_text())
    selected = candidates[:cfg['multistart']['starts']]
    if not selected:
        raise RuntimeError('No verified candidates are available.')
    print(f'Multistart: optimizing {len(selected)}/{cfg["multistart"]["starts"]} requested starts; sequential starts, {cfg["workers"]} workers each.', flush=True)
    rows = []
    for index, candidate in enumerate(selected, 1):
        name = f'start_{index:02d}'
        child = deepcopy(cfg)
        child.update(output_dir=str(root/name), initial_coordinates=candidate['coordinates'])
        child['resume'] = False
        child.pop('initial_search', None)
        child.pop('multistart', None)
        print(f'\nStart {index}/{len(selected)} | initial loss {candidate["loss"]:.8g}', flush=True)
        try:
            saved_result = root/name/'results.json'
            result = json.loads(saved_result.read_text()) if resuming and saved_result.exists() else {}
            if result.get('status') != 'completed':
                archive(root/name)
                result = run_staged_iterative_znd(child)
            if 'validation' not in result and (root/name/'results.yaml').exists():
                result['validation'] = yaml.safe_load((root/name/'results.yaml').read_text())['validation']
            loss = result['stages'][-1]['best_loss']
            passed = result.get('validation', {}).get('passed') is True
            row = dict(start=name, initial_loss=candidate['loss'], final_loss=loss,
                       status=result['status'], validation_passed=passed,
                       stop_reason=result['stages'][-1]['stop_reason'])
        except Exception as exc:
            row = dict(start=name, initial_loss=candidate['loss'], status='failed', error=str(exc))
        rows.append(row)
        valid = [r for r in rows if r.get('validation_passed')]
        best = min(valid, key=lambda r:r['final_loss']) if valid else None
        if best:
            shutil.copyfile(root/best['start']/'best_model.yaml', root/'best_model.yaml')
        report = dict(status='running' if index < len(selected) else ('completed' if best else 'failed'),
                      starts=rows, best_start=None if best is None else best['start'])
        _save(root/'results.json', report)
        (root/'results.yaml').write_text(yaml.safe_dump(report, sort_keys=False))
    if not best:
        raise RuntimeError('No optimized start passed validation; inspect results.yaml.')
    return report


def run_tuning(problem, tuning, execution):
    """Launch safely from a plain run file, including on spawn-based machines."""
    cfg = resolve_settings(problem, tuning, execution)
    with tempfile.TemporaryDirectory(prefix='mt_multistart_') as folder:
        request = Path(folder)/'request.json'
        _save(request, cfg)
        process = subprocess.Popen([sys.executable, '-u', '-m', 'mt.multistart_znd', str(request)],
            env=launcher_environment(), start_new_session=os.name == 'posix')
        code = wait_for_process(process)
        if code:
            raise RuntimeError(f'Multistart exited with status {code}; inspect {cfg["output_dir"]}.')
    return json.loads((Path(cfg['output_dir'])/'results.json').read_text())


if __name__ == '__main__':
    run_multistart(json.loads(Path(sys.argv[1]).read_text()))
