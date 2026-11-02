"""Compact, unit-labelled reports for the iterative ZND writer.

Publishing a running snapshot never removes writer files. Final cleanup is an
explicit caller assertion that the writer and its workers have exited.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import tempfile

import yaml


_ESSENTIAL = (
    'settings.json', 'provenance.json', 'bounds.json', 'uncertainty_audit.json',
    'checkpoint.json', 'initial_monotonicity.json', 'failure.json',
    'initial_search.json', 'feasible_start.json', 'original_baseline_failures.json',
)
_WRITER_FILES = _ESSENTIAL + (
    'summary.json', 'prepared_states.json', 'reference_profiles.json',
    'baseline_profiles.json', 'best_profiles.json', 'validation_profiles.json',
    'initial_sensitivity.json', 'evaluations.jsonl', 'best_model.yaml', 'profiles.png',
)


def _read(root, name, default=None):
    path = root / name
    if not path.is_file():
        path = root / 'run_info' / name
    return json.loads(path.read_text()) if path.is_file() else default


def _finite(value):
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, float):
        return float(format(value, ".8e")) if math.isfinite(value) else None
    return value


def _atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class _ReportDumper(yaml.SafeDumper):
    """Print report floats consistently; mechanisms retain their original bytes."""


def _report_float(dumper, value):
    return dumper.represent_scalar('tag:yaml.org,2002:float', format(value, '.8e'))


_ReportDumper.add_representer(float, _report_float)


def _save_yaml(path, value):
    clean = _finite(value)
    data = yaml.dump(clean, Dumper=_ReportDumper, sort_keys=False, allow_unicode=True).encode()
    _atomic_bytes(path, data)


def _copy(source, target):
    if source.is_symlink():
        raise ValueError('Refusing to copy a symbolic-link output: ' + str(source))
    if source.resolve() != target.resolve():
        data = source.read_bytes()
        _atomic_bytes(target, data)
        if target.read_bytes() != data:
            raise IOError('Output copy verification failed: ' + str(target))


def _percent(value, reference):
    if value is None or reference is None or reference == 0:
        return None
    return 100. * (value / reference - 1.)


def _exp(value):
    try:
        return math.exp(value) if value is not None else None
    except OverflowError:
        return None


def _at(values, index, default=None):
    if isinstance(values, (int, float)):
        return values
    return values[index] if values and index < len(values) else default


def _case_metrics(row, reference):
    row = row or {}
    reference = reference or {}
    temperature = row.get('T_final', row.get('final_temperature_K'))
    induction = row.get('ind_len_ZND', row.get('induction_length_m'))
    exothermic = row.get('exo_len_ZND', row.get('exothermic_length_m'))
    ref_temperature = reference.get('T_final', reference.get('final_temperature_K'))
    ref_induction = reference.get('ind_len_ZND', reference.get('induction_length_m'))
    ref_exothermic = reference.get('exo_len_ZND', reference.get('exothermic_length_m'))
    rmse = row.get('profile_rmse_K')
    if rmse is None and row.get('grid_T') and reference.get('grid_T'):
        delta = [a - b for a, b in zip(row['grid_T'], reference['grid_T'])]
        rmse = math.sqrt(sum(v * v for v in delta) / len(delta))
    return dict(final_temperature_K=temperature,
                temperature_error_K=(temperature - ref_temperature
                    if temperature is not None and ref_temperature is not None else None),
                profile_rmse_K=rmse, induction_length_m=induction,
                induction_error_percent=_percent(induction, ref_induction),
                exothermic_length_m=exothermic,
                exothermic_error_percent=_percent(exothermic, ref_exothermic))


def _round_record(info):
    screen = info.get('screen') or {}
    ranking = screen.get('ranking') or []
    changes = info.get('relative_changes')
    maximum = (100. * max(changes, default=0.)
               if changes is not None and all(v is not None for v in changes) else None)
    return dict(convergence_blockers=info.get('convergence_blockers',[]),inner_status=info.get('inner_status'),
                relative_loss_reduction=info.get('relative_loss_reduction'),stagnation_count=info.get('stagnation_count'),
                update_audit=info.get('update_audit'),active_bounds=info.get('active_bounds'),round=info.get('round'), loss=info.get('loss'),
                status=info.get('status'), converged=info.get('converged'),
                selected_reactions=info.get('selected_groups', []),
                max_A_change_percent=maximum, change_measure='Relative A change; when Ea is enabled, also scaled Ea/b changes using configured limits',
                threshold_percent=(100. * info['threshold'] if 'threshold' in info else None),
                all_changes_below_threshold=info.get('all_changes_below_threshold'),
                all_reactions_audited=info.get('all_free_audit'),
                all_sensitivities_resolved=screen.get('all_resolved'),
                unresolved_reactions=[r['group'] for r in ranking if not r.get('all_resolved')],
                projected_gradient=screen.get('projected_gradient'),
                stationary=screen.get('stationary'),
                sensitivity_ranking=[dict(reaction=r['group'], score=r.get('score'),
                    resolved=r.get('all_resolved')) for r in ranking])


def _accepted_snapshot(root):
    """Reject model/checkpoint pairs caught between the two legacy replacements."""
    checkpoint_path = root / 'checkpoint.json'
    if not checkpoint_path.is_file():
        checkpoint_path = root / 'run_info' / 'checkpoint.json'
    model_path = root / 'best_model.yaml'
    for _ in range(3):
        before = checkpoint_path.read_bytes() if checkpoint_path.is_file() else None
        model_data = model_path.read_bytes() if model_path.is_file() else None
        after = checkpoint_path.read_bytes() if checkpoint_path.is_file() else None
        if before != after:
            continue
        checkpoint = json.loads(before) if before is not None else {}
        model = yaml.safe_load(model_data) if model_data is not None else {}
        parameters = checkpoint.get('physical_parameters')
        if parameters is not None and model_data is not None:
            actual = []
            for reaction in model.get('reactions', []):
                for key in ('rate-constant', 'low-P-rate-constant',
                            'high-P-rate-constant', 'rate-constants'):
                    if key in reaction:
                        entries = reaction[key] if key == 'rate-constants' else [reaction[key]]
                        actual.extend(float(entry[field]) for entry in entries for field in checkpoint.get('physical_parameter_fields',['A']))
            if len(actual) != len(parameters) or not all(
                    math.isclose(a, b, rel_tol=1e-12, abs_tol=0.)
                    for a, b in zip(actual, parameters)):
                continue
        return checkpoint, model, model_data
    raise RuntimeError('Accepted model/checkpoint changed while publishing; retry after the checkpoint completes.')


def write_iterative_results(work_dir, output_dir=None):
    """Publish atomic compact YAML and available plots from a legacy output.

    Safe to call during a run: only accepted checkpoints are reported, and no
    writer-owned file is removed or rewritten. A model/checkpoint race raises
    RuntimeError so the caller can retry. Relative and percentage units are
    explicit; missing validation is represented by null and not_performed.
    """
    root = Path(work_dir).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else root
    if (root / 'results.yaml').is_file() and not (root / 'settings.json').is_file():
        # Finalized output no longer has raw trajectories or the large summary.
        # Reading it again must not turn a finished report into a running one.
        if output != root:
            raise ValueError('This run is already compacted; copy the output directory to relocate it.')
        existing = yaml.safe_load((root / 'results.yaml').read_text())
        return dict(status=existing['status'], results=existing,
                    artifacts=dict(results=str(root / 'results.yaml'), rounds=str(root / 'rounds.yaml')))
    cfg = _read(root, 'settings.json', {})
    summary = _read(root, 'summary.json', {})
    failure = _read(root, 'failure.json', {})
    checkpoint, model, model_data = _accepted_snapshot(root)
    refs = _read(root, 'reference_profiles.json', [])
    baseline = _read(root, 'baseline_profiles.json', [])
    bounds = _read(root, 'bounds.json', {})
    audit = _read(root, 'uncertainty_audit.json', [])
    provenance = _read(root, 'provenance.json', {})
    optimization = summary.get('optimization') or {}
    infos = optimization.get('rounds')
    if infos is None:
        infos = []
        for path in sorted(root.glob('round_[0-9][0-9][0-9]/sensitivity.json')):
            info = json.loads(path.read_text())
            if checkpoint.get('round') is not None and info.get('round', 0) <= checkpoint['round']:
                infos.append(info)
    rounds = [_round_record(info) for info in infos]
    latest = rounds[-1] if rounds else {}
    screen = optimization.get('screen') or ((infos[-1].get('screen') or {}) if infos else {})
    accepted = summary.get('cases', checkpoint.get('cases', []))
    baseline_metrics = summary.get('baseline_cases', baseline)
    cases = []
    for index in range(max(len(refs), len(baseline_metrics), len(accepted), len(cfg.get('P_bar', [])))):
        ref = _at(refs, index, {})
        reference = {key: value for key, value in _case_metrics(ref, {}).items()
                     if key in ('final_temperature_K', 'induction_length_m', 'exothermic_length_m')}
        cases.append(dict(case=index + 1, pressure_bar=_at(cfg.get('P_bar'), index),
            initial_temperature_K=_at(cfg.get('T0'), index), phi=_at(cfg.get('phi'), index),
            end_time_s=_at(cfg.get('t_end'), index), reference=reference,
            baseline=_case_metrics(_at(baseline_metrics, index, {}), ref),
            accepted=_case_metrics(_at(accepted, index, {}), ref)))
    baseline_loss = summary.get('baseline_loss')
    if baseline_loss is None and refs and len(baseline) == len(refs):
        from .iterative_znd import _score
        residual, _ = _score(baseline, refs, cfg)
        if residual is not None:
            baseline_loss = float(residual @ residual)
    best_loss = summary.get('best_loss', checkpoint.get('loss'))
    status = 'failed' if failure else summary.get('status', 'running')
    validation = summary.get('validation') or {}
    performed = validation.get('performed', False)
    check = summary.get('monotonicity', checkpoint.get('monotonicity', {}))
    threshold = cfg.get('threshold')
    coordinates = checkpoint.get('coordinates', summary.get('coordinates', []))
    numbers = checkpoint.get('reaction_numbers', bounds.get('reaction_numbers', []))
    audit_by_number = {r['reaction']: r for r in audit if 'kind' in r or r.get('status') == 'fixed'}
    reactions = []
    for index, number in enumerate(numbers):
        item = audit_by_number.get(number, {})
        multiplier = _at(checkpoint.get('rate_multipliers_from_source'), index)
        coordinate = _at(coordinates, index)
        origin = _at(bounds.get('source_coordinates'), index)
        coordinate_label = _at(checkpoint.get('coordinate_labels', bounds.get('coordinate_labels')), index, {})
        is_ea = coordinate_label.get('parameter') == 'scaled_ea'
        is_b = coordinate_label.get('parameter') == 'scaled_b'
        if multiplier is None and coordinate is not None and origin is not None and not is_ea and not is_b:
            multiplier = _exp(coordinate - origin)
        equation = _at(model.get('reactions'), number - 1, {}).get('equation')
        coordinate_label = _at(checkpoint.get('coordinate_labels', bounds.get('coordinate_labels')), index, {})
        reactions.append(dict(reaction=number, equation=equation, coordinate=coordinate_label,
            uncertainty_factor=item.get('factor'), bound_policy=item.get('kind'),
            nominal_reaction=item.get('nominal_reaction'),
            multiplier_from_source=multiplier, change_from_source_percent=_percent(multiplier, 1.),
            multiplier_from_nominal=(_exp(coordinate) if item.get('nominal_reaction') and not cfg.get('tune_ea') else None),
            ea_change_kJ_per_mol=(coordinate*coordinate_label['scale'] if is_ea and coordinate is not None else None),
            b_change=(coordinate*coordinate_label['scale'] if is_b and coordinate is not None else None)))
    result = dict(schema_version=1, status=status,fit_quality=summary.get('fit_quality'),active_bounds=summary.get('active_bounds'),
        change_measure=('max(relative A change, scaled Ea/b changes using configured limits)' if cfg.get('tune_ea') else 'relative A change'),
        accepted_round=checkpoint.get('round'),
        converged=optimization.get('converged', False if not summary else None),
        stop_reason=('failure' if failure else optimization.get('status')),
        model='best_model.yaml' if model_data is not None else None,
        model_sha256=hashlib.sha256(model_data).hexdigest() if model_data is not None else None,
        fit=dict(baseline_loss=baseline_loss, accepted_loss=best_loss,
            reduction_percent=(100. * (baseline_loss - best_loss) / baseline_loss
                if baseline_loss and best_loss is not None else None)),
        convergence=dict(threshold_percent=100. * threshold if threshold is not None else None,
            max_round_A_change_percent=latest.get('max_A_change_percent'),
            all_changes_below_threshold=latest.get('all_changes_below_threshold'),
            all_sensitivities_resolved=screen.get('all_resolved'),
            unresolved_reactions=[r['group'] for r in screen.get('ranking', []) if not r.get('all_resolved')],
            projected_gradient=screen.get('projected_gradient'), stationary=screen.get('stationary')),
        validation=dict(status=('passed' if validation.get('passed') else 'failed') if performed else 'not_performed',
            performed=performed, passed=validation.get('passed'),
            max_temperature_change_K=validation.get('max_temperature_change_K'),
            max_length_change_percent=validation.get('max_length_change_percent'),
            tighter_loss=validation.get('tighter_loss'),
            errors=[dict(case=i + 1, error=error)
                    for i, error in enumerate(validation.get('errors', [])) if error],
            plog_monotonicity_status=('not_applicable' if check.get('applicable') is False else 'checked'),
            plog_monotonicity_passed=(None if check.get('applicable') is False else check.get('pressure_monotonicity_valid',check.get('valid'))),
            rate_uncertainty_passed=check.get('rate_envelope',{}).get('valid'),
            export_max_relative_rate_error=summary.get('export_max_relative_rate_error')),
        evaluations_at_accepted_state=summary.get('evaluations', checkpoint.get('evaluations')),
        elapsed_seconds=summary.get('elapsed_seconds', failure.get('elapsed_seconds')),
        reactions=reactions, cases=cases, provenance=provenance)
    if cfg.get('tune_ea'):
        result['convergence']['max_scaled_parameter_change_percent'] = result['convergence'].pop('max_round_A_change_percent')
        for row in rounds: row['max_scaled_parameter_change_percent'] = row.pop('max_A_change_percent')
    if failure:
        result['failure'] = failure.get('error', failure)
    if model_data is not None and output != root:
        _atomic_bytes(output / 'best_model.yaml', model_data)
    artifacts = dict(results=str(output / 'results.yaml'), rounds=str(output / 'rounds.yaml'))
    for path in [root / 'profiles.png', *sorted((root / 'temperature_profiles').glob('case_[0-9][0-9].png'))]:
        if path.is_file():
            destination = output / path.relative_to(root)
            _copy(path, destination)
    snapshots = output / 'run_info' / 'input_snapshots.yaml'
    if snapshots.is_file():
        result['provenance'] = dict(provenance, input_snapshots='run_info/input_snapshots.yaml')
    _save_yaml(output / 'rounds.yaml', dict(schema_version=1, rounds=rounds))
    _save_yaml(output / 'results.yaml', result)
    return dict(status=status, artifacts=artifacts, results=_finite(result))


def _known_files(root):
    files = [root / name for name in _WRITER_FILES if (root / name).is_file()]
    for folder in root.iterdir():
        if folder.is_symlink() or not folder.is_dir():
            continue
        if re.fullmatch(r'round_\d{3}', folder.name):
            files.extend(folder / name for name in ('model.yaml', 'accepted.json', 'sensitivity.json')
                         if (folder / name).is_file())
        elif folder.name == 'solver_logs':
            files.extend(path for path in folder.iterdir()
                         if re.fullmatch(r'worker_\d+\.log', path.name) and path.is_file())
        elif folder.name == 'temperature_profiles':
            files.extend(path for path in folder.iterdir()
                         if re.fullmatch(r'case_\d{2}\.png', path.name) and path.is_file())
    return [path for path in files if not path.is_symlink()]


def finalize_iterative_output(work_dir, output_dir, diagnostics=False, *, cleanup=False):
    """Publish a finished run and preserve reproducibility inputs in run_info.

    A summary or failure record is required. By default nothing is removed.
    cleanup=True MUST be called only when all writing has finished and workers
    have exited (or after the launcher has waited for the writer to exit); completion files alone cannot prove inactivity.
    Cleanup touches only recognised writer files, after verified copies/reports.
    Unknown files and all external input paths are left untouched.
    """
    root, output = Path(work_dir).resolve(), Path(output_dir).resolve()
    if not ((root / 'summary.json').is_file() or (root / 'failure.json').is_file()):
        raise RuntimeError('Cannot finalize a run without summary.json or failure.json.')
    files = _known_files(root)
    cfg = _read(root, 'settings.json', {})
    provenance = _read(root, 'provenance.json', {})
    run_info = output / 'run_info'
    for name in _ESSENTIAL:
        if (root / name).is_file():
            _copy(root / name, run_info / name)
    snapshots = {}
    for key in ('model', 'reference', 'manifest'):
        if cfg.get(key) is None:
            continue
        source = Path(cfg[key])
        expected = provenance.get('inputs', {}).get(key, {}).get('sha256')
        if not source.is_file() and expected is None:
            # Input-validation failures can occur before an input was readable.
            snapshots[key] = dict(original_path=str(source), status='unavailable')
            continue
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError('Input changed since run began; retaining diagnostics: ' + str(source))
        suffix = source.suffix or ('.json' if key == 'manifest' else '.yaml')
        target = run_info / 'inputs' / (key + suffix)
        _atomic_bytes(target, data)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise IOError('Input snapshot verification failed: ' + str(target))
        snapshots[key] = dict(path=str(target.relative_to(output)), original_path=str(source), sha256=digest)
    _save_yaml(run_info / 'input_snapshots.yaml', snapshots)
    if diagnostics:
        for source in files:
            _copy(source, output / 'diagnostics' / source.relative_to(root))
    published = write_iterative_results(root, output)
    # Verify the published report and its accepted model before deleting evidence.
    verified = yaml.safe_load((output / 'results.yaml').read_text())
    if verified != published['results']:
        raise IOError('Published result verification failed.')
    if verified.get('model_sha256'):
        if hashlib.sha256((output / 'best_model.yaml').read_bytes()).hexdigest() != verified['model_sha256']:
            raise IOError('Published accepted model verification failed.')
    protected = {output / 'best_model.yaml', output / 'profiles.png'}
    protected.update((output / 'temperature_profiles').glob('case_[0-9][0-9].png'))
    removable = [path for path in files if path not in protected]
    published['cleanup_candidates'] = [str(path) for path in removable]
    published['run_info'] = str(run_info)
    published['diagnostics'] = str(output / 'diagnostics') if diagnostics else None
    if cleanup:
        for path in removable:
            path.unlink()
        for parent in sorted({path.parent for path in removable if path.parent != root}, key=str, reverse=True):
            try:
                parent.rmdir()
            except OSError:
                pass  # Preserve folders containing unrecognised files.
        if root != output:
            try:
                root.rmdir()
            except OSError:
                pass
    return published


def plot_round_sensitivity(event, output_dir, equations=None):
    """Plot the pre-round aggregate residual ranking, not signed length sensitivity.

    Scores are those used for reaction selection, including feasible-step
    scaling. Unresolved reactions are listed separately, never plotted as zero.
    This consumes the existing screen and performs no additional simulations.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    equations = equations or {}
    selected = set(event['selected_groups'])
    ranking = event['screen']['ranking']
    resolved = sorted((r for r in ranking if r['all_resolved']),
                      key=lambda r: -abs(r['score']))
    unresolved = [str(r['group']) for r in ranking if not r['all_resolved']]
    labels = [f"R{r['group']}: {equations[r['group']]}" if r['group'] in equations
              else f"R{r['group']}" for r in resolved]
    fig, ax = plt.subplots(figsize=(11, max(4, .36 * len(resolved) + 2)))
    try:
        positions = list(range(len(resolved)))
        ax.barh(positions, [r['score'] for r in resolved],
                color=['#2474b5' if r['group'] in selected else '#a8afb7' for r in resolved])
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set(xlabel='Aggregate residual sensitivity score (nonnegative)',
               title=f"Round {event['round']} — sensitivity before optimization")
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(handles=[Patch(color='#2474b5', label='Selected this round'),
                           Patch(color='#a8afb7', label='Other resolved reactions')],
                  loc='lower right')
        if not resolved:
            ax.text(.5, .5, 'No resolved sensitivities', transform=ax.transAxes, ha='center')
        note = ('Unresolved (excluded from bars): ' + ', '.join('R' + r for r in unresolved)
                if unresolved else 'All reaction sensitivities resolved.')
        fig.text(.02, .02, note, fontsize=9, wrap=True)
        fig.tight_layout(rect=(0, .055, 1, 1))
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / f"round_{event['round']:03d}.png"
        temporary = target.with_suffix('.tmp.png')
        fig.savefig(temporary, dpi=160)
        temporary.replace(target)
        return target
    finally:
        plt.close(fig)


def plot_coordinate_sensitivity(screen, labels, cfg, destination):
    """Plot signed length derivatives for every A coordinate and operating case."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    jac = np.asarray(screen['residual_jacobian'])
    active = screen['active_coordinates']
    stable = np.array([c['stable'] for c in screen['columns']])
    count = len(cfg['P_bar']); stride = cfg['profile_points']+3
    names = [labels[j]['label'] for j in active]
    report = dict(definition='d ln(length) / d coordinate; pivot magnitude or scaled Ea at fixed pivot rate as labelled',
                  method='Residual Jacobian with loss scaling removed; unresolved entries are null.',
                  coordinates=[labels[j] for j in active], pressure_bar=cfg['P_bar'], phi=cfg['phi'])
    for key, offset in [('induction', -2), ('exothermic', -1)]:
        if {'induction':'ind_len_ZND', 'exothermic':'exo_len_ZND'}[key] not in cfg.get('objective_names', ['ind_len_ZND','exo_len_ZND']):
            continue
        values = jac[np.arange(1, count+1)*stride+offset, :]*(1. if cfg['length_factor'] is None else np.log(cfg['length_factor']))*np.sqrt(len(cfg.get('objective_names', ['T_profile','T_ZND','ind_len_ZND','exo_len_ZND']))*count)
        values[:, ~stable] = np.nan
        report[key] = values.tolist()
        columns = min(3, count)
        fig, axes = plt.subplots((count+columns-1)//columns, columns, squeeze=False,
            figsize=(6*columns, 6*((count+columns-1)//columns)), layout='constrained')
        for case, ax in enumerate(axes.flat):
            if case >= count:
                ax.set_visible(False); continue
            order = np.argsort(-np.nan_to_num(np.abs(values[case]), nan=-1.), kind='stable')
            v = values[case, order]
            ax.barh(np.arange(len(v)), np.nan_to_num(v), color=np.where(v >= 0, '#cc6043', '#286cb0'))
            ax.set_yticks(np.arange(len(v)), [names[j]+(' [unresolved]' if not stable[j] else '') for j in order], fontsize=8)
            ax.invert_yaxis(); ax.axvline(0, color='black', lw=.7)
            ax.set_title(f"Case {case+1}: {cfg['P_bar'][case]:g} bar, phi={cfg['phi'][case]:g}")
            ax.set_xlabel('d ln(length) / d coordinate (see row label)'); ax.grid(axis='x', alpha=.2)
        fig.suptitle(key.capitalize()+' length sensitivity — largest magnitude first', fontsize=16)
        fig.savefig(destination / (key+'_bars.png'), dpi=160); plt.close(fig)
    _save_yaml(destination / 'lengths.yaml', report)
