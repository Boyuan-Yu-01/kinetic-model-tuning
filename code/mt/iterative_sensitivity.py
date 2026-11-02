"""Repeated sensitivity screening and bounded optimization of generic residuals.

Groups identify which coordinates belong together; no group is preselected.
The evaluation callback accepts a batch of coordinate vectors and returns one
residual vector (or None for an invalid evaluation) for every point.
"""
from copy import deepcopy
from .constrained_steps import gradient_mapping
import numpy as np
from .sensitivity_search import (
    DEFAULTS, _coordinate_arrays, adaptive_jacobian, sensitivity_trust_region,
)


def iterative_sensitivity_optimize(evaluate, x, low, high, groups, *,
                                   physical_parameters=None, parameter_scales=None, max_rounds=20,
                                   inner_iterations=15, top_reactions=5,
                                   threshold=0.01, steps=.01, options=None,
                                   on_round=None, sensitivity_retries=3,
                                   shrink_factor=.5, minimum_step=1e-5,
                                   on_sensitivity_retry=None, on_round_start=None,
                                   constraints=None, feasible=None, stagnation_rounds=3,
                                   stagnation_relative_tolerance=1e-5):
    """Rescreen every free coordinate after every round and record full evidence.

    Bounds may use -inf below and +inf above for physically unbounded
    coordinates; finite trust-region radii limit individual numerical steps.
    ``groups`` has one JSON-serializable scalar group label per coordinate.
    ``physical_parameters`` maps coordinates to the entire physical parameter
    vector, including inactive parameters. Relative changes use the previous
    physical value as denominator; zero-to-nonzero is represented by None in
    JSON and never satisfies convergence. The threshold is strictly less-than.
    ``parameter_scales`` optionally sets absolute change denominators; NaN
    retains relative change for that physical parameter.
    A small selected-group update receives an all-free-coordinate audit before
    convergence; exact zero updates also require full projected stationarity.
    ``stagnation_rounds`` consecutive relative loss reductions no larger than
    ``stagnation_relative_tolerance`` stop with ``loss_plateau``, without
    claiming convergence. Set the round count to zero to disable this stop.
    ``on_round`` receives each completed, JSON-serializable round record.
    ``on_round_start`` receives a snapshot of the screen and selected groups
    before that round changes any parameters.
    """
    opts = dict(DEFAULTS, **(options or {}))
    x, low, high, steps = _coordinate_arrays(x, low, high, steps)
    groups = [g.item() if isinstance(g, np.generic) else g for g in groups]
    if len(groups) != len(x):
        raise ValueError('One group label is required per coordinate.')
    for name, value in (('max_rounds', max_rounds), ('inner_iterations', inner_iterations), ('top_reactions', top_reactions)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(name + ' must be a positive integer.')
    if isinstance(stagnation_rounds, (bool, np.bool_)) or not isinstance(stagnation_rounds, (int, np.integer)) or stagnation_rounds < 0:
        raise ValueError('stagnation_rounds must be a nonnegative integer.')
    if isinstance(stagnation_relative_tolerance, (bool, np.bool_)) or not np.isfinite(stagnation_relative_tolerance) or not 0 <= stagnation_relative_tolerance < 1:
        raise ValueError('stagnation_relative_tolerance must be finite and in [0, 1).')
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('The relative-change threshold must be finite and between zero and one.')
    for group in groups:
        if not isinstance(group, (str, int, float, bool)) or isinstance(group, float) and not np.isfinite(group):
            raise ValueError('Group labels must be finite JSON scalar strings or numbers.')
    if isinstance(sensitivity_retries, bool) or not isinstance(sensitivity_retries, (int, np.integer)) or sensitivity_retries < 0:
        raise ValueError('sensitivity_retries must be a nonnegative integer.')
    if not np.isfinite(shrink_factor) or not 0 < shrink_factor < 1:
        raise ValueError('shrink_factor must be finite and between zero and one.')
    floor = np.broadcast_to(np.asarray(minimum_step, dtype=float), x.shape).copy()
    if not np.isfinite(floor).all() or np.any(floor <= 0):
        raise ValueError('minimum_step must be finite and positive.')
    floor = np.minimum(floor, steps)  # Never increase an explicitly smaller initial step.
    free = np.flatnonzero(high > low).tolist()
    transform = physical_parameters or (lambda point: point)

    def physical(point):
        values = np.asarray(transform(point.copy()), dtype=float).reshape(-1)
        if not values.size or not np.all(np.isfinite(values)):
            raise ValueError('Physical parameters must be a nonempty finite vector.')
        return values

    residual_size = None

    def batch(points):
        nonlocal residual_size
        rows = list(evaluate(points))
        if len(rows) != len(points):
            raise ValueError('Evaluation must return exactly one residual per point.')
        result = []
        for row in rows:
            if row is None:
                result.append(None)
                continue
            row = np.asarray(row, dtype=float)
            if row.ndim != 1 or not row.size or not np.all(np.isfinite(row)):
                result.append(None)
                continue
            if residual_size is None:
                residual_size = row.size
            if row.size != residual_size:
                raise ValueError('Residual dimension changed during optimization.')
            result.append(row)
        return result

    def screen(point, residual):
        if not free:
            return dict(coordinates=point.tolist(), active_coordinates=[], residual_jacobian=[],
                        columns=[], ranking=[], all_resolved=True, projected_gradient=0., stationary=True)
        jac, columns = adaptive_jacobian(batch, point, residual, low, high, free,
                                         steps, opts['initial_radius'], opts)
        gradient = jac.T @ residual
        projected = (gradient_mapping(point, free, gradient, low, high, constraints)
                     if all(c['stable'] for c in columns) else np.full(len(free), np.inf))
        movement = np.minimum(opts['initial_radius'], np.maximum(point[free] - low[free], high[free] - point[free]))
        normalization = max(float(np.linalg.norm(residual)), 1.)
        ranking = []
        for group in dict.fromkeys(groups[i] for i in free):
            positions = [j for j, i in enumerate(free) if groups[i] == group]
            reliable = [j for j in positions if columns[j]['stable']]
            ranking.append(dict(group=group, coordinates=[free[j] for j in positions],
                                reliable_coordinates=[free[j] for j in reliable],
                                all_resolved=len(reliable) == len(positions),
                                score=float(np.linalg.norm(jac[:, reliable] * movement[reliable]) / normalization),
                                projected_gradient=max((float(abs(projected[j])) for j in reliable), default=0.)))
        ranking.sort(key=lambda row: (-row['score'], -row['projected_gradient']))
        resolved = all(c['stable'] for c in columns)
        maximum = float(np.max(np.abs(projected))) if resolved else None
        return dict(coordinates=point.tolist(), active_coordinates=free,
                    residual_jacobian=jac.tolist(), columns=columns, ranking=ranking,
                    coordinate_projected_gradient=[float(v) if columns[j]['stable'] else None for j, v in enumerate(projected)],
                    all_resolved=resolved, projected_gradient=maximum,
                    stationary=bool(resolved and maximum <= opts['gradient_tolerance']))

    def rescreen(point, residual):
        """Retry a full screen with smaller base steps for every coordinate.

        The finite-difference routine still selects its own checked stencil
        around each base step. Bounds and acceptance tolerances never change.
        """
        retries = []
        for attempt in range(sensitivity_retries + 1):
            result = screen(point, residual)
            result['base_steps'] = steps.tolist()
            result['shrink_retries'] = retries.copy()
            if result['all_resolved'] or attempt == sensitivity_retries:
                return result
            smaller = np.maximum(steps * shrink_factor, floor)
            if np.array_equal(smaller, steps):
                result['step_floor_reached'] = True
                return result
            event = dict(attempt=attempt+1, limit=sensitivity_retries,
                         unresolved_groups=[row['group'] for row in result['ranking'] if not row['all_resolved']],
                         previous_steps=steps.tolist(), next_steps=smaller.tolist(),
                         shrink_factor=float(shrink_factor))
            retries.append(event)
            steps[:] = smaller
            if on_sensitivity_retry:
                on_sensitivity_retry(event)

    def changes(before, after):
        if before.shape != after.shape:
            raise ValueError('Physical parameter dimension changed during optimization.')
        delta = np.abs(after - before)
        denominator = np.abs(before)
        if parameter_scales is not None:
            scales = np.asarray(parameter_scales,dtype=float)
            if scales.shape != before.shape or np.any((~np.isnan(scales)) & ((scales<=0) | ~np.isfinite(scales))):
                raise ValueError('Parameter scales must match physical entries; NaN means relative change, otherwise positive finite.')
            denominator = np.where(np.isnan(scales),denominator,scales)
        relative = np.divide(delta, denominator, out=np.full_like(delta, np.inf), where=denominator != 0)
        relative[(before == 0) & (after == 0)] = 0.
        return relative, bool(np.all(relative < threshold)), bool(np.all(delta == 0))

    residual = batch([x])[0]
    if residual is None or (feasible is not None and not feasible(x)):
        return dict(bestx=x.tolist(), residual=None, loss=None, rounds=[], status='invalid_initial', converged=False, screen=None)
    physical(x)
    current = rescreen(x, residual)
    initial_screen = current
    history = []
    status, converged = 'maximum_rounds', False
    stagnation_count = 0
    failed_groups = set()
    failure_statuses = {'trust_region_stalled', 'loss_stalled', 'sensitivity_limited'}
    for number in range(1, max_rounds + 1):
        before_x, before_physical, before_loss = x.copy(), physical(x), float(residual @ residual)
        candidates = [row for row in current['ranking'] if row['reliable_coordinates']]
        # Deprioritize groups whose preceding attempt made no progress.
        candidates.sort(key=lambda row: (row['group'] in failed_groups, -row['score'], -row['projected_gradient']))
        chosen = candidates[:top_reactions]
        selected = sorted(i for row in chosen for i in row['reliable_coordinates'])
        trials = []
        if on_round_start:
            on_round_start(dict(round=number, screen=deepcopy(current),
                                selected_groups=[row['group'] for row in chosen]))

        def run(active, kind):
            nonlocal x, residual
            result = sensitivity_trust_region(batch, x, low, high, active, steps,
                                              max_iterations=inner_iterations, options=opts, constraints=constraints, feasible=feasible)
            trials.append(dict(kind=kind, active_coordinates=active, status=result.status,
                               converged=bool(result.converged), iterations=int(result.iterations),
                               diagnostics=result.diagnostics))
            if np.all(np.isfinite(result.residual)) and float(result.residual @ result.residual) <= float(residual @ residual):
                x, residual = result.x.copy(), result.residual.copy()
            return result

        result = run(selected, 'selected') if selected else None
        after_physical = physical(x)
        relative, small, zero = changes(before_physical, after_physical)
        stalled = result is None or result.status in failure_statuses or float(residual @ residual) >= before_loss
        # An all-free audit prevents unseen directions and failed selections
        # from masquerading as small-change convergence.
        available = sorted(i for row in current['ranking'] for i in row['reliable_coordinates'])
        skipped = sorted(set(free) - set(available))
        audited = selected == free
        if available and (stalled or small) and selected != available:
            result = run(available, 'all_free_audit' if available == free else 'available_audit')
            audited = available == free
            after_physical = physical(x)
            relative, small, zero = changes(before_physical, after_physical)
        current = rescreen(x, residual)  # Always fresh, and reused by next round.
        after_loss = float(residual @ residual)
        relative_loss_reduction = (before_loss - after_loss) / before_loss if before_loss > 0 else 0.
        stagnation_count = stagnation_count + 1 if relative_loss_reduction <= stagnation_relative_tolerance else 0
        progress = after_loss < before_loss
        if progress:
            failed_groups.clear()
        else:
            failed_groups.update(row['group'] for row in chosen)
        healthy = result is None and not free or result is not None and result.status not in failure_statuses
        converged = bool(small and current['all_resolved'] and healthy
                         and (current['stationary'] if zero else (audited or current['stationary'])))
        inner_status = result.status if result is not None else 'not_run'
        convergence_blockers = []
        if not small:
            convergence_blockers.append(dict(reason='parameter_change_threshold',
                parameter_indices=np.flatnonzero(relative >= threshold).tolist(), threshold=float(threshold)))
        if not current['all_resolved']:
            convergence_blockers.append(dict(reason='unresolved_sensitivities',
                coordinate_indices=[free[j] for j, column in enumerate(current['columns']) if not column['stable']]))
        if not healthy:
            convergence_blockers.append(dict(reason='inner_solver_status', status=inner_status))
        if zero and not current['stationary']:
            convergence_blockers.append(dict(reason='zero_update_requires_stationarity'))
        elif not zero and not (audited or current['stationary']):
            convergence_blockers.append(dict(reason='all_free_audit_or_stationarity_required'))
        if converged:
            status = 'parameter_change'
        elif not current['all_resolved']:
            # A noisy column must block convergence, not useful progress in
            # other directions. Retry every column at the next accepted state.
            usable = any(row['reliable_coordinates'] for row in current['ranking'])
            status = (('maximum_rounds' if number == max_rounds else 'iterating')
                      if usable else 'unresolved_sensitivities')
            if usable and stagnation_rounds and stagnation_count >= stagnation_rounds:
                status = 'loss_plateau_unresolved'
        elif audited and not progress:
            status = 'stalled'
        else:
            # Keep accepted progress even when the inner search stalled; its
            # failure status still blocks convergence until a fresh audit passes.
            status = 'maximum_rounds' if number == max_rounds else 'iterating'
        if (not converged and current['all_resolved'] and stagnation_rounds
                and stagnation_count >= stagnation_rounds and status in ('iterating', 'maximum_rounds')):
            status = 'loss_plateau'
        record = dict(round=number, before_coordinates=before_x.tolist(), coordinates=x.tolist(),
                      before_physical_parameters=before_physical.tolist(), physical_parameters=after_physical.tolist(),
                      relative_changes=[float(v) if np.isfinite(v) else None for v in relative],
                      all_changes_below_threshold=small, threshold=float(threshold),
                      selected_groups=[row['group'] for row in chosen], selected_coordinates=selected,
                      skipped_coordinates=skipped,
                      all_free_audit=audited, trials=trials, loss=float(residual @ residual),
                      screen=current, status=status, converged=converged,
                      convergence_blockers=convergence_blockers, inner_status=inner_status,
                      relative_loss_reduction=float(relative_loss_reduction), stagnation_count=stagnation_count)
        history.append(record)
        if on_round:
            on_round(record)
        if converged or status not in ('iterating', 'maximum_rounds'):
            break
    return dict(bestx=x.tolist(), residual=residual.tolist(), loss=float(residual @ residual),
                rounds=history, status=status, converged=converged, screen=current, initial_screen=initial_screen, final_steps=steps.tolist())
