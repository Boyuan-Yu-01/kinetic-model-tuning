"""General bounded least-squares search with adaptive numerical sensitivities.

The numerical routines know nothing about reactions, species, or ZND. A batch
callback returns a residual vector for each point, or None for an invalid solve.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear
from .constrained_steps import ConstrainedStepError, gradient_mapping, least_squares_branches


DEFAULTS = dict(fd_attempts=5, fd_relative_tolerance=.25,
                fd_absolute_tolerance=1e-5, weak_response_tolerance=1e-4,
                gradient_tolerance=1e-5, initial_radius=.35,
                maximum_radius=1., minimum_radius=1e-5,
                max_trial_steps=12, loss_tolerance=1e-7, stagnation_iterations=5,
                fd_extrapolate=False)


@dataclass
class SearchResult:
    """Separate successful execution from demonstrated numerical convergence."""
    x: np.ndarray
    residual: np.ndarray
    status: str
    converged: bool
    iterations: int
    diagnostics: dict


def _coordinate_arrays(x, low, high, steps):
    """Normalize shared search inputs and reject unusable bounds or perturbations."""
    x, low, high = [np.asarray(v, dtype=float).copy() for v in (x, low, high)]
    steps = np.broadcast_to(np.asarray(steps, dtype=float), x.shape).copy()
    if x.ndim != 1 or not len(x) or low.shape != x.shape or high.shape != x.shape:
        raise ValueError('Coordinates and bounds must have matching nonempty one-dimensional shapes.')
    # Infinite bounds express absent physical limits; finite trust radii bound trials.
    if (not np.all(np.isfinite([x, steps])) or np.any(np.isnan(low)) or np.any(np.isnan(high))
            or np.any(np.isposinf(low)) or np.any(np.isneginf(high)) or np.any(low > high)
            or np.any(x < low) or np.any(x > high) or np.any(steps <= 0)):
        raise ValueError('Finite coordinates and positive steps inside ordered bounds are required; only lower -inf and upper +inf are permitted.')
    return x, low, high, steps


def _projected_gradient(x, gradient, low, high):
    """Remove descent directions pointing outside active parameter bounds."""
    # Gradient mapping recognizes near-bound optima without requiring exact
    # floating-point equality with a bound returned by the subproblem solver.
    return x - np.clip(x-gradient, low, high)


def adaptive_jacobian(evaluate, x, residual, low, high, active, steps,
                      radius, options=None):
    """Adapt each perturbation independently; unresolved columns never enter steps.

    Compare h and h/2 stencils using absolute and relative tolerances. Try both
    larger and smaller perturbations. Central probes can fall back to a valid
    one-sided derivative if a solve fails. Weakness means the estimated residual
    change AND derivative uncertainty over the current feasible trust step are
    small. Every call reassesses all requested coordinates, including weak ones.
    Optional Richardson extrapolation removes the leading stencil truncation
    error; the original h versus h/2 discrepancy still controls reliability.
    """
    opts = dict(DEFAULTS, **(options or {}))
    active = list(active)
    supplied = opts.get("jacobian_entries")
    exact = np.asarray(supplied(x), dtype=float) if supplied is not None else None
    if exact is not None and (exact.shape != (len(residual), len(x)) or np.isinf(exact).any()):
        raise ValueError("Invalid supplied Jacobian entries.")
    estimates = {}
    pending = set(active)
    attempts = {i: 0 for i in active}
    for attempt in range(opts['fd_attempts']):
        factor = 1. if attempt == 0 else (2.**((attempt+1)//2) if attempt % 2 else 2.**(-attempt//2))
        points, probes = [], []
        for i in active:
            if i not in pending:
                continue
            left, right = x[i]-low[i], high[i]-x[i]
            requested = max(steps[i]*factor, np.broadcast_to(opts.get('fd_min_steps',0.), x.shape)[i])
            if max(left, right) <= 1e-14:
                estimates[i] = dict(column=np.zeros_like(residual), status='fixed', stable=True,
                                    error=0., effect=0., h=0., stencil='fixed', attempts=0)
                pending.remove(i)
                continue
            if min(left, right) >= .25*requested:
                h = min(requested, left, right)
                offsets = [-h, -h/2, h/2, h]
                stencil = 'central'
            else:
                direction = 1 if right >= left else -1
                h = min(requested, max(left, right)/2)
                offsets = [direction*h/2, direction*h, direction*2*h]
                stencil = 'forward' if direction == 1 else 'backward'
            start = len(points)
            for offset in offsets:
                point = x.copy();point[i] += offset;points.append(point)
            probes.append((i, h, stencil, start, len(offsets)))
        if not points:
            break
        values = evaluate(points)
        for i, h, stencil, start, count in probes:
            attempts[i] += 1
            rows = values[start:start+count]
            pairs = []
            if stencil == 'central':
                minus, half_minus, half_plus, plus = rows
                if all(v is not None for v in rows):
                    pairs.append(((plus-minus)/(2*h), (half_plus-half_minus)/h, 'central'))
                if plus is not None and half_plus is not None:
                    pairs.append(((plus-residual)/h, (half_plus-residual)/(h/2), 'forward_first_order'))
                if minus is not None and half_minus is not None:
                    pairs.append(((residual-minus)/h, (residual-half_minus)/(h/2), 'backward_first_order'))
            else:
                half, full, twice = rows
                direction = 1 if stencil == 'forward' else -1
                if all(v is not None for v in rows):
                    pairs.append((direction*(-3*residual+4*full-twice)/(2*h),
                                  direction*(-3*residual+4*half-full)/h, stencil))
                elif half is not None and full is not None:
                    pairs.append((direction*(full-residual)/h, direction*(half-residual)/(h/2), stencil+'_first_order'))
            movement = min(radius, max(x[i]-low[i], high[i]-x[i]))
            for coarse, fine, used in pairs:
                if not np.all(np.isfinite(fine)) or not np.all(np.isfinite(coarse)):
                    continue
                if exact is not None:
                    mask = np.isfinite(exact[:, i])
                    coarse = coarse.copy(); fine = fine.copy()
                    coarse[mask] = fine[mask] = exact[mask, i]
                error = float(np.linalg.norm(fine-coarse))
                column = fine
                if opts['fd_extrapolate']:
                    order = 2 if used in ('central', 'forward', 'backward') else 1
                    column = fine + (fine-coarse)/(2**order-1)
                if not np.all(np.isfinite(column)):
                    continue
                size = float(np.linalg.norm(column))
                tolerance = opts['fd_absolute_tolerance'] + opts['fd_relative_tolerance']*size
                stable = error <= tolerance
                effect = (max(size, float(np.linalg.norm(coarse))) + error)*movement
                weak = effect <= opts['weak_response_tolerance']
                status = 'weak' if weak else ('reliable' if stable else 'unresolved')
                entry = dict(column=column, status=status, stable=bool(stable), error=error,
                             effect=effect, h=float(h), stencil=used, extrapolated=bool(opts['fd_extrapolate']), attempts=attempts[i],
                             quality=error/max(tolerance, 1e-30))
                previous = estimates.get(i)
                if previous is None or entry['quality'] < previous.get('quality', np.inf):
                    estimates[i] = entry
                if stable:
                    estimates[i] = entry
                    pending.discard(i)
                    break
        # Unstable weak columns receive the same bounded retries as other columns.
    columns, diagnostics = [], []
    for i in active:
        entry = estimates.get(i, dict(column=np.zeros_like(residual), status='unavailable',
                                     stable=False,error=None,effect=None,h=None,stencil=None))
        columns.append(entry['column'])
        diagnostics.append(dict(index=i, **{k:v for k,v in entry.items() if k not in ('column','quality')},
                                total_attempts=attempts[i]))
    return np.column_stack(columns), diagnostics


def sensitivity_trust_region(evaluate, x, low, high, active, steps, *,
                             max_iterations, options=None, callback=None,
                             constraints=None, feasible=None):
    """Minimize residual squared norm using reliable columns and bounded trial steps.

    Only actual/predicted trial agreement changes the global trust radius.
    Projected-gradient convergence requires reliable estimates for all free
    directions. Loss stagnation, unresolved sensitivities, radius exhaustion,
    and iteration budgets are explicitly unconverged termination reasons.
    """
    opts = dict(DEFAULTS, **(options or {}))
    x, low, high, steps = _coordinate_arrays(x, low, high, steps)
    active = list(active)
    if not active or len(set(active)) != len(active) or any(i<0 or i>=len(x) for i in active):
        raise ValueError('Active coordinates must be distinct valid indices.')
    r = evaluate([x])[0]
    if r is None:
        raise ValueError('Initial point has invalid residuals.')
    step_scales = np.broadcast_to(np.asarray(opts.get('step_scales',1.),float),x.shape)
    if not np.isfinite(step_scales).all() or np.any(step_scales<=0):
        raise ValueError('Trust-region step scales must be positive and finite.')
    radius = opts['initial_radius'];stagnant = 0;last = {}
    for iteration in range(max_iterations):
        J, columns = adaptive_jacobian(evaluate,x,r,low,high,active,steps,radius,opts)
        all_resolved = all(c['stable'] for c in columns)
        g = gradient_mapping(x, active, J.T@r, low, high, constraints) if all_resolved else np.full(len(active), np.inf)
        # A small derivative can still matter when its residual is large.
        used = [j for j,c in enumerate(columns) if c['status']=='reliable'
                or (c['status']=='weak' and c['stable'] and abs(g[j])>opts['gradient_tolerance'])]
        if constraints is not None:
            used = [j for j, c in enumerate(columns) if c['stable']]
        gradient = float(np.max(np.abs(g))) if all_resolved else None
        last = dict(iteration=iteration+1, coordinates=x.tolist(), loss=float(r@r),
                    radius=radius, columns=columns, residual_jacobian=J.tolist(),
                    residual=r.tolist(), active_coordinates=active,
                    projected_gradient=gradient, accepted=False, rho=None)
        if all_resolved and gradient <= opts['gradient_tolerance']:
            status='projected_gradient';converged=True
        elif not used:
            status='sensitivity_limited';converged=False
        else:
            status=None;converged=False
        if status is not None:
            last['status']=status
            if callback:callback(last)
            return SearchResult(x,r,status,converged,iteration+1,last)
        selected = np.array([active[j] for j in used],dtype=int)
        model = J[:,used]
        accepted = False;old_loss=float(r@r);step_size=0.
        last['trial_steps'] = []
        for trial in range(opts['max_trial_steps']):
            lo=np.maximum(low[selected]-x[selected],-radius*step_scales[selected])
            hi=np.minimum(high[selected]-x[selected],radius*step_scales[selected])
            damping=max(1e-10,1e-6*np.linalg.norm(model,ord=2)**2)
            if constraints is None:
                fit=lsq_linear(np.vstack([model,np.sqrt(damping)*np.eye(len(used))]),
                               np.r_[-r,np.zeros(len(used))],bounds=(lo,hi),
                               tol=1e-10,lsq_solver='exact',max_iter=100)
                proposals = [fit.x]
            else:
                try:
                    proposals = least_squares_branches(model, r, x, selected, lo, hi, constraints(), damping)
                except ConstrainedStepError as exc:
                    proposals = []
                    last['trial_steps'].append(dict(attempt=trial+1,radius=float(radius),
                        accepted=False,reason='constraint_solver_failed',error=str(exc)))
                    print(f'Constraint update rejected; shrinking trust radius from {radius:.3g}.',flush=True)
            candidates = []
            for step in proposals:
                predicted=float(old_loss-np.sum((r+model@step)**2))
                if predicted <= 0 or np.max(np.abs(step)) < 1e-12:
                    continue
                point=x.copy(); point[selected]=np.clip(x[selected]+step,low[selected],high[selected])
                trial_r=evaluate([point])[0]
                valid = trial_r is not None and (feasible is None or feasible(point))
                actual=-np.inf if not valid else old_loss-float(trial_r@trial_r)
                rho=actual/predicted
                last['trial_steps'].append(dict(attempt=trial+1, radius=float(radius),
                    step_size=float(np.max(np.abs(step))), predicted_reduction=predicted,
                    valid=valid, actual_reduction=float(actual) if np.isfinite(actual) else None,
                    rho=float(rho) if np.isfinite(rho) else None, accepted=bool(actual>0 and rho>.1)))
                if actual>0 and rho>.1:
                    candidates.append((actual, point, trial_r, rho, float(np.max(np.abs(step)/step_scales[selected]))))
            if candidates:
                _, x, r, rho, step_size = max(candidates, key=lambda row: row[0])
                last['rho']=float(rho); accepted=True
                if rho>.75 and step_size>.8*radius:radius=min(opts['maximum_radius'],2*radius)
                elif rho<.25:radius*=.5
                break
            # Restrict the rejected proposal itself, not just an inactive radius.
            proposed_size = max((float(np.max(np.abs(s)/step_scales[selected])) for s in proposals), default=radius)
            radius = min(radius*.5, proposed_size*.5)
            if radius<opts['minimum_radius']:break
        last.update(accepted=accepted, accepted_coordinates=x.tolist(), loss=float(r@r),radius=radius)
        if not accepted:status='trust_region_stalled'
        else:
            stagnant = stagnant+1 if old_loss-float(r@r)<=opts['loss_tolerance']*max(1.,old_loss) else 0
            status = 'loss_stalled' if stagnant>=opts['stagnation_iterations'] else None
        last['status']=status or 'iterating'
        if callback:callback(last)
        if status:return SearchResult(x,r,status,False,iteration+1,last)
    return SearchResult(x,r,'iteration_limit',False,max_iterations,last)


def scan_coordinate(evaluate, x, index, values):
    """Return an objective profile over any chosen coordinate without optimization."""
    points=[]
    for value in values:
        point=np.asarray(x,dtype=float).copy();point[index]=value;points.append(point)
    return [dict(value=float(point[index]), valid=r is not None,
                 loss=None if r is None else float(r@r),
                 residual=None if r is None else r.tolist())
            for point,r in zip(points,evaluate(points))]
