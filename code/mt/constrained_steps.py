"""Small convex least-squares steps over a union of linear constraint sets."""
import warnings
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import LinearConstraint, OptimizeWarning, linprog, lsq_linear, minimize


class ConstrainedStepError(RuntimeError):
    """The numerical subproblem could not provide a certified feasible step."""


def _feasible(step, a, rhs, low, high):
    """Use original-coordinate tolerances consistent with physical validation."""
    return (np.isfinite(step).all() and np.all(step >= low-1e-12)
            and np.all(step <= high+1e-12) and np.all(a@step >= rhs-1e-12))


def _repair(step, a, rhs, low, high):
    """Project tiny infeasibilities without losing precision in a tiny objective."""
    if not np.isfinite(step).all(): return None
    step=np.clip(step,low,high)
    if _feasible(step,a,rhs,low,high): return step
    violation=max(0.,float(np.max(rhs-a@step,initial=0.)))
    if violation>1e-6: return None
    # Optimize a scaled correction, not a near-zero squared distance whose
    # gradient would already satisfy an optimizer's absolute stopping tolerance.
    scale=max(violation,1e-8)
    correction=minimize(lambda d:.5*float(d@d),np.zeros(len(step)),jac=lambda d:d,
        bounds=list(zip((low-step)/scale,(high-step)/scale)),
        constraints=[LinearConstraint(a,(rhs-a@step)/scale,np.inf)],
        method='SLSQP',options=dict(ftol=1e-13,maxiter=500))
    repaired=np.clip(step+scale*correction.x,low,high)
    if correction.success and _feasible(repaired,a,rhs,low,high):
        print(f'Constraint step repaired: violation {violation:.3g} -> '
              f'{max(0.,float(np.max(rhs-a@repaired,initial=0.))):.3g}',flush=True)
        return repaired
    return None


def least_squares_branches(model, residual, x, active, low, high, branches, damping=0., *, require_all=False):
    """Return feasible steps ordered by predicted damped loss.

    Each branch is C @ coordinates >= b. Inactive coordinates stay fixed.
    Use the bounded least-squares optimum directly when it satisfies a branch.
    Otherwise whiten the positive-definite quadratic before imposing the coupled
    constraints: weak parameter directions must not disappear in a global scale.
    """
    active = np.asarray(active, dtype=int)
    model, residual, x, low, high = [np.asarray(v, dtype=float) for v in (model, residual, x, low, high)]
    hessian = model.T @ model + damping*np.eye(len(active))
    gradient = model.T @ residual
    augmented = np.vstack([model, np.sqrt(damping)*np.eye(len(active))])
    target = np.r_[-residual, np.zeros(len(active))]
    bounded = lsq_linear(augmented, target, bounds=(low, high),
        tol=1e-12, lsq_solver='exact', max_iter=1000)
    if not bounded.success:
        raise ConstrainedStepError('Bounded update solve failed: '+bounded.message)
    answers = []
    unresolved = False
    transform = None
    for c, b in branches:
        c, b = np.asarray(c), np.asarray(b)
        a, rhs = c[:, active], b-c@x
        movable = np.any(a != 0, axis=1)
        if np.any(rhs[~movable] > 1e-12):
            continue
        a, rhs = a[movable], rhs[movable]
        if not len(rhs) or np.all(a@bounded.x >= rhs-1e-12):
            step = bounded.x.copy()
        else:
            initial = np.clip(np.zeros(len(active)), low, high)
            if np.any(a@initial < rhs-1e-12):
                with warnings.catch_warnings():
                    # SciPy forwards this supported HiGHS option but warns about it.
                    warnings.filterwarnings('ignore', message='Unrecognized options detected:.*threads.*', category=OptimizeWarning)
                    feasible = linprog(np.zeros(len(active)), A_ub=-a, b_ub=-rhs,
                                      bounds=list(zip(low, high)), method='highs', options={'threads': 1})
                if feasible.status == 2:
                    continue
                if not feasible.success:
                    unresolved = True
                    print('Constraint feasibility solver failed; branch rejected: '+feasible.message,flush=True)
                    continue
                initial = feasible.x
            if transform is None:
                # H = L L^T; s = L^-T z gives Hessian I in z coordinates.
                lower = np.linalg.cholesky(hessian)
                transform = solve_triangular(lower.T, np.eye(len(active)), lower=False)
                shifted = transform.T@gradient
            matrix = np.vstack([transform, a@transform])
            lb = np.r_[low, rhs]; ub = np.r_[high, np.full(len(rhs), np.inf)]
            finite = np.isfinite(lb) | np.isfinite(ub)
            matrix, lb, ub = matrix[finite], lb[finite], ub[finite]
            # Row scaling improves the constraint tolerances in transformed units.
            scales = np.maximum(np.linalg.norm(matrix, axis=1), 1e-30)
            constraint = LinearConstraint(matrix/scales[:, None], lb/scales, ub/scales)
            # Drop the constant ||shifted||² and normalize the objective: large
            # residuals otherwise make SLSQP report status 8 at feasible optima.
            objective_scale=max(1.,float(np.linalg.norm(shifted,ord=np.inf)))
            fit = minimize(lambda z: (.5*(z@z)+shifted@z)/objective_scale, lower.T@initial,
                jac=lambda z: (z+shifted)/objective_scale, constraints=[constraint], method='SLSQP',
                options=dict(ftol=1e-13, maxiter=500))
            step = _repair(transform@fit.x,a,rhs,low,high) if fit.success else None
            if step is None:
                # SLSQP can fail on nearly dependent active constraints. Solve
                # the same convex quadratic with an interior-point method; SVD
                # handles redundant pressure/temperature envelope rows.
                fit = minimize(lambda z: (.5*(z@z)+shifted@z)/objective_scale,
                    lower.T@initial, jac=lambda z: (z+shifted)/objective_scale,
                    hess=lambda z: np.eye(len(active))/objective_scale,
                    constraints=[constraint], method='trust-constr',
                    options=dict(gtol=1e-11, xtol=1e-12, barrier_tol=1e-12,
                                 initial_barrier_parameter=1e-9, initial_barrier_tolerance=1e-10,
                                 maxiter=1500, factorization_method='SVDFactorization'))
                step = (_repair(transform@fit.x,a,rhs,low,high)
                        if fit.success and fit.optimality <= 1e-8 else None)
            if step is None:
                unresolved = True
                print('Constraint update rejected: no certified feasible solution.',flush=True)
                continue
        if not _feasible(step,a,rhs,low,high):
            unresolved = True
            continue
        answers.append((float(np.sum((residual+model@step)**2)+damping*(step@step)), step))
    if unresolved and (require_all or not answers):
        raise ConstrainedStepError('Constraint solver could not certify all required branches.')
    return [step for _, step in sorted(answers, key=lambda row: row[0])]


def gradient_mapping(x, active, gradient, low, high, constraints=None):
    """Projection gradient, including coupled constraints when provided."""
    active = np.asarray(active, dtype=int)
    if constraints is None:
        return x[active]-np.clip(x[active]-gradient, low[active], high[active])
    try:
        steps = least_squares_branches(np.eye(len(active)), gradient, x, active,
            low[active]-x[active], high[active]-x[active], constraints(),require_all=True)
    except ConstrainedStepError:
        return np.full(len(active),np.inf)
    if not steps:
        return np.full(len(active),np.inf)
    return -steps[0]
