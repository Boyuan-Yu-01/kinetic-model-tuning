"""CJ-speed ZND calculation, reaction-rate sensitivity, and plotting.

Based on SDToolbox demo_ZNDCJ.py
"""

import warnings
import csv
import sys
from time import monotonic
from numbers import Integral
from pathlib import Path

import cantera as ct
from matplotlib.figure import Figure
import numpy as np
from sdtoolbox.postshock import CJspeed, PostShock_fr
from sdtoolbox.znd import zndsolve


def _sensitivity_lengths(output, t_end, *, interpolation="linear"):
    """Measure the thermicity peak and half-height width.

    Cubic interpolation reduces adaptive-grid noise in finite-difference
    sensitivities. Linear retains the original three-point peak/linear-width
    estimator for existing callers and comparison diagnostics.
    """
    if interpolation not in ('linear', 'cubic'):
        raise ValueError('Length interpolation must be linear or cubic.')
    time = np.asarray(output['time'], dtype=float)
    x = np.asarray(output['distance'], dtype=float)
    signal = np.asarray(output['thermicity'], dtype=float)
    if (time.ndim != 1 or len(time) < 5 or x.shape != time.shape
            or signal.shape != time.shape or not np.all(np.isfinite(time))
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(signal))
            or np.any(np.diff(time) <= 0) or np.any(np.diff(x) <= 0)):
        raise ValueError('ZND profiles must be finite, resolved, and strictly increasing in time/distance.')
    if not np.isclose(time[-1], t_end, rtol=1e-6, atol=1e-15):
        raise ValueError('ZND integration stopped before t_end.')
    for key in ('T', 'P', 'rho'):
        values = np.asarray(output[key], dtype=float)
        if values.shape != time.shape or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f'Invalid {key} profile.')
    peak = int(np.argmax(signal))
    height = float(signal[peak])
    if height <= 0:
        raise ValueError('No positive thermicity peak: induction/exothermic lengths are undefined.')
    if peak == 0:
        raise ValueError('Thermicity maximum is at the shock (t=0); no resolved interior peak. '
                         'Induction/exothermic sensitivities are unavailable.')
    if peak == len(x)-1:
        raise ValueError(f'Thermicity maximum is at t_end={t_end:g} s; '
                         'extend the integration to resolve the peak and its decay.')
    if interpolation == 'cubic':
        from scipy.interpolate import CubicSpline
        from scipy.optimize import brentq
        # Normalize both axes so root finding is independent of length units.
        scale = float(x[peak])
        z = x / scale
        curve = CubicSpline(z, signal / height, extrapolate=False)
        roots = curve.derivative().roots(extrapolate=False)
        roots = roots[np.isfinite(roots) & (roots >= z[peak-1]) & (roots <= z[peak+1])]
        z_peak = max([z[peak], *roots], key=lambda value: float(curve(value)))
        refined = float(curve(z_peak))
        if refined >= 2 or not np.isfinite(refined):
            raise ValueError('Thermicity peak is under-resolved; refine the integration.')
        half = refined / 2
        normalized = signal / height
        left = np.flatnonzero(normalized[:peak] <= half)
        right = np.flatnonzero(normalized[peak+1:] <= half)
        if not len(left) or not len(right):
            raise ValueError('Both half-maximum crossings must be resolved; extend/refine the integration.')
        i, j = int(left[-1]), int(peak+1+right[0])
        lo = brentq(lambda value: float(curve(value))-half, z[i], z[i+1], xtol=1e-13)
        hi = brentq(lambda value: float(curve(value))-half, z[j-1], z[j], xtol=1e-13)
        location, width = float(z_peak*scale), float((hi-lo)*scale)
        if not np.isfinite(location+width) or location <= 0 or width <= 0:
            raise ValueError('Induction and exothermic lengths must be positive and finite.')
        above = normalized > half
        pulses = int(above[0]) + int(np.count_nonzero(above[1:] & ~above[:-1]))
        return {'ind_len_ZND': location, 'exo_len_ZND': width, 'pulse_count': pulses}
    # Scale coordinates before fitting to avoid ill-conditioning for very short lengths.
    scale = x[peak+1] - x[peak-1]
    local_x = (x[peak-1:peak+2] - x[peak]) / scale
    a, b, c = np.polyfit(local_x, signal[peak-1:peak+2] / height, 2)
    location = float(x[peak])
    if a < 0:
        vertex = -b / (2*a)
        if local_x[0] <= vertex <= local_x[-1]:
            refined_height = height * float(a*vertex**2 + b*vertex + c)
            if refined_height >= 2*height:
                raise ValueError('Thermicity peak is under-resolved on the adaptive grid; reduce max_step/tolerances.')
            location += float(vertex * scale)
            height = refined_height
    half = height / 2
    left = np.flatnonzero(signal[:peak] <= half)
    right = np.flatnonzero(signal[peak+1:] <= half)
    if len(left) == 0 or len(right) == 0:
        raise ValueError('Both half-maximum crossings must be resolved; extend/refine the integration.')
    left = int(left[-1])
    right = int(peak + 1 + right[0])

    def crossing(i, j):
        """Interpolate a half-height crossing on either side of the selected peak."""
        if signal[i] == signal[j] or not min(signal[i], signal[j]) <= half <= max(signal[i], signal[j]):
            raise ValueError('Half-maximum crossing is not bracketed by sampled points.')
        return float(x[i] + (half-signal[i]) * (x[j]-x[i]) / (signal[j]-signal[i]))

    width = crossing(right-1, right) - crossing(left, left+1)
    if location <= 0 or width <= 0 or not np.isfinite(location + width):
        raise ValueError('Induction and exothermic lengths must be positive and finite.')
    above = signal > half
    pulses = int(above[0]) + int(np.count_nonzero(above[1:] & ~above[:-1]))
    return {'ind_len_ZND': location, 'exo_len_ZND': width, 'pulse_count': pulses}


def plot_znd_sensitivity(result, file_name='znd_sensitivity.pdf', top_n=None, *,
                         pdf_pages=None, title=None, objectives=None):
    """Save signed bars, largest absolute sensitivity first; return two file paths.

    Each objective has its own ordering. None shows every evaluated reaction;
    top_n limits display only. Failed sensitivities are labelled unavailable,
    placed last, and never drawn as zero. PDF is suitable for long reaction lists.
    Pass an open matplotlib PdfPages to append both plots to one document;
    file_name then identifies that document. title adds case information.
    objectives optionally selects ind_len_ZND and/or exo_len_ZND.
    """
    if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, Integral) or top_n < 1):
        raise ValueError('top_n must be a positive integer or None.')
    if objectives is None:
        objectives = ('ind_len_ZND', 'exo_len_ZND')
    if not objectives or any(key not in ('ind_len_ZND', 'exo_len_ZND') for key in objectives):
        raise ValueError('Select ind_len_ZND and/or exo_len_ZND.')
    path = Path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    for key, objective_title in (('ind_len_ZND', 'Induction length'), ('exo_len_ZND', 'Exothermic length')):
        if key not in objectives:
            continue
        values = np.asarray(result['sensitivities'][key], dtype=float)
        order = sorted(range(len(values)), key=lambda i: -abs(values[i]) if np.isfinite(values[i]) else np.inf)
        if top_n is not None:
            order = order[:top_n]
        fig = Figure(figsize=(12, max(3, 0.32*len(order) + 1.6)))
        ax = fig.subplots()
        labels = [f"R{result['reaction_indices'][i]+1}: {result['equations'][i]}" for i in order]
        for row, i in enumerate(order):
            if np.isfinite(values[i]):
                ax.barh(row, values[i], color='tab:blue')
            else:
                ax.text(0.02, row, 'unavailable', transform=ax.get_yaxis_transform(), va='center', color='tab:red')
        ax.set_yticks(range(len(order)), labels=labels)
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=0.7)
        ax.set_axisbelow(True)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlabel(r'Sensitivity: $\partial\ln\ell / \partial\ln m_j$')
        ax.set_title(f'{objective_title} sensitivity' + (f'\n{title}' if title else ''))
        fig.tight_layout()
        destination = path.with_name(f'{path.stem}_{key}{path.suffix or ".pdf"}')
        if pdf_pages is None:
            fig.savefig(destination, dpi=150, bbox_inches='tight')
        else:
            pdf_pages.savefig(fig, dpi=150, bbox_inches='tight')
            destination = path
        files[key] = str(destination)
    return files



def plot_znd_sensitivity_failure(pdf_pages, title, reason):
    """Append an explicit unavailable page, preserving case ordering in a PDF."""
    from textwrap import fill
    fig = Figure(figsize=(12, 5))
    ax = fig.subplots()
    ax.set_axis_off()
    ax.set_title(title)
    ax.text(0.05, 0.75, 'Sensitivity unavailable', transform=ax.transAxes,
            fontsize=14, color='tab:red')
    ax.text(0.05, 0.6, fill(str(reason), width=100), transform=ax.transAxes,
            va='top', fontsize=11)
    fig.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')


def znd_sensitivity(P0, T0, fuel, oxidizer, phi, mechanism, *, t_end=3e-5,
                    max_step=1e-6, rel_tol=1e-6, abs_tol=1e-9, method='BDF',
                    perturbation=0.01, reaction_indices=None, plot=False,
                    plot_file='znd_sensitivity.pdf', top_n=None,
                    progress=False, progress_label='ZND sensitivity',
                    workers=1, partial_file=None):
    """Return induction/exothermic-length sensitivities to reaction-rate multipliers.

    P0 is Pa, T0 is K, lengths are metres, and sensitivities are dimensionless.
    For each selected reaction, solve at multipliers 1 +/- perturbation and form
    [ln(length_plus)-ln(length_minus)]/[ln(1+d)-ln(1-d)]. reaction_indices are
    zero-based Cantera indices; None evaluates all reactions, including duplicates
    separately. A PLOG multiplier scales the whole reaction at every pressure.

    CJ speed and the frozen shock state are computed once: rate multipliers do
    not change thermodynamics. Each solve restores that state and all multipliers.
    workers=1 is sequential; workers>1 uses independent spawned processes.
    Both modes use 1+2*N solves and never edit the YAML. No user main guard is needed.
    Unlike calculate_znd's sampled SDToolbox lengths, this diagnostic refines the
    peak locally and interpolates the two half-height crossings around that peak.
    Check convergence versus perturbation, time step, and tolerances before using
    small sensitivities; multiple thermicity pulses can cause peak switching.

    Return baseline lengths, indices/equations, sensitivities and perturbed lengths
    in matching input order, per-reaction errors, and optional plot file paths.
    Invalid baseline profiles raise; failed perturbations return NaN (not zero).
    plot=True saves two independently magnitude-sorted horizontal bar plots.
    progress=True displays completed reactions, failures, elapsed time, and ETA.
    The bar starts after the baseline solve; failed reactions also advance it.
    partial_file optionally saves each completed reaction immediately as CSV
    (completion order, one-based reaction numbers; not a resumable checkpoint).
    """
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        raise ValueError('workers must be a positive integer.')
    settings = dict(P0=P0, T0=T0, fuel=fuel, oxidizer=oxidizer, phi=phi,
                    mechanism=str(mechanism), t_end=t_end, max_step=max_step,
                    rel_tol=rel_tol, abs_tol=abs_tol, method=method,
                    perturbation=perturbation, reaction_indices=reaction_indices,
                    plot=plot, plot_file=plot_file, top_n=top_n, progress=progress,
                    progress_label=progress_label, workers=int(workers), partial_file=partial_file)
    if workers == 1:
        return _znd_sensitivity(**settings)
    from .sensitivity_parallel import launch
    # Snapshot iterators before sending arguments to another interpreter.
    if reaction_indices is not None:
        settings['reaction_indices'] = list(reaction_indices)
    return launch(settings)


def _sensitivity_pair(index, perturbation, solve):
    """Use identical perturbation/error handling in sequential and process workers."""
    record = {'index': index, 'minus': {}, 'plus': {}, 'errors': [], 'multiple_pulses': False}
    for sign, target in ((-1, 'minus'), (1, 'plus')):
        try:
            lengths = solve(index, 1 + sign*perturbation)
            record[target] = lengths
            record['multiple_pulses'] |= lengths['pulse_count'] > 1
        except (ct.CanteraError, ValueError, RuntimeError, FloatingPointError, IndexError) as error:
            record['errors'].append(f'm={1 + sign*perturbation:g}: {error}')
    return record


def _znd_sensitivity(P0, T0, fuel, oxidizer, phi, mechanism, *, t_end=3e-5,
                    max_step=1e-6, rel_tol=1e-6, abs_tol=1e-9, method='BDF',
                    perturbation=0.01, reaction_indices=None, plot=False,
                    plot_file='znd_sensitivity.pdf', top_n=None,
                    progress=False, progress_label='ZND sensitivity',
                    workers=1, partial_file=None):
    """Compute the baseline, coordinate reaction tasks, and assemble ordered output."""
    if isinstance(perturbation, bool) or not np.isfinite(perturbation) or not 0 < perturbation < 1:
        raise ValueError('perturbation must be finite and between 0 and 1.')
    if 1 + perturbation == 1 or 1 - perturbation == 1:
        raise ValueError('perturbation is too small to represent distinct rate multipliers.')
    if not isinstance(plot, bool):
        raise ValueError('plot must be a boolean.')
    if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, Integral) or top_n < 1):
        raise ValueError('top_n must be a positive integer or None.')
    for name, value in (('P0', P0), ('T0', T0), ('phi', phi), ('t_end', t_end),
                        ('max_step', max_step), ('rel_tol', rel_tol), ('abs_tol', abs_tol)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite.')
    if method not in ('RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA'):
        raise ValueError('Unsupported integration method.')
    mechanism = str(mechanism)
    gas1 = ct.Solution(mechanism)
    indices = list(range(gas1.n_reactions)) if reaction_indices is None else list(reaction_indices)
    if not indices or any(isinstance(i, bool) or not isinstance(i, Integral)
                          or not 0 <= i < gas1.n_reactions for i in indices):
        raise ValueError('reaction_indices must contain valid zero-based reaction indices.')
    indices = [int(i) for i in indices]
    if len(set(indices)) != len(indices):
        raise ValueError('reaction_indices must not contain duplicates.')
    if progress:
        print(f'{progress_label}: computing CJ state and baseline...', file=sys.stderr, flush=True)
    gas1.TP = T0, P0
    gas1.set_equivalence_ratio(phi, fuel, oxidizer, basis='mole')
    speed = CJspeed(P0, T0, gas1.X.copy(), mechanism)
    gas = PostShock_fr(speed, P0, T0, gas1.X.copy(), mechanism)
    temperature, pressure, composition = gas.TPX
    composition = composition.copy()

    def solve(index=None, multiplier=1.0):
        """Reset the frozen state and rate multipliers before each independent solve."""
        gas.set_multiplier(1.0)
        gas.TPX = temperature, pressure, composition
        if index is not None:
            gas.set_multiplier(multiplier, index)
        output = zndsolve(gas, gas1, speed, t_end=t_end, max_step=max_step,
                          relTol=rel_tol, absTol=abs_tol, Method=method, advanced_output=False)
        return _sensitivity_lengths(output, t_end)

    upstream = (T0, P0, gas1.X.copy())
    frozen = (temperature, pressure, composition)
    baseline = solve()
    keys = ('ind_len_ZND', 'exo_len_ZND')
    minus = {key: np.full(len(indices), np.nan) for key in keys}
    plus = {key: np.full(len(indices), np.nan) for key in keys}
    sensitivities = {key: np.full(len(indices), np.nan) for key in keys}
    errors = {}
    multiple_pulses = baseline['pulse_count'] > 1
    denominator = np.log1p(perturbation) - np.log1p(-perturbation)
    started = monotonic()

    def show_progress(done):
        """Render completed reaction pairs without adding a progress-bar dependency."""
        if not progress:
            return
        elapsed = monotonic() - started
        fraction = done / len(indices)
        filled = int(24 * fraction)
        bar = '#' * filled + '-' * (24 - filled)
        eta = f'{elapsed / done * (len(indices) - done):.0f}s' if done else '--'
        end = '\r' if sys.stderr.isatty() else '\n'
        print(f'{progress_label} [{bar}] {done}/{len(indices)} '
              f'({fraction:.1%}) | failed {len(errors)} | {elapsed:.0f}s | ETA {eta}',
              end=end, file=sys.stderr, flush=True)

    equations = gas1.reaction_equations()
    row_for_index = {index: row for row, index in enumerate(indices)}
    partial = None
    records = None
    show_progress(0)
    try:
        if partial_file is not None:
            destination = Path(partial_file)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.open('w', newline='')
            writer = csv.writer(partial)
            writer.writerow(['reaction_number', 'equation', 'S_ind_len_ZND', 'S_exo_len_ZND', 'error'])
            partial.flush()
        if workers == 1:
            records = (_sensitivity_pair(index, perturbation, solve) for index in indices)
        else:
            from .sensitivity_parallel import reaction_results
            solver = dict(t_end=t_end, max_step=max_step, relTol=rel_tol, absTol=abs_tol,
                          Method=method, advanced_output=False)
            records = reaction_results(indices, perturbation, workers, mechanism,
                                       upstream, frozen, speed, solver)
        for done, record in enumerate(records, 1):
            index = record['index']
            row = row_for_index[index]
            multiple_pulses |= record['multiple_pulses']
            for key in keys:
                minus[key][row] = record['minus'].get(key, np.nan)
                plus[key][row] = record['plus'].get(key, np.nan)
            if record['errors']:
                errors[index] = '; '.join(record['errors'])
            else:
                for key in keys:
                    sensitivities[key][row] = (np.log(plus[key][row]) - np.log(minus[key][row])) / denominator
            if partial is not None:
                writer.writerow([index + 1, equations[index],
                                 *(f'{sensitivities[key][row]:.10e}' for key in keys),
                                 errors.get(index, '')])
                partial.flush()
            show_progress(done)
    finally:
        if records is not None:
            records.close()
        if partial is not None:
            partial.close()
        if progress and sys.stderr.isatty():
            print(file=sys.stderr, flush=True)
        gas.set_multiplier(1.0)
    if errors:
        warnings.warn(f'{len(errors)} reaction sensitivities are unavailable; inspect result["errors"].', RuntimeWarning)
    if multiple_pulses:
        warnings.warn('Multiple thermicity pulses detected; sensitivities may be affected by peak switching.', RuntimeWarning)
    result = {'baseline': baseline, 'cj_speed': float(speed), 'perturbation': float(perturbation),
              'reaction_indices': indices, 'equations': [equations[i] for i in indices],
              'sensitivities': sensitivities, 'lengths_minus': minus, 'lengths_plus': plus,
              'errors': errors, 'multiple_pulses': bool(multiple_pulses), 'plot_files': {}}
    if plot:
        result['plot_files'] = plot_znd_sensitivity(result, plot_file, top_n)
    return result


def _integrate_znd(gas, gas1, cj_speed, *, P0, T0, phi, t_end, max_step,
                   rel_tol, abs_tol, method):
    """Integrate prepared gas states with the same checks as calculate_znd."""
    try:
        znd_out = zndsolve(
            gas, gas1, cj_speed, t_end=t_end, max_step=max_step,
            relTol=rel_tol, absTol=abs_tol, Method=method,
            advanced_output=True,
        )
    except IndexError as error:
        # SDToolbox's advanced-output diagnostics can index M[len(M)] when
        # thermicity peaks at the first point. Treat this as solver failure,
        # not as valid induction/exothermic lengths or a search-wide crash.
        raise RuntimeError(
            'SDToolbox ZND output processing failed (index error); '
            f'P={P0:g} Pa, T={T0:g} K, phi={phi:g}, t_end={t_end:g} s. '
            'The thermicity pulse may not be resolved for this candidate.'
        ) from error
    # SDToolbox can return a partial integration without exposing solver status.
    if not np.isclose(znd_out['time'][-1], t_end, rtol=1e-6, atol=1e-15):
        raise RuntimeError(
            f"Integration stopped at {znd_out['time'][-1]:.6g} s "
            f"before requested t_end={t_end:.6g} s."
        )
    for key in ('T', 'P', 'rho'):
        values = np.asarray(znd_out[key])
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise RuntimeError(f'Invalid {key} profile; results were not exported.')
    # Unresolved widths must not appear as perfect zero-length predictions.
    if znd_out['exo_time_ZND'] <= 0 or znd_out['exo_len_ZND'] <= 0:
        warnings.warn(
            'Thermicity pulse width was not resolved; exothermic length/time '
            'are unavailable (NaN), not zero.', RuntimeWarning,
        )
        znd_out['exo_time_ZND'] = np.nan
        znd_out['exo_len_ZND'] = np.nan

    return znd_out


def calculate_znd(P0, T0, fuel, oxidizer, phi, mechanism, file_name,
                  t_end=3e-5, max_step=1e-6, rel_tol=1e-6,
                  abs_tol=1e-9, method='BDF', species_basis='Y'):
    """Return ZND results and save a text table to file_name (including extension).

    Set file_name=None to skip saving the profile table.
    P0 is in Pa, T0 in K, and integration times in seconds.
    Fuel and oxidizer compositions use relative molar proportions.
    species_basis selects text-file species columns: 'X' (mole) or 'Y' (mass).
    The returned znd_out['species'] always retains SDToolbox mass fractions.
    """
    if species_basis not in ('X', 'Y'):
        raise ValueError("species_basis must be 'X' (mole) or 'Y' (mass).")

    gas1 = ct.Solution(mechanism)
    gas1.TP = T0, P0
    gas1.set_equivalence_ratio(phi, fuel, oxidizer, basis='mole')
    x = gas1.X.copy()

    # CJ wave speed [m/s] relative to the unburned gas.
    cj_speed = CJspeed(P0, T0, x, mechanism)
    # Frozen post-shock state: compression and heating without reaction.
    # Keep this separate from gas1, which represents the upstream state.
    gas = PostShock_fr(cj_speed, P0, T0, x, mechanism)
    znd_out = _integrate_znd(
        gas, gas1, cj_speed, P0=P0, T0=T0, phi=phi, t_end=t_end,
        max_step=max_step, rel_tol=rel_tol, abs_tol=abs_tol, method=method,
    )

    znd_out['label'] = mechanism
    # Optimization uses in-memory results and skips large profile exports.
    if file_name is None:
        return znd_out

    # SDToolbox returns mass fractions; convert only the exported columns.
    # For each point: X_k = (Y_k / W_k) / sum_j(Y_j / W_j).
    species = znd_out['species'].T
    if species_basis == 'X':
        species = species / gas1.molecular_weights
        species = species / species.sum(axis=1, keepdims=True)

    # One row per solution point; species columns use the selected basis.
    # time [s], distance [m], temperature [K], pressure [Pa], density [kg/m^3],
    # shock-frame flow velocity [m/s], and thermicity [1/s].
    # Thermicity measures reaction-driven expansion, including heat release
    # and composition changes; it is not a heat-release rate in W/m^3.
    keys = ['time', 'distance', 'T', 'P', 'rho', 'U', 'thermicity']
    data = np.column_stack([znd_out[key] for key in keys]
                           + [species])
    header = (
        f'mechanism: {mechanism}\nfuel: {fuel}\noxidizer: {oxidizer}\n'
        f'phi: {phi:g}\nP0 [Pa]: {P0:g}\nT0 [K]: {T0:g}\n'
        f'CJ speed [m/s]: {cj_speed:.10g}\n'
        f'species_basis: {species_basis} (X=mole, Y=mass)\n'
        f'method: {method}; t_end [s]: {t_end:g}; max_step [s]: {max_step:g}; '
        f'rel_tol: {rel_tol:g}; abs_tol: {abs_tol:g}\n'
    )
    # Induction: location/time of maximum thermicity.
    # Exothermic width: spatial/time width of the thermicity pulse at half maximum.
    for key in ['ind_len_ZND', 'exo_len_ZND', 'ind_time_ZND', 'exo_time_ZND']:
        unit = 'm' if 'len' in key else 's'
        header += f'{key} [{unit}]: {znd_out[key]:.10g}\n'
    header += 'time_s distance_m T_K P_Pa rho_kg_m3 U_m_s thermicity_1_s '
    # X_species or Y_species identifies dimensionless mole or mass fractions.
    header += ' '.join(species_basis + '_' + name for name in gas1.species_names)
    np.savetxt(file_name, data, header=header, fmt='%.10e')
    return znd_out


def plot_temperature(znd_out, znd_out2=None, labels=None,
                     file_name='temperature.png', title=None, xlim=None, ylim=None):
    """Plot one result in blue solid and an optional second in blue dashed.

    Save to file_name without opening a window; return the output filename.
    labels supplies one label per result; title optionally sets the plot title.
    xlim=(xmin, xmax) sets the displayed distance range in mm; None shows all data.
    ylim=(Tmin, Tmax) sets the temperature range in K; None uses automatic limits.
    """
    results = [znd_out] if znd_out2 is None else [znd_out, znd_out2]
    if labels is None:
        labels = [out.get('label', f'Case {i + 1}')
                  for i, out in enumerate(results)]
    if len(labels) != len(results):
        raise ValueError('Provide one label per ZND result.')

    for name, limits, unit in (('xlim', xlim, 'mm'), ('ylim', ylim, 'K')):
        if limits is not None and (len(limits) != 2 or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]):
            raise ValueError(f'{name} must contain two finite increasing values in {unit}.')

    # Create a standalone figure, without a pyplot window or open-figure registry.
    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    # Convert metres to millimetres and show the full extent of both results.
    xmax = max(out['distance'][-1] for out in results) * 1000
    # Extend only the illustrative upstream segment when a wider view is requested.
    upstream_x = min(-0.1 * xmax, xlim[0]) if xlim is not None else -0.1 * xmax
    for out, style, label in zip(results, ['b-', 'b--'], labels):
        # Negative distance is an illustrative upstream segment at initial T.
        # Two points at x = 0 draw the vertical temperature jump at the shock.
        x_mm = np.r_[upstream_x, 0, np.asarray(out['distance']) * 1000]
        temperature = np.r_[out['gas1'].T, out['gas1'].T, out['T']]
        ax.plot(x_mm, temperature, style, linewidth=2, label=label)

    ax.set_xlabel('Distance behind shock, x (mm)')
    ax.set_ylabel('Temperature, T (K)')
    ax.set_xlim((-0.1 * xmax, xmax) if xlim is None else xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # Match the example's inward ticks on all four sides.
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(frameon=False, loc='lower right')
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(file_name, dpi=300)
    return file_name
