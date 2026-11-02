"""Reusable, plot-only-by-default validation of flame, ignition and ZND models."""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import json
import math
import os
import subprocess
import sys
import tempfile
import numpy as np
from ._execution import launcher_environment, wait_for_process


QUANTITIES = {
    'flame': ('laminar_flame_speed', 'T_flame'),
    'ignition': ('ignition_delay_time', 'T_ignition'),
    'znd': ('T_ZND', 'ind_len_ZND', 'exo_len_ZND'),
}
LABELS = {'laminar_flame_speed': ('Flame speed (cm/s)', 100.),
          'ignition_delay_time': ('Ignition delay (µs)', 1e6),
          'T_flame': ('Burned temperature (K)', 1.),
          'T_ignition': ('Final temperature (K)', 1.),
          'T_ZND': ('Final temperature (K)', 1.),
          'ind_len_ZND': ('Induction length (m)', 1.),
          'exo_len_ZND': ('Exothermic length (m)', 1.)}


def _cases(conditions, kind):
    """Expand Cartesian flame/IDT grids, but pair ZND arrays with scalar broadcasting."""
    pressure = [k for k in ('pressure_atm', 'pressure_bar') if k in conditions]
    if len(pressure) != 1:
        raise ValueError('Specify exactly one of pressure_atm or pressure_bar.')
    keys = [pressure[0], 'temperature_K', 'phi'] + (['end_time_s'] if kind == 'znd' else [])
    if set(conditions) != set(keys):
        raise ValueError(f'{kind} conditions must contain exactly {keys}.')
    arrays = [np.atleast_1d(np.asarray(conditions[k], dtype=float)) for k in keys]
    if any(a.ndim != 1 or not len(a) or not np.isfinite(a).all() or (a <= 0).any() for a in arrays):
        raise ValueError('Condition values must be nonempty, finite and positive.')
    if kind == 'znd':
        n = max(map(len, arrays))
        if any(len(a) not in (1, n) for a in arrays):
            raise ValueError('ZND arrays must have equal lengths; scalars may be broadcast.')
        combinations = zip(*(np.repeat(a, n) if len(a) == 1 else a for a in arrays))
    else:
        combinations = product(*arrays)
    unit = pressure[0].split('_')[1]
    rows = []
    for values in combinations:
        p, t, phi = map(float, values[:3])
        row = dict(pressure=p, pressure_unit=unit, P_atm=p if unit == 'atm' else p*1e5/101325., T=t, phi=phi)
        if kind == 'znd':
            row['end_time_s'] = float(values[3])
        rows.append(row)
    return rows


class Validation:
    """Compare any number of mechanisms against one reference using existing mt solvers.

    Methods calculate once and return a ValidationResult for summaries and replotting.
    Parallel workers run in a guarded subprocess, so calling scripts need no main guard.
    """
    def __init__(self, reference, models, fuel, oxidizer='O2:1, N2:3.76', *,
                 workers=8, output_dir='results', save_results=False, save_logs=False):
        if not models or 'Reference' in models or any(not isinstance(k, str) or not k for k in models):
            raise ValueError('Provide named models; Reference is reserved.')
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError('workers must be a positive integer.')
        def source(path):
            path = Path(path).expanduser()
            return str(path.resolve()) if path.exists() else str(path)
        self.models = {'Reference': source(reference), **{k: source(v) for k, v in models.items()}}
        self.fuel, self.oxidizer = fuel, oxidizer
        self.workers = workers
        self.output_dir = Path(output_dir).resolve()
        self.save_results, self.save_logs = save_results, save_logs

    def flame(self, conditions, solver=None):
        """Calculate flame speed and burned temperature on a Cartesian case grid."""
        return self._calculate('flame', conditions, solver)

    def ignition(self, conditions, solver=None):
        """Calculate ignition delay and endpoint temperature on a Cartesian grid."""
        return self._calculate('ignition', conditions, solver)

    def znd(self, conditions, solver=None):
        """Calculate ZND profiles and lengths for paired inlet conditions/end times."""
        return self._calculate('znd', conditions, solver)

    def _calculate(self, kind, conditions, solver):
        """Launch isolated single-core or process-pool calculations with temporary IPC."""
        defaults = {
            'flame': dict(width=.02, transport_model='multicomponent', soret=False,
                          slope=.015, curve=.03, timeout_seconds=900),
            'ignition': dict(marker_species='temperature', reactor_type='constant-volume',
                             rtol=1e-9, atol=1e-15, end_time=.003),
            'znd': dict(method='LSODA', rel_tol=1e-10, abs_tol=1e-14, steps=4000),
        }
        options = {**defaults[kind], **(solver or {})}
        allowed = set(defaults[kind]) | ({'max_step', 'species_basis'} if kind == 'znd' else set())
        if set(options)-allowed:
            raise ValueError(f'Unknown {kind} solver settings: {sorted(set(options)-allowed)}')
        if kind == 'znd' and (not math.isfinite(options['steps']) or options['steps'] <= 0):
            raise ValueError('steps must be finite and positive.')
        cases = _cases(conditions, kind)
        cfg = dict(kind=kind, cases=cases, models=self.models, solver=options,
                   fuel=self.fuel, oxidizer=self.oxidizer, workers=self.workers,
                   output_dir=str(self.output_dir), save_logs=self.save_logs)
        with tempfile.TemporaryDirectory(prefix='mt-validation-') as folder:
            request, response = (Path(folder)/n for n in ('request.json', 'response.json'))
            request.write_text(json.dumps(cfg))
            process = subprocess.Popen([sys.executable, '-u', '-m', 'mt.validation_worker',
                                        str(request), str(response)], env=launcher_environment(),
                                       start_new_session=os.name == 'posix')
            if wait_for_process(process):
                raise RuntimeError('Validation worker failed; see the diagnostic above.')
            rows = json.loads(response.read_text())
        result = ValidationResult(kind, cases, rows, self.output_dir)
        if self.save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir/f'{kind}_results.json').write_text(json.dumps(
                dict(kind=kind, cases=cases, models=self.models, results=rows), indent=2, allow_nan=False))
        return result


@dataclass
class ValidationResult:
    """Cached SI-unit predictions, errors and profiles; plotting never reruns a solver."""
    kind: str
    cases: list
    results: dict
    output_dir: Path

    @property
    def failures(self):
        """Return model/case-specific failures without discarding other predictions."""
        return [(name, i+1, r['error']) for name, rows in self.results.items()
                for i, r in enumerate(rows) if 'error' in r]

    def print_summary(self):
        """Print signed percentage errors and absolute log ratios (temperature: delta K)."""
        for i, case in enumerate(self.cases):
            print(f'Case {i+1}/{len(self.cases)}: {case["pressure"]:g} {case["pressure_unit"]}, '
                  f'{case["T"]:g} K, phi={case["phi"]:g}')
            for name, rows in self.results.items():
                if 'error' in rows[i]:
                    print(f'  {name}: FAILED — {rows[i]["error"]}')
            ref = self.results['Reference'][i]
            for name, rows in self.results.items():
                if name == 'Reference' or 'error' in rows[i] or 'error' in ref:
                    continue
                for q in QUANTITIES[self.kind]:
                    a, b = ref.get(q), rows[i].get(q)
                    if a is None or b is None:
                        print(f'  {name} {q}: unavailable'); continue
                    percent = f'{100*(b/a-1):+.2f}%' if a else 'unavailable'
                    extra = f'ΔT={b-a:+.3f} K' if q.startswith('T_') else (
                        f'|ln(model/reference)|={abs(math.log(b)-math.log(a)):.6g}'
                        if min(a,b)>0 else '|ln(model/reference)|=unavailable')
                    print(f'  {name} {q}: reference={a:.6g}, model={b:.6g}; error={percent}; {extra}')

    def plot(self, quantities=None, *, x=None, separate_by=None, yscales=None, styles=None):
        """Save separate case profiles or pressure panels; failed points remain gaps.

        Scalar plots group by phi for temperature axes, and by temperature for phi axes.
        Flame speeds are displayed in cm/s, delays in microseconds, distances in mm.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        profile = self.kind == 'znd' and (x is None or x == 'distance_mm')
        x = x or ('distance_mm' if profile else 'phi' if self.kind == 'flame' else '1000/T')
        separate_by = separate_by or ('case' if profile else 'pressure')
        if separate_by != ('case' if profile else 'pressure'):
            raise ValueError('Use separate_by="case" for ZND profiles or "pressure" for scalar plots.')
        quantities = list(quantities or (['T_profile'] if profile else QUANTITIES[self.kind]))
        if not quantities or (profile and quantities != ['T_profile']) or (not profile and any(q not in QUANTITIES[self.kind] for q in quantities)):
            raise ValueError('Choose supported quantities; ZND profiles use T_profile.')
        if not profile and x not in ('phi', '1000/T', 'temperature_K'):
            raise ValueError('Scalar x must be phi, 1000/T or temperature_K.')
        scales = {'ignition_delay_time': 'log', **(yscales or {})}
        if any(v not in ('linear', 'log') for v in scales.values()):
            raise ValueError('Axis scale must be linear or log.')
        styles = styles or {}
        unknown = set(styles)-set(self.results)
        if unknown:
            raise ValueError(f'Unknown model labels in styles: {sorted(unknown)}')
        def style(name, index, color):
            base = dict(color=color, linestyle=['-', '--', ':', '-.'][index % 4])
            if index:
                base.update(marker='o', markerfacecolor='none', markersize=5)
            return {**base, **styles.get(name, {})}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        groups = list(range(len(self.cases))) if profile else sorted({c['pressure'] for c in self.cases})
        for group in groups:
            fig, axes = plt.subplots(len(quantities), 1, figsize=(8, 5*len(quantities)), squeeze=False)
            try:
                for ax, q in zip(axes[:, 0], quantities):
                    if profile:
                        case = self.cases[group]
                        failed = []
                        for j, (name, rows) in enumerate(self.results.items()):
                            row = rows[group]
                            if 'error' in row:
                                failed.append(name); continue
                            extent = max((r[group]['distance_m'][-1]*1e3 for r in self.results.values()
                                          if 'distance_m' in r[group]), default=1.)
                            # Illustrative upstream state and shock jump, as in plot_temperature.
                            distance = np.r_[-.1*extent, 0., np.asarray(row['distance_m'])*1e3]
                            temperature = np.r_[case['T'], case['T'], row['temperature_K']]
                            kw = style(name, j, 'blue')
                            if j: kw.setdefault('markevery', .06)
                            ax.plot(distance, temperature, label=name, **kw)
                        ax.set(xlabel='Distance behind shock (mm)', ylabel='Temperature (K)',
                               title=f'{case["pressure"]:g} {case["pressure_unit"]}, {case["T"]:g} K, phi={case["phi"]:g}')
                        if failed:
                            ax.text(.02,.02,'Unavailable: '+', '.join(failed),transform=ax.transAxes)
                    else:
                        indices = [i for i,c in enumerate(self.cases) if c['pressure']==group]
                        key = 'T' if x=='phi' else 'phi'
                        values = sorted({self.cases[i][key] for i in indices})
                        xvalue = lambda i: self.cases[i]['phi'] if x=='phi' else 1000/self.cases[i]['T'] if x=='1000/T' else self.cases[i]['T']
                        for k, value in enumerate(values):
                            ids = sorted([i for i in indices if self.cases[i][key]==value], key=xvalue)
                            for j, (name, rows) in enumerate(self.results.items()):
                                # Keep NaN placeholders to avoid connecting across failed solves.
                                y = [rows[i].get(q) if rows[i].get(q) is not None else np.nan for i in ids]
                                label = f'{name}, {key}={value:g}'
                                ax.plot([xvalue(i) for i in ids], np.asarray(y)*LABELS[q][1],
                                        label=label, **style(name,j,f'C{k%10}'))
                        nfailed = sum('error' in rows[i] for rows in self.results.values() for i in indices)
                        ax.set(xlabel={'phi':'Equivalence ratio, φ','1000/T':'1000/T₀ (K⁻¹)',
                                       'temperature_K':'Initial temperature (K)'}[x],
                               ylabel=LABELS[q][0], title=f'{group:g} {self.cases[indices[0]]["pressure_unit"]}')
                        if nfailed: ax.text(.02,.02,f'{nfailed} failed solves (gaps)',transform=ax.transAxes)
                    ax.set_yscale(scales.get(q,'linear'))
                    ax.grid(alpha=.25)
                    if ax.lines: ax.legend(fontsize=9)
                fig.tight_layout()
                name = f'znd_case_{group+1:02d}.png' if profile else f'{self.kind}_{group:g}_{self.cases[0]["pressure_unit"]}.png'
                path = self.output_dir/name
                fig.savefig(path,dpi=200); paths.append(path)
            finally:
                plt.close(fig)
        return paths
