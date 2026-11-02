"""Sequential ignition-delay cases and incremental YAML output."""
from pathlib import Path
from itertools import product
import cantera as ct
import yaml
from .plot_settings import yscale
from .combustion_calculator import CombustionCalculator


def resolve_ignition_settings(problem):
    """Expand the declared pressure, temperature and equivalence-ratio grid."""
    conditions=problem['conditions']['ignition']
    return dict(mechanism=problem['mechanism'], output=problem['output'],
                cases=list(product(conditions['pressure_atm'], conditions['temperature_K'], conditions['phi'])),
                fuel=problem['mixture']['fuel'], oxidizer=problem['mixture']['oxidizer'],
                solver=dict(problem['solver']), comparison_plots=problem.get('comparison_plots', False),
                plot_yscales=dict(problem.get('plot_yscales', {})))


def run_ignition_cases(problem):
    """Calculate the existing OH-gradient IDT and save completed cases atomically."""
    cfg=resolve_ignition_settings(problem)
    MECHANISM=cfg['mechanism']; OUTPUT=Path(cfg['output'])
    REACTOR_TYPE=cfg['solver']['reactor_type']; END_TIME=cfg['solver']['end_time']
    results = []
    for case, (pressure, temperature, phi) in enumerate(
            cfg['cases'], 1):
        row = dict(case=case, pressure_atm=pressure, temperature_K=temperature, phi=phi)
        calculator = CombustionCalculator(
            mechanism=MECHANISM, fuel=cfg['fuel'], oxidizer=cfg['oxidizer'],
            P=pressure, phi=phi, T=temperature,
        )
        label = f"{case:02d}/{len(cfg['cases'])} | {pressure:2d} atm | {temperature} K | phi={phi:.1f}"
        try:
            delay = calculator.ignition_delay(
                **cfg['solver'],
            )
            row.update(ignition_delay_s=delay, ignition_delay_us=delay * 1e6)
            print(f'{label} | IDT={delay * 1e6:.6g} us', flush=True)
        except (ct.CanteraError, RuntimeError, ValueError) as error:
            row.update(ignition_delay_s=None, error=str(error))
            print(f'{label} | FAILED: {error}', flush=True)
        results.append(row)
        # Keep completed cases even if the script is interrupted later.
        report = dict(mechanism=str(MECHANISM), reactor=REACTOR_TYPE,
                      definition='Time of maximum d[OH]/dt; OH molar concentration',
                      fuel=cfg['fuel'], oxidizer=cfg['oxidizer'], end_time_s=END_TIME,
                      results=results)
        temporary = OUTPUT.with_suffix('.tmp.yaml')
        temporary.write_text(yaml.safe_dump(report, sort_keys=False))
        temporary.replace(OUTPUT)
    
    print(f"Finished: {sum('error' not in row for row in results)}/{len(cfg['cases'])} successful. {OUTPUT}")
    if cfg['comparison_plots']:
        save_ignition_plot(results, OUTPUT.with_suffix('.png'), cfg['plot_yscales'])
    return report


def save_ignition_plot(rows, destination, scales=None):
    """Plot valid positive ignition delays versus initial temperature, per pressure."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scale=yscale(scales,'ignition_delay_time','log')
    pressures=sorted({r['pressure_atm'] for r in rows})
    fig,axes=plt.subplots(1,len(pressures),squeeze=False,figsize=(5*len(pressures),4))
    for ax,pressure in zip(axes.flat,pressures):
        for phi in sorted({r['phi'] for r in rows if r['pressure_atm']==pressure}):
            group=sorted([r for r in rows if r['pressure_atm']==pressure and r['phi']==phi],key=lambda r:r['temperature_K'])
            delays=[r.get('ignition_delay_s') for r in group]
            values=[v*1e6 if v is not None and np.isfinite(v) and v>0 else np.nan for v in delays]
            ax.plot([r['temperature_K'] for r in group],values,'o-',label=f'phi={phi:g}')
        ax.set_yscale(scale)
        ax.set(xlabel='Initial temperature (K)',ylabel='Ignition delay (us)',title=f'{pressure:g} atm')
        ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(destination,dpi=180);plt.close(fig)
