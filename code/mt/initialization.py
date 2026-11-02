"""Explicit user-requested changes to a human-supplied mechanism."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import yaml
import cantera as ct


def add_plog(model, output, pressure_entries_bar):
    """Copy ordinary Arrhenius reactions to fixed PLOG nodes (one-based indices).

    All nodes initially use the original A, b, and Ea, so k(T,P) is unchanged.
    The input mechanism is never overwritten. Pressure entries are in bar;
    activation-energy and rate units remain those declared by the input YAML.
    """
    source, destination = Path(model).resolve(), Path(output).resolve()
    if source == destination:
        raise ValueError('PLOG preparation requires a separate output mechanism.')
    scheme=yaml.safe_load(source.read_text())
    for number, entries in pressure_entries_bar.items():
        if isinstance(number,bool) or not isinstance(number,int) or not 1<=number<=len(scheme['reactions']):
            raise ValueError('Use one-based integer reaction numbers.')
        pressures=np.asarray(entries,dtype=float)
        if (pressures.ndim!=1 or len(pressures)<2 or not np.isfinite(pressures).all()
                or np.any(pressures<=0) or np.any(np.diff(pressures)<=0)):
            raise ValueError('PLOG entries must be distinct, positive, increasing pressures in bar.')
        reaction=scheme['reactions'][number-1]
        if reaction.get('type','elementary')!='elementary' or 'rate-constant' not in reaction:
            raise ValueError(f'R{number}: conversion requires an ordinary Arrhenius reaction.')
        rate=reaction.pop('rate-constant')
        reaction['type']='pressure-dependent-Arrhenius'
        reaction['rate-constants']=[dict(P=f'{p:g} bar',**deepcopy(rate)) for p in pressures]
    text=yaml.safe_dump(scheme,sort_keys=False)
    ct.Solution(yaml=text)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.tmp.yaml'); temporary.write_text(text); temporary.replace(destination)
    return destination
