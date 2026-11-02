# Basic combustion calculations

Three independent, commented scripts demonstrate the installed `mt` API using
FFCM2 and conditions from the existing ethylene examples:

| Script | Condition | Output |
|---|---|---|
| `run_ZND.py` | 1 bar, 298 K, phi = 1 | Final temperature, induction/exothermic lengths, `znd_temperature.png` |
| `run_LFS.py` | 1 atm, 298 K, phi = 1 | Flame speed, burned temperature, flame/diffusion thicknesses |
| `run_IDT.py` | 1 atm, 1500 K, phi = 1 | Maximum d[OH]/dt ignition delay and temperature at 0.003 s |

Activate your environment and run whichever calculation you need:

```bash
conda activate ct-env
python run_ZND.py
python run_LFS.py
python run_IDT.py
```

ZND takes pressure in Pa; `CombustionCalculator` takes pressure in atm.
The scripts reuse `../ethylene_ZND/inputs/FFCM2.yaml`. They recalculate results
rather than reading cached outputs, and do not perform optimization.
