# ZND validation

Run `python plot_comparison.py` with the package installed. The script uses eight
workers and the twelve paired conditions explicitly listed in the file. It compares
`FFCM2.yaml` and `best_model.yaml` using LSODA, then prints final temperatures,
induction/exothermic lengths, signed percentage errors and absolute log ratios.

One blue solid-reference/dashed-circle-model PNG per case is saved beneath
`temperature_profiles/`. No persistent JSON or solver logs are produced by default.
Set `workers=1` for sequential execution. The reusable API is `mt.validation.Validation`.
These snapshots and cases are independent of later edits to the optimization run file.

## Twelve-case presentation

[PDF slides](results.pdf) compare all twelve temperature profiles with the supplied numerical results.
The deck includes signed percentage errors, absolute log-ratio errors, and an
all-case summary. Editable PowerPoint files stay local and are excluded from GitHub.
