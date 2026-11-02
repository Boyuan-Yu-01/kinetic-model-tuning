# ZND validation

Run `python plot_comparison.py` with the package installed. The script uses eight
workers and the nine paired conditions explicitly listed in the file. It compares
`FFCM2.yaml` and `best_model.yaml` using LSODA, then prints final temperatures,
induction/exothermic lengths, signed percentage errors and absolute log ratios.

One blue solid-reference/dashed-circle-model PNG per case is saved beneath
`temperature_profiles/`. No persistent JSON or solver logs are produced by default.
Set `workers=1` for sequential execution. The reusable API is `mt.validation.Validation`.
These snapshots and cases are independent of later edits to the optimization run file.
