"""Optional solver-backed check, run inside a Slurm allocation after stage A."""
from pathlib import Path
import json
import sys
import numpy as np
from mt.iterative_znd import _solve
from mt.executors import create_executor

if __name__=='__main__':
    root=Path(sys.argv[1])
    cfg=json.loads((root/'output/01_A/run_info/settings.json').read_text())
    cfg['output_dir']=str(root/'backend_check')
    (Path(cfg['output_dir'])/'solver_logs').mkdir(parents=True,exist_ok=True)
    states=json.loads((root/'preflight/states.json').read_text())[9:]
    model=str(root/'output/01_A/best_model.yaml')
    job=(model,states[1],None,cfg,None,False)
    serial=_solve(job)
    with create_executor('openmp',2,max_tasks_per_worker=1) as pool:
        parallel=list(pool.map(_solve,[job,job]))
    keys=('T_final','ind_len_ZND','exo_len_ZND')
    if 'error' in serial or any('error' in row for row in parallel):
        raise RuntimeError(str([serial,*parallel]))
    difference=max(abs(row[k]-serial[k])/abs(serial[k]) for row in parallel for k in keys)
    assert difference<1e-12,(serial,parallel)
    record=dict(case=2,model=model,serial={k:serial[k] for k in keys},
                max_relative_difference=difference,passed=True)
    (root/'backend_check/results.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
