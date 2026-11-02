import unittest
from pathlib import Path
from copy import deepcopy
import yaml
from mt.staged_parameters import StagedParameters
ROOT=Path(__file__).resolve().parent
class MethaneUnits(unittest.TestCase):
 def test_all_stages_and_real_mismatch(self):
  model=yaml.safe_load((ROOT/'fixtures/methane_units.yaml').read_text())
  reference=deepcopy(model);reference['reactions']=[];reference.setdefault('units',{})['time']='s'
  for stage in ['A','AE','AEB']:
   StagedParameters(model,reference,uncertainty_factors={},unbounded_reactions=[1,2,3],a_ratio=(1e-6,1e8),ea_delta=60000.,stage=stage)
  bad=deepcopy(reference);bad['units']['time']='ms'
  with self.assertRaisesRegex(ValueError,'identical YAML units'):
   StagedParameters(model,bad,uncertainty_factors={},unbounded_reactions=[1,2,3])
