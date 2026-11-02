import unittest
from types import SimpleNamespace
import numpy as np
from mt.flame_adjoint import speed_jacobian_entries
from mt.sensitivity_search import adaptive_jacobian
class AdjointTests(unittest.TestCase):
 def test_weighted_entries(self):
  space=SimpleNamespace(x0=np.zeros(6),base=SimpleNamespace(reaction_numbers=[1,2],plog_free=[]))
  cfg=dict(cases=[{},dict(kind='ignition')],objectives=dict(speed_weight=2.,temperature_weight=1.,temperature_scale_K=25),branch_weights=dict(flame=1.,ignition=3.))
  matrix=speed_jacobian_entries([dict(speed_log_rate_sensitivities=[2.,3.]),{}],cfg,space,4)
  np.testing.assert_allclose(matrix[0,:2],np.array([2.,3.])*np.sqrt(2/3/4))
  self.assertTrue(np.isnan(matrix[1:]).all());self.assertTrue(np.isnan(matrix[0,2:]).all())
  space.base.plog_free=[2]
  self.assertTrue(np.isnan(speed_jacobian_entries([dict(speed_log_rate_sensitivities=[2.,3.]),{}],cfg,space,4)[0,1]))
 def test_partial_jacobian_used(self):
  def evaluate(points):return [np.array([np.sin(1000*p[0]),3*p[0]]) for p in points]
  J,d=adaptive_jacobian(evaluate,np.zeros(1),np.zeros(2),np.array([-1.]),np.array([1.]),[0],np.array([.01]),.3,dict(jacobian_entries=lambda x:np.array([[1000.],[np.nan]])))
  np.testing.assert_allclose(J[:,0],[1000.,3.]);self.assertTrue(d[0]['stable'])
