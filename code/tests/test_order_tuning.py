"""Reaction-order anchoring, constraints, exported kinetics and solver integration."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import cantera as ct
import yaml
from mt.staged_parameters import StagedParameters
from mt.order_tuning import combine_anchors, rate_penalty_residual
from mt.flame_tuning import flame_residual
from mt.flame_adjoint import full_flame_entries
from mt.sensitivity_search import sensitivity_trust_region

ROOT=Path(__file__).resolve().parents[2]
CASE=ROOT/'example/hydrogen_LFS'
if not CASE.is_dir():CASE=ROOT/'hydrogen_LFS'
MODEL=CASE/'inputs/H2_1_step_stoi.yaml'
REF=CASE/'inputs/FFCM2H2.yaml'
ANCHOR=dict(temperature_K=1200.,concentrations_kmol_m3={'H2':.01,'O2':.02})


def space(**kw):
    settings=dict(uncertainty_factors={},unbounded_reactions=[1],stage='AEB',
                  a_ratio=(1e-6,1e8),stage_a_ratio=(1e-6,1e8),
                  orders={1:{'bounds':{'H2':[-2,2],'O2':[-2,2]}}},order_anchor=ANCHOR)
    settings.update(kw)
    return StagedParameters(MODEL,REF,**settings)


class OrderTuningTests(unittest.TestCase):
    def test_anchor_rate_preserved_and_off_anchor_response(self):
        s=space();x=s.x0.copy()
        x[s.order_indices[1,'H2']]=.5
        x[s.order_indices[1,'O2']]=-.3
        x[s.ea_index[1]]=.001
        x[s.b_index[1]]=.2
        old,new=s.make_gas(),s.make_gas(x)
        old.TPX=new.TPX=1200.,ct.gas_constant*1200.*.03,{'H2':1,'O2':2}
        self.assertAlmostEqual(new.forward_rates_of_progress[0]/old.forward_rates_of_progress[0],1.,places=11)
        old.TPX=new.TPX=1800.,ct.gas_constant*1800.*.06,{'H2':1,'O2':2}
        expected=(1800/1200)**.04*np.exp(-s.energy_si*.001/ct.gas_constant*(1/1800-1/1200))*2**(.1-.06)
        self.assertAlmostEqual(new.forward_rates_of_progress[0]/old.forward_rates_of_progress[0],expected,places=11)

    def test_export_and_native_parity_with_negative_order(self):
        s=space();x=s.x0.copy();x[s.order_indices[1,'H2']]=-6.
        native=s.make_gas();native.TPX=1600.,2e6,{'H2':1,'O2':2,'AR':3}
        native=s.apply(native,x)
        with tempfile.TemporaryDirectory() as tmp:
            loaded=ct.Solution(str(s.export(Path(tmp)/'test.yaml',x)))
            loaded.state=native.state
            np.testing.assert_allclose(native.forward_rates_of_progress,loaded.forward_rates_of_progress,rtol=1e-12)
        self.assertAlmostEqual(native.reaction(0).orders['H2'],-.2)
        self.assertEqual(len(s.physical_parameters(x)),len(s.parameter_scales))

    def test_sum_constraint_and_stage_transfer(self):
        opts={1:{'bounds':{'H2':[-2,2],'O2':[-2,2]},'sum_bounds':[1,2]}}
        a=space(stage='A',orders=opts);x=a.x0.copy();x[a.order_indices[1,'H2']]=-.5
        ae=space(stage='AE',orders=opts,center=x)
        np.testing.assert_array_equal(ae.x0,x)
        self.assertEqual(ae.lower[ae.b_index[1]],ae.upper[ae.b_index[1]])
        bad=x.copy();bad[ae.order_indices[1,'O2']]=1.
        self.assertFalse(ae.rate_envelope(bad,300,3000)['valid'])
        c,d=ae.linear_monotonicity(300,3000)[0]
        self.assertTrue(np.all(c@x>=d-1e-12));self.assertLess(np.min(c@bad-d),0.)

    def test_saved_start_does_not_rebase_source(self):
        s=space(b_delta=9.)
        x=s.encode(CASE/'inputs/start_model.yaml')
        self.assertEqual(s.base._source['reactions'][0]['rate-constant']['Ea'],15000.)
        self.assertAlmostEqual(s.scheme(x)['reactions'][0]['rate-constant']['Ea'],49666.13474338099)
        np.testing.assert_allclose(s.scheme(x)['reactions'][0]['orders']['H2'],1.2942258989987447)

    def test_other_reaction_orders_unchanged(self):
        data=yaml.safe_load(MODEL.read_text());data['reactions'][0]['duplicate']=True
        data['reactions'].append(deepcopy(data['reactions'][0]))
        s=StagedParameters(data,REF,uncertainty_factors={},unbounded_reactions=[1,2],
            orders={1:{'bounds':{'H2':[-2,2],'O2':[-2,2]}}},order_anchor=ANCHOR,stage='AEB')
        self.assertEqual(len(s.x0),3*2+2)
        x=s.x0.copy();x[s.order_indices[1,'H2']]=.5
        self.assertEqual(s.scheme(x)['reactions'][1]['orders'],data['reactions'][1]['orders'])

    def test_reference_anchor_family_weighting(self):
        def row(t,h):return dict(order_anchor=dict(temperature_K=t,concentrations_kmol_m3={'H2':h}))
        anchor=combine_anchors([row(1000,1),row(1000,1),row(2000,4)],[{}, {}, {'kind':'ignition'}])
        self.assertAlmostEqual(anchor['temperature_K'],1/(.5/1000+.5/2000))
        self.assertAlmostEqual(anchor['concentrations_kmol_m3']['H2'],2.)

    def test_hydrogen_loss_and_original_penalty(self):
        ref=[dict(speed_m_s=1.,burnt_temperature_K=1000.,flame_thickness_m=1.)]
        row=[dict(speed_m_s=np.e,burnt_temperature_K=1025.,flame_thickness_m=np.e)]
        r=flame_residual(row,ref,temperature_scale_K=25.,thickness_weight=1.)
        self.assertAlmostEqual(r@r,1.)
        s=space();x=s.x0.copy();x[0]=np.log(10)
        r=rate_penalty_residual(s,x,{'weight':.01})
        self.assertAlmostEqual(r@r,.01)

    def test_adjoint_does_not_supply_order_columns(self):
        s=space();rows=[dict(speed_log_rate_sensitivities=[2.],flame_parameter_adjoints=[[2.,3.,4.],[5.,6.,7.]])]
        cfg=dict(cases=[{}],objectives=dict(speed_weight=1.,temperature_weight=1.,temperature_scale_K=25.,thickness_weight=1.))
        jac=full_flame_entries(rows,cfg,s,3)
        self.assertTrue(np.isnan(jac[:,list(s.order_indices.values())]).all())
        self.assertAlmostEqual(jac[0,0],2/np.sqrt(3))
        self.assertTrue(np.isnan(jac[2]).all())

    def test_joint_update_can_fit_rate_and_orders(self):
        s=space(stage='A')
        # Independent concentration states make all three active directions identifiable.
        c=np.array([[.01,.02],[.02,.02],[.01,.04],[.03,.01]])
        matrix=np.zeros((4,len(s.x0)));matrix[:,0]=1
        for k,sp in enumerate(('H2','O2')):
            matrix[:,s.order_indices[1,sp]]=.2*np.log(c[:,k]/ANCHOR['concentrations_kmol_m3'][sp])
        target=s.x0.copy();target[0]=.1;target[3]=.5;target[4]=-.5
        evaluate=lambda points:[matrix@(x-target) for x in points]
        result=sensitivity_trust_region(evaluate,s.x0,s.lower,s.upper,[0,3,4],s.sensitivity_steps(),
            max_iterations=10,constraints=lambda:s.linear_monotonicity(300,3000),
            feasible=lambda x:s.rate_envelope(x,300,3000)['valid'],
            options={'step_scales':s.trust_step_scales()})
        self.assertLess(result.residual@result.residual,1e-9)
        self.assertGreater(abs(result.x[3]),.1);self.assertGreater(abs(result.x[4]),.1)

    def test_plog_transfer_preserves_species_coordinates(self):
        from mt.initialization import add_plog
        from mt.plog_adaptation import transfer_coordinates,check_rate_equivalence
        old=space();x=old.x0.copy();x[3]=.4;x[4]=-.2
        with tempfile.TemporaryDirectory() as tmp:
            path=add_plog(MODEL,Path(tmp)/'plog.yaml',{1:[1,50,100]})
            new=StagedParameters(path,REF,uncertainty_factors={},unbounded_reactions=[1],
                orders=old.orders,order_anchor=ANCHOR,stage='AEB',a_ratio=(1e-6,1e8),stage_a_ratio=(1e-6,1e8),plog_free=[1])
            y=transfer_coordinates(old,new,x)
            self.assertTrue(check_rate_equivalence(old,new,x,y)['valid'])
            self.assertEqual(new.scheme(y)['reactions'][0]['orders'],old.scheme(x)['reactions'][0]['orders'])

if __name__=='__main__':unittest.main()
