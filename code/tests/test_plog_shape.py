"""Independent pressure-node shape, export parity, and interior reversal checks."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import cantera as ct
import numpy as np
from test_plog_adaptation import MODEL, REFERENCE, FACTORS
from mt.staged_parameters import StagedParameters
from mt.plog_adaptation import transfer_coordinates, check_rate_equivalence
from mt.initialization import add_plog
from mt.uncertainty import plog_monotonicity


class ShapeTests(unittest.TestCase):
    def setUp(self):
        self.folder=TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        expanded=add_plog(MODEL,Path(self.folder.name)/'expanded.yaml',{2:[1.,50.,100.]})
        self.options=dict(uncertainty_factors=FACTORS,unbounded_reactions=[1,2,3],
                          a_ratio=(.01,100.),stage_a_ratio=(.01,100.),stage='AEB')
        self.space=StagedParameters(expanded,REFERENCE,plog_free=[2],plog_shape_free=True,**self.options)

    def test_transfer_and_native_export_parity(self):
        old=StagedParameters(MODEL,REFERENCE,**self.options)
        x=old.x0.copy();x[old.ea_index[2]]=.001;x[old.b_index[2]]=.02
        s=self.space;y=transfer_coordinates(old,s,x)
        self.assertTrue(check_rate_equivalence(old,s,x,y)['valid'])
        indices=[i for i,n in enumerate(s.base.reaction_numbers) if n==2]
        y[s.ea_index[s.shape_for_a[indices[1]]]]+=.002
        y[s.b_index[s.shape_for_a[indices[2]]]]+=.03
        gas=s.make_gas(y);native=s.apply(s.make_gas(),y)
        path=s.export(Path(self.folder.name)/'out.yaml',y);loaded=ct.Solution(str(path))
        for t in (400.,1100.,2500.):
            for p in (1e5,1e6,5e6,1e7):
                for g in (gas,native,loaded):g.TP=t,p
                np.testing.assert_allclose(native.forward_rate_constants,gas.forward_rate_constants,rtol=1e-12)
                np.testing.assert_allclose(loaded.forward_rate_constants,gas.forward_rate_constants,rtol=1e-12)

    def test_stage_transfer_retains_accepted_boundary_slack(self):
        s=self.space;x=s.x0.copy()
        i=s.base.reaction_numbers.index(4)
        x[i]=s.a_low[i]-5e-13
        opts=dict(self.options,stage='A',center=x)
        new=StagedParameters(s.base._source,REFERENCE,plog_free=[2],plog_shape_free=True,**opts)
        np.testing.assert_array_equal(new.x0,x)
        self.assertTrue(new.rate_envelope(x,1000.,3000.)['valid'])
        x[i]-=1e-8
        with self.assertRaises(ValueError):
            StagedParameters(s.base._source,REFERENCE,plog_free=[2],plog_shape_free=True,**dict(opts,center=x))

    def test_pressure_cuts_allow_reversed_a_order(self):
        s=self.space;x=s.x0.copy()
        indices=[i for i,n in enumerate(s.base.reaction_numbers) if n==2]
        for i in indices[1:]:
            x[s.ea_index[s.shape_for_a[i]]]=-ct.gas_constant*1000./s.energy_si
            x[i]=-.1+1000./s.pivot
        self.assertLess((s.a_map@x)[indices[1]],(s.a_map@x)[indices[0]])
        self.assertTrue(plog_monotonicity(s.scheme(x),300.,3000.,'inc')['valid'])
        c,d=s._pressure_branches(300.,3000.,'inc')[0]
        self.assertGreaterEqual(np.min(c@x-d),-1e-12)

    def test_interior_reversal_is_cut_and_exactly_rejected(self):
        s=self.space;x=s.x0.copy()
        indices=[i for i,n in enumerate(s.base.reaction_numbers) if n==2]
        # Adjacent log-rate ratio = db*(ln(T/1000)+1000/T-1)-epsilon.
        # Positive endpoints but a negative minimum at T=1000 K.
        db=.1;de=-ct.gas_constant*db*1000.;epsilon=.001
        da=-db*np.log(1000.)-db-epsilon
        for i in indices[1:]:
            x[s.ea_index[s.shape_for_a[i]]]=de/s.energy_si
            x[s.b_index[s.shape_for_a[i]]]=db/s.b_scale
            x[i]=da-de/(ct.gas_constant*s.pivot)+db*np.log(s.pivot)
        report=plog_monotonicity(s.scheme(x),300.,3000.,'inc')
        self.assertFalse(report['valid'])
        pair=next(p for p in report['pairs'] if p['reaction']==2)
        self.assertGreater(pair['log_rate_ratios'][0],0)
        self.assertGreater(pair['log_rate_ratios'][1],0)
        s.x0=x.copy()
        c,d=s._pressure_branches(300.,3000.,'inc')[0]
        self.assertLess(np.min(c@x-d),-.0009)

if __name__=='__main__':unittest.main()
