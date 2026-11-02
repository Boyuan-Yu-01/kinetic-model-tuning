"""Profile alignment, coverage, and objective normalization regressions."""
import unittest
import numpy as np
from mt.flame_profile import prepare_profile_reference, profile_residual, profile_settings
from mt.flame_tuning import flame_residual


def row(shift=0.,width=1.,points=1001):
    x=np.linspace(-10,10,points)
    return dict(distance_m=(x+shift).tolist(),temperature_K=(300+2000/(1+np.exp(-x/width))).tolist(),speed_m_s=1.,burnt_temperature_K=2300.)


class ProfileTests(unittest.TestCase):
    def test_translation_and_different_grids(self):
        ref=prepare_profile_reference(row(),{})
        self.assertLess(np.max(np.abs(profile_residual(row(shift=8,points=1401),ref))),1e-4)
        self.assertGreater(np.linalg.norm(profile_residual(row(width=1.3),ref)),1.)

    def test_settings_and_fraction_scale(self):
        ref=prepare_profile_reference(row(),dict(points=200,lower_fraction=.02,upper_fraction=.98,margin_fraction=.05,scale={'reference_temperature_rise_fraction':.01}))
        self.assertEqual(len(ref['grid_m']),200)
        self.assertAlmostEqual(ref['scale_K'],(row()['temperature_K'][-1]-row()['temperature_K'][0])*.01)
        for options in ({'points':1},{'scale':0},{'lower_fraction':.6},{'margin_fraction':-1}):
            with self.assertRaises(ValueError):profile_settings(options)

    def test_no_extrapolation(self):
        ref=prepare_profile_reference(row(),{})
        short=row();short['distance_m']=short['distance_m'][400:600];short['temperature_K']=short['temperature_K'][400:600]
        with self.assertRaises(ValueError):profile_residual(short,ref)

    def test_loss_normalization_and_coverage(self):
        ref=row();ref['profile_reference']=prepare_profile_reference(ref,{})
        candidate=row();candidate['temperature_K']=(np.asarray(candidate['temperature_K'])+25).tolist()
        candidate['burnt_temperature_K']+=25;candidate['speed_m_s']=np.e
        r=flame_residual([candidate],[ref],temperature_scale_K=25,profile={})
        self.assertAlmostEqual(r@r,1.)
        self.assertIsNone(flame_residual([dict(candidate,distance_m=[0,1],temperature_K=[300,2300])],[ref],profile={}))
