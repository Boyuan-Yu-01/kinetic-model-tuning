import unittest
from mt.plog_settings import resolve_plog_settings

class PlogSettingsTests(unittest.TestCase):
    model={'reactions':[{'type':'elementary'},{'type':'pressure-dependent-Arrhenius'},{'type':'pressure-dependent-Arrhenius'}]}
    def test_boolean_scope(self):
        self.assertEqual(resolve_plog_settings({'enabled':False,'plog_free':True},self.model)['plog_free'],[2,3])
        self.assertIsNone(resolve_plog_settings({'enabled':False,'plog_free':False},self.model)['plog_free'])
    def test_insertion_and_defaults(self):
        c=resolve_plog_settings({'enabled':True,'plog_free':True,'reactions':[1]},self.model)
        self.assertEqual(c['plog_candidates'],[1]);self.assertEqual(c['plog_trial_rounds'],5)
        with self.assertRaises(ValueError):resolve_plog_settings({'enabled':True,'plog_free':False},self.model)
    def test_legacy(self):
        old={'enable_plog':True,'plog_free':[2]}
        self.assertEqual(resolve_plog_settings(old,self.model),old)
