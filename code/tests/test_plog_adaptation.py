"""Fast PLOG registry tests on the example mechanism; no ZND integration."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import yaml
from mt.plog_adaptation import rank_plog_candidates, transfer_coordinates, check_rate_equivalence
from mt.staged_parameters import StagedParameters
from mt.initialization import add_plog

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE=ROOT.parent/'example/ethylene_ZND'
MODEL = EXAMPLE/'inputs/model_populated.yaml'
REFERENCE = EXAMPLE/'inputs/FFCM2.yaml'
FACTORS = {4:1.15, 5:1.6, 6:1.6, 7:1.2, 8:1.4, 9:3., 10:3., 11:5.,
           12:3.2, 13:1.5, 14:3., 15:10., 16:1.2, 17:1.5}


class AdaptationTests(unittest.TestCase):
    def test_rank_requires_resolved_artificial_arrhenius(self):
        scheme = yaml.safe_load(MODEL.read_text())
        screen = {'ranking': [dict(group=n, score=score, reliable_coordinates=[n-1], all_resolved=resolved)
                              for n,score,resolved in [(4, 20., True), (2, 10., True), (1, 9., False), (3, .1, True)]]}
        self.assertEqual(rank_plog_candidates(screen, scheme, [1,2,3], top=3), [2])
        self.assertEqual(rank_plog_candidates(screen, scheme, [1,2,3], top=3, min_relative_score=0), [2,3])
        scheme['reactions'][1]['borrowed'] = True
        self.assertEqual(rank_plog_candidates(screen, scheme, [1,2,3]), [])
        scheme['reactions'][1].pop('borrowed')
        scheme['reactions'][1]['type'] = 'three-body'
        self.assertEqual(rank_plog_candidates(screen, scheme, [1,2,3]), [])
        with self.assertRaises(ValueError):
            rank_plog_candidates(screen, scheme, [True])

    def test_actual_mapping_preserves_rates_and_shared_ea_b(self):
        options = dict(uncertainty_factors=FACTORS, unbounded_reactions=[1,2,3],
                       a_ratio=(.01,100.), stage_a_ratio=(.01,100.), stage='AEB')
        old = StagedParameters(MODEL, REFERENCE, **options)
        x = old.x0.copy()
        # Perturb the accepted artificial rate in all physical parameter families.
        x[old.base.reaction_numbers.index(2)] += .1
        x[old.ea_index[2]] = .001
        x[old.b_index[2]] = .005
        source_before = MODEL.read_bytes()
        with TemporaryDirectory() as folder:
            expanded = add_plog(MODEL, Path(folder)/'expanded.yaml', {2:[1.,50.,100.]})
            new = StagedParameters(expanded, REFERENCE, plog_free=[2], **options)
            mapped = transfer_coordinates(old,new,x)
            self.assertEqual(len(mapped),len(x)+2)
            for index,label in enumerate(new.coordinate_labels):
                if label['reaction'] == 2 and label['parameter'] == 'log_k_pivot':
                    self.assertEqual(mapped[index],x[old.base.reaction_numbers.index(2)])
            self.assertEqual(mapped[new.ea_index[2]],x[old.ea_index[2]])
            self.assertEqual(mapped[new.b_index[2]],x[old.b_index[2]])
            check = check_rate_equivalence(old,new,x,mapped, pressures=(5e4,1e5,1e6,5e6,7e6,1e7,2e7))
            self.assertTrue(check['valid'], check)
            changed = mapped.copy()
            changed[new.base.reaction_numbers.index(2)] += .01
            self.assertFalse(check_rate_equivalence(old,new,x,changed)['valid'])
            new.pivot += 1
            with self.assertRaises(ValueError):
                transfer_coordinates(old,new,x)
        self.assertEqual(source_before,MODEL.read_bytes())


if __name__ == '__main__':
    unittest.main()
