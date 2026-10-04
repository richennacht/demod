import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from validate_stages import accuracy, paired


class ValidationMetricTests(unittest.TestCase):
    def test_abstention_does_not_inflate_closed_accuracy(self):
        r=accuracy([0,0,1,1],[0,1,1,0],[True,False,True,False])
        self.assertEqual(r['accuracy'],.5)
        self.assertEqual(r['accepted_accuracy'],1.)
        self.assertEqual(r['coverage'],.5)
        self.assertLess(r['accepted_wilson95'][0],1.)

    def test_paired_delta_and_direction(self):
        r=paired([0,0,1,1],[0,0,1,1],[0,1,1,0])
        self.assertEqual(r['delta_percentage_points'],50.)
        self.assertEqual(r['proposed_only_correct'],2)
        self.assertEqual(r['baseline_only_correct'],0)
        self.assertEqual(paired([0,1],[0,1],[0,1])['delta_bootstrap95_pp'],[0.,0.])

    def test_confusion_accounts_for_unknown_prediction(self):
        r=accuracy(['code','code'],['code','unknown'])
        self.assertEqual(np.asarray(r['confusion']).sum(),2)
        self.assertAlmostEqual(r['macro_f1'],2/3)

    def test_no_accepted_examples(self):
        r=accuracy([0,1],[0,0],[False,False])
        self.assertIsNone(r['accepted_accuracy'])
        self.assertIsNone(r['accepted_wilson95'])


if __name__=='__main__': unittest.main()
