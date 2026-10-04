import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'research'))
from library_receiver import receive,libraries,constellation,SUPPORTED
from eval_receivers import fixture


class LibraryReceiverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: libraries()
        except ValueError as e: raise unittest.SkipTest(str(e))

    def test_independent_known_labels(self):
        points,labels=constellation('16qam')
        np.testing.assert_array_equal(labels[[0,1,2,3,4,15]],[[0,0,0,0],[0,0,0,1],[0,0,1,1],[0,0,1,0],[0,1,0,0],[1,0,1,0]])
        np.testing.assert_array_equal(receive([-1,-1,1,1]*4,1000,'bpsk',2)['bits'],[0,1]*4)

    def test_controlled_all_seven_modes(self):
        for mod in SUPPORTED:
            with self.subTest(mod=mod):
                x,t,labels,tones=fixture(np.random.default_rng(44),mod,n=512,snr=40)
                r=receive(x,250000,mod,8,3,750.,phase_radians=.17,fsk_tones_hz=tones)
                np.testing.assert_array_equal(np.asarray(r['bits']).reshape(-1,labels.shape[1])[12:-12],labels[t][12:-12])
                if tones is not None:
                    inferred=receive(x,250000,mod,8,3,750.)
                    np.testing.assert_array_equal(np.asarray(inferred['bits']).reshape(-1,labels.shape[1])[12:-12],labels[t][12:-12])

    def test_static_search_and_rrc(self):
        x,t,labels,_=fixture(np.random.default_rng(15),'16qam',pulse='rrc',snr=40)
        r=receive(x,250000,'16qam',8,None,750.,pulse='rrc',phase_radians=None)
        self.assertEqual(r['configuration']['timing_offset_samples'],3)
        np.testing.assert_array_equal(np.asarray(r['bits']).reshape(-1,4)[12:-12],labels[t][12:-12])

    def test_phase_ambiguity_is_not_claimed_resolved(self):
        x,t,labels,_=fixture(np.random.default_rng(15),'qpsk',snr=40,phase=.17+np.pi/2)
        r=receive(x,250000,'qpsk',8,3,750.,phase_radians=None)
        self.assertIn('ambiguity unresolved',r['configuration']['phase_source'])
        self.assertGreater(np.mean(np.asarray(r['bits']).reshape(-1,2)[12:-12]!=labels[t][12:-12]),.1)

    def test_rejects_invalid_inputs(self):
        for kwargs in ({'sps':0},{'sps':2,'timing':3},{'sps':2,'fs':0},{'sps':2,'pulse':'gauss'},
                       {'sps':2,'phase_radians':float('nan')}):
            args=dict(samples=[-1.,1.]*10,fs=1000,modulation='bpsk',sps=2);args.update(kwargs)
            with self.assertRaises(ValueError): receive(**args)

    def test_api_qam_and_tone_validation(self):
        from local_comparison_api import ComparisonService
        x,t,labels,_=fixture(np.random.default_rng(11),'16qam',n=64,snr=40,cfo=0.,phase=0.,offset=0)
        service=ComparisonService(ROOT/'data/recipes/mvp-recipes.json',examples=2)
        report=service.demodulate_bytes(x.astype('<c8').tobytes(),'f32le',250000,'16qam',8,backend='komm_scipy')
        self.assertEqual(report['bit_count'],256)
        with self.assertRaises(ValueError): receive(x,250000,'4fsk',8,fsk_tones_hz=[1,2])


if __name__=='__main__': unittest.main()
