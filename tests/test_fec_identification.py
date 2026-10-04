import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fec_identification import (encode, permute, syndromes, hamming_matrix, gf2_rank,
    analyse_bits, validate_bits, candidate_evidence, INTERLEAVERS, MODEL_PATH)
from demodulation import demodulate


class FECTests(unittest.TestCase):
    def test_hamming_independent_known_vectors(self):
        # One-based parity positions 1,2,4; payload positions 3,5,6,7.
        self.assertEqual(encode([1,0,1,1],'hamming7').tolist(),[0,1,1,0,0,1,1])
        self.assertEqual(encode([1,1,0,1],'hamming7').tolist(),[1,0,1,0,1,0,1])
        for code in ('hamming7','hamming15','repetition3','conv_k3_7_5'):
            a=encode(np.random.default_rng(5).integers(0,2,1000),code)
            self.assertEqual(int(syndromes(a,code).sum()),0)

    def test_convolutional_independent_polynomial_vector(self):
        # Impulse [1,0,0,0], generators current xor delay1 xor delay2 and current xor delay2.
        self.assertEqual(encode([1,0,0,0],'conv_k3_7_5').tolist(),[1,1,1,0,1,1,0,0])

    def test_matrix_inverse_and_bit_order(self):
        a=np.arange(840)
        for inter in INTERLEAVERS:
            np.testing.assert_array_equal(permute(permute(a,inter),inter,True),a)
        self.assertEqual(permute(a,'matrix420_r4')[:8].tolist(),[0,105,210,315,1,106,211,316])

    def test_binary_validation_and_abstention(self):
        for bad in ('','012','1 0',[0,2],[.5,1],[[0,1]]):
            with self.assertRaises(ValueError): validate_bits(bad)
        self.assertEqual(analyse_bits('01'*20)['status'],'abstained')
        self.assertEqual(analyse_bits('0'*3360)['status'],'abstained')
        with self.assertRaises(ValueError): analyse_bits('01'*2000,-1)
        with self.assertRaises(ValueError): analyse_bits('01'*2000,4000)

    def test_finite_field_rank(self):
        self.assertEqual(gf2_rank(np.array([[1,1,0],[1,0,1],[0,1,1]])),2)
        self.assertEqual(gf2_rank(hamming_matrix(15)),4)

    def test_codeword_offset_search(self):
        b=encode(np.random.default_rng(74).integers(0,2,2200),'hamming7')[:3360]
        e=candidate_evidence(np.r_[1,0,1,b][:3360])
        candidate=next(c for c in e if c['code']=='hamming7' and c['interleaver']=='none')
        self.assertEqual(candidate['codeword_offset_bits'],3)
        self.assertEqual(candidate['syndrome_violation_rate'],0)

    @unittest.skipUnless(MODEL_PATH.exists(),'trained FEC model required')
    def test_model_and_full_receiver_bits(self):
        rng=np.random.default_rng(917); b=encode(rng.integers(0,2,2200),'hamming7')[:3360]
        r=analyse_bits(b)
        self.assertEqual(r['status'],'candidate_identified')
        self.assertEqual(r['selected_candidate']['code'],'hamming7')
        self.assertEqual(r['selected_candidate']['interleaver'],'none')
        samples=np.repeat(2*b.astype(float)-1,4).astype(complex).tolist()
        d=demodulate(samples,16000,'bpsk',4)
        self.assertEqual(len(d['bits_preview']),512)
        self.assertEqual(d['fec_identification']['bit_count_supplied'],3360)
        self.assertEqual(d['fec_identification']['selected_candidate']['code'],'hamming7')

    def test_model_absence_is_not_fabricated_result(self):
        b=np.random.default_rng(918).integers(0,2,3360)
        r=analyse_bits(b,model_path=ROOT/'data/models/nonexistent-fec-test.npz')
        self.assertEqual(r['status'],'abstained')
        self.assertIn('manual_baseline',r)


if __name__=='__main__': unittest.main()
