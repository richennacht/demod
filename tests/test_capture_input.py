import io
import sys
import unittest
import wave
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from capture_input import decode_capture, analytic_signal, capture_provenance
from rate_estimation import wav_metadata


def pcm_wav(values, fs=16000, width=2):
    a=np.asarray(values)
    if a.ndim==1: a=a[:,None]
    buf=io.BytesIO()
    with wave.open(buf,'wb') as w:
        w.setnchannels(a.shape[1]);w.setsampwidth(width);w.setframerate(fs)
        w.writeframes(a.astype('<i2').tobytes())
    return buf.getvalue()


class WavInputTests(unittest.TestCase):
    def test_stereo_iq_and_qi_preserve_samples_and_header_source(self):
        raw=pcm_wav([[16384,-8192],[-16384,8192]])
        iq,fs,meta=decode_capture(raw,'s8',0,'stereo_iq')
        qi,_,_=decode_capture(raw,'s16le',16000,'stereo_qi')
        self.assertEqual(iq,[.5-.25j,-.5+.25j]);self.assertEqual(qi,[-.25+.5j,.25-.5j])
        self.assertEqual(fs,16000)
        p=capture_provenance(raw,'s8',fs,1000000,None,'capture_log',meta)
        import hashlib
        self.assertEqual(p['sha256'],hashlib.sha256(raw).hexdigest())
        self.assertEqual(p['capture']['sample_rate_source'],'wav_header')
        self.assertEqual(p['capture']['centre_frequency_source'],'capture_log')

    def test_mono_if_analytic_signal_and_translation(self):
        n=np.arange(1024); tone=np.cos(2*np.pi*n/16)
        np.testing.assert_allclose(analytic_signal(tone),np.exp(2j*np.pi*n/16),atol=1e-12)
        raw=pcm_wav(np.rint(tone*16384))
        iq,_,meta=decode_capture(raw,'s16le',0,'real_if',1000)
        np.testing.assert_allclose(iq,.5,atol=5e-5)
        self.assertFalse(meta['resampled'])

    def test_no_implicit_channel_role_or_rate_override(self):
        raw=pcm_wav(np.zeros(128))
        for role,fs in [('unspecified',0),('stereo_iq',0),('real_if',8000)]:
            with self.assertRaises(ValueError): decode_capture(raw,'s16le',fs,role)
        with self.assertRaises(ValueError): wav_metadata(raw[:-2])

    def test_pcm24_sign_extension(self):
        buf=io.BytesIO()
        with wave.open(buf,'wb') as w:
            w.setnchannels(1);w.setsampwidth(3);w.setframerate(16000)
            w.writeframes(bytes([0,0,128,255,255,127,0,0,0]))
        meta,a=wav_metadata(buf.getvalue())
        np.testing.assert_allclose(a[:,0],[-1,1-2**-23,0])
        self.assertEqual(meta['bits_per_channel_sample'],24)
