"""Experimental candidate-set FEC/interleaver evidence on hard bits, not decryption.

Known parity checks plus a small trained feature MLP. No arbitrary-code or
arbitrary-permutation reconstruction. Matrix interleaver boundaries are assumed.
"""
from pathlib import Path
import hashlib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / 'data/models/fec_identifier.npz'
MAX_BITS = 8400
MIN_BITS = 1680
PERIOD = 420
CODES = ('repetition3', 'hamming7', 'hamming15', 'conv_k3_7_5')
INTERLEAVERS = ('none', 'matrix420_r4', 'matrix420_r10', 'matrix420_r20')
CANDIDATES = tuple((c, i) for c in CODES for i in INTERLEAVERS)
LABELS = tuple(f'{c}:{i}' for c, i in CANDIDATES) + ('unknown',)


def validate_bits(bits):
    if isinstance(bits, str):
        if not bits or any(c not in '01' for c in bits):
            raise ValueError('Bits must be a nonempty string containing only 0 and 1.')
        return np.frombuffer(bits.encode('ascii'), dtype=np.uint8) - 48
    a = np.asarray(bits)
    if a.ndim != 1 or not len(a) or not np.all((a == 0) | (a == 1)):
        raise ValueError('Bits must be a nonempty one-dimensional binary sequence.')
    return a.astype(np.uint8)


def hamming_matrix(n):
    if n not in (7, 15):
        raise ValueError('Supported Hamming lengths are 7 and 15.')
    return ((np.arange(1, n+1)[None, :] >> np.arange(int(np.log2(n+1)))[:, None]) & 1).astype(np.uint8)


def encode(payload, code):
    """Recipe encoder: parity at one-based positions 1,2,4,8; convolution MSB first."""
    b = validate_bits(payload)
    if code == 'repetition3':
        return np.repeat(b, 3)
    if code in ('hamming7', 'hamming15'):
        n = int(code[7:]); h = hamming_matrix(n)
        data = [j for j in range(n) if (j+1) & j]
        count = len(b)//len(data)
        if not count:
            raise ValueError('Insufficient payload for one codeword.')
        words = np.zeros((count, n), dtype=np.uint8)
        words[:, data] = b[:count*len(data)].reshape(count, len(data))
        syndrome = (words @ h.T) & 1
        for j in range(len(h)):
            words[:, (1 << j)-1] = syndrome[:, j]
        return words.ravel()
    if code == 'conv_k3_7_5':
        prev1 = np.r_[0, b[:-1]]; prev2 = np.r_[0, 0, b[:-2]][:len(b)]
        return np.column_stack((b ^ prev1 ^ prev2, b ^ prev2)).astype(np.uint8).ravel()
    raise ValueError('Unknown code recipe.')


def permute(bits, interleaver, inverse=False):
    if interleaver == 'none':
        return bits.copy()
    if interleaver not in INTERLEAVERS[1:]:
        raise ValueError('Unsupported interleaver hypothesis.')
    rows = int(interleaver.rsplit('r', 1)[1]); cols = PERIOD // rows
    count = len(bits)//PERIOD
    frames = bits[:count*PERIOD].reshape(count, PERIOD)
    if inverse:
        return frames.reshape(count, cols, rows).transpose(0, 2, 1).ravel()
    return frames.reshape(count, rows, cols).transpose(0, 2, 1).ravel()


def syndromes(bits, code, offset=0):
    a = bits[offset:]
    if code == 'conv_k3_7_5':
        y = a[:len(a)//2*2].reshape(-1, 2)
        # (1+D^2)y0 + (1+D+D^2)y1 = 0; first two states excluded.
        return (y[2:, 0] ^ y[:-2, 0] ^ y[2:, 1] ^ y[1:-1, 1] ^ y[:-2, 1])[:, None]
    n = 3 if code == 'repetition3' else int(code[7:])
    w = a[:len(a)//n*n].reshape(-1, n)
    if code == 'repetition3':
        return np.column_stack((w[:, 0] ^ w[:, 1], w[:, 1] ^ w[:, 2]))
    return (w @ hamming_matrix(n).T) & 1


def candidate_evidence(bits):
    out = []
    for code, inter in CANDIDATES:
        a = permute(bits, inter, inverse=True)
        n = 2 if code.startswith('conv') else (3 if code == 'repetition3' else int(code[7:]))
        choices = []
        for off in range(n):
            s = syndromes(a, code, off)
            rate = float(s.mean())
            choices.append((rate, off, s))
        rate, off, s = min(choices, key=lambda v: v[0])
        halves = np.array_split(s, 2)
        out.append({'code': code, 'interleaver': inter, 'codeword_offset_bits': off,
                    'syndrome_violation_rate': rate, 'check_count': int(s.size),
                    'first_half_rate': float(halves[0].mean()), 'second_half_rate': float(halves[1].mean()),
                    'zero_syndrome_word_fraction': float(np.all(s == 0, axis=1).mean())})
    return out


def features(bits, evidence=None):
    e = evidence if evidence is not None else candidate_evidence(bits)
    x = [v for c in e for v in (c['syndrome_violation_rate'], c['first_half_rate'],
                                c['second_half_rate'], c['zero_syndrome_word_fraction'])]
    bipolar = 1.-2.*bits.astype(float)
    x.extend([float(bits.mean()), float((bits[1:] != bits[:-1]).mean())])
    x.extend(float(np.mean(bipolar[:-lag]*bipolar[lag:])) for lag in (1, 2, 3, 7, 15, 20))
    return np.asarray(x)


def gf2_rank(a):
    a = a.copy().astype(np.uint8); rank = 0
    for col in range(a.shape[1]):
        piv = np.flatnonzero(a[rank:, col])
        if not len(piv):
            continue
        p = rank + piv[0]; a[[rank, p]] = a[[p, rank]]
        for row in np.flatnonzero(a[:, col]):
            if row != rank:
                a[row] ^= a[rank]
        rank += 1
        if rank == a.shape[0]:
            break
    return rank


def rank_baseline(bits):
    """Small GF(2) rank-deficiency baseline, not arbitrary parity-matrix recovery."""
    scores = []
    for code, inter in CANDIDATES:
        a = permute(bits, inter, inverse=True)
        n = {'repetition3': 3, 'hamming7': 7, 'hamming15': 15, 'conv_k3_7_5': 16}[code]
        expected_dimension = {'repetition3': 1, 'hamming7': 4, 'hamming15': 11, 'conv_k3_7_5': 10}[code]
        best = 0.
        for off in range(n if not code.startswith('conv') else 2):
            w = a[off:off+(len(a)-off)//n*n].reshape(-1, n)
            # Rank saturates under errors; repeated 2n-row blocks expose deficiency.
            # Normalize by each candidate's redundancy, not n: raw deficiency
            # otherwise unfairly favors the lowest-rate (repetition) candidate.
            vals = [(n-gf2_rank(w[j:j+2*n]))/(n-expected_dimension) for j in range(0, min(len(w)-2*n+1, 8*n), 2*n)]
            if vals:
                best = max(best, float(np.mean(vals)))
        scores.append(best)
    return int(np.argmax(scores)) if max(scores) > .05 else len(CANDIDATES)


def ml_probabilities(x, model):
    a = (x-model['mean'])/model['scale']
    h = np.maximum(a @ model['w1'] + model['b1'], 0)
    z = (h @ model['w2'] + model['b2'])/float(model['temperature'])
    p = np.exp(z-np.max(z, axis=-1, keepdims=True))
    return p/p.sum(axis=-1, keepdims=True)


def analyse_bits(bits, frame_offset_bits=0, model_path=MODEL_PATH):
    original = validate_bits(bits)
    if type(frame_offset_bits) is not int or frame_offset_bits < 0 or frame_offset_bits >= len(original):
        raise ValueError('Frame offset must be an integer within the supplied bitstream.')
    a = original[frame_offset_bits:frame_offset_bits+MAX_BITS]
    common = {'stage': 'experimental_fec_identification', 'bit_count_supplied': len(original),
              'bit_count_analysed': len(a), 'input_bits_sha256': hashlib.sha256(original.tobytes()).hexdigest(),
              'hash_representation': 'one unsigned byte per bit, values 0 and 1',
              'frame_offset_bits': frame_offset_bits, 'frame_boundary_source': 'analyst_hypothesis',
              'frame_alignment_verified': False, 'codeword_offset_reference': 'after candidate inverse permutation',
              'scope': 'Known candidate codes and 420-bit matrix permutations; simulated hard-bit training only.',
              'limitations': ['No FEC decoding, CRC, descrambling or decryption is performed.',
                             'An absent parity match does not prove uncoded or encrypted data.',
                             'Matrix frame alignment is unverified. A class match does not establish the exact transmitted permutation or boundary.',
                             'Receiver phase/mapping errors, puncturing, whitening and bit slips can invalidate hypotheses.']}
    if len(a) < MIN_BITS or np.mean(a) < .1 or np.mean(a) > .9:
        return {**common, 'status': 'abstained', 'reason': 'Need at least 1680 nondegenerate bits after the frame offset.'}
    a = a[:len(a)//PERIOD*PERIOD]; common['bit_count_analysed'] = len(a)
    e = candidate_evidence(a); rates = np.asarray([c['syndrome_violation_rate'] for c in e])
    order = np.argsort(rates); best = e[int(order[0])]
    common['manual_baseline'] = {'method': 'minimum hard parity-check violation rate over candidates and codeword offsets',
                                 'candidate': best, 'score_gap': float(rates[order[1]]-rates[order[0]])}
    common['candidate_evidence'] = sorted(e, key=lambda c: c['syndrome_violation_rate'])
    if not Path(model_path).exists():
        return {**common, 'status': 'abstained', 'reason': 'Learned weights are unavailable; algebraic evidence only.'}
    with np.load(model_path, allow_pickle=False) as m:
        if tuple(m['labels'].tolist()) != LABELS:
            raise ValueError('FEC model candidate schema does not match runtime.')
        p = ml_probabilities(features(a, e), m)
        threshold = float(m['threshold']); digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    winner = int(p.argmax()); confidence = float(p[winner])
    unknown = winner == len(CANDIDATES)
    reason = None
    if unknown:
        reason = 'No supported code/interleaver selected; this is not evidence of encryption.'
    elif confidence < threshold:
        reason = 'Learned candidate probability below the validation-selected acceptance threshold.'
    elif winner != int(order[0]) or rates[winner] > .32 or rates[order[1]]-rates[order[0]] < .015:
        reason = 'Learned and algebraic evidence disagree, weak redundancy, or ambiguous candidate permutation.'
    elif max(e[winner]['first_half_rate'], e[winner]['second_half_rate']) > .36:
        reason = 'Parity evidence is not stable across both halves of the selected bit region.'
    return {**common, 'status': 'abstained' if reason else 'candidate_identified', 'reason': reason,
            'selected_candidate': None if reason else e[winner],
            'learned': {'model': 'FECFeatureMLP-v1', 'model_sha256': digest, 'confidence': confidence,
                        'acceptance_threshold': threshold, 'training_scope': 'simulation only',
                        'ranking': [{'label': LABELS[j], 'probability': float(p[j])} for j in np.argsort(-p)]}}
