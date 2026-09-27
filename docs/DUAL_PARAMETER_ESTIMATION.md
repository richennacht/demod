# Dual Parameter Estimation

The first DEmod dual system independently estimates DC offset and carrier-frequency offset (CFO):

1. **DSP branch:** I/Q means and a fourth-power phase-increment CFO estimator.
2. **Learned branch:** a small 7-to-12-to-3 MLP trained from ephemeral recipe-generated IQ.
3. **Evidence layer:** returns both estimates and their absolute disagreement; it does not silently choose one.

The model sees only the complex samples. Recipe ID, seed, source type, and realised impairments are retained only as audit metadata. The learned model is intentionally small and dependency-free for the MVP; its role is to establish the dual-estimation contract, not to claim production accuracy.

Run it with:

```powershell
python src/dual_parameter_estimator.py --examples 48
```

The next extensions use the same pattern: DSP versus learned symbol-rate estimation, modulation candidate ranking, and FEC/interleaver candidate ranking. Each needs an independent verifier (constellation likelihood, framing/CRC, or syndrome), because agreement alone does not prove correctness.
