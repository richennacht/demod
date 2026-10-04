#!/usr/bin/env bash
# v2 = robustness-augmented SpecCFO (with absolute-position channels) and DemodAMC.
# Augmentation: more pulse shapes, 1 sample/symbol, interferers, phase noise, impulses, longer captures.
# Resumable like run_all_training.sh. V2_CFO_STEPS / V2_AMC_STEPS / BATCH override the budgets.
set -u
cd "$(dirname "$0")/.."
V2_CFO_STEPS=${V2_CFO_STEPS:-4000}
V2_AMC_STEPS=${V2_AMC_STEPS:-2500}
BATCH=${BATCH:-64}
mkdir -p research/logs research/checkpoints data/models
if [ ! -f research/checkpoints/speccfo_v2.npz ]; then
  python research/train_cfo.py --model speccfo --augment --coords --tag _v2 --steps "$V2_CFO_STEPS" --batch "$BATCH" >> research/logs/train_speccfo_v2.log 2>&1
fi
# copy only when absent: tune_cfo_fallback.py writes the tuned fallback threshold into the shipped copy
[ -f data/models/speccfo_v2.npz ] || cp research/checkpoints/speccfo_v2.npz data/models/speccfo_v2.npz
if [ ! -f research/checkpoints/demod_amc_v2.npz ]; then
  python research/train_amc.py --model demod_amc --augment --tag _v2 --cfo-model data/models/speccfo_v2.npz --steps "$V2_AMC_STEPS" --batch "$BATCH" >> research/logs/train_demod_amc_v2.log 2>&1
fi
echo done > research/logs/v2_done
