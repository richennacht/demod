#!/usr/bin/env bash
# Train every model in order. Safe to re-run: finished checkpoints are skipped and
# interrupted runs resume from their last periodic checkpoint.
#   CFO_STEPS / AMC_STEPS / BATCH override the budgets (GPU runs use much larger values).
set -u
cd "$(dirname "$0")/.."
CFO_STEPS=${CFO_STEPS:-3000}
AMC_STEPS=${AMC_STEPS:-2000}
BATCH=${BATCH:-64}
mkdir -p research/logs research/checkpoints data/models
run() { # script model steps
  if [ -f "research/checkpoints/$2.npz" ] && [ "${FORCE:-0}" != "1" ]; then echo "skip $2 (done)"; return; fi
  python "research/$1" --model "$2" --steps "$3" --batch "$BATCH" >> "research/logs/train_$2.log" 2>&1
}
run train_cfo.py speccfo "$CFO_STEPS"
cp research/checkpoints/speccfo.npz data/models/speccfo.npz
run train_amc.py demod_amc "$AMC_STEPS"
run train_amc.py demod_amc_nocomp "$AMC_STEPS"
run train_cfo.py iq_resnet "$CFO_STEPS"
run train_cfo.py oshea_cfo "$CFO_STEPS"
run train_amc.py vtcnn2 "$AMC_STEPS"
run train_amc.py lstm_ap "$AMC_STEPS"
[ -f research/checkpoints/demod_amc.npz ] && cp research/checkpoints/demod_amc.npz data/models/demod_amc.npz
echo done > research/logs/all_done
