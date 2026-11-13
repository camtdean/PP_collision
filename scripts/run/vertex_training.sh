#!/bin/bash
# Full vertex training of both heads (hybrid, pure) on gpuserver0, in the background, with the track finder's
# settings (scripts/configs/mamba_tracking.yaml) on the live fp32 backbone.
#
#   scripts/run/vertex_training.sh             start
#   scripts/run/vertex_training.sh --resume    continue each head from its last checkpoint
#   further options go to train_vertex_reconstruction.py (the later one wins), e.g. --max_epochs 45
#
# Settings, each overridable from the environment:
#   CHECKPOINT_DIR  logs, checkpoints, the run's output (.out) and process id (.pid)
#   EVENTS          training events
#   BATCH           training batch size
#   VAL_EVENTS      validation events per epoch
#   SEED            random seed
#   HYBRID_MAX_LR, HYBRID_MIN_LR   hybrid's learning rates (pure: the config's)
set -euo pipefail

CHECKPOINT_DIR=${CHECKPOINT_DIR:-/home/bmartin/checkpoint}
EVENTS=${EVENTS:-70000}
BATCH=${BATCH:-16}
VAL_EVENTS=${VAL_EVENTS:-2000}
SEED=${SEED:-42}
HYBRID_MAX_LR=${HYBRID_MAX_LR:-1e-3}
HYBRID_MIN_LR=${HYBRID_MIN_LR:-1e-4}

out="$CHECKPOINT_DIR/vertex_full_d$EVENTS.out"
pid="$CHECKPOINT_DIR/vertex_full_d$EVENTS.pid"

if [ -f "$pid" ] && kill -0 "$(cat "$pid")" 2>/dev/null; then
    echo "already running (process $(cat "$pid")): kill it first" >&2
    exit 1
fi
mkdir -p "$CHECKPOINT_DIR"
# a fresh start overwrites the output file, --resume appends to it
[[ " $* " == *" --resume "* ]] || : > "$out"

cd "$(dirname "$0")/../../train/downstream"
nohup python -u train_vertex_reconstruction.py \
    --run_config ../../scripts/configs/vertex_m5_gpuserver0.yaml \
    --checkpoint_dir "$CHECKPOINT_DIR" --vertex_method hybrid pure --eventnumber "$EVENTS" \
    --train_batch_size "$BATCH" --seed "$SEED" --val_events "$VAL_EVENTS" \
    --hybrid_max_lr "$HYBRID_MAX_LR" --hybrid_min_lr "$HYBRID_MIN_LR" "$@" >> "$out" 2>&1 &
echo $! > "$pid"
echo "started process $(cat "$pid"); output: $out"
echo "stop: kill \$(cat $pid)"
