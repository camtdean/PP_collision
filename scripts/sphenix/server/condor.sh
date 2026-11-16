#!/usr/bin/env bash
# generate.sh split over HTCondor jobs: prepares once here (macros, patched macro, SpacepointDumper build), then
# submits <jobs> runs of <events per job>, job k with seed SEED_BASE + k, writing <work dir>/out/spacepoints_<k>.root.
#   server/condor.sh <work dir> [jobs (100)] [events per job (10)]
# Env: SEED_BASE (1), REQUEST_MEMORY (4GB), SL7_IMAGE (as generate.sh). Logs in <work dir>/condor. Then:
#   python scripts/sphenix/spacepoints_to_dataset.py --in <work dir>/out/spacepoints_*.root --out <data_root> --split test
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mkdir -p "$1" && cd "$1" && pwd)
JOBS=${2:-100}
PER_JOB=${3:-10}

TPCPP_STEP=prepare bash "$HERE/generate.sh" "$WORK"
mkdir -p "$WORK/condor" "$WORK/out"
cat > "$WORK/condor/tpcpp.sub" <<EOF
universe       = vanilla
executable     = /bin/bash
seed           = \$(Process) + ${SEED_BASE:-1}
arguments      = $HERE/generate.sh $WORK $PER_JOB $WORK/out/spacepoints_\$(Process).root \$INT(seed)
environment    = "TPCPP_STEP=run SL7_IMAGE=${SL7_IMAGE:-/cvmfs/sphenix.sdcc.bnl.gov/singularity/rhic_sl7.sif}"
getenv         = false
request_memory = ${REQUEST_MEMORY:-4GB}
output         = $WORK/condor/\$(Process).out
error          = $WORK/condor/\$(Process).err
log            = $WORK/condor/tpcpp.log
queue $JOBS
EOF
condor_submit "$WORK/condor/tpcpp.sub"
echo "watch: condor_q; progress: tail $WORK/condor/<k>.out"
