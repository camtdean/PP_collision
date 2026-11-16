#!/usr/bin/env bash
# TPCpp-10M-like events on the sPHENIX software (ana.435, macros 661f781): Pythia8 Detroit minimum-bias pp at
# 200 GeV, full sPHENIX Geant4 simulation and reconstruction (PHSimpleVertexFinder), SpacepointDumper's ROOT file.
#   server/generate.sh <work dir> [events (1000)] [output (<work dir>/spacepoints.root)]
# Self-contained: this folder (scripts/sphenix/server) is all the sPHENIX server needs. On the host it checks out the
# macros into <work dir>/macros and patches the macro; ana.435 is an SL7 build (gcc-12.1.0), so it then builds
# SpacepointDumper into <work dir>/install and runs inside the SL7 container (SL7_IMAGE, default below). Then, where
# uproot and mmap_ninja are installed:
#   python scripts/sphenix/spacepoints_to_dataset.py --in <output> --out <data_root> --split test
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mkdir -p "$1" && cd "$1" && pwd)
EVENTS=${2:-1000}
OUTPUT=$(realpath -m "${3:-$WORK/spacepoints.root}")
MACROS_COMMIT=661f781db23352a3fa72055ce7dbf1a0ee1c2167
SL7_IMAGE=${SL7_IMAGE:-/cvmfs/sphenix.sdcc.bnl.gov/singularity/rhic_sl7.sif}
ANA_SETUP=/cvmfs/sphenix.sdcc.bnl.gov/gcc-12.1.0/opt/sphenix/core/bin

if [[ ${TPCPP_IN_SL7:-} != 1 ]]; then
    # host: macros at the pinned commit, patched macro in run/ (its side outputs go beside its plain-name outputFile)
    [[ -d $WORK/macros ]] || git clone -q https://github.com/sPHENIX-Collaboration/macros.git "$WORK/macros"
    git -C "$WORK/macros" checkout -q "$MACROS_COMMIT"
    mkdir -p "$WORK/run"
    python3 "$HERE/patch_macro.py" "$WORK/macros/detectors/sPHENIX/Fun4All_G4_sPHENIX.C" "$WORK/run/Fun4All_TPCpp.C"
    [[ -f $SL7_IMAGE ]] || { echo "no SL7 image at $SL7_IMAGE (set SL7_IMAGE; see ls /cvmfs/sphenix.sdcc.bnl.gov/singularity)" >&2; exit 1; }
    binds=$(printf '%s\n' /cvmfs "$WORK" "$HERE" "$(dirname "$OUTPUT")" | sort -u | paste -sd, -)
    exec env TPCPP_IN_SL7=1 singularity exec --bind "$binds" "$SL7_IMAGE" bash "$HERE/generate.sh" "$WORK" "$EVENTS" "$OUTPUT"
fi

# SL7 container: ana.435, SpacepointDumper (once), the run
set +u
source "$ANA_SETUP/sphenix_setup.sh" -n ana.435
set -u
[[ $OFFLINE_MAIN == *ana.435* ]] || { echo "OFFLINE_MAIN=$OFFLINE_MAIN is not ana.435" >&2; exit 1; }
if [[ $(cat "$WORK/install/built_with" 2>/dev/null) != "$OFFLINE_MAIN" ]]; then     # (re)build for this release
    rm -rf "$WORK/build" "$WORK/install"
    mkdir -p "$WORK/build"
    (cd "$WORK/build" && sh "$HERE/SpacepointDumper/autogen.sh" --prefix="$WORK/install" && make -j4 install)
    echo "$OFFLINE_MAIN" > "$WORK/install/built_with"
fi
set +u
source "$ANA_SETUP/setup_local.sh" "$WORK/install"
set -u
export ROOT_INCLUDE_PATH=$WORK/macros/detectors/sPHENIX:$WORK/macros/common:${ROOT_INCLUDE_PATH:-}

cd "$WORK/run"
root -b -q "Fun4All_TPCpp.C($EVENTS, \"\", \"spacepoints.root\")"   # (nEvents, inputFile (unused), outputFile)
[[ $OUTPUT == "$WORK/run/spacepoints.root" ]] || mv spacepoints.root "$OUTPUT"
echo "wrote $OUTPUT"
