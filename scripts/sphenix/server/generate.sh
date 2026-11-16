#!/usr/bin/env bash
# TPCpp-10M-like events on the sPHENIX software (ana.435, macros 661f781): Pythia8 Detroit minimum-bias pp at
# 200 GeV, full sPHENIX Geant4 simulation and reconstruction (PHSimpleVertexFinder), SpacepointDumper's ROOT file.
#   server/generate.sh <work dir> [events (1000)] [output (<work dir>/spacepoints.root)]
# Self-contained: this folder (scripts/sphenix/server) is all the sPHENIX server needs. Builds SpacepointDumper into
# <work dir>/install and checks out the macros into <work dir>/macros (once each). Then, where uproot and mmap_ninja
# are installed: python scripts/sphenix/spacepoints_to_dataset.py --in <output> --out <data_root> --split test
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mkdir -p "$1" && cd "$1" && pwd)
EVENTS=${2:-1000}
OUTPUT=$(realpath -m "${3:-$WORK/spacepoints.root}")
MACROS_COMMIT=661f781db23352a3fa72055ce7dbf1a0ee1c2167

source /opt/sphenix/core/bin/sphenix_setup.sh -n ana.435
set -euo pipefail

if [[ ! -f $WORK/install/lib/libspacepointdumper.so ]]; then
    mkdir -p "$WORK/build"
    (cd "$WORK/build" && "$HERE/SpacepointDumper/autogen.sh" --prefix="$WORK/install" && make -j4 install)
fi
set +u
source /opt/sphenix/core/bin/setup_local.sh "$WORK/install"
set -u

if [[ ! -d $WORK/macros ]]; then
    git clone -q https://github.com/sPHENIX-Collaboration/macros.git "$WORK/macros"
fi
git -C "$WORK/macros" checkout -q "$MACROS_COMMIT"
export ROOT_INCLUDE_PATH=$WORK/macros/detectors/sPHENIX:$WORK/macros/common:${ROOT_INCLUDE_PATH:-}

# the macro's side outputs (QA, evaluators) go beside its outputFile, which must be a plain file name: run in run/
mkdir -p "$WORK/run"
cd "$WORK/run"
python3 "$HERE/patch_macro.py" "$WORK/macros/detectors/sPHENIX/Fun4All_G4_sPHENIX.C" Fun4All_TPCpp.C
root -b -q "Fun4All_TPCpp.C($EVENTS, \"\", \"spacepoints.root\")"   # (nEvents, inputFile (unused), outputFile)
[[ $OUTPUT == "$WORK/run/spacepoints.root" ]] || mv spacepoints.root "$OUTPUT"
echo "wrote $OUTPUT"
