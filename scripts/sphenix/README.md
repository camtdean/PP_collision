# TPCpp-10M-like events on the sPHENIX software

The same kind of events as TPCpp-10M, regenerated on the sPHENIX stack, so HybridVertexHead and sPHENIX's production
vertexing (PHSimpleVertexFinder) run on identical events with full truth.

TPCpp-10M (arXiv 2509.05792): Pythia 8.307 Detroit-tune minimum-bias pp at 200 GeV, Geant4 FTFP_BERT_HP, as-built
sPHENIX in 1.4 T, sPHENIX TPC digitization and clustering; build ana.435, macros 661f781, coresoftware b849eba,
calibrations a3e66e6. It stores per TPC cluster (ADC, x, y, z in cm); tracks with fewer than 5 clusters and events
with fewer than 20 are dropped.

```bash
# sPHENIX server (copy scripts/sphenix/server/ there; nothing else from the repository is needed):
# build SpacepointDumper, check out the macros, simulate and reconstruct
server/generate.sh <work dir> 1000                   # -> <work dir>/spacepoints.root
# or over HTCondor: 100 jobs of 10 events, seeds 1..100
server/condor.sh <work dir> 100 10                   # -> <work dir>/out/spacepoints_<k>.root

# where uproot and mmap_ninja are installed: the dataset layout, as eval reads it
python scripts/sphenix/spacepoints_to_dataset.py --in <work dir>/out/spacepoints_*.root --out <data_root> --split test
```

## Pieces

| file | what |
| --- | --- |
| `server/generate.sh` | macros at 661f781, patched macro, SpacepointDumper build, then the run in `<work dir>/run/<output name>`; ana.435 is SL7-only, so the build and run go inside the SL7 container (`SL7_IMAGE`); optional seed (else /dev/urandom) |
| `server/condor.sh` | prepares once, then submits `generate.sh` runs as HTCondor jobs (seed `SEED_BASE` + job, `REQUEST_MEMORY`), logs in `<work dir>/condor` |
| `server/patch_macro.py` | the standard `Fun4All_G4_sPHENIX.C` with Pythia8 (`phpythia8_detroit_minBias.cfg`), the seed from `$TPCPP_SEED`, the pp beam parameters (vertex width 120 um, 120 um, 10 cm) and SpacepointDumper; exact-match edits only |
| `server/SpacepointDumper/` | `SubsysReco`: per event the TPC clusters (ADC, global x y z, layer) with the truth particle of each (most energy: track and parent id, PDG, momentum, energy, charge, production vertex), the truth PV and the `SvtxVertexMap` vertices |
| `spacepoints_to_dataset.py` (not on the server) | TPCpp-10M's cuts, then `features`, `seg_target`, `pid_target`, `reg_target`, `mid_target` (RaggedMmap, as `prepare_data.py`) with real truth, and `vertices_<split>.npz` (truth PV, sPHENIX's vertices) |

## Not pinned by the paper

Vertex distribution (here sPHENIX's pp default), pileup (none), `TRACKING::pp_mode` (macro default, off), field map
(macro default). Compare the regenerated events with TPCpp-10M (ADC, r, clusters per track and per event, truth PV
spread) before using them.

The full simulation includes the silicon trackers, so PHSimpleVertexFinder uses silicon + TPC tracks while
HybridVertexHead sees the TPC only.
