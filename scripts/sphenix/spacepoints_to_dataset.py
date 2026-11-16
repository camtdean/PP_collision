#!/usr/bin/env python3
"""
spacepoints_to_dataset.py: SpacepointDumper's ROOT files (sPHENIX simulation, TPC clusters with truth) to the
RaggedMmap layout TPCBatchDataset opens (as scripts/prepare_data.py writes from the Zenodo release), with the real
truth that release lacks:

    python scripts/sphenix/spacepoints_to_dataset.py --in <dump.root> [<dump.root> ...] --out <data_root> \
        --split pretrain|test [--min_track_points 5] [--min_event_points 20] [--keep_unmatched]

Writes under --out, per event (TPCpp-10M's cuts: clusters of truth tracks with fewer than --min_track_points dropped,
events with fewer than --min_event_points rejected):
- features_<split>      (n, 4) float32  E (cluster ADC), x, y, z (cm)
- seg_target_<split>    (n,)   int64    truth track id (Geant4: primaries > 0, secondaries < 0; 0: no truth match,
                                          with --keep_unmatched)
- pid_target_<split>    (n,)   int64    PDG code (downstream_util.get_pidlabel maps it to a class)
- reg_target_<split>    (n, 8) float32  px, py, pz (GeV), vtx_x, vtx_y, vtx_z (production vertex, cm), q, e (GeV)
- mid_target_<split>    (n,)   int64    parent track id (0: primary)
- vertices_<split>.npz  per event: file (index in --in), event (within the file), truth_pv (E, 3) cm, and sPHENIX's
  reconstructed vertices (reco_n, and reco_x/y/z, reco_ntracks, reco_cov (V, 6: xx xy yy xz yz zz) flattened over events in event order)

Needs uproot and mmap_ninja.
"""
import argparse
import os

import numpy as np

POINT = ("adc", "x", "y", "z", "track_id", "parent_id", "pid", "px", "py", "pz", "vtx_x", "vtx_y", "vtx_z",
         "charge", "e")
EVENT = ("event", "truth_pv_x", "truth_pv_y", "truth_pv_z", "reco_x", "reco_y", "reco_z", "reco_ntracks", "reco_cov")


def events(paths):
    """(input file index, each dumped event as a dict of numpy arrays (SpacepointDumper's tree 'spacepoints'))."""
    import uproot
    for i, path in enumerate(paths):
        with uproot.open(path) as f:
            for chunk in f["spacepoints"].iterate(POINT + EVENT, library="np", step_size=1000):
                for k in range(len(chunk["event"])):
                    yield i, {name: np.asarray(chunk[name][k]) for name in POINT + EVENT}


def select(ev, min_track_points, keep_unmatched):
    """The event's kept clusters: matched clusters on truth tracks with >= min_track_points clusters (and unmatched
    ones with keep_unmatched)."""
    ids = ev["track_id"]
    matched = ids != 0
    unique, counts = np.unique(ids[matched], return_counts=True)
    long_enough = np.isin(ids, unique[counts >= min_track_points])
    return long_enough | (keep_unmatched & ~matched)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="inputs", nargs="+", required=True, help="SpacepointDumper ROOT files")
    parser.add_argument("--out", required=True, help="output data_root")
    parser.add_argument("--split", required=True, choices=("pretrain", "test"),
                        help="'pretrain' for the training loader, 'test' for eval (as prepare_data.py)")
    parser.add_argument("--min_track_points", type=int, default=5, help="TPCpp-10M: truth tracks with fewer dropped")
    parser.add_argument("--min_event_points", type=int, default=20, help="TPCpp-10M: events with fewer rejected")
    parser.add_argument("--keep_unmatched", action="store_true", help="keep clusters without a truth track (id 0)")
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()
    from mmap_ninja import RaggedMmap

    kept, vertices = [], {k: [] for k in ("file", "event", "truth_pv", "reco_n", "reco_x", "reco_y", "reco_z",
                                          "reco_ntracks", "reco_cov")}
    n_read = 0
    for i, ev in events(args.inputs):
        n_read += 1
        keep = select(ev, args.min_track_points, args.keep_unmatched)
        if keep.sum() < args.min_event_points:
            continue
        kept.append({k: ev[k][keep] for k in POINT})
        vertices["file"].append(i)
        vertices["event"].append(int(ev["event"]))
        vertices["truth_pv"].append([float(ev[f"truth_pv_{a}"]) for a in "xyz"])
        vertices["reco_n"].append(len(ev["reco_x"]))
        for k in ("reco_x", "reco_y", "reco_z", "reco_ntracks"):
            vertices[k].extend(ev[k].tolist())
        vertices["reco_cov"].extend(np.reshape(ev["reco_cov"], (-1, 6)).tolist())

    specs = {
        "features": lambda p: np.stack([p["adc"], p["x"], p["y"], p["z"]], axis=1).astype(np.float32),
        "seg_target": lambda p: p["track_id"].astype(np.int64),
        "pid_target": lambda p: p["pid"].astype(np.int64),
        "reg_target": lambda p: np.stack([p[k] for k in ("px", "py", "pz", "vtx_x", "vtx_y", "vtx_z", "charge", "e")],
                                         axis=1).astype(np.float32),
        "mid_target": lambda p: p["parent_id"].astype(np.int64),
    }
    os.makedirs(args.out, exist_ok=True)
    for name, fn in specs.items():
        out_dir = os.path.join(args.out, f"{name}_{args.split}")
        RaggedMmap.from_generator(out_dir=out_dir, sample_generator=(fn(p) for p in kept), batch_size=args.batch_size,
                                  verbose=False)
    np.savez(os.path.join(args.out, f"vertices_{args.split}.npz"),
             **{k: np.asarray(v, dtype=np.int64 if k in ("file", "event", "reco_n", "reco_ntracks") else np.float64)
                for k, v in vertices.items()})

    sizes = np.array([len(p["adc"]) for p in kept])
    print(f"{len(kept)} of {n_read} events kept -> {args.out} (split {args.split}); points per event: median "
          f"{np.median(sizes):.0f}, mean {sizes.mean():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
