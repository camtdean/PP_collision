#!/usr/bin/env python3
"""
Run vertex reconstruction after a trained track-finding head.

Config, data paths and the backbone checkpoint come from the mamba_5m_downstream entry of
scripts/configs/mamba_tracking.yaml by default (width-256 Mamba2 backbone, the same
architecture as train_track_finding.py's d9_m1_k30_p20). Run from train/downstream:

    python train_vertex_reconstruction.py --root_dir out/ --track_ckpt <head.pth> [--save_csv]

    config     : --config (default mamba_5m_downstream) in --yaml_config
    backbone   : --pretrained_ckpt, else pretrained_ckpt from that config
    track head : --track_ckpt, else
                 {checkpoint_dir}/{track_config}_nerf_tracking_head_d{eventnumber}_{track_run_num}_seed{seed}_checkpoint.pth
                 where track_config is the --config the head was trained with (default d9_m1_k30_p20)

The vertex options --bz and --outlier_dca_cut fall back to the YAML keys vertex_bz and
vertex_outlier_dca_cut, then to the defaults in vertex_reconstruction_trainer.py / vertexhead.py.
"""
import os
import sys
import argparse
import csv
import json

import numpy as np

sys.path.append('../..')

from fm4npp.utils import YParams
from vertex_reconstruction_trainer import VertexTrainer


def _get(params, key, default=None):
    try:
        return getattr(params, key)
    except Exception:
        return default


def default_track_ckpt(params, config, eventnumber, track_run_num, seed, usepretrain=True):
    """Checkpoint name train_track_finding.py + track_finding_trainer.py write for these args."""
    base = f"{config}_nerf_tracking_head_d{eventnumber}_{track_run_num}_seed{seed}"
    if not usepretrain:
        base += "_nopretrain"
    return os.path.join(params.checkpoint_dir, base + "_checkpoint.pth")


DEFAULT_YAML = "../../scripts/configs/mamba_tracking.yaml"
DEFAULT_CONFIG = "mamba_5m_downstream"
DEFAULT_TRACK_CONFIG = "d9_m1_k30_p20"    # train_track_finding.py's name for the same backbone


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml_config", default=DEFAULT_YAML, help=f"default: {DEFAULT_YAML}")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"config name inside the YAML; default: {DEFAULT_CONFIG}")
    p.add_argument("--track_config", default=DEFAULT_TRACK_CONFIG,
                   help="--config the track head was trained with; only names the derived track checkpoint; "
                        f"default: {DEFAULT_TRACK_CONFIG}")
    p.add_argument("--pretrained_ckpt", default=None,
                   help="frozen backbone checkpoint; default: pretrained_ckpt from the YAML")
    p.add_argument("--track_ckpt", default=None,
                   help="train_track_finding.py checkpoint; default: derived from config/run/seed")
    p.add_argument("--track_run_num", default="0", help="--run_num the track head was trained with")
    p.add_argument("--seed", default=42, type=int, help="--seed the track head was trained with")
    p.add_argument("--no-pretrain-track", dest="track_usepretrain", action="store_false",
                   help="track head was trained with --no-pretrain (affects the derived name)")
    p.add_argument("--root_dir", required=True)
    p.add_argument("--run_num", default="0")
    p.add_argument("--global_log_dir", default="globallogs")
    p.add_argument("--eventnumber", default=70000, type=int,
                   help="must match the track head's --eventnumber for the derived checkpoint name")
    p.add_argument("--max_events", default=None, type=int)
    p.add_argument("--bz", default=None, type=float, help="solenoid field, Tesla; default: YAML vertex_bz or 1.4")
    p.add_argument("--outlier_dca_cut", default=None, type=float,
                   help="drop tracks passing farther than this (cm) from the vertex, worst first, and refit; "
                        "0 disables; default: YAML vertex_outlier_dca_cut or vertexhead.OUTLIER_DCA_CUT")
    p.add_argument("--data_root_test", default=None)
    p.add_argument("--save_csv", action="store_true",
                   help="also write eval_track_finding.py's per-point CSV, with vtx_x/y/z the reconstructed "
                        "vertex; test_* columns (truth vertex, residual) are for testing only")
    p.add_argument("--csv_output_path", default=None,
                   help="per-point CSV path; default: {run dir}/vertex_per_point_data.csv")
    args = p.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # --- checkpoints: CLI overrides YAML / naming convention ---
    if args.pretrained_ckpt:
        params.pretrained_ckpt = args.pretrained_ckpt
    elif not _get(params, "pretrained_ckpt"):
        sys.exit(f"config {args.config!r} has no pretrained_ckpt; add one to the YAML or pass --pretrained_ckpt")

    track_ckpt = args.track_ckpt or default_track_ckpt(
        params, args.track_config, args.eventnumber, args.track_run_num, args.seed, args.track_usepretrain)
    if not os.path.isfile(track_ckpt):
        sys.exit(f"track-head checkpoint not found: {track_ckpt}\n"
                 f"(pass --track_ckpt, or check --track_config/--track_run_num/--seed/--eventnumber "
                 f"against the training run)")

    # --- fixed by how VertexTrainer consumes the loader ---
    params.limit_data = True
    params.limit_size = args.eventnumber
    params.batch_size = 1
    params.valid_batch_size = 1
    params.num_embedder_layers = 0
    params.return_reg_test = True        # val loader then yields reg as a 4th element
    params.return_dict = False           # VertexTrainer unpacks tuples, like track_finding_trainer

    # --- vertex options: CLI flag if given, else YAML key, else trainer default ---
    if args.bz is not None:
        params.vertex_bz = args.bz
    if args.outlier_dca_cut is not None:
        params.vertex_outlier_dca_cut = args.outlier_dca_cut
    if args.data_root_test:
        params.data_root_test = args.data_root_test

    params.log_file_name = f"{args.config}_vertex_{args.run_num}.log"
    print(f"[vertex] backbone   : {params.pretrained_ckpt}")
    print(f"[vertex] track head : {track_ckpt}")

    trainer = VertexTrainer(params, args)
    trainer.launch()
    trainer.setup_vertexing(track_ckpt)
    result = trainer.run_vertexing(pretrain=True, max_events=args.max_events, save_csv=args.save_csv)
    if trainer.world_rank == 0:
        save_results(result, trainer, params, args, track_ckpt)
    trainer.cleanup()


def _jsonable(v):
    if hasattr(v, "item") and getattr(v, "numel", lambda: 1)() == 1:
        return v.item()
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


def save_results(result, trainer, params, args, track_ckpt):
    """Write per-event vertices (.npz) and metrics (.json) to the run dir; append metrics to the performance log."""
    out_dir = params.experiment_dir            # {root_dir}/{config}/{run_num}, set by launch()
    tag = "vertex"

    npz_path = os.path.join(out_dir, f"{tag}_results.npz")
    np.savez(npz_path,
             vertex_estimate=result["vertex_estimate"].float().numpy(),
             vertex_target=result["vertex_target"].float().numpy(),
             fit_is_valid=result["fit_is_valid"].numpy())

    metrics = {k: _jsonable(v) for k, v in result["metrics"].items()}
    summary = {
        "config": args.config,
        "yaml_config": os.path.abspath(args.yaml_config),
        "pretrained_ckpt": params.pretrained_ckpt,
        "track_ckpt": os.path.abspath(track_ckpt),
        "bz": trainer.vertex_head.Bz,
        "outlier_dca_cut": trainer.vertex_head.outlier_dca_cut,
        "max_events": args.max_events,
        "metrics": metrics,
    }
    json_path = os.path.join(out_dir, f"{tag}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    trainer.log_infile("[vertex] " + json.dumps(metrics))
    print(f"[vertex] wrote {npz_path}\n[vertex] wrote {json_path}")

    if "per_point" in result:
        csv_path = args.csv_output_path or os.path.join(out_dir, f"{tag}_per_point_data.csv")
        write_per_point_csv(result["per_point"], csv_path)
        print(f"[vertex] wrote {csv_path} ({len(result['per_point']['batch_idx'])} points)")


def write_per_point_csv(columns, path):
    """
    Write {column: 1-D array} as CSV in the format pandas' to_csv(index=False) gives
    track_finding_trainer.inference(): shortest round-trip floats, NaN as an empty field,
    os.linesep line endings. No pandas dependency.
    """
    names = list(columns)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator=os.linesep)
        writer.writerow(names)
        for row in zip(*(columns[n].tolist() for n in names)):
            writer.writerow(["" if v != v else v for v in row])    # v != v only for NaN


if __name__ == "__main__":
    main()
