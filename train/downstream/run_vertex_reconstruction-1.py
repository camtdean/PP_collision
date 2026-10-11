#!/usr/bin/env python3
"""
Run vertex reconstruction after a trained track-finding head.

Everything comes from the same YAML config + config name used for train_track_finding.py
(scripts/configs/mamba_tracking.yaml). The backbone checkpoint and the track-head checkpoint
default to what that config and the track-finding naming convention imply:

    python run_vertex_reconstruction.py \
        --yaml_config ../../scripts/configs/mamba_tracking.yaml --config d9_m1_k30_p20 \
        --root_dir out/ [--track_run_num 0] [--seed 42] [--eventnumber 70000]

    backbone   : --pretrained_ckpt, else pretrained_ckpt from the YAML
    track head : --track_ckpt, else
                 {checkpoint_dir}/{config}_nerf_tracking_head_d{eventnumber}_{track_run_num}_seed{seed}_checkpoint.pth

Vertex options (--bz, --track_params, ...) fall back to the YAML keys vertex_bz,
vertex_track_params, vertex_use_quality_mask, vertex_silicon_tpc_mask, then to the
defaults in vertex_reconstruction_trainer.py.
"""
import os
import sys
import argparse

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml_config", required=True, help="normally scripts/configs/mamba_tracking.yaml")
    p.add_argument("--config", required=True, help="config name inside the YAML, e.g. d9_m1_k30_p20")
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
    p.add_argument("--no_quality_mask", action="store_true", help="don't pass noise/validity labels to VertexHead")
    p.add_argument("--silicon_tpc_mask", action="store_true", help="drop silicon hits whose predicted track has no TPC hit")
    p.add_argument("--track_params", choices=("hits", "truth"), default=None,
                   help="track parameters + quality labels source; default: YAML vertex_track_params or 'hits'")
    p.add_argument("--data_root_test", default=None)
    args = p.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # --- checkpoints: CLI overrides YAML / naming convention ---
    if args.pretrained_ckpt:
        params.pretrained_ckpt = args.pretrained_ckpt
    elif not _get(params, "pretrained_ckpt"):
        sys.exit(f"config {args.config!r} has no pretrained_ckpt; add one to the YAML or pass --pretrained_ckpt")

    track_ckpt = args.track_ckpt or default_track_ckpt(
        params, args.config, args.eventnumber, args.track_run_num, args.seed, args.track_usepretrain)
    if not os.path.isfile(track_ckpt):
        sys.exit(f"track-head checkpoint not found: {track_ckpt}\n"
                 f"(pass --track_ckpt, or check --track_run_num/--seed/--eventnumber against the training run)")

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
    if args.no_quality_mask:
        params.vertex_use_quality_mask = False
    if args.silicon_tpc_mask:
        params.vertex_silicon_tpc_mask = True
    if args.track_params is not None:
        params.vertex_track_params = args.track_params
    if args.data_root_test:
        params.data_root_test = args.data_root_test

    params.log_file_name = f"{args.config}_vertex_{args.run_num}.log"
    print(f"[vertex] backbone   : {params.pretrained_ckpt}")
    print(f"[vertex] track head : {track_ckpt}")

    trainer = VertexTrainer(params, args)
    trainer.launch()
    trainer.setup_vertexing(track_ckpt)
    trainer.run_vertexing(pretrain=True, max_events=args.max_events)
    trainer.cleanup()


if __name__ == "__main__":
    main()
