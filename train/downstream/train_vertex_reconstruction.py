#!/usr/bin/env python3
"""
Vertex reconstruction training, laid out like train_track_finding.py: LearnedVertexHead (--vertex_method learned)
on the frozen backbone's features. Evaluate with eval_vertex_reconstruction.py. Artifacts:
{config}_nerf_vertex_{method}_d{N}_{run}_seed{seed}.
"""
import os
import sys

import argparse
import gc
import random
import numpy as np

import torch

# ensure fm4npp modules can be found
sys.path.append('../..')


from fm4npp.utils import YParams
from vertex_reconstruction_trainer import DownstreamTrainer

DEFAULT_YAML = "../../scripts/configs/mamba_tracking.yaml"
DEFAULT_CONFIG = "mamba_5m_downstream"


def set_seed(seed):
    """Set all random seeds (torch, CUDA, NumPy, random), as train_track_finding.py does."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)

    print(f"Random seed set to: {seed}")


def main():
    parser = argparse.ArgumentParser(description="Downstream vertex reconstruction training script")
    parser.add_argument("--yaml_config", default=DEFAULT_YAML, type=str, help="Path to YAML config file")
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=str, help="Model config name")
    parser.add_argument("--pretrained_ckpt", default=None, type=str,
                        help="Frozen backbone checkpoint (default: pretrained_ckpt from the config)")
    parser.add_argument("--run_num", default='0', type=str, help="Sub run number")
    parser.add_argument("--root_dir", default=None, type=str, help="Root dir to store results")
    parser.add_argument("--global_log_dir", default='globallogs', type=str, help="Global dir to store logging only")
    parser.add_argument("--eventnumber", default=70000, type=int, help="downstream training event number")
    parser.add_argument(
        "--usepretrain",
        dest="usepretrain",
        action="store_true",
        help="enable using the pretrained model (default)",
    )
    parser.add_argument(
        "--no-pretrain",
        dest="usepretrain",
        action="store_false",
        help="disable pretrained model",
    )
    parser.set_defaults(usepretrain=True)
    parser.add_argument("--train_batch_size", default=32, type=int, help="train batch size")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--vertex_method", default="learned", choices=["learned"], help="the vertexing method trained")
    parser.add_argument("--truth_radius", default=1.0, type=float,
                        help="cm: hits whose particle starts within this of the beam line make the truth vertices")
    parser.add_argument("--detector", default=None, type=str, help="detector file (its beam_position_cm)")
    parser.add_argument("--checkpoint_dir", default=None, type=str,
                        help="where the log and checkpoints go (default: checkpoint_dir from the config)")
    parser.add_argument("--val_events", default=None, type=int, help="validation events per epoch (default: all)")
    parser.add_argument("--val_on_train", action="store_true",
                        help="validate on the training set (the overfit test: the log's terms are then the training set's)")
    parser.add_argument("--max_lr", default=None, type=float, help="peak learning rate (default: the config's)")
    parser.add_argument("--min_lr", default=None, type=float, help="lowest learning rate (default: the config's)")
    parser.add_argument("--max_epochs", default=None, type=int, help="epochs (default: max_epochs from the config)")
    parser.add_argument("--min_sigma_xy", default=None, type=float,
                        help="cm: floor of the predicted x, y position uncertainty (default: none)")
    parser.add_argument("--min_sigma_z", default=None, type=float,
                        help="cm: floor of the predicted z position uncertainty (default: none)")
    parser.add_argument("--run_config", default=None, type=str,
                        help="YAML file of options (the eval's, e.g. scripts/configs/vertex_m5_gpuserver0.yaml): its "
                             "keys this script has set the defaults (others ignored); command-line options "
                             "override it, relative paths are relative to the file")
    path_keys = {"yaml_config", "pretrained_ckpt", "checkpoint_dir", "root_dir", "detector"}
    run_config = parser.parse_known_args()[0].run_config
    if run_config:
        import yaml
        with open(run_config) as f:
            run = yaml.safe_load(f) or {}
        base = os.path.dirname(os.path.abspath(run_config))
        # each value converted as the command line would be (e.g. run_num: 2 -> "2")
        types = {a.dest: a.type for a in parser._actions if a.type is not None}
        parser.set_defaults(**{
            k: os.path.join(base, os.path.expanduser(v)) if k in path_keys and isinstance(v, str)
            else types[k](v) if k in types and v is not None else v
            for k, v in run.items() if k in types and k not in ("eventnumber", "run_num", "vertex_method")})
    args = parser.parse_args()
    # paths: ~ expanded, absolute (not relative to this folder once the working directory changes)
    for key in path_keys:
        if getattr(args, key, None):
            setattr(args, key, os.path.abspath(os.path.expanduser(getattr(args, key))))

    if not args.root_dir:
        parser.error("--root_dir is required (on the command line or in --run_config)")

    # seed before anything touches an RNG
    set_seed(args.seed)

    # Initialize parameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if args.pretrained_ckpt:
        params.pretrained_ckpt = args.pretrained_ckpt
    elif not getattr(params, "pretrained_ckpt", None):
        sys.exit(f"config {args.config!r} has no pretrained_ckpt; add one to the YAML or pass --pretrained_ckpt")
    params.continue_from_best = True
    params.batch_size = int(args.train_batch_size)
    params.limit_data = True
    params.limit_size = int(args.eventnumber)
    params.valid_batch_size = 1
    # the seed appears in artifact names so runs at different seeds don't overwrite each other
    base_name = f"{args.config}_nerf_vertex_{args.vertex_method}_d{params.limit_size}_{args.run_num}_seed{args.seed}"
    if args.usepretrain:
        params.log_file_name = base_name + ".log"
    else:
        params.log_file_name = base_name + "_nopretrain.log"
    params.num_embedder_layers = 0
    params.return_reg_test = True        # loaders then yield reg (truth) as a 4th element
    params.return_dict = False
    params.vertex_method, params.truth_radius = args.vertex_method, args.truth_radius
    for key, value in (("vertex_detector", args.detector), ("learned_vertex_val_events", args.val_events),
                       ("max_epochs", args.max_epochs), ("max_lr", args.max_lr), ("min_lr", args.min_lr),
                       ("learned_vertex_min_sigma_xy", args.min_sigma_xy),
                       ("learned_vertex_min_sigma_z", args.min_sigma_z),
                       ("learned_vertex_val_on_train", args.val_on_train or None), ("checkpoint_dir", args.checkpoint_dir)):
        if value is not None:
            params[key] = value

    # Launch and train
    trainer = DownstreamTrainer(params, args)
    trainer.launch()
    checkpoint_path = None
    trainer.train(pretrain=args.usepretrain, train_from_checkpoint=False, checkpoint_path=checkpoint_path)

    # Cleanup
    trainer.cleanup()
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
