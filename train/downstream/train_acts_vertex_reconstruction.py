"""
train_acts_vertex_reconstruction.py
-------------------------------------
CLI entry point for ACTS-based vertex reconstruction.

Mirrors train_vertex_reconstruction.py but uses ActsVertexTrainer
instead of VertexTrainer.

Usage
-----
    python train_acts_vertex_reconstruction.py \
        --track_ckpt path/to/track.pth \
        [--config path/to/config.yaml] \
        [--max_events 500] \
        [--n_geometry_layers 48] \
        [--save_csv] \
        [--output_suffix _acts]

The script will:
  1. Load the pretrained backbone and MambaAttentionHead from track_ckpt.
  2. Build the ACTS TPC cylindrical geometry.
  3. Run ACTS Kalman Filter + AMVF on each test event.
  4. Save vertex_results_acts.npz, vertex_metrics_acts.json.
"""

import argparse
import os
import sys

# Run from train/downstream/ (same convention as train_vertex_reconstruction.py)
# '../..' adds PP_collision/ so 'from train.downstream import ...' works
sys.path.append('../..')
# Also add this directory for the local acts_* modules
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from acts_vertex_reconstruction_trainer import ActsVertexTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="ACTS-based vertex reconstruction for FM4NPP"
    )
    parser.add_argument(
        "--track_ckpt",
        type=str,
        required=True,
        help="Path to the track-finding checkpoint (.pth)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: scripts/configs/mamba_tracking.yaml "
             "relative to repository root)",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=-1,
        help="Maximum number of test events to process (-1 = all)",
    )
    parser.add_argument(
        "--n_geometry_layers",
        type=int,
        default=48,
        help="Number of cylindrical TPC layers for ACTS geometry (default 48)",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Write per-point CSV file in addition to .npz / .json",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_acts",
        help="Suffix appended to output filenames (default _acts)",
    )
    parser.add_argument(
        "--bz",
        type=float,
        default=1.4,
        help="Solenoid field strength in Tesla (default 1.4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("FM4NPP  |  ACTS Vertex Reconstruction")
    print("=" * 60)
    print(f"  track_ckpt         : {args.track_ckpt}")
    print(f"  config             : {args.config or '(default)'}")
    print(f"  max_events         : {args.max_events}")
    print(f"  n_geometry_layers  : {args.n_geometry_layers}")
    print(f"  Bz                 : {args.bz} T")
    print(f"  save_csv           : {args.save_csv}")
    print(f"  output_suffix      : {args.output_suffix}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Build trainer
    # ------------------------------------------------------------------
    trainer = ActsVertexTrainer.launch(config_path=args.config)
    # Override Bz if provided
    trainer.Bz = args.bz

    # Set up neural networks (frozen backbone + track head)
    trainer.setup_vertexing(track_ckpt=args.track_ckpt)

    # ------------------------------------------------------------------
    # Run ACTS evaluation
    # ------------------------------------------------------------------
    trainer.run_acts_vertexing(
        max_events        = args.max_events,
        save_csv          = args.save_csv,
        n_geometry_layers = args.n_geometry_layers,
        output_suffix     = args.output_suffix,
    )

    print("Done.")


if __name__ == "__main__":
    main()
