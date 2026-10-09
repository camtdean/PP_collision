#!/usr/bin/env python3
"""
Run vertex reconstruction after a trained track-finding head.

    python run_vertex_reconstruction.py --yaml_config cfg.yaml --config d9_m64_k30_p20 \
        --pretrained_ckpt backbone.ckpt --track_ckpt <..._seed42_checkpoint.pth> \
        --root_dir out/ [--bz 1.4] [--max_events 2000] [--no_quality_mask] [--silicon_tpc_mask] [--track_params hits|truth]
"""
import os
import sys
import argparse

sys.path.append('../..')

from fm4npp.utils import YParams
from vertex_reconstruction_trainer import VertexTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml_config", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--pretrained_ckpt", required=True, help="frozen backbone checkpoint")
    p.add_argument("--track_ckpt", required=True, help="checkpoint from train_track_finding.py")
    p.add_argument("--root_dir", required=True)
    p.add_argument("--run_num", default="0")
    p.add_argument("--global_log_dir", default="globallogs")
    p.add_argument("--eventnumber", default=70000, type=int)
    p.add_argument("--max_events", default=None, type=int)
    p.add_argument("--bz", default=1.4, type=float, help="solenoid field, Tesla")
    p.add_argument("--no_quality_mask", action="store_true", help="don't pass noise/validity labels to VertexHead")
    p.add_argument("--silicon_tpc_mask", action="store_true", help="drop silicon hits whose predicted track has no TPC hit")
    p.add_argument("--track_params", choices=("hits", "truth"), default="hits",
                   help="track parameters + quality labels: fitted from predicted clusters' hits, or truth placeholder")
    p.add_argument("--data_root_test", default=None)
    args = p.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params.pretrained_ckpt = args.pretrained_ckpt
    params.limit_data = True
    params.limit_size = args.eventnumber
    params.batch_size = 1
    params.valid_batch_size = 1
    params.num_embedder_layers = 0
    params.return_reg_test = True        # val loader then yields reg as a 4th element
    params.vertex_bz = args.bz
    params.vertex_use_quality_mask = not args.no_quality_mask
    params.vertex_silicon_tpc_mask = args.silicon_tpc_mask
    params.vertex_track_params = args.track_params
    params.log_file_name = f"{args.config}_vertex_{args.run_num}.log"
    if args.data_root_test:
        params.data_root_test = args.data_root_test

    trainer = VertexTrainer(params, args)
    trainer.launch()
    trainer.setup_vertexing(args.track_ckpt)
    trainer.run_vertexing(pretrain=True, max_events=args.max_events)
    trainer.cleanup()


if __name__ == "__main__":
    main()
