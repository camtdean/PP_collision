"""
vertex_reconstruction_trainer.py: VertexHead after a trained track-finding head.

    frozen backbone -> frozen track finder -> VertexHead (PV + SVs)

Entry point: inference() (eval_vertex_reconstruction.py); train() is a phase-2 hook. Truth is used only for
scoring. Per-point CSV: track finding's columns (vtx_*: truth production vertex), reco_vtx_* (reconstructed PV)
and test_* truth columns.
"""

import numpy as np
import copy
import json
import os, sys, time
import inspect
import torch

import torch.nn.init as init
import gc, torch.distributed as dist
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

from torch.nn.parallel import DistributedDataParallel

from tqdm import tqdm 
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts # pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'

sys.path.append('../..')

from fm4npp.utils import *
from fm4npp.datasets.dataset import *
from fm4npp.models.mambagpt import MambaGPT, Mamba1GPT
from fm4npp.models.longformer_gpt import LongformerGPT
from fm4npp.models.linformer_gpt import LinformerGPT


from trackinghead import *
import csv

import vertexing
from vertexing import VertexHead
from vertexing.learnedvertex import LearnedVertexHead, LearnedVertexing
from downstream_util import get_vertex_label
from loss import compute_vertex_metrics, assign_points_to_masks, VertexHungarianMatcher, compute_vertex_set_loss

PAD_VALUE = -100

# track finding's per-point CSV columns (batch_idx: the event's index; vtx_*: the hit's particle's truth production
# vertex, cm)
PER_POINT_COLUMNS = ("batch_idx", "point_idx", "E_norm", "eta_norm", "phi_norm", "r_norm",
                     "px", "py", "pz", "vtx_x", "vtx_y", "vtx_z", "charge", "energy",
                     "seg_target", "pred_assignment", "confidence")

# the reconstructed PV, cm
RECO_COLUMNS = ("reco_vtx_x", "reco_vtx_y", "reco_vtx_z")

# TESTING ONLY: the truth PV (get_vertex_label) and reconstructed - truth, cm, NaN where undefined; the rank of
# the hit's particle's primary vertex (0: the PV, 1, 2...: pileup; -1: none)
TEST_COLUMNS = ("test_truth_vtx_x", "test_truth_vtx_y", "test_truth_vtx_z",
                "test_residual_x", "test_residual_y", "test_residual_z", "test_residual_3d", "test_truth_pv_index")

CSV_COLUMNS = PER_POINT_COLUMNS + RECO_COLUMNS + TEST_COLUMNS


def write_metrics_log(metrics, logfile):
    """
    Append metrics to logfile as track_finding_trainer.inference() writes them: a tab-separated header line,
    then the values.
    """
    with open(logfile, "a") as f:
        f.write("\t".join(metrics) + "\n")
        f.write("\t".join(str(v) for v in metrics.values()) + "\n")


def write_per_point_csv(columns, path):
    """
    Write {column: 1-D array} as pandas' to_csv(index=False) would (round-trip floats, NaN as
    empty), without pandas.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator=os.linesep)
        writer.writerow(columns)
        for row in zip(*(c.tolist() for c in columns.values())):
            writer.writerow(["" if v != v else v for v in row])    # v != v only for NaN


def per_point_columns(first_event, grouped, label, reg, mask, class_probs, mask_probs, vertex, truth_vertex,
                      hit_vertex):
    """
    One batch of the per-point CSV.

    Args:
        first_event: index of the batch's first event (batch_idx = first_event + its row); grouped: (B, N, 4); label: (B, N) truth track ID; reg: (B, N, 8) truth;
        mask: (B, N) real hits; class_probs, mask_probs: raw track-head outputs; vertex: (B, 3) reconstructed PV;
        truth_vertex: (B, 3), NaN where undefined; hit_vertex: (B, N) the hit's primary-vertex rank (get_vertex_label).

    Returns:
        dict column -> 1-D array (CSV_COLUMNS), one entry per real hit.
    """
    assign = assign_points_to_masks({"pred_probs": class_probs, "pred_masks": mask_probs.permute(0, 2, 1)},
                                    option=2)["assignments"]
    b, n = mask.nonzero(as_tuple=True)

    def col(t):
        # float64 (exact for float32) so the CSV prints each value as track_finding_trainer.inference()'s .item() does
        t = t[b, n] if t.dim() == 2 else t[b]
        return t.double().cpu().numpy()

    columns = {
        "batch_idx": (first_event + b).cpu().numpy(),
        "point_idx": n.cpu().numpy(),
        **{name: col(grouped[..., k]) for k, name in enumerate(("E_norm", "eta_norm", "phi_norm", "r_norm"))},
        **{name: col(reg[..., k]) for k, name in enumerate(("px", "py", "pz"))},
        **{name: col(reg[..., k]) for k, name in zip((3, 4, 5), ("vtx_x", "vtx_y", "vtx_z"))},
        **{name: col(vertex[:, k]) for k, name in enumerate(RECO_COLUMNS)},
        "charge": col(reg[..., 6]), "energy": col(reg[..., 7]),
        "seg_target": label.to(mask.device)[b, n].cpu().numpy(),
        "pred_assignment": assign[b, n].cpu().numpy(),
        "confidence": col(mask_probs.max(dim=-1).values),
    }
    # TESTING ONLY
    residual = vertex.double() - truth_vertex.double()
    for k, axis in enumerate("xyz"):
        columns[f"test_truth_vtx_{axis}"] = col(truth_vertex[:, k])
        columns[f"test_residual_{axis}"] = col(residual[:, k])
    columns["test_residual_3d"] = col(residual.norm(dim=-1))
    columns["test_truth_pv_index"] = hit_vertex.to(mask.device)[b, n].cpu().numpy()
    return {k: columns[k] for k in CSV_COLUMNS}


class DownstreamTrainer():

    # ------------------------------------------------------------------------------------------
    # Infrastructure as in the other downstream trainers (track_finding_trainer, point_classification_trainer)
    # ------------------------------------------------------------------------------------------

    def __init__(self, params, args):
        
        ''' init vars for distributed training (ddp) and logging'''
        self.root_dir = args.root_dir
        self.config = args.config 
        self.run_num = args.run_num
        self.world_size = 1
        
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])

        self.local_rank = 0
        self.world_rank = 0
        
        if self.world_size > 1: # multigpu, use DDP with standard NCCL backend for communication routines
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            torch.backends.cudnn.benchmark = True

        self.log_to_screen = (self.world_rank==0)
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        
        self.params = params
        self.use_amp = bool(getattr(self.params, 'use_amp', True))
        # how vertices are found: fit (VertexHead) or learned (LearnedVertexing)
        self.vertex_method = getattr(self.params, "vertex_method", None) or "fit"
        print("running on rank {} with world size {}".format(self.world_rank, self.world_size))



        
    def init_exp_dir(self, exp_dir):
        """
        Run folder exp_dir with a new logs/<name>_<N>.log (parameters, then metrics; name: params.run_name, default
        vertex_eval); its per-point CSV goes to data/ (csv_path), its metrics JSON to metrics/ (metrics_path). No
        files with no_output.
        """
        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        if getattr(self.params, "no_output", False):
            self.logfile = self.csv_path = self.metrics_path = os.devnull
            return
        logs, idx = os.path.join(exp_dir, "logs"), 0
        name = getattr(self.params, "run_name", None) or "vertex_eval"
        if self.world_rank == 0:
            os.makedirs(logs, exist_ok=True)
            os.makedirs(os.path.join(exp_dir, "data"), exist_ok=True)
            os.makedirs(os.path.join(exp_dir, "metrics"), exist_ok=True)
            while os.path.exists(os.path.join(logs, f'{name}_{idx}.log')):
                idx += 1
        self.logfile = os.path.join(logs, f'{name}_{idx}.log')
        self.csv_path = os.path.join(exp_dir, "data", f'{name}_{idx}_per_point_data.csv')
        self.metrics_path = os.path.join(exp_dir, "metrics", f'{name}_{idx}.json')
        if self.world_rank == 0:
            with open(self.logfile, 'w') as f:
                f.write('Initialized at: {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        if dist.is_initialized():
            dist.barrier()

    def log_infile(self, log):
        with open(self.logfile, "a") as f:
            f.write("{}\n".format(log))

    def parse_exp_details(self, D):
        """The run's parameters, one per line, for the run log."""
        return 'Important Details:\n' + ''.join(f'{a}: {b}\n' for a, b in D.items())

    def launch(self, train=True):
        """Run folder, backbone (pretrained checkpoint), optimizer and loaders; train=False (the eval): the test
        loader only."""
        # the run folder: params.run_folder (the eval: <backbone>/<track head>), else <config>/<run_num>
        exp_dir = os.path.join(self.root_dir, getattr(self.params, "run_folder", None) or os.path.join(self.config, self.run_num))
        self.init_exp_dir(exp_dir)

        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = int(self.params.batch_size//self.world_size)
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = int(self.params.valid_batch_size//self.world_size)

        print('batch size: ', self.params['global_batch_size'])
        print('local batch size: ', self.params['local_batch_size'])

        self.log_infile(self.parse_exp_details(self.params.params))       

        # get the pretrained model
        self.klen = self.params.klen
        if self.params.mambaversion == 'mamba1':
            self.model = Mamba1GPT(embed_dim=self.params.embed_dim, num_layers=self.params.num_layers_backbone,
                                d_state=self.params.d_state, d_conv=4, expand=2, klen=self.klen, dropout=self.params.dropout,
                                embed_method=self.params.embed_method, pe_method=self.params.pe_method)
        elif self.params.mambaversion == 'longformer':
            self.model = LongformerGPT(
                embed_dim=self.params.embed_dim,
                num_layers=self.params.num_layers_backbone,
                num_heads=self.params.num_heads_backbone,
                window_size=getattr(self.params, 'window_size', 256),
                mlp_ratio=getattr(self.params, 'mlp_ratio', 2.0),
                klen=self.klen,
                dropout=self.params.dropout,
                embed_method=self.params.embed_method,
                pe_method=self.params.pe_method
            )
        elif self.params.mambaversion == 'linformer':
            self.model = LinformerGPT(
                embed_dim=self.params.embed_dim,
                num_layers=self.params.num_layers_backbone,
                num_heads=self.params.num_heads_backbone,
                seq_len=getattr(self.params, 'seq_len', 512),
                proj_dim=getattr(self.params, 'proj_dim', 256),
                mlp_ratio=getattr(self.params, 'mlp_ratio', 2.0),
                klen=self.klen,
                dropout=self.params.dropout,
                embed_method=self.params.embed_method,
                pe_method=self.params.pe_method
            )
        else:
            self.model = MambaGPT(embed_dim=self.params.embed_dim, num_layers=self.params.num_layers_backbone,
                    d_state=self.params.d_state, d_conv=4, expand=2, klen=self.klen, dropout=self.params.dropout,
                    embed_method=self.params.embed_method, pe_method=self.params.pe_method)
        



        def initialize_mamba2(model, d_state, embed_dim):
            """ Properly initializes Mamba v2 to ensure stable learning. """
            
            with torch.no_grad():
                for name, param in model.named_parameters():

                    if "lin_B" in name:
                        param.normal_(mean=0.0, std=(d_state / embed_dim)**0.5)

                    elif "lin_C" in name:
                        param.normal_(mean=0.0, std=(1.0 / (embed_dim*d_state))**0.5)

                    elif "norm.weight" in name:
                        init.ones_(param)

                    # Bias Terms
                    elif "bias" in name:
                        init.zeros_(param)

                print(f"✅ Mamba v2 Model Initialized")
                
        Nu = self.params.embed_dim
        Nx = getattr(self.params, 'd_state', 16)

        # Only initialize Mamba models with Mamba-specific initialization
        if self.params.mambaversion in ['mamba1', 'mamba2']:
            initialize_mamba2(self.model, Nx, Nu)
        else:
            if self.world_rank == 0:
                print(f"✅ {self.params.mambaversion.capitalize()} Model Initialized")

        self.model = self.model.to(self.device)
        if self.world_rank == 0:
            print('Nparams: ', count_parameters(self.model))

        # distributed wrapper for data parallel
        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                device_ids=[self.local_rank],
                                                output_device=[self.local_rank],
                                                find_unused_parameters=True)

            

        # set an optimizer and learning rate scheduler   
        params_a   = []
        params_b   = []
        params_c   = []
        params_else= []

        for name, p in self.model.named_parameters():
            if "A_log" in name:
                params_a.append(p)   # might do LR ~ Nu
            elif "lin_B" in name:
                params_b.append(p)   # might do LR ~ Nx / sqrt(Nu)
            elif "lin_C" in name:
                params_c.append(p)   # might do LR ~ sqrt(Nu) / Nx
            else:
                params_else.append(p)
                
        self.optimizer = torch.optim.AdamW([
            {"params": params_a,   "lr": self.params.min_lr * Nu},                   # e.g. for A
            {"params": params_b,   "lr": self.params.min_lr * Nx / (Nu**0.5)},       # e.g. for B
            {"params": params_c,   "lr": self.params.min_lr * (Nu**0.5) / Nx},       # e.g. for C
            {"params": params_else,"lr": self.params.min_lr},
        ], weight_decay=0.1, betas=(0.9, 0.95))

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer,
                                          first_cycle_steps=getattr(self.params, 'total_steps', 200),
                                          max_lr=self.params.max_lr,
                                          min_lr=self.params.min_lr,
                                          warmup_steps=self.params.warmup_steps)

        
        # get the dataloaders
        if not train:
            # the eval: the test loader. get_data_loader (shared) builds the training set too; limited to its first
            # event it is not scanned (all its events otherwise)
            loader_params = copy.copy(self.params)
            loader_params.limit_data, loader_params.limit_size = True, 1
            self.train_data_loader = self.train_sampler = None
            _, _, self.val_data_loader, _ = get_data_loader(loader_params, dist.is_initialized())
        else:
            self.train_data_loader, self.train_sampler, self.val_data_loader, _ = get_data_loader(self.params,
                                                                                                  dist.is_initialized())

        # checkpointing
        self.iters = 0
        self.startEpoch = 0
        self.resumed = False

        ##### Pretraining checkpoint
        print("Loading checkpoint %s"%self.params.pretrained_ckpt)
        self.restore_checkpoint(self.params.pretrained_ckpt)
        self.resumed = True

        self.startEpoch = 0
        self.epoch = self.startEpoch
        self.logs = {}

        # 
        #  training


    def cleanup(self):
        # 1) break references to big objects
        for obj in ("model", "down_model", "vertex_head", "learned_vertex",
                    "optimizer", "vertex_optimizer",
                    "scheduler", "vertex_scheduler",
                    "train_data_loader", "val_data_loader"):
            if hasattr(self, obj):
                delattr(self, obj)

        # 2) empty CUDA cache and run GC
        torch.cuda.empty_cache()
        gc.collect()

        # 3) tear down DDP if we set it up
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        print("Cleanup complete. All resources released.")

    # VertexHead's optimizer and scheduler (made by training) and their keys in a checkpoint
    _TRAINING_STATE = (("optimizer_state_dict", "vertex_optimizer"), ("scheduler_state_dict", "vertex_scheduler"))

    def _trained_head(self):
        """The module a checkpoint holds: LearnedVertexHead with vertex_method learned, else VertexHead."""
        head = self.learned_vertex.head if self.vertex_method == "learned" else self.vertex_head
        return getattr(head, "module", head)          # DDP-wrapped or not

    def _save_checkpoint(self, filename, epoch, is_best, loss):
        """The trained head's weights (_trained_head; the backbone and track head stay frozen) and, once training has
        made them, its optimizer and scheduler, with vertex_method and the run's params, to
        params.checkpoint_dir/filename."""
        checkpoint = {"epoch": epoch, "model_state_dict": self._trained_head().state_dict(),
                      "best_loss": getattr(self, "best_loss", None), "current_loss": loss, "is_best": is_best,
                      "vertex_method": self.vertex_method, "params": vars(self.params)}
        for key, name in self._TRAINING_STATE:
            if getattr(self, name, None) is not None:
                checkpoint[key] = getattr(self, name).state_dict()
        torch.save(checkpoint, os.path.join(self.params.checkpoint_dir, filename))

    def load_checkpoint(self, checkpoint_path, inference=False):
        """The trained head's weights (_trained_head) from a _save_checkpoint file (strict: its settings and
        vertex_method must match); without inference also its optimizer and scheduler where training made them, the
        next epoch and the best loss."""
        checkpoint, state = self._load_checkpoint_state(checkpoint_path, "model_state_dict")
        if checkpoint.get("vertex_method", "fit") != self.vertex_method:
            raise ValueError(f"{checkpoint_path} is a vertex_method {checkpoint.get('vertex_method', 'fit')!r} "
                             f"checkpoint; this run's is {self.vertex_method!r}")
        self._trained_head().load_state_dict(state)
        if not inference:
            for key, name in self._TRAINING_STATE:
                if getattr(self, name, None) is not None and key in checkpoint:
                    getattr(self, name).load_state_dict(checkpoint[key])
            self.startEpoch = checkpoint.get("epoch", 0) + 1
            self.best_loss = checkpoint.get("best_loss") or float("inf")
        if self.log_to_screen:
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    @staticmethod
    def _load_checkpoint_state(path, key):
        """A checkpoint (on the CPU) and its key's state dict without DDP's "module." prefixes."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return checkpoint, {k.replace("module.", ""): v for k, v in checkpoint[key].items()}

    def restore_checkpoint(self, checkpoint_path):
        checkpoint, state = self._load_checkpoint_state(checkpoint_path, "model_state")
        self.model.load_state_dict(state)

        self.iters = checkpoint['iters']
        if self.train_data_loader is not None:     # the eval builds no train loader
            self.startEpoch = checkpoint['epoch']+1 if self.iters % len(self.train_data_loader) == 0 else checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # ------------------------------------------------------------------------------------------
    # Training: LearnedVertexHead (vertex_method learned) with the backbone frozen; same names and signatures as the
    # other trainers. VertexHead's learnable parts (vertex_method fit) are not trained yet.
    # ------------------------------------------------------------------------------------------

    # the terms of the training loss and their weights (params: learned_vertex_<term>_weight)
    LEARNED_LOSS_WEIGHTS = {"loss_matched_ce": 0.5, "loss_unmatched_ce": 0.1, "loss_dice": 1.0, "loss_focal": 30.0,
                            "loss_position": 1.0}

    def setup_learned_vertexing(self):
        """
        Build LearnedVertexing (after launch()). From params: any LearnedVertexHead or LearnedVertexing argument as
        learned_vertex_<argument> (e.g. learned_vertex_n_vertex_slots, learned_vertex_real_threshold), each only where
        set; pe_method; the track slots: max_gt_classes.
        """
        settings = lambda cls: {name: getattr(self.params, "learned_vertex_" + name)  # noqa: E731
                                for name in inspect.signature(cls.__init__).parameters
                                if getattr(self.params, "learned_vertex_" + name, None) is not None}
        head = LearnedVertexHead(**{
            "input_dim": self.params.embed_dim, "pe_method": getattr(self.params, "pe_method", "nerf"),
            "num_feature_layers": 1 if getattr(self, "cache_combined", False) else self.params.num_layers_backbone,
            **settings(LearnedVertexHead)})
        self.learned_vertex = LearnedVertexing(
            head, **{"n_track_slots": self.params.max_gt_classes, **settings(LearnedVertexing)}).to(self.device)

    def _beam_position(self):
        """The beam line (x, y) cm of the truth labels: VertexHead's when built, else params.vertex_beam_position,
        the detector file's (vertex_detector) beam_position_cm, TrackBuilder's default."""
        if hasattr(self, "vertex_head"):
            return self.vertex_head.tracks.beam_position
        beam = getattr(self.params, "vertex_beam_position", None)
        if beam is None and getattr(self.params, "vertex_detector", None):
            import yaml
            with open(self.params.vertex_detector) as f:
                beam = yaml.safe_load(f).get("beam_position_cm")
        return tuple(beam or inspect.signature(vertexing.tracks.TrackBuilder.__init__).parameters["beam_position"].default)

    def _truth_labels(self, reg, label, mask):
        """get_vertex_label over the hits whose particle starts within params.truth_radius of the beam line (padded
        rows excluded: their zeros would pass); the truth PV: the origin shared by the most particles."""
        beam = self._beam_position()
        return get_vertex_label(reg, particle_id=label, valid_tracks=(torch.hypot(
            reg[..., 3] - beam[0], reg[..., 4] - beam[1]) < self.params.truth_radius) & mask)

    def _learned_vertex_loss(self, batch, pretrain, aux=True):
        """
        One loader batch (grouped, label, knearest, reg) through the frozen backbone and LearnedVertexing.

        Returns (loss, terms, residual): the loss (LEARNED_LOSS_WEIGHTS over compute_vertex_set_loss's terms; with
        aux the same on each of the head's aux outputs added), its terms of the final output as floats, and the PV
        residual (cm) of the events with a predicted and a truth PV.
        """
        if len(batch) != 4:
            raise RuntimeError("loader must yield (grouped, label, knearest, reg): the test set with "
                               "params.return_reg_test = True before launch(), a training set with its "
                               "dataset's return_reg = True")
        grouped, label, _, reg = (t.to(self.device) if torch.is_tensor(t) else t for t in batch)
        grouped = grouped.reshape(grouped.size(0), -1, grouped.size(-1))
        mask = grouped[..., 0] != PAD_VALUE
        with torch.no_grad():
            _, pre_embed, _ = self.model(grouped, return_z=True)
            truth = self._truth_labels(reg, label, mask)
        outputs, raw = self.learned_vertex(grouped, self._layer_features(pre_embed), mask)
        targets = [{"labels": torch.ones(n, dtype=torch.long, device=self.device),
                    "masks": (hit_vertex == torch.arange(n, device=self.device).unsqueeze(-1)).float(),
                    "positions": vertices[:n].float()}
                   for n, hit_vertex, vertices in zip(truth["n_primary_vertices"].tolist(), truth["hit_vertex"],
                                                      truth["primary_vertices"])]
        loss, terms = 0.0, None
        for out in [raw] + (raw["aux_list"] if aux else []):
            losses = compute_vertex_set_loss(
                {"pred_probs": out["vertex_class_probs"], "pred_masks": out["vertex_mask_probs"].permute(0, 2, 1),
                 "pred_positions": out["vertex_position"], "pred_log_variance": out["vertex_log_variance"]},
                targets, mask, self.matcher)
            loss = loss + sum(weight * losses[term] for term, weight in self.loss_weights.items())
            terms = terms or {term: losses[term].item() for term in self.loss_weights}
        ok = outputs["fit_is_valid"] & truth["vertex_valid"]
        return loss, terms, (outputs["vertex_estimate"] - truth["vertex_target"])[ok].norm(dim=-1).detach().cpu()

    def train(self, pretrain=True, train_from_checkpoint=False, checkpoint_path=None):
        """
        Train LearnedVertexHead (vertex_method learned) on the frozen backbone's features, in fp32. AdamW (max_lr,
        weight decay 1e-4), gradient norm clipped to 1, cosine schedule with warmup stepped per epoch
        (first_cycle_steps, default max_epochs; min_lr; warmup_steps). Each epoch: the training set, then the
        validation set (its first learned_vertex_val_events events, default all).

        Writes to params.checkpoint_dir: the log (log_file_name: per epoch the losses, the PV residual medians, the
        validation loss terms), <log name>_checkpoint.pth at the best validation loss and
        <log name>_best_pv_checkpoint.pth at the best validation PV residual median. Stops after
        learned_vertex_patience epochs (default 15) without a better validation loss (by learned_vertex_min_delta,
        default 1e-4), once past learned_vertex_early_stop_warmup epochs (default 60).

        Args:
            pretrain: backbone features (must be True); train_from_checkpoint, checkpoint_path: resume.
        """
        if self.vertex_method != "learned":
            raise NotImplementedError(f"training with vertex_method {self.vertex_method!r} is not written yet")
        if self.world_size > 1:
            raise RuntimeError("the vertex training runs in one process: run it with python, not torchrun")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False     # full fp32
        self.setup_learned_vertexing()
        head = self.learned_vertex.head
        print(f"Total parameters in LearnedVertexHead: {sum(p.numel() for p in head.parameters())}")
        self.train_data_loader.dataset.return_reg = True        # the training set yields reg too
        get = lambda name, default: (lambda v: default if v is None else v)(  # noqa: E731
            getattr(self.params, "learned_vertex_" + name, None))
        self.loss_weights = {term: get(term + "_weight", weight) for term, weight in self.LEARNED_LOSS_WEIGHTS.items()}
        self.matcher = VertexHungarianMatcher(
            cost_class=self.loss_weights["loss_matched_ce"], cost_dice=self.loss_weights["loss_dice"],
            cost_focal=self.loss_weights["loss_focal"], cost_position=get("cost_position_weight", 1.0),
            position_scale=head.position_scale.tolist()).to(self.device)
        self.vertex_optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],
                                                  lr=self.params.max_lr, weight_decay=0.0001)
        self.vertex_scheduler = CosineAnnealingWarmupRestarts(
            self.vertex_optimizer, max_lr=self.params.max_lr, min_lr=self.params.min_lr,
            first_cycle_steps=int(getattr(self.params, "first_cycle_steps", None) or self.params.max_epochs),
            warmup_steps=self.params.warmup_steps)
        self.best_loss = self.best_pv = float("inf")
        if train_from_checkpoint:
            self.load_checkpoint(checkpoint_path, inference=False)

        os.makedirs(self.params.checkpoint_dir, exist_ok=True)
        log_file_path = os.path.join(self.params.checkpoint_dir, self.params.log_file_name)
        stem = self.params.log_file_name.split('.')[0]
        terms = list(self.loss_weights)
        if not train_from_checkpoint:
            with open(log_file_path, "w") as f:
                f.write("\t".join(["Epoch", "Train_Loss", "Val_Loss", "Train_PV_median_cm", "Val_PV_median_cm",
                                   "Val_PV_rate"] + terms + ["Time"]) + "\n")
        stagnation = 0
        for epoch in range(self.startEpoch, self.params.max_epochs):
            self.epoch, start = epoch, time.time()
            self.down_results = {"train": [], "train_residual": [], "val": [], "val_residual": [], "val_events": 0,
                                 **{term: [] for term in terms}}
            self.downstream_end_to_end_one_epoch(pretrain=pretrain)
            val_loss = self.validate_end_to_end_one_epoch(pretrain=pretrain)
            median = lambda key: (lambda r: r.median().item() if len(r) else float("nan"))(  # noqa: E731
                torch.cat(self.down_results[key]))
            row = [np.mean(self.down_results["train"]), val_loss, median("train_residual"), median("val_residual"),
                   sum(map(len, self.down_results["val_residual"])) / max(self.down_results["val_events"], 1)
                   ] + [np.mean(self.down_results[term]) for term in terms] + [time.time() - start]
            with open(log_file_path, "a") as f:
                f.write("\t".join([str(epoch)] + [f"{v:.8f}" for v in row]) + "\n")
            if self.log_to_screen:
                print(f"Epoch {epoch}/{self.params.max_epochs - 1} | Time: {row[-1]:.2f}s\n"
                      f"  Train Loss: {row[0]:.6f} | Val Loss: {row[1]:.6f}\n"
                      f"  PV residual median (cm): train {row[2]:.4f} | val {row[3]:.4f} (PV found: {row[4]:.3f})\n  "
                      + " | ".join(f"{term}: {v:.6f}" for term, v in zip(terms, row[5:])) + "\n" + "-" * 80)
            if row[3] < self.best_pv:
                self.best_pv = row[3]
                self._save_checkpoint(stem + "_best_pv_checkpoint.pth", epoch, True, val_loss)
            if val_loss < self.best_loss - get("min_delta", 1e-4):
                self.best_loss, stagnation = val_loss, 0
                self._save_checkpoint(stem + "_checkpoint.pth", epoch, True, val_loss)
            elif epoch >= get("early_stop_warmup", 60):
                stagnation += 1
                if stagnation >= get("patience", 15):
                    print(f"Early stopping at epoch {epoch}: best validation loss {self.best_loss:.4f}")
                    break
            self.vertex_scheduler.step()

    def downstream_end_to_end_one_epoch(self, pretrain=False):
        """One epoch over the training set: LearnedVertexHead optimised, the backbone frozen."""
        self.model.eval()
        self.learned_vertex.train()
        for batch in tqdm(self.train_data_loader, disable=not self.log_to_screen):
            loss, _, residual = self._learned_vertex_loss(batch, pretrain)
            self.vertex_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.learned_vertex.head.parameters(), max_norm=1.0)
            self.vertex_optimizer.step()
            self.down_results["train"].append(loss.item())
            self.down_results["train_residual"].append(residual)

    @torch.no_grad()
    def validate_end_to_end_one_epoch(self, pretrain=False):
        """The validation set (its first learned_vertex_val_events events, default all); returns the mean loss (of
        the final output: no aux terms)."""
        self.model.eval()
        self.learned_vertex.eval()
        max_events = getattr(self.params, "learned_vertex_val_events", None)
        for batch in tqdm(self.val_data_loader, disable=not self.log_to_screen):
            loss, terms, residual = self._learned_vertex_loss(batch, pretrain, aux=False)
            self.down_results["val"].append(loss.item())
            self.down_results["val_residual"].append(residual)
            self.down_results["val_events"] += batch[0].size(0)
            for term, value in terms.items():
                self.down_results[term].append(value)
            if max_events and self.down_results["val_events"] >= max_events:
                break
        return float(np.mean(self.down_results["val"]))

    # ------------------------------------------------------------------------------------------
    # VertexHead evaluation
    # ------------------------------------------------------------------------------------------

    def setup_vertexing(self, track_ckpt):
        """
        Load the track head and build VertexHead (after launch()). From params: vertex_detector (detector file),
        vertex_device, and any VertexHead argument as vertex_<argument> (e.g. vertex_Bz, vertex_learn_weights), each
        only where set.
        """
        n_feat = 1 if getattr(self, "cache_combined", False) else self.params.num_layers_backbone
        self.down_model = MambaAttentionHead(
            input_dim=self.params.embed_dim, num_layers=0,
            num_embedder_layers=self.params.num_embedder_layers,
            d_state=64, d_conv=4, expand=2, num_feature_layers=n_feat,
            num_prototypes=self.params.max_gt_classes,
            embed_method=getattr(self.params, "embed_method", "add")).to(self.device)

        # strict: a mismatched checkpoint raises instead of half-loading
        self.down_model.load_state_dict(self._load_checkpoint_state(track_ckpt, "model_state_dict")[1])
        self.down_model.eval()

        # VertexHead arguments set in params as vertex_<argument>
        names = VertexHead.setting_names()
        settings = {name: getattr(self.params, "vertex_" + name) for name in names
                    if getattr(self.params, "vertex_" + name, None) is not None}
        # VertexHead's device (vertex_device, e.g. cuda or cpu; default: the trainer's)
        self.vertex_device = torch.device(getattr(self.params, "vertex_device", None) or self.device)
        detector = getattr(self.params, "vertex_detector", None)
        head = VertexHead.from_detector(detector, **settings) if detector else VertexHead(**settings)
        self.vertex_head = head.to(self.vertex_device).eval()
        self.vertex_head.tracks.start_acts()       # boots while the first batches run

    def _vertex_head_chunk(self, records):
        """
        VertexHead once over several loader batches, padded to a common size; returns each batch's
        (vertex_estimate, fit_is_valid, per event: usable, acts_failed, acts_refit_failed (0 without) track counts and
        vertex_from_fallback). Each event keeps only the track slots VertexHead can fit, those at least
        min_hits real hits are assigned to (vertexing.assign_hits), and the hits of the others stay unassigned: the
        same result, without the track head's mostly empty max_gt_classes slots. VertexHead runs on vertex_device;
        the results come back to the CPU.
        """
        events = []
        for r in records:
            for e in range(r["mask"].shape[0]):
                probs, real = r["mask_probs"][e].float(), r["mask"][e]
                best, assigned = vertexing.assign_hits(probs, real)
                keep = torch.bincount(best[assigned], minlength=probs.shape[-1]) >= self.vertex_head.tracks.min_hits
                slots = keep.nonzero(as_tuple=True)[0]
                probs = probs * keep[best].unsqueeze(-1)    # hits on the other slots: unassigned
                events.append((r["class_probs"][e, slots].float(), probs[:, slots], r["grouped"][e], real))
        n_hits = max(len(real) for *_, real in events)
        n_tracks = max(1, max(len(c) for c, *_ in events))
        pad = lambda t, dims, value=0.0: torch.nn.functional.pad(t, dims, value=value)  # noqa: E731
        d = self.vertex_device
        result = self.vertex_head(
            class_probs=torch.stack([pad(c, (0, 0, 0, n_tracks - len(c))) for c, *_ in events]).to(d),
            mask_probs=torch.stack([pad(m, (0, n_tracks - m.shape[1], 0, n_hits - m.shape[0])) for _, m, *_ in events]).to(d),
            points=torch.stack([pad(p, (0, 0, 0, n_hits - len(p))) for *_, p, _ in events]).to(d),
            padding_mask=torch.stack([pad(real, (0, n_hits - len(real)), value=False) for *_, real in events]).to(d))
        sizes = [r["mask"].shape[0] for r in records]
        zeros = torch.zeros_like(result["track_quality_mask"])
        counts = torch.stack([result.get(k, zeros).sum(dim=-1) for k in ("track_quality_mask", "acts_failed",
                                                                          "acts_refit_failed")]
                             + [result["vertex_from_fallback"].to(zeros.dtype)], dim=-1)
        return list(zip(result["vertex_estimate"].cpu().split(sizes), result["fit_is_valid"].cpu().split(sizes),
                        counts.cpu().split(sizes)))

    def inference(self, checkpoint_path, max_events=None, pretrain=True, save_csv=False, csv_output_path=None):
        """
        setup_vertexing, then run_vertexing over at most max_events (None: all); appends the metrics to the run log,
        writes the per-point CSV with save_csv (csv_output_path, default the run folder's data/) and, unless
        no_output, the metrics JSON (metrics/: inputs, the track settings (TrackBuilder.track_fit_config), metrics).

        Args:
            checkpoint_path: track-head checkpoint; pretrain: backbone features (True) or raw points.
            save_csv, csv_output_path: the per-point CSV.
        """
        if self.world_size > 1:
            raise RuntimeError("the vertex eval runs in one process (its metrics are not gathered across ranks): "
                               "run it with python, not torchrun")
        self.setup_vertexing(checkpoint_path)
        result = self.run_vertexing(pretrain=pretrain, max_events=max_events, save_csv=save_csv)
        if self.world_rank == 0:
            write_metrics_log(result["metrics"], self.logfile)
            if save_csv:
                result["csv_output_path"] = csv_output_path or self.csv_path
                write_per_point_csv(result["per_point"], result["csv_output_path"])
                print(f"[vertex] wrote {result['csv_output_path']} ({len(result['per_point']['batch_idx'])} points)")
            if not getattr(self.params, "no_output", False):
                with open(self.metrics_path, "w") as f:
                    json.dump({"config": self.config, "yaml_config": os.path.abspath(self.params._yaml_filename),
                               "pretrained_ckpt": self.params.pretrained_ckpt, "track_ckpt": os.path.abspath(checkpoint_path),
                               "track_fit_config": self.vertex_head.tracks.track_fit_config(), "max_eval_events": max_events,
                               "metrics": result["metrics"]}, f, indent=2)
                print(f"[vertex] wrote {self.logfile}\n[vertex] wrote {self.metrics_path}")
        return result

    def _layer_features(self, pre_embed):
        """
        torch.stack(pre_embed), collapsed with eval_combine_weights when set (a head trained on a
        --combine_layers cache).
        """
        feature = torch.stack(pre_embed)
        weights = getattr(self, "eval_combine_weights", None)
        if weights is None:
            return feature
        return torch.einsum("lbnd,l->bnd", feature, feature.new_tensor(weights)).unsqueeze(0)

    @torch.no_grad()
    def run_vertexing(self, pretrain=True, max_events=None, save_csv=False):
        """
        Backbone and track head per loader batch; truth labels and VertexHead per vertex_chunk_events events (default
        32), with the metrics inputs and CSV rows, in a background thread on CPU copies while the GPU runs the next
        batches.

        Args:
            pretrain: backbone features (True) or raw points; max_events: limit (None: all); save_csv: build the CSV.

        Returns:
            dict: metrics (incl. time_<stage>_ms_per_event), vertex_estimate, vertex_target (n, 3), fit_is_valid (n,)
            (the fit is valid and the truth PV is defined),
            timing (seconds busy per stage; truth, vertex_head and csv overlap the others), per_point (with
            save_csv).
        """
        self.model.eval()
        # full fp32: no TF32 in matrix multiplies or convolutions
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        est, tgt, reco_valid, truth_valid, track_counts, per_point = [], [], [], [], [], []
        chunk_events = int(getattr(self.params, "vertex_chunk_events", None) or 32)
        pending = []        # loader batches waiting for VertexHead (CPU copies)
        job = [None]
        # VertexHead on the GPU queues on its own stream, so the main loop's syncs and copies do not wait for it
        stream = torch.cuda.Stream(self.vertex_device) if self.vertex_device.type == "cuda" else None

        @torch.no_grad()        # grad mode is per thread: this one runs in the pool
        def finish(records):
            """Truth labels and VertexHead over these batches, then their metrics inputs and CSV rows (the background
            thread)."""
            t0 = time.perf_counter()
            for r in records:
                r["truth"] = self._truth_labels(r["reg"], r["label"], r["mask"])
            t1 = time.perf_counter()
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                fitted = self._vertex_head_chunk(records)
            t2 = time.perf_counter()
            for r, (vertex, valid, counts) in zip(records, fitted):
                if save_csv:
                    truth_vertex = r["truth"]["vertex_target"].masked_fill(~r["truth"]["vertex_valid"][:, None],
                                                                           float("nan"))
                    per_point.append(per_point_columns(r["first_event"], r["grouped"], r["label"], r["reg"], r["mask"],
                                                       r["class_probs"], r["mask_probs"], vertex, truth_vertex,
                                                       r["truth"]["hit_vertex"]))
                est.append(vertex)
                tgt.append(r["truth"]["vertex_target"])
                reco_valid.append(valid)
                truth_valid.append(r["truth"]["vertex_valid"])
                track_counts.append(counts)
            timing["truth"] += t1 - t0
            timing["vertex_head"] += t2 - t1
            timing["csv"] += time.perf_counter() - t2

        def flush():
            """Hands the pending batches to the background thread once it has finished the previous chunk."""
            if job[0] is not None:
                job[0].result()
                job[0] = None
                lap("waiting")
            if pending:
                job[0] = pool.submit(finish, list(pending))
                pending.clear()

        stages = ("data", "backbone", "track_head", "to_cpu", "waiting", "truth", "vertex_head", "csv")
        timing = dict.fromkeys(stages, 0.0)
        vertexing.ACTS_TIMING.update(start_s=0.0, propagate_s=0.0, states=0, restarts=0)
        # the main loop's own stream only (VertexHead runs on another)
        sync = torch.cuda.current_stream().synchronize if torch.cuda.is_available() else (lambda: None)
        clock = [time.perf_counter()]
        start, n_seen = clock[0], 0

        def lap(stage):
            sync()
            now = time.perf_counter()
            timing[stage] += now - clock[0]
            clock[0] = now

        with ThreadPoolExecutor(max_workers=1) as pool:     # shut down on errors too
            for batch in tqdm(self.val_data_loader):
                if len(batch) != 4:
                    raise RuntimeError("loader must yield (grouped, label, knearest, reg): the test set with "
                                       "params.return_reg_test = True before launch(), a training set with its "
                                       "dataset's return_reg = True")
                grouped, label, _, reg = batch
                if max_events:      # the last batch: only the events still wanted
                    grouped, label, reg = (t[:max_events - n_seen] for t in (grouped, label, reg))
                b, c = grouped.size(0), grouped.size(-1)
                grouped = grouped.reshape(b, -1, c)
                mask = grouped[..., 0] != PAD_VALUE
                inputs, input_mask = grouped.to(self.device), mask.to(self.device)
                lap("data")

                if pretrain:
                    _, pre_embed, _ = self.model(inputs, return_z=True)
                    lap("backbone")
                    out = self.down_model(inputs, self._layer_features(pre_embed), pretrain=True, padding_mask=input_mask)
                else:
                    out = self.down_model(inputs, feature=None)
                lap("track_head")

                pending.append({"first_event": n_seen, "grouped": grouped, "label": label, "reg": reg, "mask": mask,
                                "class_probs": out["class_probs"].cpu(), "mask_probs": out["mask_probs"].cpu()})
                lap("to_cpu")
                n_seen += b
                last = bool(max_events) and n_seen >= max_events
                if last or n_seen - pending[0]["first_event"] >= chunk_events:
                    flush()
                if last:
                    break
            flush()              # the loader ended with a partial chunk
            flush()              # and waits for it
        timing["wall"] = wall = time.perf_counter() - start

        est, tgt, reco_valid, truth_valid = torch.cat(est), torch.cat(tgt), torch.cat(reco_valid), torch.cat(truth_valid)
        ok = reco_valid & truth_valid
        metrics = compute_vertex_metrics(est, tgt, ok)
        # fit_valid_rate: valid fits among the events with a truth PV (the accuracies are over those fits)
        metrics.update(n_events=len(ok), truth_valid_rate=truth_valid.float().mean().item(),
                       fit_valid_rate=reco_valid[truth_valid].float().mean().item())
        counts = torch.cat(track_counts).double()
        usable, acts_failed, refit_failed, _ = counts.mean(dim=0).tolist()
        metrics.update(usable_tracks_per_event=usable)
        # the fallback PVs (found with a looser preselection): their share of the valid fits and both groups' rmse
        fallback, error = counts[:, 3].bool(), (est - tgt).double().norm(dim=-1)
        rmse = lambda sel: error[sel].pow(2).mean().sqrt().item() if sel.any() else float("nan")  # noqa: E731
        metrics.update(pv_fallback_rate=fallback[ok].double().mean().item() if ok.any() else float("nan"),
                       rmse_3d_finder=rmse(ok & ~fallback), rmse_3d_fallback=rmse(ok & fallback))
        if self.vertex_head.tracks.use_acts:
            metrics.update(acts_failed_per_event=acts_failed, acts_refit_failed_per_event=refit_failed,
                           acts_worker_restarts=vertexing.ACTS_TIMING["restarts"])
        timing["acts_start"] = vertexing.ACTS_TIMING["start_s"]
        timing["acts_propagate"] = vertexing.ACTS_TIMING["propagate_s"]
        n_events = max(len(ok), 1)
        metrics.update({f"time_{k}_ms_per_event": v / n_events * 1e3 for k, v in timing.items()})
        if self.log_to_screen:
            print("[vertex] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in metrics.items() if not k.startswith("time_")))
            print(f"[vertex] time over {len(ok)} events: {wall:.1f} s ({wall / n_events * 1e3:.1f} ms/event); busy "
                  f"time per stage (truth, vertex_head and csv run alongside the others; waiting: for them)")
            for k in stages:
                print(f"[vertex]   {k:14s} {timing[k]:8.2f} s  {timing[k] / n_events * 1e3:8.2f} ms/event  "
                      f"{timing[k] / max(wall, 1e-12):6.1%}")
            if timing["acts_start"] or timing["acts_propagate"]:
                print(f"[vertex]     of which ACTS: worker start {timing['acts_start']:.2f} s, propagation "
                      f"{timing['acts_propagate']:.2f} s ({vertexing.ACTS_TIMING['states']} track states)")
        output = {"metrics": metrics, "vertex_estimate": est.cpu(), "vertex_target": tgt.cpu(),
                  "fit_is_valid": ok.cpu(), "timing": timing}
        if save_csv:
            output["per_point"] = {k: np.concatenate([p[k] for p in per_point]) for k in CSV_COLUMNS}
        return output
