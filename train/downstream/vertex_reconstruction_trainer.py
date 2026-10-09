"""
Route A trainer/evaluator: attaches VertexHead to the OUTPUT of an
already-trained track-finding head.

VertexHead is now the closed-form geometric fit (see vertex_head.py). Its
default configuration (learn_weights=False) has ZERO learnable parameters --
it's a fixed linear-algebra solve, not a network -- so there is nothing to
train in that mode; this script becomes a pure evaluation loop and skips
optimizer/backward entirely. Set learn_weights=True on VertexHead (a small
MLP that reweights tracks before the same linear solve) if you want a
trainable variant; the loop below handles both without changes.

Three tiers, in order of "how much this task retrains":
    1. self.model        -- the foundation-model backbone (MambaGPT/etc).
                             Frozen. Always was, for every downstream task.
    2. self.track_model  -- the ALREADY-TRAINED MambaAttentionHead track
                             finder. Frozen HERE specifically for Route A:
                             we're building a new task on top of a solved
                             one, not re-learning it.
    3. self.down_model   -- VertexHead. Zero params by default; only
                             trainable at all if learn_weights=True.

DDP setup, full logging/checkpoint-rotation ceremony, and early-stopping
bookkeeping are trimmed for clarity -- port those back in from
track_finding_trainer.py's __init__/init_exp_dir if you need them, they
don't change for this task.
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from model import MambaAttentionHead
from train.downstream import track_finding_trainer
from vertex_head import VertexHead
from downstream_util import get_vertex_label
from loss import *
from cached_dataset import get_cached_data_loader


class VertexTrainer (track_finding_trainer.DownstreamTrainer):

    def __init__(self, params, args):
        self.params = params
        self.root_dir = args.root_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_to_screen = True
        self.iters = 0

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    def launch(self):
        """
        Loads the two frozen tiers (foundation backbone + trained track
        finder) and builds VertexHead. Call once, before train()/evaluate().
        """
        # --- Tier 1: frozen foundation-model backbone -------------------
        from fm4npp.models.mambagpt import MambaGPT  # or whichever variant params selects
        self.model = MambaGPT(
            embed_dim=self.params.embed_dim,
            num_layers=self.params.num_layers_backbone,
            d_state=self.params.d_state,
            embed_method=self.params.embed_method,
            pe_method=self.params.pe_method,
        ).to(self.device)
        self._load_weights_only(self.model, self.params.pretrained_ckpt, key="model_state")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # --- Tier 2: frozen, ALREADY-TRAINED track finder ----------------
        self.track_model = MambaAttentionHead(
            input_dim=self.params.embed_dim,
            num_feature_layers=self.params.num_layers_backbone,
            num_prototypes=self.params.max_gt_classes,
            embed_method=self.params.embed_method,
        ).to(self.device)
        self._load_weights_only(self.track_model, self.params.track_finder_ckpt, key="model_state_dict")
        self.track_model.eval()
        for p in self.track_model.parameters():
            p.requires_grad_(False)

        # --- Tier 3: the geometric vertex fit ----------------------------
        self.down_model = VertexHead(
            learn_weights=getattr(self.params, "vertex_learn_weights", False),
        ).to(self.device)

        trainable_params = [p for p in self.down_model.parameters() if p.requires_grad]
        print(f"Total trainable parameters in VertexHead: {sum(p.numel() for p in trainable_params)}")

        if trainable_params:
            self.down_optimizer = optim.AdamW(trainable_params, lr=self.params.max_lr, weight_decay=0.0001)
            self.down_scheduler = CosineAnnealingWarmupRestarts(
                self.down_optimizer,
                first_cycle_steps=200,
                max_lr=self.params.max_lr,
                min_lr=self.params.min_lr,
                warmup_steps=20,
            )
        else:
            # Zero-parameter default: nothing to optimize. train() below
            # detects this and just evaluates once instead of looping
            # epochs against an empty optimizer.
            self.down_optimizer = None
            self.down_scheduler = None
            if self.log_to_screen:
                print("VertexHead has no trainable parameters (learn_weights=False) -- "
                      "train() will run a single evaluation pass, not an optimization loop.")

        self.train_data_loader, self.val_data_loader = self._get_data_loaders()

        self.best_loss = np.inf
        self.down_results = {
            "train": [], "val": [], "chi2": [], "fit_valid_rate": [],
            "accuracy@0.5cm": [], "accuracy@1.0cm": [], "accuracy@2.0cm": [],
            "mae_x": [], "mae_y": [], "mae_z": [], "rmse_3d": [],
        }

    def _load_weights_only(self, module, ckpt_path, key):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in ckpt[key].items()}
        module.load_state_dict(state_dict)
        if self.log_to_screen:
            print(f"Loaded frozen weights for {module.__class__.__name__} from {ckpt_path}")

    def _get_data_loaders(self):
        # Wire this up to whatever get_data_loader(...) your repo already
        # uses for track finding -- the batch needs the same fields:
        # points/grouped (B,N,4) and reg (B,N,8) for get_vertex_label.
        raise NotImplementedError("reuse the existing get_data_loader(...) used by track_finding_trainer.py")

    # ------------------------------------------------------------------
    # Forward through the frozen tiers -> VertexHead
    # ------------------------------------------------------------------
    def _forward(self, grouped, mask):
        """
        grouped: (B, N, 4) raw points, mask: (B, N) bool (points[...,0] != -100)
        Returns VertexHead's output dict (vertex_estimate, chi_square, fit_is_valid, ...).
        """
        with torch.no_grad():
            _, pre_embed, _ = self.model(grouped, return_z=True)
            feature = torch.stack(pre_embed)  # (S, B, N, D)
            track_out = self.track_model(grouped, feature, pretrain=True, padding_mask=mask)

        # Only matters at all if learn_weights=True -- otherwise this call
        # has no parameters to take gradients w.r.t. anyway.
        vertex_out = self.down_model(
            class_probs=track_out["class_probs"],
            track_reg_result=track_out["track_reg_result"],
            mask_probs=track_out["mask_probs"],
            points=grouped,
            padding_mask=mask,
        )
        return vertex_out

    def _combined_valid(self, vertex_out, vertex_label):
        """
        An event only gets scored if BOTH the ground truth had a defined
        vertex (vertex_label['vertex_valid']) AND the geometric fit itself
        had enough independent tracks to solve
        (vertex_out['fit_is_valid']) -- vertex_estimate is NaN otherwise,
        which would silently corrupt a mean if included.
        """
        return vertex_label["vertex_valid"] & vertex_out["fit_is_valid"]

    # ------------------------------------------------------------------
    # Training loop (only exercises the optimizer if learn_weights=True)
    # ------------------------------------------------------------------
    def train(self, num_epochs=None):
        if self.down_optimizer is None:
            print("No trainable parameters -- running a single evaluation pass instead of training.")
            self._validate_one_epoch()
            self._report_epoch(epoch=0, train_loss=float("nan"), t0=time.time())
            return

        num_epochs = num_epochs or self.params.max_epochs
        for epoch in range(num_epochs):
            t0 = time.time()
            self._train_one_epoch()
            val_loss = self._validate_one_epoch()
            train_loss = np.mean(self.down_results["train"]) if self.down_results["train"] else float("nan")
            self._report_epoch(epoch, train_loss, t0)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint(epoch, is_best=True, loss=val_loss)

            self.down_scheduler.step()
            for k in self.down_results:
                self.down_results[k] = []

    def _report_epoch(self, epoch, train_loss, t0):
        val_loss = np.mean(self.down_results["val"])
        acc_1cm = np.mean(self.down_results["accuracy@1.0cm"])
        mae_x = np.mean(self.down_results["mae_x"])
        mae_y = np.mean(self.down_results["mae_y"])
        mae_z = np.mean(self.down_results["mae_z"])
        rmse_3d = np.mean(self.down_results["rmse_3d"])
        chi2 = np.mean(self.down_results["chi2"])
        fit_valid_rate = np.mean(self.down_results["fit_valid_rate"])

        # Same shape of per-epoch log line as track_finding_trainer.py's
        # "Epoch\tTrain_Loss\tVal_Loss\tARI\t...", with chi2 (a real
        # goodness-of-fit number, not a learned metric) and fit_valid_rate
        # (how often the event even had >=2 usable tracks) alongside the
        # regression metrics.
        print(
            f"Epoch {epoch}\tTrain_Loss {train_loss:.4f}\tVal_Loss {val_loss:.4f}\t"
            f"Acc@1cm {acc_1cm:.4f}\tMAE(x,y,z) ({mae_x:.3f},{mae_y:.3f},{mae_z:.3f})\t"
            f"RMSE_3D {rmse_3d:.4f}\tChi2 {chi2:.3f}\tFit_valid_rate {fit_valid_rate:.3f}\t"
            f"Time {time.time() - t0:.1f}s"
        )

    def _train_one_epoch(self):
        self.track_model.eval()
        self.model.eval()
        self.down_model.train()

        for i, inputdict in enumerate(tqdm(self.train_data_loader)):
            self.iters += 1
            grouped = inputdict["points"].to(self.device)
            b, c = grouped.size(0), grouped.size(-1)
            grouped = grouped.reshape(b, -1, c)
            mask = grouped[..., 0] != -100
            reg = inputdict["reg_target"].to(self.device)  # (B, N, 8)

            vertex_label = get_vertex_label(reg)
            vertex_out = self._forward(grouped, mask)
            valid = self._combined_valid(vertex_out, vertex_label)

            self.down_optimizer.zero_grad()
            losses = vertex_reg_loss(vertex_out["vertex_estimate"], vertex_label["vertex_target"], valid)
            loss = losses["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.down_model.parameters() if p.requires_grad], max_norm=1.0
            )
            self.down_optimizer.step()

            self.down_results["train"].append(loss.item())

    @torch.no_grad()
    def _validate_one_epoch(self):
        self.track_model.eval()
        self.model.eval()
        self.down_model.eval()

        for i, inputdict in enumerate(tqdm(self.val_data_loader)):
            grouped = inputdict["points"].to(self.device)
            b, c = grouped.size(0), grouped.size(-1)
            grouped = grouped.reshape(b, -1, c)
            mask = grouped[..., 0] != -100
            reg = inputdict["reg_target"].to(self.device)

            vertex_label = get_vertex_label(reg)
            vertex_out = self._forward(grouped, mask)
            valid = self._combined_valid(vertex_out, vertex_label)

            losses = vertex_reg_loss(vertex_out["vertex_estimate"], vertex_label["vertex_target"], valid)
            metrics = compute_vertex_metrics(vertex_out["vertex_estimate"], vertex_label["vertex_target"], valid)

            self.down_results["val"].append(losses["loss"].item())
            self.down_results["chi2"].append(vertex_out["chi_square"][valid].mean().item() if valid.any() else float("nan"))
            self.down_results["fit_valid_rate"].append(vertex_out["fit_is_valid"].float().mean().item())
            for k, v in metrics.items():
                self.down_results[k].append(v)

        return float(np.mean(self.down_results["val"]))

    # ------------------------------------------------------------------
    # Checkpointing -- only meaningful if learn_weights=True; harmless
    # (and small) to call either way.
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch, is_best, loss):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.down_model.state_dict(),
            "optimizer_state_dict": self.down_optimizer.state_dict() if self.down_optimizer else None,
            "scheduler_state_dict": self.down_scheduler.state_dict() if self.down_scheduler else None,
            "best_loss": self.best_loss,
            "current_loss": loss,
            "params": vars(self.params),
        }
        os.makedirs(self.params.checkpoint_dir, exist_ok=True)
        fname = "vertex_head_checkpoint.pth"
        torch.save(checkpoint, os.path.join(self.params.checkpoint_dir, fname))
        if self.log_to_screen:
            print(f"Saved {'best ' if is_best else ''}checkpoint at epoch {epoch} with loss {loss:.4f}")
