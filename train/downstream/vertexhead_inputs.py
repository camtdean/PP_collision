"""
vertex_head_inputs.py
=====================
Raw FM4NPP event data -> VertexHead.forward inputs (truth-level, batch size 1).

    load_event          .mmap dataset -> per-hit features, seg_target, reg_target
    build_truth_tracks  per-hit arrays -> per-truth-track tensors (td)
    apply_silicon_tpc_mask   optional: drop silicon hits whose track has no TPC hit
    vertex_head_inputs  td -> exactly the keyword arguments VertexHead.forward takes
    load_vertex_head_inputs  all of the above for one event

Usage:
    inputs, td = load_vertex_head_inputs(data_dir, event_idx)
    result = VertexHead(Bz=1.4)(**inputs)
"""
import os

import numpy as np
import torch

DEFAULT_DATA_DIR = os.path.expanduser("~/Storage/PP_collision/train/downstream/test/combined_0")
DEFAULT_EVENT_IDX = 1   # event 0 has only 10 hits; event 1 has ~1951 hits
DEFAULT_BZ = 1.4        # Tesla
NOISE_PT_THRESHOLD = 0.25


# reads FM4NPP .mmap dirs: data.ninja + dtype.ninja + starts/ and ends/ index arrays
class RaggedMmap:
    def __init__(self, mmap_dir: str, n_cols: int):
        self.n_cols = n_cols
        self.dtype = np.dtype(open(os.path.join(mmap_dir, "dtype.ninja")).read().strip())
        self.flat = np.memmap(os.path.join(mmap_dir, "data.ninja"), dtype=self.dtype, mode="r")
        self.starts = np.fromfile(os.path.join(mmap_dir, "starts", "data.ninja"), dtype=np.int64)
        self.ends = np.fromfile(os.path.join(mmap_dir, "ends", "data.ninja"), dtype=np.int64)
        assert len(self.starts) == len(self.ends), "starts/ends length mismatch"

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx: int) -> np.ndarray:
        data = np.array(self.flat[self.starts[idx]:self.ends[idx]])   # copy out of memmap
        return data if self.n_cols == 1 else data.reshape(-1, self.n_cols)


# features (E, x, y, z), seg_target (track ID), reg_target (px, py, pz, vx, vy, vz, q, e)
def load_event(data_dir: str, event_idx: int):
    features   = RaggedMmap(os.path.join(data_dir, "features.mmap"),   n_cols=4)[event_idx]
    seg_target = RaggedMmap(os.path.join(data_dir, "seg_target.mmap"), n_cols=1)[event_idx]
    reg_target = RaggedMmap(os.path.join(data_dir, "reg_target.mmap"), n_cols=8)[event_idx]
    return features.astype(np.float32), seg_target.astype(np.int32), reg_target.astype(np.float64)


# per-truth-track tensors (batch dim 1); every unique seg_target is a track, negative IDs included
def build_truth_tracks(features, seg_target, reg_target, noise_pt_threshold=NOISE_PT_THRESHOLD):
    unique_ids = np.unique(seg_target)
    n_hits, n_tracks = len(seg_target), len(unique_ids)
    id_to_col = {tid: col for col, tid in enumerate(unique_ids)}

    # all hits of a truth track share its reg_target row; take the first
    track_px  = np.zeros(n_tracks, dtype=np.float64)
    track_py  = np.zeros(n_tracks, dtype=np.float64)
    track_pz  = np.zeros(n_tracks, dtype=np.float64)
    track_vtx = np.zeros((n_tracks, 3), dtype=np.float64)
    track_q   = np.zeros(n_tracks, dtype=np.float64)
    track_seen = np.zeros(n_tracks, dtype=bool)
    mask_probs_np = np.zeros((n_hits, n_tracks), dtype=np.float32)

    for hit_idx, tid in enumerate(seg_target):
        col = id_to_col[int(tid)]
        mask_probs_np[hit_idx, col] = 1.0
        if not track_seen[col]:
            track_px[col]  = reg_target[hit_idx, 0]
            track_py[col]  = reg_target[hit_idx, 1]
            track_pz[col]  = reg_target[hit_idx, 2]
            track_vtx[col] = reg_target[hit_idx, 3:6]
            track_q[col]   = reg_target[hit_idx, 6]
            track_seen[col] = True

    pT    = np.sqrt(track_px**2 + track_py**2)
    phi   = np.arctan2(track_py, track_px)
    theta = np.arctan2(pT, track_pz)
    rho   = track_q / (pT + 1.0)

    # track_reg_result layout: (rho, theta, sin_phi, cos_phi), rho = q / (pT + 1)
    track_reg = np.stack([rho, theta, np.sin(phi), np.cos(phi)], axis=-1).astype(np.float32)

    # same noise/validity cuts as get_trackinfo_noiselabel
    vtx_r = np.sqrt(track_vtx[:, 0]**2 + track_vtx[:, 1]**2)
    noise = (pT < noise_pt_threshold).astype(np.int64)   # 1=noise, 0=good
    valid = (vtx_r < 1.0).astype(np.int32)               # 1=valid, 0=invalid

    class_probs = torch.zeros(1, n_tracks, 2, dtype=torch.float32)
    class_probs[..., 1] = 1.0                            # all truth tracks are real

    return {
        "track_ids":           unique_ids,
        "track_reg_result":    torch.from_numpy(track_reg).unsqueeze(0),
        "noise_labels":        torch.from_numpy(noise).unsqueeze(0),
        "valid_tracks":        torch.from_numpy(valid).unsqueeze(0),
        "mask_probs":          torch.from_numpy(mask_probs_np).unsqueeze(0),
        "points":              torch.from_numpy(features).float().unsqueeze(0),
        "padding_mask":        torch.ones(1, n_hits, dtype=torch.bool),
        "class_probs":         class_probs,
        "truth_vtx_per_track": torch.from_numpy(track_vtx.astype(np.float32)),
        "pT_per_track":        torch.from_numpy(pT.astype(np.float32)),
    }


# zeroes mask_probs rows of silicon hits whose track has no TPC hit, in place (apply=False: info only)
def apply_silicon_tpc_mask(td, features, seg_target, apply=True):
    from downstream_util import get_silicon_tpc_match_mask
    info = get_silicon_tpc_match_mask(features, seg_target)
    if apply:
        keep = torch.as_tensor(np.asarray(info["keep_mask"]), dtype=torch.bool)
        td["mask_probs"][0, ~keep, :] = 0.0
    return info


# exactly VertexHead.forward's keyword arguments; quality labels omitted when use_quality_mask=False
def vertex_head_inputs(td, use_quality_mask=True):
    return {
        "class_probs":      td["class_probs"],
        "track_reg_result": td["track_reg_result"],
        "mask_probs":       td["mask_probs"],
        "points":           td["points"],
        "padding_mask":     td["padding_mask"],
        "noise_labels":     td["noise_labels"] if use_quality_mask else None,
        "valid_tracks":     td["valid_tracks"] if use_quality_mask else None,
    }


def load_vertex_head_inputs(data_dir=DEFAULT_DATA_DIR, event_idx=DEFAULT_EVENT_IDX,
                            use_silicon_tpc_mask=False, use_quality_mask=True,
                            noise_pt_threshold=NOISE_PT_THRESHOLD):
    features, seg_target, reg_target = load_event(data_dir, event_idx)
    td = build_truth_tracks(features, seg_target, reg_target, noise_pt_threshold)
    if use_silicon_tpc_mask:
        td["silicon_info"] = apply_silicon_tpc_mask(td, features, seg_target)
    return vertex_head_inputs(td, use_quality_mask), td
