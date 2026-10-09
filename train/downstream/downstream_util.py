import numpy as np
import torch

def get_trackinfo_noiselabel(reg, noise_pt_threshold=0.06):
    """
    Extract track information from the reg tensor.
    reg: B X N X 8 tensor containing track information (
    'px': reg[..., 0],
    'py': reg[..., 1],
    'pz': reg[..., 2],
    'vtx_x': reg[..., 3],
    'vtx_y': reg[..., 4],
    'vtx_z': reg[..., 5],
    'q': reg[..., 6],
    'e': reg[..., 7],)
    Returns: dictionary of
        "track_info" : B X N X 4 tensor with track information (q/(pT + 1), theta, sin_phi, cos_phi)
        "valid_tracks" : B X N tensor indicating valid tracks (1 for valid, 0 for invalid) we require track's production vertex is within 1cm
        "noise_labels" : B X N tensor with noise labels (1 for noise, 0 for valid points)
    """
    # Extract the relevant columns from reg
    px = reg[..., 0] 
    py = reg[..., 1]
    pz = reg[..., 2]
    vtx_x = reg[..., 3]
    vtx_y = reg[..., 4]
    q = reg[..., 6]
    pt = torch.sqrt(px**2 + py**2)  # Calculate transverse momentum
    transformed_pt = q / (pt + 1)  # Add 1 to avoid division by zero
    theta = torch.atan2(pt, pz)  # Calculate theta angle
    sin_phi = py / pt  # Calculate sine of phi
    cos_phi = px / pt  # Calculate cosine of phi
    vtx_r = torch.sqrt(vtx_x**2 + vtx_y**2)  # Calculate radial distance of vertex
    valid_tracks = (vtx_r < 1.0).int()  # Check if vertex is within 1 cm radius
    noise_labels = (pt < noise_pt_threshold).long()  # Identify noise points based on transverse momentum threshold
    # Create the track_info tensor
    
    track_info = torch.stack(
        [transformed_pt, theta, sin_phi, cos_phi], 
        dim=-1
    )
    return {
        "track_info": track_info,  # B X N X 4 tensor with track information
        "valid_tracks": valid_tracks,  # B X N tensor indicating valid tracks
        "noise_labels": noise_labels  # B X N tensor with noise labels
    }

def get_pidlabel(pid):
    """
    Extract track information from the reg tensor.
    pid: B X N tensor containing particle IDs
    Returns: dictionary of
        "pid_class" : B X N tensor with particle class information (pi, k, p, e), 0 if not belong to any of these classes
    """
    # Extract the relevant columns from reg
    # pion abs(pid) == 211
    # kaon abs(pid) == 321
    # proton abs(pid) == 2212
    # electron abs(pid) == 11
    pid_class = torch.zeros_like(pid, dtype=torch.long)  # Initialize with zeros
    pid_class[pid.abs() == 211] = 1
    pid_class[pid.abs() == 321] = 2
    pid_class[pid.abs() == 2212] = 3
    pid_class[pid.abs() == 11] = 4
    return {
        "pid_class": pid_class,  # B X N tensor with particle class information, 0 if not belong to any of these classes
    }

def get_weakdecaylabel(mid):
    """
    Extract weak decay labels from the mid tensor.
    mid: B X N tensor containing mother IDs
    Returns: dictionary of
        "weak_decay_class" : B X N tensor with weak decay labels (k_0)
    """
    # Define weak decay mother IDs
    weak_decay_class = torch.zeros_like(mid, dtype=torch.long)  # Initialize with zeros
    weak_decay_class[mid == 130] = 1  # K0
    weak_decay_class[mid == 310] = 1
    return {
        "weak_decay_class": weak_decay_class,  # B X N tensor with weak decay labels (1 for K0, 0 otherwise)
    }

def get_vertex_label(reg, valid_tracks=None, valid_radius_cm=1.0, z_cut_cm=None):
    """
    Ground-truth vertex position for VertexHead: mean (vtx_x, vtx_y, vtx_z)
    over hits/tracks near the origin, aggregated per event.

    Args:
        reg: (B, N, 8) - per-hit regression target tensor (same columns as get_trackinfo_noiselabel)
        valid_tracks: (B, N) bool/int, optional - if None, recomputed exactly like
            get_trackinfo_noiselabel: vtx_r = sqrt(vtx_x^2 + vtx_y^2) < valid_radius_cm
            (NOTE: transverse-only cut, same as the original)
        valid_radius_cm: transverse radius cut used when valid_tracks is None
        z_cut_cm: optional float - if set, ALSO requires |vtx_z| < z_cut_cm

    Returns:
        vertex_target: (B, 3) - mean (vtx_x, vtx_y, vtx_z) over hits/tracks flagged valid
            for that event; the regression target for VertexHead
        vertex_valid: (B,) bool - whether the event had >=1 valid hit to average (events
            with none should be excluded from the loss, not trained toward a meaningless
            zero-vector target)
        n_valid: (B,) long - how many hits contributed, for logging/debugging class imbalance
    """
    vtx_x = reg[..., 3]
    vtx_y = reg[..., 4]
    vtx_z = reg[..., 5]

    if valid_tracks is None:
        vtx_r = torch.sqrt(vtx_x ** 2 + vtx_y ** 2)
        valid_tracks = vtx_r < valid_radius_cm  # (B, N)
        if z_cut_cm is not None:
            valid_tracks = valid_tracks & (vtx_z.abs() < z_cut_cm)

    valid_f = valid_tracks.float()  # (B, N)
    n_valid = valid_f.sum(dim=-1)  # (B,)
    counts = n_valid.clamp(min=1)  # avoid div-by-zero; masked out via vertex_valid anyway

    vx = (vtx_x * valid_f).sum(dim=-1) / counts
    vy = (vtx_y * valid_f).sum(dim=-1) / counts
    vz = (vtx_z * valid_f).sum(dim=-1) / counts

    return {
        "vertex_target": torch.stack([vx, vy, vz], dim=-1),  # (B, 3)
        "vertex_valid": n_valid > 0,  # (B,)
        "n_valid": n_valid.long(),  # (B,)
    }
    
def get_silicon_tpc_match_mask(
    features,
    seg_target,
    silicon_r_min_cm=0.0,
    silicon_r_max_cm=15.0,
    tpc_r_min_cm=25.0,
    tpc_r_max_cm=100.0,
):
    """
    Mask Silicon hits whose truth track has no TPC hit.

    Silicon: 0 < R < 15 cm
    TPC:     25 < R < 100 cm

    Returns:
        keep_mask:              True for hits to keep.
        silicon_mask:           True for Silicon hits.
        unmatched_silicon_mask: True for Silicon hits with no TPC hit.
    """
    features = np.asarray(features)
    seg_target = np.asarray(seg_target)

    x = features[:, 1]
    y = features[:, 2]

    r = np.sqrt(x**2 + y**2)

    silicon_mask = ((r > silicon_r_min_cm) & (r < silicon_r_max_cm))

    tpc_mask = ((r > tpc_r_min_cm) & (r < tpc_r_max_cm))

    tpc_track_ids = np.unique(seg_target[tpc_mask])

    track_has_tpc = np.isin(seg_target, tpc_track_ids)

    unmatched_silicon_mask = silicon_mask & ~track_has_tpc

    keep_mask = np.ones(len(features), dtype=bool)

    keep_mask[unmatched_silicon_mask] = False

    return {
        "keep_mask": keep_mask,
        "silicon_mask": silicon_mask,
        "unmatched_silicon_mask": unmatched_silicon_mask,
    }