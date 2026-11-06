"""
SkillCorner Body Pose — Metrics
===============================
"""

import numpy as np
from typing import Dict, Optional
from constants import ALL_JOINTS

def compute_stride_length(
    joint_trajectories: Dict[str, np.ndarray],
    left_ankle_joint: str = "lAnkle",
    right_ankle_joint: str = "rAnkle",
    left_heel_joint: str = "lHeel",
    right_heel_joint: str = "rHeel",
    height_threshold_m: Optional[float] = None,
) -> np.ndarray:
    """
    Compute per-frame stride length (foot-to-foot Euclidean distance in the
    x/y pitch plane) from lower-body joint trajectory data.

    Ankle joints are used as the primary contact point; heel joints are used
    as a per-frame fallback when the corresponding ankle is NaN for that frame.

    Frames where either resolved foot's z-coordinate exceeds ``height_threshold_m``
    are excluded (NaN) to avoid measuring stride during high kicks.

    Parameters:
    -----------
    - joint_trajectories: dict[str, np.ndarray]
        Mapping of joint name -> (n_frames, 3) array of [x, y, z] in metres.
    - left_ankle_joint / right_ankle_joint: str
        Primary left/right foot joint names.
    - left_heel_joint / right_heel_joint: str
        Fallback left/right foot joint names used when the ankle is NaN.
    - height_threshold_m: float or None
        Maximum z (metres) allowed for either foot. If None, defaults to the
        mean z of all knee joints (lKnee + rKnee) across the sequence; if no
        valid knee data exists, no height filtering is applied.

    Returns:
    --------
    - stride_lengths: np.ndarray of shape (n_frames,)
        Stride length in metres per frame. NaN where either foot is missing
        or above the height threshold.
    """
    n_frames = len(joint_trajectories[ALL_JOINTS[0]])

    def resolve_foot(ankle_name: str, heel_name: str):
        ankle = joint_trajectories.get(ankle_name)
        heel = joint_trajectories.get(heel_name)
        if ankle is None and heel is None:
            return None
        xs = np.full(n_frames, np.nan)
        ys = np.full(n_frames, np.nan)
        zs = np.full(n_frames, np.nan)
        if ankle is not None:
            a = np.asarray(ankle, dtype=np.float64)
            valid = ~np.isnan(a[:, 0])
            xs[valid] = a[valid, 0]
            ys[valid] = a[valid, 1]
            zs[valid] = a[valid, 2]
        if heel is not None:
            h = np.asarray(heel, dtype=np.float64)
            fallback = np.isnan(xs)
            if np.any(fallback):
                xs[fallback] = h[fallback, 0]
                ys[fallback] = h[fallback, 1]
                zs[fallback] = h[fallback, 2]
        return xs, ys, zs

    left = resolve_foot(left_ankle_joint, left_heel_joint)
    right = resolve_foot(right_ankle_joint, right_heel_joint)

    stride_lengths = np.full(n_frames, np.nan, dtype=np.float64)

    if left is None or right is None:
        return stride_lengths

    lx, ly, lz = left
    rx, ry, rz = right

    if height_threshold_m is None:
        knee_zs = []
        for kj in ("lKnee", "rKnee"):
            kj_arr = joint_trajectories.get(kj)
            if kj_arr is not None:
                k = np.asarray(kj_arr, dtype=np.float64)
                valid = ~np.isnan(k[:, 0])
                knee_zs.append(k[valid, 2])
        if knee_zs:
            height_threshold_m = float(np.mean(np.concatenate(knee_zs)))
        else:
            height_threshold_m = np.inf

    both_valid = (~np.isnan(lx)) & (~np.isnan(ly)) & (~np.isnan(rx)) & (~np.isnan(ry))
    below_threshold = (lz <= height_threshold_m) & (rz <= height_threshold_m)
    valid = both_valid & below_threshold

    if np.any(valid):
        dx = lx[valid] - rx[valid]
        dy = ly[valid] - ry[valid]
        stride_lengths[valid] = np.sqrt(dx * dx + dy * dy)

    return stride_lengths
