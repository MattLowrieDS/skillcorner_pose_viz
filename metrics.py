"""
SkillCorner Body Pose — Metrics
===============================
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from constants import ALL_JOINTS
from scipy.signal import find_peaks, savgol_filter


@dataclass(frozen=True)
class StepMetrics:
    """Peak-based step metrics for a single trajectory window.

    Attributes:
    -----------
    - steps_total: int
        Number of speed-gated step-separation peaks detected.
    - step_frames: np.ndarray of shape (steps_total,)
        Absolute frame indices of the detected peaks.
    - peak_separations_m: np.ndarray of shape (steps_total,)
        Smoothed horizontal foot-to-foot separation at each detected peak,
        i.e. the step length at maximum extension (near ground contact).
    - cadence_spm: float
        Mean cadence (steps per minute) across active step intervals.
        0.0 when fewer than two steps are detected.
    - peak_separation_mean_m: float
        Mean of ``peak_separations_m``; NaN when no peak was detected.
    - smoothed_separation_max_m: float
        Maximum of the smoothed foot-separation span; NaN when no smoothed
        span exists. Used as the zero-peak fallback by feature_eng.py.
    - foot_separation_max_m: float
        Maximum of the raw per-frame foot separation; NaN when no valid
        frame exists.
    """

    steps_total: int
    step_frames: np.ndarray
    peak_separations_m: np.ndarray
    cadence_spm: float
    peak_separation_mean_m: float
    smoothed_separation_max_m: float
    foot_separation_max_m: float


def _empty_step_metrics(foot_separation_max_m: float = float("nan")) -> StepMetrics:
    """Return a zero-step StepMetrics instance for empty/degenerate inputs."""
    return StepMetrics(
        steps_total=0,
        step_frames=np.array([], dtype=int),
        peak_separations_m=np.array([], dtype=float),
        cadence_spm=0.0,
        peak_separation_mean_m=float("nan"),
        smoothed_separation_max_m=float("nan"),
        foot_separation_max_m=float(foot_separation_max_m),
    )


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
                valid = ~np.isnan(k[:, 0]) & ~np.isnan(k[:, 2])
                if np.any(valid):
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


def compute_step_metrics(
    joint_trajectories: Dict[str, np.ndarray],
    fps: float = 25.0,
    min_prominence_m: float = 0.07,
    min_speed_mps: float = 0.35,
    max_cadence_spm: float = 260.0,
    max_step_interval_s: float = 1.5,
    smooth_window_frames: int = 9,
    left_ankle_joint: str = "lAnkle",
    right_ankle_joint: str = "rAnkle",
    left_heel_joint: str = "lHeel",
    right_heel_joint: str = "rHeel",
    height_threshold_m: Optional[float] = None,
) -> StepMetrics:
    """
    Detect steps and compute peak step separations, step frames, and cadence
    (steps per minute) from 3D body pose trajectories.

    Step events are identified by detecting periodic foot separation peaks
    (strikes) in the horizontal pitch plane, filtered with Savitzky-Golay
    smoothing and gated by player displacement speed to reject stationary jitter.

    Parameters:
    -----------
    - joint_trajectories: Dict[str, np.ndarray]
        Mapping of joint name -> (n_frames, 3) array of [x, y, z] in metres.
    - fps: float, default=25.0
        Video tracking frame rate in frames per second.
    - min_prominence_m: float, default=0.07
        Minimum peak prominence (in metres) of the foot separation signal
        required to register a foot strike event.
    - min_speed_mps: float, default=0.35
        Minimum horizontal player velocity (m/s) at midHip required to count active
        locomotion steps (prevents stationary fidgeting from registering steps).
    - max_cadence_spm: float, default=260.0
        Physiological upper bound on cadence used to prevent double-counting.
    - max_step_interval_s: float, default=1.5
        Maximum duration between steps before locomotion is considered paused.
    - smooth_window_frames: int, default=9
        Window length for Savitzky-Golay signal smoothing (must be odd).
    - left_ankle_joint / right_ankle_joint: str
        Primary left/right foot joint names.
    - left_heel_joint / right_heel_joint: str
        Fallback left/right foot joint names used when the ankle is NaN.
    - height_threshold_m: float or None
        Maximum z (metres) allowed for foot contacts (passed to compute_stride_length).

    Returns:
    --------
    - StepMetrics
        Dataclass containing:
          * ``steps_total`` — number of speed-gated peaks detected;
          * ``step_frames`` — absolute frame indices of the detected peaks;
          * ``peak_separations_m`` — smoothed foot separation at each peak
            (step length at maximum extension);
          * ``cadence_spm`` — mean cadence across active intervals;
          * ``peak_separation_mean_m`` — mean of the peak separations (NaN
            when no peak was detected);
          * ``smoothed_separation_max_m`` — max of the smoothed span (zero-peak
            fallback; NaN when no smoothed span exists);
          * ``foot_separation_max_m`` — max of the raw per-frame separation.
    """
    if not joint_trajectories:
        return _empty_step_metrics()

    # Determine sequence length
    first_arr = next(iter(joint_trajectories.values()), None)
    if first_arr is None or len(first_arr) == 0:
        return _empty_step_metrics()
    n_frames = len(first_arr)

    # Compute horizontal foot-to-foot distance
    strides = compute_stride_length(
        joint_trajectories,
        left_ankle_joint=left_ankle_joint,
        right_ankle_joint=right_ankle_joint,
        left_heel_joint=left_heel_joint,
        right_heel_joint=right_heel_joint,
        height_threshold_m=height_threshold_m,
    )

    foot_separation_max_m = (
        float(np.nanmax(strides)) if np.any(~np.isnan(strides)) else float("nan")
    )

    valid_mask = ~np.isnan(strides)
    valid_indices = np.where(valid_mask)[0]

    # Need enough valid frames to detect steps
    if len(valid_indices) < max(5, smooth_window_frames):
        return _empty_step_metrics(foot_separation_max_m)

    # Calculate midHip horizontal speed for velocity gating if midHip is present
    mid_hip = joint_trajectories.get("midHip")
    speed_smooth = np.full(n_frames, np.nan, dtype=np.float64)
    if mid_hip is not None and len(mid_hip) == n_frames:
        mh = np.asarray(mid_hip, dtype=np.float64)
        mh_valid = ~np.isnan(mh[:, 0]) & ~np.isnan(mh[:, 1])
        if np.sum(mh_valid) >= 2:
            dx = np.full(n_frames, 0.0, dtype=np.float64)
            dy = np.full(n_frames, 0.0, dtype=np.float64)
            valid_mh_idx = np.where(mh_valid)[0]
            # Compute finite differences between consecutive valid midHip frames
            dt_frames = np.diff(valid_mh_idx)
            # Avoid division by zero
            dt_frames[dt_frames == 0] = 1
            inst_speeds = (
                np.sqrt(np.diff(mh[valid_mh_idx, 0]) ** 2 + np.diff(mh[valid_mh_idx, 1]) ** 2)
                / (dt_frames / fps)
            )
            raw_speeds = np.zeros(n_frames, dtype=np.float64)
            raw_speeds[valid_mh_idx[1:]] = inst_speeds
            raw_speeds[valid_mh_idx[0]] = inst_speeds[0] if len(inst_speeds) > 0 else 0.0

            # Smooth speed over 5 frames
            kernel = np.ones(min(5, len(raw_speeds))) / min(5, len(raw_speeds))
            speed_smooth = np.convolve(raw_speeds, kernel, mode="same")
        else:
            speed_smooth.fill(min_speed_mps)  # Fall back to passing if no hip tracked
    else:
        speed_smooth.fill(min_speed_mps)

    # Process contiguous segment of stride data
    first_v, last_v = valid_indices[0], valid_indices[-1]
    s_segment = strides[first_v : last_v + 1].copy()

    # Interpolate internal NaNs in segment
    seg_nan = np.isnan(s_segment)
    if np.any(seg_nan):
        x_axis = np.arange(len(s_segment))
        s_segment[seg_nan] = np.interp(x_axis[seg_nan], x_axis[~seg_nan], s_segment[~seg_nan])

    # Ensure smoothing window length is valid (must be odd and <= len(s_segment))
    w_len = smooth_window_frames if smooth_window_frames % 2 == 1 else smooth_window_frames - 1
    w_len = min(w_len, len(s_segment) if len(s_segment) % 2 == 1 else len(s_segment) - 1)

    if w_len >= 5:
        smoothed = savgol_filter(s_segment, window_length=w_len, polyorder=2)
    else:
        smoothed = s_segment

    smoothed_separation_max_m = (
        float(np.max(smoothed)) if len(smoothed) > 0 else float("nan")
    )

    # Minimum distance between steps based on maximum physiological cadence
    min_dist = max(3, int(np.floor(fps * 60.0 / max_cadence_spm)))

    peaks_rel, _ = find_peaks(smoothed, distance=min_dist, prominence=min_prominence_m)
    detected_frames = peaks_rel + first_v

    # Filter out peaks where player was stationary
    gated_mask = np.array(
        [
            np.isnan(speed_smooth[f]) or speed_smooth[f] >= min_speed_mps
            for f in detected_frames
        ],
        dtype=bool,
    )

    step_frames = np.array(detected_frames[gated_mask], dtype=int)
    total_steps = int(len(step_frames))

    if total_steps > 0:
        peak_separations_m = np.asarray(
            smoothed[peaks_rel[gated_mask]], dtype=np.float64
        )
        peak_separation_mean_m = float(np.mean(peak_separations_m))
    else:
        peak_separations_m = np.array([], dtype=np.float64)
        peak_separation_mean_m = float("nan")

    cadence_spm = 0.0
    if total_steps >= 2:
        # Calculate intervals and mean SPM across active locomotion intervals
        intervals_s = np.diff(step_frames) / fps
        valid_intervals = intervals_s[intervals_s <= max_step_interval_s]
        if len(valid_intervals) > 0:
            cadence_spm = float(round(np.mean(60.0 / valid_intervals), 2))

    return StepMetrics(
        steps_total=total_steps,
        step_frames=step_frames,
        peak_separations_m=peak_separations_m,
        cadence_spm=cadence_spm,
        peak_separation_mean_m=peak_separation_mean_m,
        smoothed_separation_max_m=smoothed_separation_max_m,
        foot_separation_max_m=foot_separation_max_m,
    )


def compute_num_steps(
    joint_trajectories: Dict[str, np.ndarray],
    fps: float = 25.0,
    min_prominence_m: float = 0.07,
    min_speed_mps: float = 0.35,
    max_cadence_spm: float = 260.0,
    max_step_interval_s: float = 1.5,
    smooth_window_frames: int = 9,
    left_ankle_joint: str = "lAnkle",
    right_ankle_joint: str = "rAnkle",
    left_heel_joint: str = "lHeel",
    right_heel_joint: str = "rHeel",
    height_threshold_m: Optional[float] = None,
) -> Tuple[int, np.ndarray, float]:
    """
    Calculate the number of steps and steps per minute (cadence in SPM)
    from 3D body pose trajectories.

    Thin backward-compatible wrapper around :func:`compute_step_metrics`;
    see that function for the parameter documentation.

    Returns:
    --------
    - total_steps: int
        Total number of valid steps detected in the sequence.
    - step_frames: np.ndarray of shape (total_steps,)
        0-indexed frame indices where each foot strike peak was detected.
    - mean_spm: float
        Average cadence in steps per minute across active step intervals.
        Returns 0.0 if fewer than 2 steps are detected or player was stationary.
    """
    metrics = compute_step_metrics(
        joint_trajectories,
        fps=fps,
        min_prominence_m=min_prominence_m,
        min_speed_mps=min_speed_mps,
        max_cadence_spm=max_cadence_spm,
        max_step_interval_s=max_step_interval_s,
        smooth_window_frames=smooth_window_frames,
        left_ankle_joint=left_ankle_joint,
        right_ankle_joint=right_ankle_joint,
        left_heel_joint=left_heel_joint,
        right_heel_joint=right_heel_joint,
        height_threshold_m=height_threshold_m,
    )
    return metrics.steps_total, metrics.step_frames, metrics.cadence_spm


def compute_3point_angle(
    p1: np.ndarray,
    p2_vertex: np.ndarray,
    p3: np.ndarray,
    radians: bool = False,
) -> np.ndarray:
    """
    Compute the 3D planar angle at vertex p2 between rays (p2 -> p1) and (p2 -> p3).

    The angle is computed directly in the 3D plane spanned by the three points,
    making it perspective-invariant and free from camera or pitch foreshortening.
    This function is generalized for any 3-point joint triad (e.g. hip-knee-ankle,
    shoulder-elbow-wrist).

    Parameters:
    -----------
    - p1: np.ndarray of shape (..., 3)
        Point 1 (e.g., hip or shoulder coordinates).
    - p2_vertex: np.ndarray of shape (..., 3)
        Vertex point where the angle is formed (e.g., knee or elbow coordinates).
    - p3: np.ndarray of shape (..., 3)
        Point 3 (e.g., ankle or wrist coordinates).
    - radians: bool, default=False
        If True, return the angle in radians [0, pi].
        If False (default), return the angle in degrees [0, 180].

    Returns:
    --------
    - angles: np.ndarray of shape matching leading dimensions of inputs
        Planar angle in degrees (or radians). Returns NaN where any coordinate is NaN
        or where either vector has zero length.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2_vertex, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    # Segment vectors radiating from vertex p2
    v1 = p1 - p2
    v2 = p3 - p2

    # Dot product and Euclidean lengths along the last axis
    dot = np.sum(v1 * v2, axis=-1)
    norm1 = np.linalg.norm(v1, axis=-1)
    norm2 = np.linalg.norm(v2, axis=-1)
    denom = norm1 * norm2

    # Safe division avoiding zero-norm warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = dot / denom
        cosine = np.clip(cosine, -1.0, 1.0)
        angles = np.arccos(cosine)

        if not radians:
            angles = np.degrees(angles)

        # Invalidate degenerate cases where either segment has zero length
        valid_denom = denom > 0
        angles = np.where(valid_denom, angles, np.nan)

    return angles


def compute_knee_angles(
    joint_trajectories: Dict[str, np.ndarray],
    radians: bool = False,
    angle_convention: str = "included",
    smooth_window_frames: Optional[int] = None,
    use_heel_fallback: bool = False,
    left_hip_joint: str = "lHip",
    left_knee_joint: str = "lKnee",
    left_ankle_joint: str = "lAnkle",
    left_heel_joint: str = "lHeel",
    right_hip_joint: str = "rHip",
    right_knee_joint: str = "rKnee",
    right_ankle_joint: str = "rAnkle",
    right_heel_joint: str = "rHeel",
) -> Dict[str, np.ndarray]:
    """
    Compute per-frame back-of-knee angle for both left and right legs in the
    angled 3D coordinate plane defined by the hip -> knee -> ankle joints.

    Parameters:
    -----------
    - joint_trajectories: Dict[str, np.ndarray]
        Mapping of joint name -> (n_frames, 3) array of [x, y, z] in metres.
    - radians: bool, default=False
        If True, return angles in radians.
        If False (default), return angles in degrees.
    - angle_convention: str, default="included"
        - "included": Posterior / back-of-knee angle.
            180 deg (or pi rad) at full extension (straight leg);
            decreases toward ~40 deg as the knee flexes.
        - "flexion": Biomechanical knee flexion angle.
            0 deg at full extension;
            increases up to ~140 deg as the knee flexes.
    - smooth_window_frames: Optional[int], default=None
        If provided (>= 3), applies a Savitzky-Golay filter to smooth angle
        trajectories over time.
    - use_heel_fallback: bool, default=False
        If True, uses the heel joint as a per-frame fallback when the ankle
        is NaN/occluded.
    - left/right_*_joint: str
        Names of the anatomical joints in joint_trajectories.

    Returns:
    --------
    - Dict[str, np.ndarray]:
        {"left": np.ndarray, "right": np.ndarray} where each array has shape
        (n_frames,) containing the computed knee angles.
    """
    if angle_convention not in ("included", "flexion"):
        raise ValueError(
            f"Unknown angle_convention: '{angle_convention}'. Must be 'included' or 'flexion'."
        )

    if not joint_trajectories:
        empty = np.array([], dtype=np.float64)
        return {"left": empty, "right": empty}

    # Determine sequence length
    first_arr = next(iter(joint_trajectories.values()), None)
    if first_arr is None or len(first_arr) == 0:
        empty = np.array([], dtype=np.float64)
        return {"left": empty, "right": empty}
    n_frames = len(first_arr)

    def _resolve_distal(ankle_name: str, heel_name: str) -> Optional[np.ndarray]:
        ankle_arr = joint_trajectories.get(ankle_name)
        if not use_heel_fallback:
            return ankle_arr
        heel_arr = joint_trajectories.get(heel_name)
        if ankle_arr is None:
            return heel_arr
        if heel_arr is None:
            return ankle_arr
        resolved = np.asarray(ankle_arr, dtype=np.float64).copy()
        h = np.asarray(heel_arr, dtype=np.float64)
        missing = np.isnan(resolved[:, 0])
        resolved[missing] = h[missing]
        return resolved

    def _compute_single_leg(hip_name: str, knee_name: str, ankle_name: str, heel_name: str) -> np.ndarray:
        hip = joint_trajectories.get(hip_name)
        knee = joint_trajectories.get(knee_name)
        distal = _resolve_distal(ankle_name, heel_name)

        if hip is None or knee is None or distal is None:
            return np.full(n_frames, np.nan, dtype=np.float64)

        h = np.asarray(hip, dtype=np.float64)
        k = np.asarray(knee, dtype=np.float64)
        d = np.asarray(distal, dtype=np.float64)

        angles = compute_3point_angle(h, k, d, radians=radians)

        if angle_convention == "flexion":
            full_ext = np.pi if radians else 180.0
            angles = full_ext - angles

        # Optional temporal smoothing
        if smooth_window_frames is not None and smooth_window_frames >= 3:
            valid_mask = ~np.isnan(angles)
            if np.sum(valid_mask) >= smooth_window_frames:
                valid_idx = np.where(valid_mask)[0]
                first_v, last_v = valid_idx[0], valid_idx[-1]
                seg = angles[first_v : last_v + 1].copy()
                seg_nan = np.isnan(seg)
                if np.any(seg_nan):
                    x_axis = np.arange(len(seg))
                    seg[seg_nan] = np.interp(x_axis[seg_nan], x_axis[~seg_nan], seg[~seg_nan])
                w_len = smooth_window_frames if smooth_window_frames % 2 == 1 else smooth_window_frames - 1
                w_len = min(w_len, len(seg) if len(seg) % 2 == 1 else len(seg) - 1)
                if w_len >= 3:
                    poly = min(2, w_len - 1)
                    smoothed_seg = savgol_filter(seg, window_length=w_len, polyorder=poly)
                    angles[first_v : last_v + 1] = np.where(
                        valid_mask[first_v : last_v + 1], smoothed_seg, np.nan
                    )

        return angles

    left_angles = _compute_single_leg(left_hip_joint, left_knee_joint, left_ankle_joint, left_heel_joint)
    right_angles = _compute_single_leg(right_hip_joint, right_knee_joint, right_ankle_joint, right_heel_joint)

    return {"left": left_angles, "right": right_angles}


def compute_hip_orientation(
    joint_trajectories: Dict[str, np.ndarray],
    radians: bool = False,
    left_hip_joint: str = "lHip",
    right_hip_joint: str = "rHip",
    mid_hip_joint: str = "midHip",
) -> np.ndarray:
    """
    Compute per-frame pelvic orientation angle in the horizontal (x, y) pitch plane
    from the lHip -> midHip -> rHip anatomical connection.

    The pelvic lateral axis is directed from right hip to left hip:
    vector = lHip - rHip.
    If either hip is occluded or NaN, midHip is used as a per-frame fallback
    (e.g., 2 * (lHip - midHip) or 2 * (midHip - rHip)).

    Parameters:
    -----------
    - joint_trajectories: Dict[str, np.ndarray]
        Mapping of joint name -> (n_frames, 3) array of [x, y, z] in metres.
    - radians: bool, default=False
        If True, return angle in radians in range (-pi, pi].
        If False (default), return angle in degrees in range (-180, 180].
    - left_hip_joint / right_hip_joint / mid_hip_joint: str
        Names of the hip joints in joint_trajectories.

    Returns:
    --------
    - angles: np.ndarray of shape (n_frames,)
        Pelvic orientation angle per frame. NaN where hip joints cannot be resolved.
    """
    if not joint_trajectories:
        return np.array([], dtype=np.float64)

    first_arr = next(iter(joint_trajectories.values()), None)
    if first_arr is None or len(first_arr) == 0:
        return np.array([], dtype=np.float64)
    n_frames = len(first_arr)

    l_hip = joint_trajectories.get(left_hip_joint)
    r_hip = joint_trajectories.get(right_hip_joint)
    m_hip = joint_trajectories.get(mid_hip_joint)

    vx = np.full(n_frames, np.nan, dtype=np.float64)
    vy = np.full(n_frames, np.nan, dtype=np.float64)

    l_arr = np.asarray(l_hip, dtype=np.float64) if l_hip is not None else None
    r_arr = np.asarray(r_hip, dtype=np.float64) if r_hip is not None else None
    m_arr = np.asarray(m_hip, dtype=np.float64) if m_hip is not None else None

    l_valid = (l_arr is not None) and (~np.isnan(l_arr[:, 0])) & (~np.isnan(l_arr[:, 1]))
    r_valid = (r_arr is not None) and (~np.isnan(r_arr[:, 0])) & (~np.isnan(r_arr[:, 1]))
    m_valid = (m_arr is not None) and (~np.isnan(m_arr[:, 0])) & (~np.isnan(m_arr[:, 1]))

    # Case 1: Both left and right hips valid
    both_valid = l_valid & r_valid
    if np.any(both_valid):
        vx[both_valid] = l_arr[both_valid, 0] - r_arr[both_valid, 0]
        vy[both_valid] = l_arr[both_valid, 1] - r_arr[both_valid, 1]

    # Case 2: Only left hip and midHip valid
    left_only = l_valid & (~r_valid) & m_valid
    if np.any(left_only):
        vx[left_only] = 2.0 * (l_arr[left_only, 0] - m_arr[left_only, 0])
        vy[left_only] = 2.0 * (l_arr[left_only, 1] - m_arr[left_only, 1])

    # Case 3: Only right hip and midHip valid
    right_only = (~l_valid) & r_valid & m_valid
    if np.any(right_only):
        vx[right_only] = 2.0 * (m_arr[right_only, 0] - r_arr[right_only, 0])
        vy[right_only] = 2.0 * (m_arr[right_only, 1] - r_arr[right_only, 1])

    valid = (~np.isnan(vx)) & (~np.isnan(vy))
    angles = np.full(n_frames, np.nan, dtype=np.float64)
    if np.any(valid):
        lengths = np.hypot(vx[valid], vy[valid])
        non_zero = lengths > 1e-6
        valid_indices = np.where(valid)[0][non_zero]
        raw_angles = np.arctan2(vy[valid_indices], vx[valid_indices])
        if radians:
            angles[valid_indices] = raw_angles
        else:
            angles[valid_indices] = np.degrees(raw_angles)

    return angles


def compute_turning_velocity(
    joint_trajectories: Dict[str, np.ndarray],
    fps: float = 25.0,
    radians: bool = False,
    absolute: bool = False,
    axis: str = "vertical",
    smooth_window_frames: Optional[int] = 7,
    left_hip_joint: str = "lHip",
    right_hip_joint: str = "rHip",
    mid_hip_joint: str = "midHip",
    neck_joint: str = "neck",
) -> np.ndarray:
    """
    Calculate player turning velocity (rate of rotation of the pelvic
    lHip -> midHip -> rHip connection around the midHip axis) in deg/s or rad/s.

    Supports rotation around either:
    - "vertical": Standard pitch-plane yaw turning velocity around the vertical
      axis (0, 0, 1) passing through midHip.
    - "spine": 3D anatomical body yaw turning velocity around the spine axis
      vector (midHip -> neck).

    Features:
    - Phase unwrapping to avoid artificial 360-degree boundary cut jumps.
    - Savitzky-Golay numerical derivative smoothing to filter out pose jitter.
    - Graceful fallback for single-hip occlusions using midHip.

    Parameters:
    -----------
    - joint_trajectories: Dict[str, np.ndarray]
        Mapping of joint name -> (n_frames, 3) array of [x, y, z] in metres.
    - fps: float, default=25.0
        Video tracking frame rate in frames per second.
    - radians: bool, default=False
        If True, return turning velocity in radians/second.
        If False (default), return turning velocity in degrees/second.
    - absolute: bool, default=False
        If True, return turning speed (magnitude >= 0).
        If False (default), return signed turning velocity (positive = counter-clockwise /
        turning left; negative = clockwise / turning right).
    - axis: str, default="vertical"
        Rotation axis: "vertical" (pitch z-axis) or "spine" (midHip -> neck axis).
    - smooth_window_frames: Optional[int], default=7
        Savitzky-Golay filter window length (must be odd and >= 3). If None or < 3,
        uses standard numerical gradient without smoothing.
    - left_hip_joint / right_hip_joint / mid_hip_joint / neck_joint: str
        Joint names in joint_trajectories.

    Returns:
    --------
    - turning_velocity: np.ndarray of shape (n_frames,)
        Turning velocity per frame. NaN where pose is missing or invalid.
    """
    if axis not in ("vertical", "spine"):
        raise ValueError(f"Unknown axis: '{axis}'. Must be 'vertical' or 'spine'.")

    if not joint_trajectories:
        return np.array([], dtype=np.float64)

    first_arr = next(iter(joint_trajectories.values()), None)
    if first_arr is None or len(first_arr) == 0:
        return np.array([], dtype=np.float64)
    n_frames = len(first_arr)

    dt = 1.0 / fps if fps > 0 else 1.0 / 25.0

    # Resolve 3D pelvic vector
    l_hip = joint_trajectories.get(left_hip_joint)
    r_hip = joint_trajectories.get(right_hip_joint)
    m_hip = joint_trajectories.get(mid_hip_joint)

    l_arr = np.asarray(l_hip, dtype=np.float64) if l_hip is not None else None
    r_arr = np.asarray(r_hip, dtype=np.float64) if r_hip is not None else None
    m_arr = np.asarray(m_hip, dtype=np.float64) if m_hip is not None else None

    l_valid = (l_arr is not None) and (~np.isnan(l_arr[:, 0])) & (~np.isnan(l_arr[:, 1]))
    r_valid = (r_arr is not None) and (~np.isnan(r_arr[:, 0])) & (~np.isnan(r_arr[:, 1]))
    m_valid = (m_arr is not None) and (~np.isnan(m_arr[:, 0])) & (~np.isnan(m_arr[:, 1]))

    v_3d = np.full((n_frames, 3), np.nan, dtype=np.float64)

    both = l_valid & r_valid
    if np.any(both):
        v_3d[both] = l_arr[both] - r_arr[both]

    left_only = l_valid & (~r_valid) & m_valid
    if np.any(left_only):
        v_3d[left_only] = 2.0 * (l_arr[left_only] - m_arr[left_only])

    right_only = (~l_valid) & r_valid & m_valid
    if np.any(right_only):
        v_3d[right_only] = 2.0 * (m_arr[right_only] - r_arr[right_only])

    v_valid = ~np.isnan(v_3d[:, 0]) & ~np.isnan(v_3d[:, 1])

    turning_velocity = np.full(n_frames, np.nan, dtype=np.float64)

    if not np.any(v_valid):
        return turning_velocity

    if axis == "vertical":
        raw_angles = np.full(n_frames, np.nan, dtype=np.float64)
        raw_angles[v_valid] = np.arctan2(v_3d[v_valid, 1], v_3d[v_valid, 0])

        valid_indices = np.where(~np.isnan(raw_angles))[0]
        if len(valid_indices) < 2:
            return turning_velocity

        # Split into contiguous segments (gaps > 2 frames treated as separate runs)
        split_points = np.where(np.diff(valid_indices) > 2)[0]
        segment_ranges = []
        start_i = 0
        for sp in split_points:
            segment_ranges.append((valid_indices[start_i], valid_indices[sp]))
            start_i = sp + 1
        segment_ranges.append((valid_indices[start_i], valid_indices[-1]))

        for seg_start, seg_end in segment_ranges:
            seg_len = seg_end - seg_start + 1
            if seg_len < 2:
                continue

            seg_angles = raw_angles[seg_start : seg_end + 1].copy()
            seg_mask = ~np.isnan(seg_angles)
            if np.sum(seg_mask) < 2:
                continue

            if np.any(~seg_mask):
                x_ax = np.arange(seg_len)
                unwrapped_known = np.unwrap(seg_angles[seg_mask])
                seg_unwrapped = np.interp(x_ax, x_ax[seg_mask], unwrapped_known)
            else:
                seg_unwrapped = np.unwrap(seg_angles)

            w_len = smooth_window_frames if smooth_window_frames is not None else 0
            if w_len >= 3:
                w_len = w_len if w_len % 2 == 1 else w_len - 1
                w_len = min(w_len, seg_len if seg_len % 2 == 1 else seg_len - 1)

            if w_len >= 3:
                poly = min(2, w_len - 1)
                seg_vel = savgol_filter(seg_unwrapped, window_length=w_len, polyorder=poly, deriv=1, delta=dt)
            else:
                seg_vel = np.gradient(seg_unwrapped, dt)

            seg_vel[~seg_mask] = np.nan
            turning_velocity[seg_start : seg_end + 1] = seg_vel

    else:
        # axis == "spine": rotation around spine vector (midHip -> neck)
        neck = joint_trajectories.get(neck_joint)
        neck_arr = np.asarray(neck, dtype=np.float64) if neck is not None else None

        axis_vecs = np.zeros((n_frames, 3), dtype=np.float64)
        axis_vecs[:, 2] = 1.0  # default vertical

        if neck_arr is not None and m_arr is not None:
            spine = neck_arr - m_arr
            spine_lens = np.linalg.norm(spine, axis=1, keepdims=True)
            valid_spine = (~np.isnan(spine_lens[:, 0])) & (spine_lens[:, 0] > 1e-4)
            if np.any(valid_spine):
                axis_vecs[valid_spine] = spine[valid_spine] / spine_lens[valid_spine]

        # Project hip vector onto plane orthogonal to axis_vecs
        dot_a = np.sum(v_3d * axis_vecs, axis=1, keepdims=True)
        v_perp = v_3d - dot_a * axis_vecs
        perp_lens = np.linalg.norm(v_perp, axis=1, keepdims=True)
        valid_u = v_valid & (perp_lens[:, 0] > 1e-4)

        u = np.full((n_frames, 3), np.nan, dtype=np.float64)
        u[valid_u] = v_perp[valid_u] / perp_lens[valid_u]

        valid_indices = np.where(valid_u)[0]
        if len(valid_indices) < 2:
            return turning_velocity

        split_points = np.where(np.diff(valid_indices) > 2)[0]
        segment_ranges = []
        start_i = 0
        for sp in split_points:
            segment_ranges.append((valid_indices[start_i], valid_indices[sp]))
            start_i = sp + 1
        segment_ranges.append((valid_indices[start_i], valid_indices[-1]))

        for seg_start, seg_end in segment_ranges:
            seg_len = seg_end - seg_start + 1
            if seg_len < 2:
                continue

            seg_u = u[seg_start : seg_end + 1]
            seg_axis = axis_vecs[seg_start : seg_end + 1]
            seg_mask = ~np.isnan(seg_u[:, 0])
            if np.sum(seg_mask) < 2:
                continue

            dtheta = np.zeros(seg_len, dtype=np.float64)
            for i in range(seg_len - 1):
                if seg_mask[i] and seg_mask[i + 1]:
                    avg_axis = seg_axis[i] + seg_axis[i + 1]
                    norm_a = np.linalg.norm(avg_axis)
                    if norm_a > 1e-4:
                        avg_axis = avg_axis / norm_a
                    else:
                        avg_axis = np.array([0.0, 0.0, 1.0])
                    cross_u = np.cross(seg_u[i], seg_u[i + 1])
                    sin_dt = np.dot(cross_u, avg_axis)
                    cos_dt = np.dot(seg_u[i], seg_u[i + 1])
                    dtheta[i + 1] = np.arctan2(sin_dt, cos_dt)

            cum_theta = np.cumsum(dtheta)

            w_len = smooth_window_frames if smooth_window_frames is not None else 0
            if w_len >= 3:
                w_len = w_len if w_len % 2 == 1 else w_len - 1
                w_len = min(w_len, seg_len if seg_len % 2 == 1 else seg_len - 1)

            if w_len >= 3:
                poly = min(2, w_len - 1)
                seg_vel = savgol_filter(cum_theta, window_length=w_len, polyorder=poly, deriv=1, delta=dt)
            else:
                seg_vel = np.gradient(cum_theta, dt)

            seg_vel[~seg_mask] = np.nan
            turning_velocity[seg_start : seg_end + 1] = seg_vel

    if not radians:
        turning_velocity = np.degrees(turning_velocity)

    if absolute:
        turning_velocity = np.abs(turning_velocity)

    return turning_velocity


