"""
SkillCorner Body Pose Animation Tool
====================================
Animates 3D soccer body pose estimation data (29 anatomical joints) from
SkillCorner broadcast tracking .jsonl files.

Inspired by the Sport Performance Lab (SPL) basketball pose animation tool:
https://github.com/Sport-Performance-Lab/SPL-Open-Data/blob/main/basketball/freethrow/animate.py

Features:
- Full 29-joint anatomical skeleton connections for soccer biomechanics.
- Animate individual players or all visible players simultaneously on the pitch.
- Streamlined .jsonl streaming parser with fast seeking to maintain minimal memory footprint.
- Optional limb-side coloring (distinguishing left/right limbs, torso, and head).
- Integrated 3D ball animation when ball tracking data is present.
- Pitch markings & ground plane depth cues at ground level (z=0).
- Flexible camera view modes (dynamic player follow-mode, local bounding box, or whole pitch fixed view).
- CLI interface with flags for frame duration, all-players mode, whole-pitch view, player ID, and file export.
- Reusable Python function for Jupyter notebooks or custom analysis scripts.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from constants import ALL_JOINTS, CONNECTIONS, SIDE_COLORS
from data_loading import (
    get_line_offsets,
    load_pose_data,
    compute_stride_length,
    find_first_player_frame,
    locate_active_players,
    load_player_team_map,
)
from flags import build_arg_parser


def get_connection_color(
    connection: Tuple[str, str],
    default_color: str = "royalblue",
    color_by_side: bool = True,
) -> str:
    """Return the line color for a given anatomical joint pair."""
    if not color_by_side:
        return default_color

    j1, j2 = connection
    if j1.startswith("l") and j2.startswith("l"):
        return SIDE_COLORS["left"]
    if j1.startswith("r") and j2.startswith("r"):
        return SIDE_COLORS["right"]

    head_joints = {"nose", "neck", "lEye", "rEye", "lEar", "rEar"}
    if j1 in head_joints and j2 in head_joints:
        return SIDE_COLORS["head"]

    return SIDE_COLORS["center"]


def get_joint_color(
    joint_name: str,
    default_color: str = "royalblue",
    color_by_side: bool = True,
) -> str:
    """Return the marker color for an individual joint."""
    if not color_by_side:
        return default_color

    if joint_name.startswith("l"):
        return SIDE_COLORS["left"]
    if joint_name.startswith("r"):
        return SIDE_COLORS["right"]
    if joint_name in {"nose", "lEye", "rEye", "lEar", "rEar"}:
        return SIDE_COLORS["head"]

    return SIDE_COLORS["center"]


# ---------------------------------------------------------------------------
# Soccer Pitch Geometry (Ground Plane z=0)
# ---------------------------------------------------------------------------

def draw_soccer_pitch_3d(
    ax,
    line_color: str = "#27ae60",
    line_alpha: float = 0.65,
    line_width: float = 1.0,
    show_goals: bool = True,
):
    """
    Draw 3D soccer pitch markings at ground level (z=0) based on standard
    FIFA/SkillCorner pitch dimensions (105m x 68m, centered at (0, 0)).
    """
    # Outer pitch boundary
    pitch_x = [-52.5, 52.5, 52.5, -52.5, -52.5]
    pitch_y = [-34.0, -34.0, 34.0, 34.0, -34.0]
    ax.plot(pitch_x, pitch_y, [0] * 5, color=line_color, alpha=line_alpha, lw=line_width)

    # Halfway line
    ax.plot([0, 0], [-34.0, 34.0], [0, 0], color=line_color, alpha=line_alpha, lw=line_width)

    # Center circle (radius 9.15m)
    theta = np.linspace(0, 2 * np.pi, 80)
    ax.plot(
        9.15 * np.cos(theta),
        9.15 * np.sin(theta),
        np.zeros_like(theta),
        color=line_color,
        alpha=line_alpha,
        lw=line_width,
    )
    # Center spot
    ax.plot([0], [0], [0], "o", color=line_color, markersize=3, alpha=line_alpha)

    # Penalty areas (16.5m depth from goal line, 40.3m wide)
    # Left penalty area
    ax.plot(
        [-52.5, -36.0, -36.0, -52.5],
        [-20.15, -20.15, 20.15, 20.15],
        [0, 0, 0, 0],
        color=line_color,
        alpha=line_alpha,
        lw=line_width,
    )
    # Right penalty area
    ax.plot(
        [52.5, 36.0, 36.0, 52.5],
        [-20.15, -20.15, 20.15, 20.15],
        [0, 0, 0, 0],
        color=line_color,
        alpha=line_alpha,
        lw=line_width,
    )

    # Goal areas (5.5m depth, 18.3m wide)
    # Left goal area
    ax.plot(
        [-52.5, -47.0, -47.0, -52.5],
        [-9.15, -9.15, 9.15, 9.15],
        [0, 0, 0, 0],
        color=line_color,
        alpha=line_alpha,
        lw=line_width,
    )
    # Right goal area
    ax.plot(
        [52.5, 47.0, 47.0, 52.5],
        [-9.15, -9.15, 9.15, 9.15],
        [0, 0, 0, 0],
        color=line_color,
        alpha=line_alpha,
        lw=line_width,
    )

    # Penalty spots (11m from goal line)
    ax.plot([-41.5, 41.5], [0, 0], [0, 0], "o", color=line_color, markersize=3, alpha=line_alpha)

    # 3D Goal frames (7.32m wide, 2.44m high)
    if show_goals:
        goal_w = 3.66
        goal_h = 2.44
        for gx in [-52.5, 52.5]:
            # Posts
            ax.plot([gx, gx], [-goal_w, -goal_w], [0, goal_h], color=line_color, lw=line_width * 1.5, alpha=line_alpha)
            ax.plot([gx, gx], [goal_w, goal_w], [0, goal_h], color=line_color, lw=line_width * 1.5, alpha=line_alpha)
            # Crossbar
            ax.plot([gx, gx], [-goal_w, goal_w], [goal_h, goal_h], color=line_color, lw=line_width * 1.5, alpha=line_alpha)



# ---------------------------------------------------------------------------
# Core Animation Engine
# ---------------------------------------------------------------------------

def animate_soccer_pose(
    path_to_jsonl: str,
    start_frame: Optional[int] = None,
    num_frames: int = 100,
    player_id: Optional[int] = None,
    all_players: bool = False,
    connections: List[Tuple[str, str]] = CONNECTIONS,
    xbuffer: float = 2.5,
    ybuffer: float = 2.5,
    zlim: float = 2.5,
    elev: float = 20.0,
    azim: float = 45.0,
    player_color: str = "royalblue",
    player_lw: float = 2.5,
    color_by_side: bool = True,
    show_joints: bool = True,
    joint_size: float = 25.0,
    show_ball: bool = True,
    ball_color: str = "#e67e22",
    ball_size: float = 35.0,
    show_pitch: bool = True,
    show_hud: bool = True,
    follow_player: bool = True,
    whole_pitch: Optional[bool] = None,
    pitch_xlim: Tuple[float, float] = (-56.0, 56.0),
    pitch_ylim: Tuple[float, float] = (-38.0, 38.0),
    pitch_zlim: float = 15.0,
    max_error_cm: Optional[float] = None,
    output_path: Optional[str] = None,
    fps: int = 25,
    dpi: int = 100,
    notebook_mode: bool = False,
    no_axis: bool = False,
    offsets: Optional[np.ndarray] = None,
) -> FuncAnimation:
    """
    Animate 3D body pose tracking data for soccer players and save or display the animation.

    Parameters:
    -----------
    - path_to_jsonl: str
        The path to the SkillCorner .jsonl file containing 3D pose data.
    - start_frame: int, optional
        The starting frame number (0-indexed). Auto-detected if None.
    - num_frames: int
        The number of frames to animate (native dataset is 25 fps).
    - player_id: int, optional
        The SkillCorner player identifier (used when all_players=False). Auto-detected if None.
    - all_players: bool
        If True, animates all visible players present on the pitch for each frame.
    - connections: list of tuples
        List of (joint_1, joint_2) pairs defining the skeleton.
    - xbuffer: float
        Horizontal half-width buffer in metres around player hip (used in follow mode).
    - ybuffer: float
        Vertical half-height buffer in metres around player hip (used in follow mode).
    - zlim: float
        Height limit in metres above pitch for the z-axis (default: 2.5m).
    - elev: float
        Elevation view angle in degrees for the 3D plot.
    - azim: float
        Azimuth view angle in degrees for the 3D plot.
    - player_color: str
        Fallback uniform color for the player skeleton lines.
    - player_lw: float
        Line width for the player skeleton segments.
    - color_by_side: bool
        Whether to color left/right limbs differently for depth perception.
    - show_joints: bool
        Whether to draw scatter markers at anatomical joint positions.
    - joint_size: float
        Size of joint scatter markers.
    - show_ball: bool
        Whether to render the 3D ball trajectory.
    - ball_color: str
        Color of the soccer ball marker.
    - ball_size: float
        Marker size of the soccer ball.
    - show_pitch: bool
        Whether to draw soccer pitch field markings on the ground plane (z=0).
    - show_hud: bool
        Whether to show an on-screen HUD with frame, timestamp, and player stats.
    - follow_player: bool
        If True, camera pans dynamically to follow player/centroid.
        If False, keeps fixed bounds.
    - whole_pitch: bool, optional
        If True, fixes the camera to display the whole pitch.
        Defaults to True when all_players=True, and False when all_players=False.
    - pitch_xlim: tuple of (float, float)
        Fixed x-axis limits in metres for the whole pitch view (default: (-56.0, 56.0)).
    - pitch_ylim: tuple of (float, float)
        Fixed y-axis limits in metres for the whole pitch view (default: (-38.0, 38.0)).
    - pitch_zlim: float
        Fixed z-axis upper limit in metres for whole pitch view (default: 15.0m).
    - max_error_cm: float, optional
        Filter out joints with p90_mae_cm above this value.
    - output_path: str, optional
        If provided, saves the animation to this file (.mp4 or .gif).
    - fps: int
        Frame rate for the animation (default: 25).
    - dpi: int
        DPI resolution for the exported video/gif.
    - notebook_mode: bool
        If True, configures matplotlib for inline HTML5 rendering in Jupyter.
    - no_axis: bool
        If True, turns off the 3D coordinate pane axes via ax.set_axis_off().
    - offsets: np.ndarray, optional
        Pre-computed line byte-offsets array for instant O(1) seeking.

    Returns:
    --------
    - anim: matplotlib.animation.FuncAnimation
        The constructed matplotlib FuncAnimation object.
    """
    if notebook_mode:
        plt.rcParams["animation.html"] = "jshtml"

    # Default whole_pitch choice: True for all_players, False for single player
    if whole_pitch is None:
        whole_pitch = True if all_players else False

    # Load sequence from jsonl
    data = load_pose_data(
        jsonl_path=path_to_jsonl,
        start_frame=start_frame,
        num_frames=num_frames,
        player_id=player_id,
        all_players=all_players,
        max_error_cm=max_error_cm,
        offsets=offsets,
    )

    actual_frames = len(data["frames"])
    if actual_frames == 0:
        raise RuntimeError(f"No frames could be read from {path_to_jsonl}.")

    ball_data = data["ball_xyz"]
    ball_detected = data["ball_detected"]
    timestamps = data["timestamps"]
    frame_indices = data["frames"]

    # 3D Figure Setup
    actual_zlim = pitch_zlim if (whole_pitch and (zlim == 2.5 or zlim is None)) else zlim
    figsize = (13, 8) if whole_pitch else (9, 9)
    fig = plt.figure(figsize=figsize, facecolor="#1a1a1a")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a1a")

    ax.view_init(elev=elev, azim=azim)

    if whole_pitch:
        ax.set_xlim(pitch_xlim)
        ax.set_ylim(pitch_ylim)
        ax.set_zlim([0, actual_zlim])
        ax.set_box_aspect([pitch_xlim[1] - pitch_xlim[0], pitch_ylim[1] - pitch_ylim[0], 18.0])
    else:
        ax.set_zlim([0, zlim])

    if no_axis:
        ax.set_axis_off()
    else:
      # Clean transparent panes
      ax.xaxis.pane.fill = False
      ax.yaxis.pane.fill = False
      ax.zaxis.pane.fill = False
      ax.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.05))
      ax.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.05))
      ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.05))
      ax.grid(False)

      # Style coordinate axes
      ax.tick_params(colors="#888888", labelsize=8)
      ax.set_xlabel("Pitch X (m)", color="#aaaaaa", labelpad=10, fontsize=9)
      ax.set_ylabel("Pitch Y (m)", color="#aaaaaa", labelpad=10, fontsize=9)
      ax.set_zlabel("Height Z (m)", color="#aaaaaa", labelpad=6, fontsize=9)

    # Pitch markings
    if show_pitch:
        draw_soccer_pitch_3d(ax, line_color="#2ecc71", line_alpha=0.6, line_width=1.2)

    # HUD (Heads-Up Display)
    hud_text = None
    if show_hud:
        hud_text = ax.text2D(
            0.03, 0.95, "",
            transform=ax.transAxes,
            fontsize=10,
            fontfamily="monospace",
            color="#ffffff",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#111111",
                alpha=0.8,
                edgecolor="#333333",
            ),
        )

    # --- Mode-specific setup ---
    if all_players:
        all_players_frames = data["all_players_frames"]

        # Vectorized line collection for all skeletons
        eff_lw = player_lw if not whole_pitch else max(1.5, player_lw * 0.7)
        skeleton_coll = Line3DCollection([], linewidths=eff_lw)
        ax.add_collection(skeleton_coll)

        # Pre-allocate joint scatter markers
        joint_scatter_plots: Dict[str, Any] = {}
        if show_joints:
            groups = {
                "left": (SIDE_COLORS["left"] if color_by_side else player_color),
                "right": (SIDE_COLORS["right"] if color_by_side else player_color),
                "head": (SIDE_COLORS["head"] if color_by_side else player_color),
                "center": (SIDE_COLORS["center"] if color_by_side else player_color),
            }
            marker_size = np.sqrt(joint_size) if not whole_pitch else max(2.0, np.sqrt(joint_size) * 0.6)
            for g_name, g_color in groups.items():
                pts, = ax.plot(
                    [], [], [], "o",
                    markersize=marker_size,
                    color=g_color,
                    markeredgecolor="#ffffff",
                    markeredgewidth=0.4 if whole_pitch else 0.6,
                    alpha=0.95,
                )
                joint_scatter_plots[g_name] = pts

        # Ground position markers for all visible players
        ground_markers, = ax.plot(
            [], [], [], "o",
            markersize=max(3.0, np.sqrt(joint_size) * 0.75),
            color="#3498db",
            markeredgecolor="#ffffff",
            markeredgewidth=0.5,
            alpha=0.6,
        )

        # Ball marker
        ball_marker, = ax.plot(
            [], [], [], "o",
            markersize=np.sqrt(ball_size) * (1.2 if whole_pitch else 1.5),
            color=ball_color,
            markeredgecolor="#ffffff",
            markeredgewidth=1.0,
            zorder=10,
        )

        def update_frame(frame_i: int):
            players_in_frame = all_players_frames[frame_i]

            # 1. Update Camera Position & Bounds (when not whole pitch)
            if not whole_pitch:
                if follow_player and players_in_frame:
                    valid_coords = [[p["x"], p["y"]] for p in players_in_frame if not np.isnan(p.get("x", np.nan))]
                    if valid_coords:
                        cx = float(np.mean([c[0] for c in valid_coords]))
                        cy = float(np.mean([c[1] for c in valid_coords]))
                        ax.set_xlim([cx - xbuffer, cx + xbuffer])
                        ax.set_ylim([cy - ybuffer, cy + ybuffer])
                        ax.set_box_aspect([2 * xbuffer, 2 * ybuffer, actual_zlim])

            # 2. Update Skeletons and Joints
            segments = []
            seg_colors = []
            grouped_pts = {"left": [[], [], []], "right": [[], [], []], "head": [[], [], []], "center": [[], [], []]}
            ground_xs, ground_ys = [], []
            num_pose_players = 0

            for p in players_in_frame:
                px = p.get("x")
                py = p.get("y")
                if px is not None and py is not None and not np.isnan(px) and not np.isnan(py):
                    ground_xs.append(px)
                    ground_ys.append(py)

                joints = p.get("joints")
                if joints:
                    has_conn = False
                    for c in connections:
                        j1, j2 = c
                        if j1 in joints and j2 in joints:
                            p1 = joints[j1]
                            p2 = joints[j2]
                            if not np.isnan(p1[0]) and not np.isnan(p2[0]):
                                segments.append([p1, p2])
                                seg_colors.append(get_connection_color(c, default_color=player_color, color_by_side=color_by_side))
                                has_conn = True

                    if has_conn:
                        num_pose_players += 1

                    if show_joints:
                        for jname, pt in joints.items():
                            if not np.isnan(pt[0]):
                                if jname.startswith("l"):
                                    tg = "left"
                                elif jname.startswith("r"):
                                    tg = "right"
                                elif jname in {"nose", "neck", "lEye", "rEye", "lEar", "rEar"}:
                                    tg = "head"
                                else:
                                    tg = "center"
                                grouped_pts[tg][0].append(pt[0])
                                grouped_pts[tg][1].append(pt[1])
                                grouped_pts[tg][2].append(pt[2])

            skeleton_coll.set_segments(segments)
            skeleton_coll.set_color(seg_colors)

            if show_joints:
                for g_name, g_coords in grouped_pts.items():
                    joint_scatter_plots[g_name].set_data_3d(g_coords[0], g_coords[1], g_coords[2])

            ground_markers.set_data_3d(ground_xs, ground_ys, np.zeros(len(ground_xs)))

            # 3. Update Ball
            if show_ball and ball_detected[frame_i]:
                bx, by, bz = ball_data[frame_i]
                ball_marker.set_data_3d([bx], [by], [bz])
            else:
                ball_marker.set_data_3d([], [], [])

            # 4. HUD
            if hud_text:
                ts = timestamps[frame_i] or "--:--:--"
                f_idx = frame_indices[frame_i]
                hud_lines = [
                    "Mode: All Visible Players",
                    f"Frame: {f_idx} (Idx: {frame_i + 1}/{actual_frames})",
                    f"Clock: {ts}",
                    f"Visible Players: {len(players_in_frame)} ({num_pose_players} with 3D pose)",
                ]
                if show_ball and ball_detected[frame_i]:
                    bx, by, bz = ball_data[frame_i]
                    hud_lines.append(f"Ball: ({bx:.1f}m, {by:.1f}m, z={bz:.2f}m)")
                hud_text.set_text("\n".join(hud_lines))

    else:
        # Single Player Mode
        resolved_player_id = data["player_id"]
        joint_data = data["joint_trajectories"]
        stride_lengths = compute_stride_length(joint_data)

        # Local ground grid beneath player (depth perception)
        ground_grid_lines = []
        num_grid = 7
        if not whole_pitch:
            for _ in range(num_grid * 2):
                gl, = ax.plot([], [], [], color="#444444", lw=0.6, alpha=0.5, linestyle=":")
                ground_grid_lines.append(gl)

        # Pre-allocate Skeleton Connection Lines
        skeleton_lines: Dict[Tuple[str, str], Any] = {}
        for c in connections:
            c_color = get_connection_color(c, default_color=player_color, color_by_side=color_by_side)
            line, = ax.plot([], [], [], color=c_color, lw=player_lw, solid_capstyle="round")
            skeleton_lines[c] = line

        # Pre-allocate Joint Markers grouped by side
        joint_scatter_plots: Dict[str, Any] = {}
        if show_joints:
            groups = {
                "left": (SIDE_COLORS["left"] if color_by_side else player_color),
                "right": (SIDE_COLORS["right"] if color_by_side else player_color),
                "head": (SIDE_COLORS["head"] if color_by_side else player_color),
                "center": (SIDE_COLORS["center"] if color_by_side else player_color),
            }
            marker_size = np.sqrt(joint_size) if not whole_pitch else max(2.5, np.sqrt(joint_size) * 0.7)
            for g_name, g_color in groups.items():
                pts, = ax.plot(
                    [], [], [], "o",
                    markersize=marker_size,
                    color=g_color,
                    markeredgecolor="#ffffff",
                    markeredgewidth=0.4 if whole_pitch else 0.6,
                    alpha=0.95,
                )
                joint_scatter_plots[g_name] = pts

        # Ball marker
        ball_marker, = ax.plot(
            [], [], [], "o",
            markersize=np.sqrt(ball_size) * (1.2 if whole_pitch else 1.5),
            color=ball_color,
            markeredgecolor="#ffffff",
            markeredgewidth=1.0 if whole_pitch else 1.2,
            zorder=10,
        )

        # Compute static field bounds if fixed view is requested without whole_pitch
        fixed_xlim = None
        fixed_ylim = None
        if not follow_player and not whole_pitch:
            all_xs = joint_data["midHip"][:, 0]
            all_ys = joint_data["midHip"][:, 1]
            valid_x = all_xs[~np.isnan(all_xs)]
            valid_y = all_ys[~np.isnan(all_ys)]
            if len(valid_x) > 0 and len(valid_y) > 0:
                fixed_xlim = [np.min(valid_x) - xbuffer, np.max(valid_x) + xbuffer]
                fixed_ylim = [np.min(valid_y) - ybuffer, np.max(valid_y) + ybuffer]
            else:
                fixed_xlim = [-xbuffer, xbuffer]
                fixed_ylim = [-ybuffer, ybuffer]

        def update_frame(frame_i: int):
            # 1. Update Camera Position & Bounds
            mid_hip = joint_data["midHip"][frame_i]
            valid_centroid = not np.isnan(mid_hip[0])

            if whole_pitch:
                pass  # Fixed view of the whole pitch
            elif follow_player and valid_centroid:
                cx, cy = mid_hip[0], mid_hip[1]
                ax.set_xlim([cx - xbuffer, cx + xbuffer])
                ax.set_ylim([cy - ybuffer, cy + ybuffer])
                ax.set_box_aspect([2 * xbuffer, 2 * ybuffer, zlim])

                if show_pitch and ground_grid_lines:
                    gx = np.linspace(cx - xbuffer, cx + xbuffer, num_grid)
                    gy = np.linspace(cy - ybuffer, cy + ybuffer, num_grid)
                    for idx, x_val in enumerate(gx):
                        ground_grid_lines[idx].set_data_3d([x_val, x_val], [cy - ybuffer, cy + ybuffer], [0, 0])
                    for idx, y_val in enumerate(gy):
                        ground_grid_lines[num_grid + idx].set_data_3d([cx - xbuffer, cx + xbuffer], [y_val, y_val], [0, 0])
            elif not follow_player and fixed_xlim and fixed_ylim:
                ax.set_xlim(fixed_xlim)
                ax.set_ylim(fixed_ylim)
                ax.set_box_aspect([fixed_xlim[1] - fixed_xlim[0], fixed_ylim[1] - fixed_ylim[0], zlim])

            # 2. Update Skeleton Connection Lines
            for c in connections:
                part1, part2 = c
                p1 = joint_data[part1][frame_i]
                p2 = joint_data[part2][frame_i]
                if np.isnan(p1[0]) or np.isnan(p2[0]):
                    skeleton_lines[c].set_data_3d([], [], [])
                else:
                    skeleton_lines[c].set_data_3d(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                    )

            # 3. Update Joint Scatter Points
            if show_joints:
                grouped_pts = {"left": [[], [], []], "right": [[], [], []], "head": [[], [], []], "center": [[], [], []]}
                for jname in ALL_JOINTS:
                    pt = joint_data[jname][frame_i]
                    if not np.isnan(pt[0]):
                        if jname.startswith("l"):
                            target_g = "left"
                        elif jname.startswith("r"):
                            target_g = "right"
                        elif jname in {"nose", "neck", "lEye", "rEye", "lEar", "rEar"}:
                            target_g = "head"
                        else:
                            target_g = "center"
                        grouped_pts[target_g][0].append(pt[0])
                        grouped_pts[target_g][1].append(pt[1])
                        grouped_pts[target_g][2].append(pt[2])

                for g_name, g_coords in grouped_pts.items():
                    joint_scatter_plots[g_name].set_data_3d(g_coords[0], g_coords[1], g_coords[2])

            # 4. Update Ball
            if show_ball and ball_detected[frame_i]:
                bx, by, bz = ball_data[frame_i]
                ball_marker.set_data_3d([bx], [by], [bz])
            else:
                ball_marker.set_data_3d([], [], [])

            # 5. Update HUD Text
            if hud_text:
                ts = timestamps[frame_i] or "--:--:--"
                f_idx = frame_indices[frame_i]
                status_pose = "TRACKED" if valid_centroid else "OCCLUDED"
                hud_lines = [
                    f"Player ID: {resolved_player_id} [{status_pose}]",
                    f"Frame: {f_idx} (Idx: {frame_i + 1}/{actual_frames})",
                    f"Clock: {ts}",
                ]
                if valid_centroid:
                    hud_lines.append(f"Position: ({mid_hip[0]:.2f}m, {mid_hip[1]:.2f}m)")
                    stride = stride_lengths[frame_i]
                    if not np.isnan(stride):
                        hud_lines.append(f"Stride Length: {stride:.2f}m")
                if show_ball and ball_detected[frame_i]:
                    bx, by, bz = ball_data[frame_i]
                    hud_lines.append(f"Ball: ({bx:.1f}m, {by:.1f}m, z={bz:.2f}m)")

                hud_text.set_text("\n".join(hud_lines))

    # Construct FuncAnimation
    plt.tight_layout()
    anim = FuncAnimation(
        fig,
        update_frame,
        frames=actual_frames,
        interval=1000.0 / fps,
        blit=False,
    )

    # Save animation if output path is provided
    if output_path:
        save_animation(anim, output_path=output_path, fps=fps, dpi=dpi, total_frames=actual_frames)

    return anim


def save_animation(
    anim: FuncAnimation,
    output_path: str,
    fps: int = 25,
    dpi: int = 100,
    total_frames: Optional[int] = None,
):
    """
    Save the matplotlib animation to a local file (.mp4 or .gif) with progress reporting.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    is_gif = output_path.lower().endswith(".gif")

    print(f"[*] Saving animation to: {output_path} ({'GIF' if is_gif else 'MP4'}, {fps} fps, {dpi} dpi)...")

    # Set up progress callback
    pbar = None
    if tqdm is not None and total_frames:
        pbar = tqdm(total=total_frames, desc="Rendering Frames", unit="frame")

    def progress_callback(current_frame: int, max_frames: int):
        if pbar is not None:
            pbar.update(1)
        elif current_frame % max(1, max_frames // 10) == 0 or current_frame == max_frames - 1:
            print(f"    Frame {current_frame + 1}/{max_frames} ({((current_frame + 1) / max_frames) * 100:.1f}%)")

    try:
        if is_gif:
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer, dpi=dpi, progress_callback=progress_callback)
        else:
            if not FFMpegWriter.isAvailable():
                print("[!] Warning: ffmpeg writer not found. Falling back to default writer.")
                anim.save(output_path, fps=fps, dpi=dpi, progress_callback=progress_callback)
            else:
                writer = FFMpegWriter(
                    fps=fps,
                    codec="libx264",
                    extra_args=["-pix_fmt", "yuv420p"],
                )
                anim.save(output_path, writer=writer, dpi=dpi, progress_callback=progress_callback)
    finally:
        if pbar is not None:
            pbar.close()

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[✓] Animation successfully saved: {output_path} ({size_mb:.2f} MB)")
    else:
        print(f"[!] Warning: Output file was not found at {output_path}")




def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Construct input path from HF download root and match id
    jsonl_path = os.path.join(args.hf_dir, "raw", f"{args.match_id}.jsonl")

    if not os.path.exists(jsonl_path):
        print(f"Error: Target file '{jsonl_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Input file: {jsonl_path}")

    # Load / compute frame offsets index for instant O(1) seeking
    offsets = get_line_offsets(jsonl_path)

    # Handle --list-players utility
    if args.list_players:
        scan_start = args.start_frame if args.start_frame is not None else 0
        scan_count = max(args.num_frames, 250)
        print(f"[*] Scanning {scan_count} frames starting at frame {scan_start} for active players...")
        team_map = load_player_team_map(args.sc_opendata_dir, args.match_id)
        candidates = locate_active_players(
            jsonl_path,
            start_frame=scan_start,
            scan_frames=scan_count,
            offsets=offsets,
            team_map=team_map,
        )
        if not candidates:
            print("No players found in this frame window.")
        else:
            print(f"{'Player ID':<12} | {'Team ID':<10} | {'Valid Pose Frames':<18} | {'Coverage (%)':<12}")
            print("-" * 61)
            for pid, count, team_id in candidates:
                pct = (count / scan_count) * 100.0
                team_str = str(team_id) if team_id is not None else "N/A"
                print(f"{pid:<12} | {team_str:<10} | {count:<18} | {pct:>6.1f}%")
        sys.exit(0)

    # Determine output path and resolve start_frame/player_id if not specified
    output_path = args.output_path
    s_frame = args.start_frame

    if args.all_players:
        if args.player_id is not None:
            print("[*] Note: Animating all visible players (--all-players is set; ignoring specified --player-id)")
        if s_frame is None:
            s_frame, _ = find_first_player_frame(jsonl_path, None, offsets=offsets)
            args.start_frame = s_frame
        if output_path is None:
            output_path = f"pose_all_players_f{s_frame}_{args.num_frames}frames.mp4"
    else:
        p_id = args.player_id
        if s_frame is None and p_id is None:
            s_frame, p_id = find_first_player_frame(jsonl_path, None, offsets=offsets)
            args.start_frame = s_frame
            args.player_id = p_id
        elif s_frame is None and p_id is not None:
            s_frame, _ = find_first_player_frame(jsonl_path, p_id, offsets=offsets)
            args.start_frame = s_frame
        elif s_frame is not None and p_id is None:
            # Auto-detect active player in the specific target window
            candidates = locate_active_players(jsonl_path, start_frame=s_frame, scan_frames=max(args.num_frames, 50), offsets=offsets)
            if candidates and candidates[0][1] > 0:
                p_id = candidates[0][0]
                args.player_id = p_id

        if output_path is None:
            output_path = f"pose_player_{p_id or 'auto'}_f{s_frame}_{args.num_frames}frames.mp4"

    # Run animation
    animate_soccer_pose(
        path_to_jsonl=jsonl_path,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        player_id=args.player_id,
        all_players=args.all_players,
        connections=CONNECTIONS,
        xbuffer=args.xbuffer,
        ybuffer=args.ybuffer,
        zlim=args.zlim,
        elev=args.elev,
        azim=args.azim,
        player_color=args.player_color,
        color_by_side=args.color_by_side,
        show_joints=args.show_joints,
        show_ball=args.show_ball,
        show_pitch=args.show_pitch,
        follow_player=args.follow_player,
        whole_pitch=args.whole_pitch,
        max_error_cm=args.max_error_cm,
        output_path=output_path,
        fps=args.fps,
        dpi=args.dpi,
        no_axis=args.no_axis,
        offsets=offsets,
    )


if __name__ == "__main__":
    main()
