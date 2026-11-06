"""
SkillCorner Body Pose — Data Loading & Streaming
=================================================
Byte-offset indexing, caching, frame seeking, player scanning,
and pose data extraction from SkillCorner .jsonl tracking files.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from constants import ALL_JOINTS


# ---------------------------------------------------------------------------
# Data Loading & Streaming (Byte-Offset Indexing & Fast Seeking)
# ---------------------------------------------------------------------------

def build_line_index(jsonl_path: str, chunk_size: int = 4 * 1024 * 1024) -> np.ndarray:
    """
    Build an array of byte offsets for each line in a .jsonl file using fast chunked binary reads.
    Each entry offsets[i] is the starting byte position of frame i.
    """
    offsets = [0]
    pos = 0
    with open(jsonl_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            start = 0
            while True:
                idx = chunk.find(b"\n", start)
                if idx == -1:
                    break
                offsets.append(pos + idx + 1)
                start = idx + 1
            pos += len(chunk)

    if len(offsets) > 1 and offsets[-1] == pos:
        offsets.pop()

    return np.array(offsets, dtype=np.uint64)


def get_line_offsets(
    jsonl_path: str,
    cache_dir: str = ".cache",
    force_rebuild: bool = False,
) -> np.ndarray:
    """
    Retrieve or compute a cached array of byte offsets for line-indexed O(1) seeking.
    Offsets are saved to .cache/<filename>.idx.npy.
    """
    base_name = os.path.basename(jsonl_path)
    os.makedirs(cache_dir, exist_ok=True)
    idx_path = os.path.join(cache_dir, f"{base_name}.idx.npy")

    # Check if cache is valid (exists, non-empty, newer than source jsonl)
    if not force_rebuild and os.path.exists(idx_path) and os.path.getsize(idx_path) > 0:
        try:
            if os.path.getmtime(idx_path) >= os.path.getmtime(jsonl_path):
                return np.load(idx_path, mmap_mode="r")
        except Exception:
            pass  # Fall back to rebuilding if corrupt

    t0_wall = time.time()
    print(f"[*] Building frame index for {base_name}...")
    offsets = build_line_index(jsonl_path)
    try:
        np.save(idx_path, offsets)
        elapsed = time.time() - t0_wall
        print(f"[✓] Indexed {len(offsets):,} frames in {elapsed:.2f}s -> cached to {idx_path}")
        return np.load(idx_path, mmap_mode="r")
    except Exception as e:
        print(f"[!] Warning: Could not write cache to {idx_path}: {e}")
        return offsets


def seek_to_frame(f, offsets: Optional[np.ndarray], frame_idx: int) -> bool:
    """
    Seek directly to frame_idx using byte offset index if available,
    or fallback to sequential readline skipping.
    Returns True if successfully positioned at frame_idx, False if EOF.
    """
    if frame_idx <= 0:
        f.seek(0)
        return True

    if offsets is not None:
        if frame_idx >= len(offsets):
            f.seek(0, os.SEEK_END)
            return False
        f.seek(int(offsets[frame_idx]))
        return True
    else:
        f.seek(0)
        for _ in range(frame_idx):
            if not f.readline():
                return False
        return True


def load_player_team_map(opendata_dir: str, match_id: str) -> Dict[int, int]:
    """
    Build a player_id -> team_id mapping from the complementary SkillCorner
    open-data match metadata for the same match id, if available.
    Returns an empty dict when the match metadata cannot be located.
    """
    match_path = os.path.join(opendata_dir, "opendata", "data", "matches", match_id, f"{match_id}_match.json")
    if not os.path.exists(match_path):
        return {}
    try:
        with open(match_path, "r", encoding="utf-8") as f:
            match = json.load(f)
        return {
            int(p["id"]): int(p["team_id"])
            for p in match.get("players", [])
            if p.get("id") is not None and p.get("team_id") is not None
        }
    except Exception:
        return {}


def locate_active_players(
    jsonl_path: str,
    start_frame: int = 0,
    scan_frames: int = 300,
    offsets: Optional[np.ndarray] = None,
    team_map: Optional[Dict[int, int]] = None,
) -> List[Tuple[int, int, Optional[int]]]:
    """
    Scan a window of frames to discover all players present and count
    how many frames each player has valid 3D pose data.
    Returns a sorted list of (player_id, count_of_pose_frames, team_id).
    """
    if offsets is None:
        offsets = get_line_offsets(jsonl_path)

    player_counts: Dict[int, int] = {}
    player_teams: Dict[int, Optional[int]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        if not seek_to_frame(f, offsets, start_frame):
            return []

        for _ in range(scan_frames):
            line = f.readline()
            if not line:
                break
            try:
                frame_data = json.loads(line)
            except json.JSONDecodeError:
                continue

            for player in frame_data.get("player_data", []):
                pid = player.get("player_id")
                if pid is not None:
                    if pid not in player_counts:
                        player_counts[pid] = 0
                        player_teams[pid] = team_map.get(pid) if team_map else None
                    if player.get("joints"):
                        player_counts[pid] += 1

    return sorted(
        ((pid, count, player_teams[pid]) for pid, count in player_counts.items()),
        key=lambda x: x[1],
        reverse=True,
    )


def find_first_player_frame(
    jsonl_path: str,
    player_id: Optional[int] = None,
    max_scan_frames: int = 50000,
    offsets: Optional[np.ndarray] = None,
    start_frame: int = 0,
) -> Tuple[int, int]:
    """
    Find the first frame where player_id has valid pose data.
    If player_id is None, finds the first frame containing ANY player with pose data.
    Returns (start_frame, resolved_player_id).
    """
    if offsets is None:
        offsets = get_line_offsets(jsonl_path)

    pid_str = f'"player_id": {player_id}' if player_id is not None else None

    with open(jsonl_path, "r", encoding="utf-8") as f:
        if start_frame > 0:
            if not seek_to_frame(f, offsets, start_frame):
                raise ValueError(f"Start frame {start_frame} is beyond end of file.")

        for idx in range(start_frame, start_frame + max_scan_frames):
            line = f.readline()
            if not line:
                break

            # Fast text pre-filter if searching for a specific player
            if pid_str is not None and pid_str not in line:
                continue

            try:
                frame_data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip pre-match intervals without active match clock
            if frame_data.get("period") is None:
                continue

            for player in frame_data.get("player_data", []):
                if player.get("joints"):
                    pid = player.get("player_id")
                    if player_id is None:
                        return frame_data.get("frame", idx), pid
                    if pid == player_id:
                        return frame_data.get("frame", idx), pid

    player_str = f"player {player_id}" if player_id is not None else "any player"
    raise ValueError(f"No valid pose frames found for {player_str} within {max_scan_frames} frames from frame {start_frame}.")


def load_pose_data(
    jsonl_path: str,
    start_frame: Optional[int] = None,
    num_frames: int = 100,
    player_id: Optional[int] = None,
    all_players: bool = False,
    max_error_cm: Optional[float] = None,
    offsets: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Load a sequence of pose and ball tracking frames from a SkillCorner .jsonl file.

    Parameters:
    -----------
    - jsonl_path: str
        Path to the SkillCorner .jsonl file.
    - start_frame: int or None
        The starting frame number (0-indexed). If None, automatically seeks to
        the first frame where valid pose data exists.
    - num_frames: int
        Number of frames to extract.
    - player_id: int or None
        The player ID to animate (ignored if all_players=True). If None and all_players=False,
        the most active player in the frame range is automatically detected.
    - all_players: bool
        If True, loads 3D pose and tracking data for all visible players on the pitch.
    - max_error_cm: float or None
        If set, filters out any joint observation where p90_mae_cm exceeds this threshold.
    - offsets: np.ndarray, optional
        Pre-computed byte offset array for O(1) seeking. Computed automatically if None.

    Returns:
    --------
    - data: dict
        Contains 'frames', 'timestamps', 'ball_xyz', 'ball_detected', and either:
        - if all_players=False: 'joint_trajectories', 'player_id', 'player_xy', 'valid_pose_mask'.
        - if all_players=True: 'all_players_frames', 'unique_player_ids', 'all_players'=True.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    if offsets is None:
        offsets = get_line_offsets(jsonl_path)

    if all_players:
        if start_frame is None:
            start_frame, _ = find_first_player_frame(jsonl_path, None, offsets=offsets)
            print(f"[*] Auto-detected starting frame: {start_frame} for all players")
    else:
        # Determine start_frame or player_id if not explicitly provided
        if start_frame is None:
            start_frame, first_pid = find_first_player_frame(jsonl_path, player_id, offsets=offsets)
            if player_id is None:
                player_id = first_pid
            print(f"[*] Auto-detected starting frame: {start_frame} for player ID: {player_id}")

        if player_id is None:
            # Detect active players in the target frame window
            candidates = locate_active_players(jsonl_path, start_frame=start_frame, scan_frames=max(num_frames, 50), offsets=offsets)
            if not candidates or candidates[0][1] == 0:
                raise ValueError(f"No players with valid pose data detected around frame {start_frame}.")
            player_id = candidates[0][0]
            pose_count = candidates[0][1]
            print(f"[*] Auto-selected player ID: {player_id} ({pose_count}/{num_frames} frames with valid pose)")
            if len(candidates) > 1:
                print(f"    Other available players in window: {', '.join(f'{pid} ({c})' for pid, c, _ in candidates[1:6])}")

    # Initialize data structures
    frames: List[int] = []
    timestamps: List[Optional[str]] = []
    ball_xyz_list: List[List[float]] = []
    ball_detected_list: List[bool] = []

    if all_players:
        all_players_frames: List[List[Dict[str, Any]]] = []
        all_player_ids = set()
    else:
        player_xy_list: List[List[float]] = []
        valid_pose_mask: List[bool] = []
        joint_trajectories: Dict[str, List[List[float]]] = {
            j: [] for j in ALL_JOINTS
        }

    with open(jsonl_path, "r", encoding="utf-8") as f:
        # Fast-seek directly to target frame in O(1)
        if not seek_to_frame(f, offsets, start_frame):
            raise ValueError(f"Requested start_frame {start_frame} is beyond total frames ({len(offsets)}) in {jsonl_path}.")

        # Read the requested number of frames
        for _ in range(num_frames):
            line = f.readline()
            if not line:
                break
            try:
                frame_data = json.loads(line)
            except json.JSONDecodeError:
                continue

            current_frame = frame_data.get("frame", start_frame + len(frames))
            frames.append(current_frame)
            timestamps.append(frame_data.get("timestamp"))

            # Ball tracking
            bd = frame_data.get("ball_data")
            if bd and bd.get("x") is not None and bd.get("is_detected"):
                ball_xyz_list.append([bd["x"], bd["y"], bd.get("z", 0.0)])
                ball_detected_list.append(True)
            else:
                ball_xyz_list.append([np.nan, np.nan, np.nan])
                ball_detected_list.append(False)

            if all_players:
                frame_players: List[Dict[str, Any]] = []
                for p in frame_data.get("player_data", []):
                    is_detected = p.get("is_detected", False)
                    raw_joints = p.get("joints")
                    # Include visible players: visually detected or with valid joints
                    if not is_detected and not raw_joints:
                        continue

                    pid = p.get("player_id")
                    if pid is not None:
                        all_player_ids.add(pid)

                    filtered_joints = {}
                    if raw_joints:
                        for jname in ALL_JOINTS:
                            if jname in raw_joints:
                                j_info = raw_joints[jname]
                                if max_error_cm is not None and j_info.get("p90_mae_cm", 0.0) > max_error_cm:
                                    continue
                                filtered_joints[jname] = j_info["xyz"]

                    frame_players.append({
                        "player_id": pid,
                        "x": p.get("x", np.nan),
                        "y": p.get("y", np.nan),
                        "is_detected": is_detected,
                        "joints": filtered_joints if filtered_joints else None,
                    })
                all_players_frames.append(frame_players)
            else:
                target_player = None
                for p in frame_data.get("player_data", []):
                    if p.get("player_id") == player_id:
                        target_player = p
                        break

                if target_player:
                    player_xy_list.append([target_player.get("x", np.nan), target_player.get("y", np.nan)])
                    joints = target_player.get("joints")
                else:
                    player_xy_list.append([np.nan, np.nan])
                    joints = None

                has_valid_pose = joints is not None and len(joints) > 0
                valid_pose_mask.append(has_valid_pose)

                for jname in ALL_JOINTS:
                    if joints and jname in joints:
                        j_info = joints[jname]
                        # Filter noisy joint if max_error_cm threshold is set
                        if max_error_cm is not None and j_info.get("p90_mae_cm", 0.0) > max_error_cm:
                            joint_trajectories[jname].append([np.nan, np.nan, np.nan])
                        else:
                            joint_trajectories[jname].append(j_info["xyz"])
                    else:
                        joint_trajectories[jname].append([np.nan, np.nan, np.nan])

    if all_players:
        return {
            "frames": frames,
            "timestamps": timestamps,
            "all_players_frames": all_players_frames,
            "unique_player_ids": sorted(list(all_player_ids)),
            "ball_xyz": np.array(ball_xyz_list, dtype=np.float64),
            "ball_detected": np.array(ball_detected_list, dtype=bool),
            "all_players": True,
            "player_id": None,
        }
    else:
        # Convert to numpy arrays for vectorized indexing
        np_joint_trajectories = {
            k: np.array(v, dtype=np.float64) for k, v in joint_trajectories.items()
        }

        return {
            "frames": frames,
            "timestamps": timestamps,
            "joint_trajectories": np_joint_trajectories,
            "ball_xyz": np.array(ball_xyz_list, dtype=np.float64),
            "ball_detected": np.array(ball_detected_list, dtype=bool),
            "player_id": player_id,
            "player_xy": np.array(player_xy_list, dtype=np.float64),
            "valid_pose_mask": np.array(valid_pose_mask, dtype=bool),
            "all_players": False,
        }
