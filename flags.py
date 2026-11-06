"""
SkillCorner Body Pose — CLI Argument Declarations
==================================================
All argparse flag definitions for the animate.py command-line interface.
Self-contained: no imports from other project modules.
"""

import argparse
import glob
import os


# ---------------------------------------------------------------------------
# CLI & Argument Parsing
# ---------------------------------------------------------------------------

def find_default_jsonl() -> str:
    """Find the first available .jsonl file in the raw/ directory."""
    matches = glob.glob("raw/*.jsonl")
    if matches:
        return sorted(matches)[0]
    return "raw/1925299.jsonl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Animate 3D soccer pose data from SkillCorner .jsonl tracking files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core required/frequent flags
    parser.add_argument(
        "-m", "--match-id",
        dest="match_id",
        type=str,
        default=None,
        help="SkillCorner match identifier (string).",
    )
    parser.add_argument(
        "-s", "--start-frame",
        dest="start_frame",
        type=int,
        default=None,
        help="Starting video frame (0-indexed). Defaults to first frame player has pose.",
    )
    parser.add_argument(
        "-n", "--num-frames",
        dest="num_frames",
        type=int,
        default=100,
        help="Number of frames to animate (25 fps = 4 seconds for 100 frames).",
    )
    parser.add_argument(
        "-p", "--player-id",
        dest="player_id",
        type=int,
        default=None,
        help="Player ID to animate. If omitted, the most active player in the sequence is auto-selected.",
    )
    parser.add_argument(
        "-a", "--all-players",
        dest="all_players",
        action="store_true",
        default=False,
        help="Animate all visible players on the pitch instead of a single player.",
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_path",
        type=str,
        default=None,
        help="Path to save the output animation (.mp4 or .gif). Default: pose_<player_id|all_players>_f<start>.mp4",
    )

    # Visualization settings
    parser.add_argument("--fps", type=int, default=25, help="Frame rate for the animation.")
    parser.add_argument("--dpi", type=int, default=100, help="DPI resolution for exported video/gif.")
    parser.add_argument("--elev", type=float, default=20.0, help="Camera 3D elevation angle in degrees.")
    parser.add_argument("--azim", type=float, default=45.0, help="Camera 3D azimuth angle in degrees.")
    parser.add_argument("--xbuffer", type=float, default=2.5, help="Horizontal half-buffer in metres around player.")
    parser.add_argument("--ybuffer", type=float, default=2.5, help="Vertical half-buffer in metres around player.")
    parser.add_argument("--zlim", type=float, default=2.5, help="Maximum height limit for z-axis in metres.")
    parser.add_argument("--player-color", type=str, default="royalblue", help="Uniform color for skeleton lines.")
    parser.add_argument(
        "--no-color-by-side",
        dest="color_by_side",
        action="store_false",
        help="Disable anatomical side coloring (uses uniform player-color instead).",
    )
    parser.add_argument(
        "--no-ball",
        dest="show_ball",
        action="store_false",
        help="Do not render the soccer ball trajectory.",
    )
    parser.add_argument(
        "--no-pitch",
        dest="show_pitch",
        action="store_false",
        help="Do not draw ground pitch lines at z=0.",
    )
    parser.add_argument(
        "--no-joints",
        dest="show_joints",
        action="store_false",
        help="Do not render joint markers (lines only).",
    )
    parser.add_argument(
        "--no-axis",
        dest="no_axis",
        action="store_true",
        default=False,
        help="Turn off the 3D coordinate pane axis for a clean view.",
    )
    parser.add_argument(
        "--fixed-view",
        dest="follow_player",
        action="store_false",
        help="Fix camera to player movement region instead of tracking/following player hip.",
    )
    parser.add_argument(
        "--whole-pitch", "--fixed-pitch",
        dest="whole_pitch",
        action="store_true",
        default=None,
        help="Fix camera view to the whole soccer pitch. (Default when --all-players is enabled).",
    )
    parser.add_argument(
        "--no-whole-pitch",
        dest="whole_pitch",
        action="store_false",
        help="Disable whole pitch camera view and follow player/centroid instead.",
    )
    parser.add_argument(
        "--max-error-cm",
        type=float,
        default=None,
        help="Filter out joints with p90_mae_cm above this threshold in cm.",
    )
    parser.add_argument(
        "--hf-dir",
        dest="hf_dir",
        type=str,
        default=os.getcwd(),
        help="Path to the Hugging Face data download root directory.",
    )
    parser.add_argument(
        "--sc-opendata-dir",
        dest="sc_opendata_dir",
        type=str,
        default=os.getcwd(),
        help="Path to the cloned SkillCorner opendata repository root directory.",
    )
    parser.add_argument(
        "--list-players",
        action="store_true",
        help="Scan and list players detected in the frame range, then exit.",
    )

    return parser
