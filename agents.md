# SkillCorner Open Data — Body Pose Dataset Reference

This document provides a technical summary and operational guide for agents, data scientists, and engineers working with the SkillCorner Body Pose dataset located in the `raw/` directory.

---

## 1. Overview & Dataset Context

- **Source:** SkillCorner in collaboration with PySport ([SkillCorner Open Data](https://github.com/SkillCorner/opendata)).
- **Content:** 3D body pose estimations (29 anatomical joints) derived from broadcast video, integrated with broadcast tracking and ball tracking for two complete Australian A-League soccer matches.
- **License:** MIT License.
- **Complementary Data:** Both matches correspond directly to matches published in the SkillCorner open data repository (competition edition 870), allowing join operations with 10 fps tracking, physical dynamic events, and tactical phases of play.

---

## 2. Match Files & Metadata

| Match ID | Match Details | Date | File Name | Size (Uncompressed) | Size (Compressed) | Total Frames |
|---|---|---|---|---|---|---|
| **1925299** | Brisbane Roar vs. Perth Glory | 2024-12-21 | `raw/1925299.jsonl` | ~3.1 GB | 595 MB (`.zip`) | 153,252 (frames 0–153,251) |
| **1996435** | Sydney FC vs. Adelaide United | 2025-02-01 | `raw/1996435.jsonl` | ~3.1 GB | 614 MB (`.zip`) | 144,052 (frames 0–144,051) |

*Format:* Each file is a Line-delimited JSON (`.jsonl`) file where each line represents a single broadcast video frame.

---

## 3. Frame Rate & Synchronization with Tracking Data

| Data Source | Frame Rate | Approximate Frame Count per Match |
|---|---|---|
| **Body Pose Dataset (`.jsonl`)** | **25 fps** | ~144,000 – 153,250 frames |
| **Standard SkillCorner Tracking (`opendata`)** | **10 fps** | ~57,600 – 61,300 frames |

### Alignment Formula
Both datasets share the same wall-clock timeline:
$$\text{pose\_frame} = 2.5 \times \text{tracking\_frame}$$

- **Exact alignment:** Every 5th pose frame (where `pose_frame % 5 == 0`) maps exactly to an even-numbered tracking frame (`tracking_frame = pose_frame / 2.5`).
- **Interpolation note:** The 2D coordinates `(x, y)` in the pose `.jsonl` files are linearly interpolated up to 25 fps from the native 10 fps tracking data.

---

## 4. Coordinate System & Physical Units

- **Units:** All spatial coordinates (`x`, `y`, `z`) are expressed in **metres**.
- **Origin `(0, 0, 0)`:** Pitch center at ground level.
- **X-axis:** Along the pitch length (long axis, typically from $-52.5\text{ m}$ to $+52.5\text{ m}$).
- **Y-axis:** Along the pitch width (short axis, typically from $-34.0\text{ m}$ to $+34.0\text{ m}$).
- **Z-axis (Vertical):**
  - **Player Joints:** Joint `z` is **not** an absolute elevation above the pitch plane. It represents vertical pose geometry relative to the player's centroid/root. Values typically range between $\sim -0.1\text{ m}$ (foot level) and $\sim 1.9\text{ m}$ (head level), but negative values can occur due to centroid-relative fitting.
  - **Ball:** Unlike the 10 fps tracking dataset (which is 2D only), `ball_data` here includes a true physical `z` coordinate representing estimated ball elevation in metres above the pitch (observed heights reach up to $13+\text{ m}$ during high aerial balls).

---

## 5. JSON Schema & Record Specification

Each line in the `.jsonl` file is a JSON object with the following schema:

```json
{
  "frame": 2438,
  "timestamp": "00:01:36.52",
  "period": 1,
  "ball_data": {
    "x": 0.35,
    "y": -22.22,
    "z": 7.074,
    "is_detected": true
  },
  "possession": {
    "player_id": null,
    "group": "away team"
  },
  "image_corners_projection": {
    "x_top_left": -44.26,
    "y_top_left": 39.0,
    "x_bottom_left": -15.02,
    "y_bottom_left": -36.53,
    "x_bottom_right": 15.51,
    "y_bottom_right": -36.44,
    "x_top_right": 48.84,
    "y_top_right": 39.0
  },
  "player_data": [
    {
      "player_id": 809166,
      "x": -17.36,
      "y": -22.06,
      "is_detected": true,
      "joints": {
        "nose": {"xyz": [-17.31, -22.02, 1.71], "p90_mae_cm": 8.09},
        "neck": {"xyz": [-17.33, -22.04, 1.52], "p90_mae_cm": 5.88},
        "..." : "..."
      }
    }
  ]
}
```

### Field Definitions

| Field | Type | Description |
|---|---|---|
| `frame` | `int` | Sequential 0-indexed frame identifier (25 fps). |
| `timestamp` | `string \| null` | Match clock in `HH:MM:SS.ss` format. `null` outside active play (pre-kickoff, halftime, post-match). |
| `period` | `int \| null` | `1` (1st half), `2` (2nd half), or `null` during pre-game/intervals. |
| `ball_data.x`, `y`, `z` | `float \| null` | Ball position in metres. `z` represents ball height. |
| `ball_data.is_detected` | `bool \| null` | `true` if the ball is actively identified by the visual detector in this frame. |
| `possession.player_id` | `int \| null` | Player ID currently possessing the ball (if individual possession resolved). |
| `possession.group` | `string \| null` | Team with possession: `"home team"`, `"away team"`, or `null`. |
| `image_corners_projection` | `dict` | Coordinates `(x, y)` in metres of the camera frustum's 4 corners projected onto the pitch surface: `x/y_top_left`, `x/y_bottom_left`, `x/y_bottom_right`, `x/y_top_right`. |
| `player_data` | `list[dict]` | List of all players tracked in this frame. |
| `player_data[i].player_id` | `int` | Unique player identifier (matches SkillCorner open data player IDs). |
| `player_data[i].x`, `y` | `float` | 2D pitch coordinates in metres (interpolated 25 fps). |
| `player_data[i].is_detected` | `bool` | `true` if player was detected directly in the broadcast frame; `false` if position was extrapolated/interpolated while off-screen or occluded. |
| `player_data[i].joints` | `dict \| null` | `null` if pose could not be extracted. When present, contains exactly 29 joint dictionary entries. |

---

## 6. Biomechanical Joint Structure (29 Joints)

When `joints` is non-null, it contains exactly **29 joints**. Each joint contains:
- `xyz`: `[x, y, z]` (3 floats in metres; `x` and `y` on the pitch coordinate grid; `z` relative height).
- `p90_mae_cm`: `float` (accuracy estimate: 90th percentile Mean Absolute Error in centimetres).

### Joint Taxonomy & Mean Error Rates

The 29 joints are divided into four anatomical groups:

```
                          [nose]
                       /    |    \
              [lEye] [rEye] |
                 |      |   |
              [lEar] [rEar] |
                     \  [neck]  /
                      /   |   \
           [lShoulder]    |    [rShoulder]
              |           |           |
           [lElbow]       |        [rElbow]
              |           |           |
           [lWrist]       |        [rWrist]
           /      \       |        /      \
      [lThumb]  [lPinky]  |   [rThumb]  [rPinky]
                          |
                      [midHip]
                      /      \
                 [lHip]      [rHip]
                   |            |
                [lKnee]      [rKnee]
                   |            |
                [lAnkle]     [rAnkle]
                /     \      /     \
           [lHeel] [lBigToe] [rHeel] [rBigToe]
                      |                 |
                 [lSmallToe]       [rSmallToe]
```

| Anatomical Region | Joint Key | Typical Mean Error (`p90_mae_cm`) | Accuracy Tier |
|---|---|---|---|
| **Head & Neck** | `neck` | ~5.88 cm | Very High |
| | `lEye`, `rEye` | ~7.72 – 7.91 cm | High |
| | `lEar`, `rEar` | ~7.79 – 8.02 cm | High |
| | `nose` | ~8.09 cm | High |
| **Torso & Pelvis** | `lShoulder`, `rShoulder` | ~6.49 cm | Very High |
| | `lHip`, `rHip` | ~8.49 – 8.66 cm | High |
| | `midHip` | ~10.06 cm | Moderate |
| **Arms & Hands** | `lElbow`, `rElbow` | ~9.54 – 9.67 cm | Moderate |
| | `lWrist`, `rWrist` | ~15.07 – 15.55 cm | Low |
| | `lThumb`, `rThumb` | ~18.30 – 18.60 cm | Low |
| | `lPinky`, `rPinky` | ~20.57 – 20.61 cm | Low |
| **Legs & Feet** | `lKnee`, `rKnee` | ~11.42 – 11.46 cm | Moderate |
| | `lAnkle`, `rAnkle` | ~14.65 – 14.69 cm | Low |
| | `lHeel`, `rHeel` | ~16.99 – 17.01 cm | Low |
| | `lBigToe`, `rBigToe` | ~17.47 – 17.65 cm | Low |
| | `lSmallToe`, `rSmallToe` | ~17.31 – 17.35 cm | Low |

---

## 7. Quality & Coverage Statistics (Match 1925299)

- **Total Video Frames:** 153,252
- **Frames with At Least One Pose:** 106,757 (69.7% of match frames)
- **Total Player-Frame Instances:** 4,902,560
- **Player-Frames with Valid Pose:** 1,540,715 (31.4% overall coverage across all 22+ players)
- **Average Visible Players with Pose per Active Frame:** 14.4 players
- **Total Joint Observations:** 44,680,735

---

## 8. Critical Guidelines & Limitations for Agents

1. **Memory Management (Do Not Read Entire Match into Memory at Once):**
   - Each uncompressed `.jsonl` file is $\sim 3.1\text{ GB}$. Parsing all JSON records simultaneously will exhaust memory ($>10\text{ GB}$ in RAM).
   - **Recommended:** Process line-by-line (streaming) or convert filtered subsets to columnar formats such as Apache Parquet or DuckDB tables.
2. **Confidence Filtering via `p90_mae_cm`:**
   - Coordinates are reported to millimetre precision in floats, but actual accuracy is bounded by `p90_mae_cm`.
   - Before computing fine-grained joint angles or velocity derivatives on extremities (e.g. ankles, wrists, toes), filter out observations where `p90_mae_cm > threshold` (e.g., discard $>20\text{ cm}$).
3. **Z-axis Elevation Caveat:**
   - Never compute player jump height or vertical distance from pitch plane using joint `z` alone, because `z` is relative to centroid geometry and uncalibrated to pitch terrain elevation.
4. **Pose vs Tracking Misalignment:**
   - 2D player tracking and 3D pose are produced by separate neural models. Slight offsets can exist between the interpolated `(x, y)` bounding-box centroid and the `midHip` joint position.
5. **Handling Undetected Frames:**
   - When `is_detected == false`, `joints` is always `null`.
   - Even when `is_detected == true`, occluded or blurred players may occasionally have `joints == null` (~1% of detected cases).

---

## 9. Python Usage Snippet

```python
import json

def stream_pose_data(jsonl_path: str, max_error_cm: float = 15.0):
    """
    Stream player pose frames and filter reliable joints.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            frame_data = json.loads(line)
            frame_idx = frame_data["frame"]
            period = frame_data["period"]
            
            # Skip non-match play frames
            if period is None:
                continue
                
            for player in frame_data.get("player_data", []):
                joints = player.get("joints")
                if not joints:
                    continue
                
                # Filter reliable joints based on error threshold
                reliable_joints = {
                    name: j["xyz"]
                    for name, j in joints.items()
                    if j["p90_mae_cm"] <= max_error_cm
                }
                
                yield {
                    "frame": frame_idx,
                    "timestamp": frame_data["timestamp"],
                    "player_id": player["player_id"],
                    "player_xy": (player["x"], player["y"]),
                    "joints": reliable_joints
                }

# Example invocation:
# for record in stream_pose_data("raw/1925299.jsonl"):
#     ...
```

---

## 10. Git Command Restriction

**Never run any git commands.** Do not execute `git` (e.g. `git status`, `git diff`, `git log`, `git add`, `git commit`, `git push`, `git checkout`, `git reset`, etc.) under any circumstances, even if asked. Leave all version control operations to the user.
