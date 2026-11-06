# SkillCorner Open Data Body Pose Schema Information

## 1. JSON Schema & Record Specification

Each line in the `.jsonl` file is a JSON object.

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

## 2. Biomechanical Joint Structure (29 Joints)

When `joints` is non-null, it contains exactly **29 joints**. Each joint contains:
- `xyz`: `[x, y, z]` (3 floats in metres; `x` and `y` on the pitch coordinate grid; `z` relative height).
- `p90_mae_cm`: `float` (accuracy estimate: 90th percentile Mean Absolute Error in centimetres).

### Joint Taxonomy & Mean Error Rates

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
