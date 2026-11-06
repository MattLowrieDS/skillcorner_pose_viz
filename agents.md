# SkillCorner Open Data Body Pose Dataset Reference

Technical summary and operational guide for agents when working with the SkillCorner Body Pose dataset located in the `raw/` directory.

---

## 1. Match Files & Metadata

| Match ID | Match Details | Date | File Name | Total Frames |
|---|---|---|---|---|---|---|
| 1925299 | Brisbane Roar vs. Perth Glory | 2024-12-21 | `raw/1925299.jsonl` | 153,252 (frames 0–153,251) |
| 1996435 | Sydney FC vs. Adelaide United | 2025-02-01 | `raw/1996435.jsonl` | 144,052 (frames 0–144,051) |

*Format:* Each file is a Line-delimited JSON (`.jsonl`) file where each line represents a single broadcast video frame.

---

## 2. Frame Rate & Synchronization with Tracking Data

- Pose dataset (.jsonl) is 25 fps; standard SkillCorner tracking (opendata) is 10 fps; both share the same wall-clock timeline.
- Alignment: `pose_frame = 2.5 * tracking_frame`. Exact alignment when `pose_frame % 5 == 0` (maps to an even-numbered tracking frame).
- The 2D coordinates `(x, y)` in the pose `.jsonl` files are linearly interpolated up to 25 fps from the native 10 fps tracking data.

---

## 3. Coordinate System & Physical Units

- **Units:** All spatial coordinates (`x`, `y`, `z`) are expressed in **metres**.
- **Origin `(0, 0, 0)`:** Pitch center at ground level.
- **X-axis:** Along the pitch length (long axis, typically from $-52.5\text{ m}$ to $+52.5\text{ m}$).
- **Y-axis:** Along the pitch width (short axis, typically from $-34.0\text{ m}$ to $+34.0\text{ m}$).
- **Z-axis (Vertical):**
  - **Player Joints:** Joint `z` is **not** an absolute elevation above the pitch plane. It represents vertical pose geometry relative to the player's centroid/root. Values typically range between ~ -0.1m (foot level) and ~ 1.9m (head level), but negative values can occur due to centroid-relative fitting.
  - **Ball:** Unlike the 10 fps tracking dataset (which is 2D only), `ball_data` here includes a true physical `z` coordinate representing estimated ball elevation in metres above the pitch (observed heights reach up to 13+m during high aerial balls).

---

## 4. JSON Schema & Joint Structure

- **Full record schema (field definitions) and the 29-joint taxonomy (keys, `p90_mae_cm` error table):** see `docs/schema.md`.
- Consult `docs/schema.md` when you need exact field types, joint key names, or per-joint accuracy values; it is not needed for most tasks.

## 5. Quality & Coverage Statistics (Match 1925299)

- **Total Video Frames:** 153,252
- **Frames with At Least One Pose:** 106,757 (69.7% of match frames)
- **Total Player-Frame Instances:** 4,902,560
- **Player-Frames with Valid Pose:** 1,540,715 (31.4% overall coverage across all 22+ players)
- **Average Visible Players with Pose per Active Frame:** 14.4 players
- **Total Joint Observations:** 44,680,735

---

## 7. Critical Guidelines & Limitations for Agents

1. **Memory Management (Do Not Read Entire Match into Memory at Once):**
   - Each uncompressed `.jsonl` file is $\sim 3.1\text{ GB}$. Parsing all JSON records simultaneously will exhaust memory ($>10\text{ GB}$ in RAM).
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

## 8. Git Command Restriction

**Never run any git commands.** Do not execute `git` (e.g. `git status`, `git diff`, etc.). Leave all version control operations to the user.
