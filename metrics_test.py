"""
Unit tests for SkillCorner Body Pose Metrics (metrics.py).
Tests compute_stride_length and compute_num_steps.
"""

import os
import unittest
import numpy as np

from constants import ALL_JOINTS
from metrics import (
    compute_stride_length,
    compute_num_steps,
    compute_3point_angle,
    compute_knee_angles,
    compute_hip_orientation,
    compute_turning_velocity,
)


def _create_mock_joint_trajectories(n_frames: int) -> dict:
    """Create a dictionary of joint trajectories initialized to NaNs."""
    return {j: np.full((n_frames, 3), np.nan, dtype=np.float64) for j in ALL_JOINTS}


class TestComputeNumSteps(unittest.TestCase):
    """Test suite for compute_num_steps in metrics.py."""

    def test_synthetic_running_cadence_180(self):
        """Simulate a runner moving at 4.0 m/s with 180 SPM cadence."""
        fps = 25.0
        n_frames = 100  # 4 seconds
        target_spm = 180.0
        step_freq = target_spm / 60.0  # 3.0 steps/sec

        t = np.arange(n_frames) / fps
        jt = _create_mock_joint_trajectories(n_frames)

        # MidHip moving forward along x at 4 m/s
        jt["midHip"][:, 0] = 4.0 * t
        jt["midHip"][:, 1] = 0.0
        jt["midHip"][:, 2] = 0.95

        # Knees below pelvis
        jt["lKnee"][:, 2] = 0.50
        jt["rKnee"][:, 2] = 0.50
        jt["lKnee"][:, 0] = jt["midHip"][:, 0]
        jt["rKnee"][:, 0] = jt["midHip"][:, 0]

        # Feet oscillating forward and backward in antiphase
        # Stride amplitude 0.45m forward/back -> peak-to-peak foot separation ~0.9m
        phase = 2.0 * np.pi * (step_freq / 2.0) * t  # full gait cycle frequency = step_freq / 2
        foot_offset_l = 0.45 * np.sin(phase)
        foot_offset_r = -0.45 * np.sin(phase)

        jt["lAnkle"][:, 0] = jt["midHip"][:, 0] + foot_offset_l
        jt["lAnkle"][:, 1] = 0.15
        jt["lAnkle"][:, 2] = 0.10

        jt["rAnkle"][:, 0] = jt["midHip"][:, 0] + foot_offset_r
        jt["rAnkle"][:, 1] = -0.15
        jt["rAnkle"][:, 2] = 0.10

        total_steps, step_frames, mean_spm = compute_num_steps(jt, fps=fps)

        # In 4 seconds at 3 steps/sec, expect ~11-12 steps
        self.assertGreaterEqual(total_steps, 10)
        self.assertLessEqual(total_steps, 14)
        self.assertEqual(len(step_frames), total_steps)
        # SPM should be within 3% of 180 SPM
        self.assertAlmostEqual(mean_spm, target_spm, delta=6.0)

    def test_synthetic_jogging_cadence_120(self):
        """Simulate a jogger moving at 2.5 m/s with 120 SPM cadence."""
        fps = 25.0
        n_frames = 125  # 5 seconds
        target_spm = 120.0
        step_freq = target_spm / 60.0  # 2.0 steps/sec

        t = np.arange(n_frames) / fps
        jt = _create_mock_joint_trajectories(n_frames)

        jt["midHip"][:, 0] = 2.5 * t
        jt["midHip"][:, 1] = 0.0
        jt["midHip"][:, 2] = 0.95
        jt["lKnee"][:, 2] = 0.50
        jt["rKnee"][:, 2] = 0.50

        phase = 2.0 * np.pi * (step_freq / 2.0) * t
        foot_offset_l = 0.35 * np.sin(phase)
        foot_offset_r = -0.35 * np.sin(phase)

        jt["lAnkle"][:, 0] = jt["midHip"][:, 0] + foot_offset_l
        jt["lAnkle"][:, 1] = 0.15
        jt["lAnkle"][:, 2] = 0.10

        jt["rAnkle"][:, 0] = jt["midHip"][:, 0] + foot_offset_r
        jt["rAnkle"][:, 1] = -0.15
        jt["rAnkle"][:, 2] = 0.10

        total_steps, step_frames, mean_spm = compute_num_steps(jt, fps=fps)

        # In 5 seconds at 2 steps/sec, expect ~9-11 steps
        self.assertGreaterEqual(total_steps, 8)
        self.assertLessEqual(total_steps, 12)
        self.assertAlmostEqual(mean_spm, target_spm, delta=5.0)

    def test_stationary_player_rejected(self):
        """Stationary player with low velocity and minor pose jitter should return 0 steps."""
        fps = 25.0
        n_frames = 80
        jt = _create_mock_joint_trajectories(n_frames)

        # midHip barely moving (< 0.1 m/s)
        np.random.seed(42)
        jt["midHip"][:, 0] = 5.0 + 0.005 * np.random.randn(n_frames)
        jt["midHip"][:, 1] = 10.0 + 0.005 * np.random.randn(n_frames)
        jt["midHip"][:, 2] = 0.95

        # Ankles with minor millimetre noise
        jt["lAnkle"][:, 0] = 5.0 + 0.01 * np.random.randn(n_frames)
        jt["lAnkle"][:, 1] = 10.2 + 0.01 * np.random.randn(n_frames)
        jt["lAnkle"][:, 2] = 0.10

        jt["rAnkle"][:, 0] = 5.0 + 0.01 * np.random.randn(n_frames)
        jt["rAnkle"][:, 1] = 9.8 + 0.01 * np.random.randn(n_frames)
        jt["rAnkle"][:, 2] = 0.10

        total_steps, step_frames, mean_spm = compute_num_steps(jt, fps=fps)

        self.assertEqual(total_steps, 0)
        self.assertEqual(len(step_frames), 0)
        self.assertEqual(mean_spm, 0.0)

    def test_nan_resilience_and_heel_fallback(self):
        """Test fallback to heel when ankle is NaN, and interpolation across NaN blinks."""
        fps = 25.0
        n_frames = 100
        step_freq = 3.0  # 180 SPM
        t = np.arange(n_frames) / fps
        jt = _create_mock_joint_trajectories(n_frames)

        jt["midHip"][:, 0] = 4.0 * t
        jt["midHip"][:, 1] = 0.0
        jt["midHip"][:, 2] = 0.95

        phase = 2.0 * np.pi * (step_freq / 2.0) * t
        offset_l = 0.45 * np.sin(phase)
        offset_r = -0.45 * np.sin(phase)

        jt["lAnkle"][:, 0] = jt["midHip"][:, 0] + offset_l
        jt["lAnkle"][:, 1] = 0.15
        jt["lAnkle"][:, 2] = 0.10

        jt["rAnkle"][:, 0] = jt["midHip"][:, 0] + offset_r
        jt["rAnkle"][:, 1] = -0.15
        jt["rAnkle"][:, 2] = 0.10

        # Replace frames 30-32 of lAnkle with NaN, but provide lHeel
        jt["lHeel"][:, 0] = jt["lAnkle"][:, 0] - 0.05
        jt["lHeel"][:, 1] = 0.15
        jt["lHeel"][:, 2] = 0.10
        jt["lAnkle"][30:33, :] = np.nan

        # Inject 2 completely missing NaN frames at frames 50:52
        jt["lAnkle"][50:52, :] = np.nan
        jt["lHeel"][50:52, :] = np.nan

        total_steps, step_frames, mean_spm = compute_num_steps(jt, fps=fps)

        self.assertGreaterEqual(total_steps, 10)
        self.assertAlmostEqual(mean_spm, 180.0, delta=8.0)

    def test_empty_and_all_nan_inputs(self):
        """Empty dictionaries or all-NaN inputs must return zero steps safely."""
        # Empty dict
        total_steps, step_frames, mean_spm = compute_num_steps({})
        self.assertEqual(total_steps, 0)
        self.assertEqual(len(step_frames), 0)
        self.assertEqual(mean_spm, 0.0)

        # All-NaN 50-frame dict
        jt_nan = _create_mock_joint_trajectories(50)
        total_steps, step_frames, mean_spm = compute_num_steps(jt_nan)
        self.assertEqual(total_steps, 0)
        self.assertEqual(len(step_frames), 0)
        self.assertEqual(mean_spm, 0.0)

        # Very short sequence (< 5 frames)
        jt_short = _create_mock_joint_trajectories(3)
        total_steps, step_frames, mean_spm = compute_num_steps(jt_short)
        self.assertEqual(total_steps, 0)
        self.assertEqual(len(step_frames), 0)
        self.assertEqual(mean_spm, 0.0)

    def test_real_match_data_if_available(self):
        """Test on real match data from Match 1925299 if present."""
        raw_path = "../raw/1925299.jsonl"
        if not os.path.exists(raw_path):
            self.skipTest("Raw match data file not found.")

        from data_loading import load_pose_data

        # 1. Running sequence: pid 50982 around frame 10000
        run_data = load_pose_data(raw_path, start_frame=10000, num_frames=100, player_id=50982)
        total_steps, step_frames, mean_spm = compute_num_steps(run_data["joint_trajectories"])
        self.assertGreaterEqual(total_steps, 7)
        self.assertGreater(mean_spm, 140.0)
        self.assertLess(mean_spm, 240.0)

        # 2. Stationary sequence: pid 560985 around frame 5016
        stand_data = load_pose_data(raw_path, start_frame=5016, num_frames=100, player_id=560985)
        stand_steps, stand_frames, stand_spm = compute_num_steps(stand_data["joint_trajectories"])
        # Should detect 0 steps or 0.0 SPM for stationary player
        self.assertEqual(stand_spm, 0.0)

    def test_synthetic_sprint_cadence_210(self):
        """Simulate a sprint at 6.0 m/s with 210 SPM cadence."""
        fps = 25.0
        n_frames = 100
        target_spm = 210.0
        step_freq = target_spm / 60.0  # 3.5 steps/sec

        t = np.arange(n_frames) / fps
        jt = _create_mock_joint_trajectories(n_frames)

        jt["midHip"][:, 0] = 6.0 * t
        jt["midHip"][:, 1] = 0.0
        jt["midHip"][:, 2] = 0.95
        jt["lKnee"][:, 2] = 0.50
        jt["rKnee"][:, 2] = 0.50
        jt["lKnee"][:, 0] = jt["midHip"][:, 0]
        jt["rKnee"][:, 0] = jt["midHip"][:, 0]

        phase = 2.0 * np.pi * (step_freq / 2.0) * t
        foot_offset_l = 0.50 * np.sin(phase)
        foot_offset_r = -0.50 * np.sin(phase)

        jt["lAnkle"][:, 0] = jt["midHip"][:, 0] + foot_offset_l
        jt["lAnkle"][:, 1] = 0.15
        jt["lAnkle"][:, 2] = 0.10

        jt["rAnkle"][:, 0] = jt["midHip"][:, 0] + foot_offset_r
        jt["rAnkle"][:, 1] = -0.15
        jt["rAnkle"][:, 2] = 0.10

        total_steps, step_frames, mean_spm = compute_num_steps(jt, fps=fps)

        self.assertGreaterEqual(total_steps, 12)
        self.assertAlmostEqual(mean_spm, target_spm, delta=6.0)

    def test_return_types_and_shapes(self):
        """Verify the exact return types and numpy array properties."""
        jt = _create_mock_joint_trajectories(50)
        res = compute_num_steps(jt)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 3)
        total_steps, step_frames, mean_spm = res
        self.assertIsInstance(total_steps, int)
        self.assertIsInstance(step_frames, np.ndarray)
        self.assertIsInstance(mean_spm, float)

    def test_compute_stride_length_basic(self):
        """Test compute_stride_length returns valid distances."""
        jt = _create_mock_joint_trajectories(10)
        jt["lAnkle"][:, :] = np.array([0.0, 0.0, 0.1])
        jt["rAnkle"][:, :] = np.array([3.0, 4.0, 0.1])  # 3-4-5 triangle -> dist = 5.0m
        strides = compute_stride_length(jt)
        self.assertEqual(len(strides), 10)
        np.testing.assert_allclose(strides, 5.0)


class TestCompute3PointAngle(unittest.TestCase):
    """Test suite for generalized compute_3point_angle in metrics.py."""

    def test_straight_line_180(self):
        """Opposite rays should yield exactly 180 degrees (or pi radians)."""
        p1 = np.array([0.0, 0.0, 1.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 0.0, -1.0])

        deg = compute_3point_angle(p1, p2, p3, radians=False)
        self.assertAlmostEqual(float(deg), 180.0, places=6)

        rad = compute_3point_angle(p1, p2, p3, radians=True)
        self.assertAlmostEqual(float(rad), np.pi, places=6)

    def test_right_angle_90(self):
        """Perpendicular rays should yield 90 degrees (or pi/2 radians)."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])

        deg = compute_3point_angle(p1, p2, p3)
        self.assertAlmostEqual(float(deg), 90.0, places=6)

        rad = compute_3point_angle(p1, p2, p3, radians=True)
        self.assertAlmostEqual(float(rad), np.pi / 2.0, places=6)

    def test_acute_45_degrees(self):
        """Test 45 degree angle between vectors."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])

        deg = compute_3point_angle(p1, p2, p3)
        self.assertAlmostEqual(float(deg), 45.0, places=5)

    def test_3d_arbitrary_plane_rotational_invariance(self):
        """Verify angle calculation in an angled 3D plane is invariant to 3D rotation."""
        # Define 3 points with an exact 60-degree included angle
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.5, np.sqrt(3) / 2.0, 0.0])

        # Create an arbitrary 3D rotation matrix via Euler angles (yaw, pitch, roll)
        alpha, beta, gamma = 0.45, 0.82, 1.15
        Rz = np.array([
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha),  np.cos(alpha), 0],
            [0, 0, 1],
        ])
        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)],
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(gamma), -np.sin(gamma)],
            [0, np.sin(gamma),  np.cos(gamma)],
        ])
        R = Rz @ Ry @ Rx

        # Translate and rotate all points into an arbitrary angled plane in 3D
        offset = np.array([12.3, -5.7, 8.4])
        p1_rot = p1 @ R.T + offset
        p2_rot = p2 @ R.T + offset
        p3_rot = p3 @ R.T + offset

        deg_rot = compute_3point_angle(p1_rot, p2_rot, p3_rot)
        self.assertAlmostEqual(float(deg_rot), 60.0, places=5)

    def test_generalized_arm_elbow_angle(self):
        """Verify the generalized function works for upper-body triads (shoulder-elbow-wrist)."""
        n_frames = 20
        shoulder = np.tile([0.0, 0.2, 1.4], (n_frames, 1))
        elbow = np.tile([0.0, 0.2, 1.1], (n_frames, 1))
        # Forearm extended forward at 90 degrees
        wrist = np.tile([0.3, 0.2, 1.1], (n_frames, 1))

        elbow_angles = compute_3point_angle(shoulder, elbow, wrist)
        self.assertEqual(elbow_angles.shape, (n_frames,))
        np.testing.assert_allclose(elbow_angles, 90.0)

    def test_degenerate_zero_norm_and_nans(self):
        """Degenerate points (zero-length segments) and NaNs must return NaN without crash."""
        p_zero = np.array([0.0, 0.0, 0.0])
        p_nan = np.array([np.nan, 0.0, 0.0])
        p_valid = np.array([1.0, 0.0, 0.0])

        # p1 == p2 -> zero length
        res_zero = compute_3point_angle(p_zero, p_zero, p_valid)
        self.assertTrue(np.isnan(res_zero))

        # NaN in one point
        res_nan = compute_3point_angle(p_nan, p_zero, p_valid)
        self.assertTrue(np.isnan(res_nan))


class TestComputeKneeAngles(unittest.TestCase):
    """Test suite for compute_knee_angles in metrics.py."""

    def test_return_dictionary_structure(self):
        """Verify dictionary keys and array types/shapes."""
        n_frames = 15
        jt = _create_mock_joint_trajectories(n_frames)
        res = compute_knee_angles(jt)

        self.assertIsInstance(res, dict)
        self.assertIn("left", res)
        self.assertIn("right", res)
        self.assertEqual(res["left"].shape, (n_frames,))
        self.assertEqual(res["right"].shape, (n_frames,))

    def test_straight_legs_included_convention(self):
        """Straight legs should have ~180 degree included back-of-knee angle."""
        n_frames = 10
        jt = _create_mock_joint_trajectories(n_frames)

        # Left leg straight vertical
        jt["lHip"][:, :] = [0.1, 0.0, 1.0]
        jt["lKnee"][:, :] = [0.1, 0.0, 0.5]
        jt["lAnkle"][:, :] = [0.1, 0.0, 0.0]

        # Right leg straight along x-axis
        jt["rHip"][:, :] = [0.0, -0.1, 0.0]
        jt["rKnee"][:, :] = [0.5, -0.1, 0.0]
        jt["rAnkle"][:, :] = [1.0, -0.1, 0.0]

        res = compute_knee_angles(jt, angle_convention="included")
        np.testing.assert_allclose(res["left"], 180.0)
        np.testing.assert_allclose(res["right"], 180.0)

    def test_flexion_convention(self):
        """Flexion angle should be 0 deg when straight and 90 deg when bent at right angle."""
        n_frames = 5
        jt = _create_mock_joint_trajectories(n_frames)

        # Left leg straight
        jt["lHip"][:, :] = [0.0, 0.1, 1.0]
        jt["lKnee"][:, :] = [0.0, 0.1, 0.5]
        jt["lAnkle"][:, :] = [0.0, 0.1, 0.0]

        # Right leg bent at 90 degrees (shank horizontal backward)
        jt["rHip"][:, :] = [0.0, -0.1, 1.0]
        jt["rKnee"][:, :] = [0.0, -0.1, 0.5]
        jt["rAnkle"][:, :] = [-0.5, -0.1, 0.5]

        res_flex = compute_knee_angles(jt, angle_convention="flexion")
        np.testing.assert_allclose(res_flex["left"], 0.0)
        np.testing.assert_allclose(res_flex["right"], 90.0)

    def test_radians_option(self):
        """Test radians=True produces output in [0, pi]."""
        n_frames = 4
        jt = _create_mock_joint_trajectories(n_frames)

        jt["lHip"][:, :] = [0.0, 0.1, 1.0]
        jt["lKnee"][:, :] = [0.0, 0.1, 0.5]
        jt["lAnkle"][:, :] = [0.0, 0.1, 0.0]

        res_rad = compute_knee_angles(jt, radians=True, angle_convention="included")
        np.testing.assert_allclose(res_rad["left"], np.pi)

        res_flex_rad = compute_knee_angles(jt, radians=True, angle_convention="flexion")
        np.testing.assert_allclose(res_flex_rad["left"], 0.0)

    def test_heel_fallback(self):
        """Test fallback to heel joint when ankle joint has NaNs."""
        n_frames = 6
        jt = _create_mock_joint_trajectories(n_frames)

        jt["lHip"][:, :] = [0.0, 0.1, 1.0]
        jt["lKnee"][:, :] = [0.0, 0.1, 0.5]
        # Ankle is all NaN
        jt["lAnkle"][:, :] = np.nan
        # Heel is valid and straight below knee
        jt["lHeel"][:, :] = [0.0, 0.1, 0.0]

        # Without fallback -> NaN
        res_no_fb = compute_knee_angles(jt, use_heel_fallback=False)
        self.assertTrue(np.all(np.isnan(res_no_fb["left"])))

        # With fallback -> valid 180 degrees
        res_fb = compute_knee_angles(jt, use_heel_fallback=True)
        np.testing.assert_allclose(res_fb["left"], 180.0)

    def test_temporal_smoothing(self):
        """Verify that smooth_window_frames smooths high-frequency noise."""
        np.random.seed(42)
        n_frames = 30
        jt = _create_mock_joint_trajectories(n_frames)

        # Baseline right-angle knee
        jt["lHip"][:, :] = [0.0, 0.1, 1.0]
        jt["lKnee"][:, :] = [0.0, 0.1, 0.5]
        jt["lAnkle"][:, :] = [0.4, 0.1, 0.5]

        # Add zero-mean jitter to ankle z coordinate to perturb the angle around 90 deg
        noise = np.random.normal(0.0, 0.03, size=n_frames)
        jt["lAnkle"][:, 2] += noise

        raw = compute_knee_angles(jt, smooth_window_frames=None)["left"]
        smoothed = compute_knee_angles(jt, smooth_window_frames=5)["left"]

        # Smoothed variance should be lower than raw variance
        self.assertLess(np.var(smoothed), np.var(raw))
        # Mean should remain approximately 90 degrees
        self.assertAlmostEqual(float(np.mean(smoothed)), 90.0, delta=1.0)

    def test_invalid_convention_raises(self):
        """Unknown angle_convention must raise ValueError."""
        jt = _create_mock_joint_trajectories(5)
        with self.assertRaises(ValueError):
            compute_knee_angles(jt, angle_convention="unsupported_convention")

    def test_empty_trajectories(self):
        """Empty input dict should return empty arrays."""
        res = compute_knee_angles({})
        self.assertEqual(len(res["left"]), 0)
        self.assertEqual(len(res["right"]), 0)


class TestComputeTurningVelocity(unittest.TestCase):
    """Test suite for compute_hip_orientation and compute_turning_velocity in metrics.py."""

    def test_compute_hip_orientation_basic(self):
        """Verify hip orientation in cardinal directions."""
        jt = _create_mock_joint_trajectories(4)
        jt["midHip"][:, :] = [0.0, 0.0, 0.0]

        # Frame 0: Facing +x -> hips along y-axis (lHip at (0, 0.2), rHip at (0, -0.2))
        # Vector lHip - rHip points along +y -> angle = +90 degrees
        jt["lHip"][0] = [0.0, 0.2, 0.0]
        jt["rHip"][0] = [0.0, -0.2, 0.0]

        # Frame 1: Facing +y -> hips along x-axis (lHip at (-0.2, 0), rHip at (0.2, 0))
        # Vector points along -x -> angle = 180 degrees
        jt["lHip"][1] = [-0.2, 0.0, 0.0]
        jt["rHip"][1] = [0.2, 0.0, 0.0]

        # Frame 2: Facing -x -> hips along y-axis (lHip at (0, -0.2), rHip at (0, 0.2))
        # Vector points along -y -> angle = -90 degrees
        jt["lHip"][2] = [0.0, -0.2, 0.0]
        jt["rHip"][2] = [0.0, 0.2, 0.0]

        # Frame 3: Facing -y -> hips along x-axis (lHip at (0.2, 0), rHip at (-0.2, 0))
        # Vector points along +x -> angle = 0 degrees
        jt["lHip"][3] = [0.2, 0.0, 0.0]
        jt["rHip"][3] = [-0.2, 0.0, 0.0]

        deg = compute_hip_orientation(jt, radians=False)
        self.assertAlmostEqual(deg[0], 90.0, delta=1e-3)
        self.assertAlmostEqual(abs(deg[1]), 180.0, delta=1e-3)
        self.assertAlmostEqual(deg[2], -90.0, delta=1e-3)
        self.assertAlmostEqual(deg[3], 0.0, delta=1e-3)

        rad = compute_hip_orientation(jt, radians=True)
        self.assertAlmostEqual(rad[0], np.pi / 2, delta=1e-3)
        self.assertAlmostEqual(abs(rad[1]), np.pi, delta=1e-3)
        self.assertAlmostEqual(rad[2], -np.pi / 2, delta=1e-3)
        self.assertAlmostEqual(rad[3], 0.0, delta=1e-3)

    def test_constant_counter_clockwise_spin(self):
        """Simulate continuous counter-clockwise rotation at 90 deg/s."""
        fps = 25.0
        n_frames = 100  # 4 seconds
        target_deg_s = 90.0
        omega = np.radians(target_deg_s)  # pi/2 rad/s

        t = np.arange(n_frames) / fps
        theta = omega * t  # monotonically increasing angle

        jt = _create_mock_joint_trajectories(n_frames)
        half_w = 0.2
        # Pelvic lateral axis rotating at angle theta
        # lHip at +half_w in direction theta, rHip at -half_w
        jt["midHip"][:, :] = [0.0, 0.0, 0.95]
        jt["lHip"][:, 0] = half_w * np.cos(theta)
        jt["lHip"][:, 1] = half_w * np.sin(theta)
        jt["lHip"][:, 2] = 0.95

        jt["rHip"][:, 0] = -half_w * np.cos(theta)
        jt["rHip"][:, 1] = -half_w * np.sin(theta)
        jt["rHip"][:, 2] = 0.95

        vel = compute_turning_velocity(jt, fps=fps, smooth_window_frames=7)

        # Excluding edges affected by boundary filters
        valid_vel = vel[5:-5]
        self.assertAlmostEqual(float(np.mean(valid_vel)), target_deg_s, delta=1.5)
        self.assertTrue(np.all(valid_vel > 0))  # CCW is positive

    def test_constant_clockwise_spin(self):
        """Simulate continuous clockwise rotation at -120 deg/s."""
        fps = 25.0
        n_frames = 75  # 3 seconds
        target_deg_s = -120.0
        omega = np.radians(target_deg_s)

        t = np.arange(n_frames) / fps
        theta = omega * t

        jt = _create_mock_joint_trajectories(n_frames)
        half_w = 0.18
        jt["midHip"][:, :] = [5.0, -10.0, 0.9]
        jt["lHip"][:, 0] = jt["midHip"][:, 0] + half_w * np.cos(theta)
        jt["lHip"][:, 1] = jt["midHip"][:, 1] + half_w * np.sin(theta)
        jt["lHip"][:, 2] = 0.9

        jt["rHip"][:, 0] = jt["midHip"][:, 0] - half_w * np.cos(theta)
        jt["rHip"][:, 1] = jt["midHip"][:, 1] - half_w * np.sin(theta)
        jt["rHip"][:, 2] = 0.9

        vel_signed = compute_turning_velocity(jt, fps=fps, absolute=False, smooth_window_frames=7)
        vel_abs = compute_turning_velocity(jt, fps=fps, absolute=True, smooth_window_frames=7)

        valid_signed = vel_signed[5:-5]
        valid_abs = vel_abs[5:-5]

        self.assertAlmostEqual(float(np.mean(valid_signed)), target_deg_s, delta=2.0)
        self.assertAlmostEqual(float(np.mean(valid_abs)), abs(target_deg_s), delta=2.0)

    def test_branch_cut_wrap_around(self):
        """Ensure rotation across +/- 180 deg boundary does not create a 360-deg velocity spike."""
        fps = 25.0
        n_frames = 50
        # Start at 160 deg and rotate CCW to 260 deg (which wraps across +180/-180)
        target_deg_s = 50.0
        omega = np.radians(target_deg_s)

        t = np.arange(n_frames) / fps
        theta = np.radians(160.0) + omega * t

        jt = _create_mock_joint_trajectories(n_frames)
        half_w = 0.2
        jt["midHip"][:, :] = [0.0, 0.0, 1.0]
        jt["lHip"][:, 0] = half_w * np.cos(theta)
        jt["lHip"][:, 1] = half_w * np.sin(theta)
        jt["lHip"][:, 2] = 1.0
        jt["rHip"][:, 0] = -half_w * np.cos(theta)
        jt["rHip"][:, 1] = -half_w * np.sin(theta)
        jt["rHip"][:, 2] = 1.0

        vel = compute_turning_velocity(jt, fps=fps, smooth_window_frames=7)
        valid_vel = vel[5:-5]

        # Without unwrapping, crossing +/-180 would yield ~9000 deg/s spike
        self.assertLess(float(np.max(np.abs(valid_vel))), 100.0)
        self.assertAlmostEqual(float(np.mean(valid_vel)), target_deg_s, delta=1.5)

    def test_stationary_player(self):
        """Player moving straight without turning should have 0 turning velocity."""
        fps = 25.0
        n_frames = 50
        jt = _create_mock_joint_trajectories(n_frames)

        # Player walking along x-axis, constant orientation (hips along y)
        x = np.linspace(0, 10, n_frames)
        jt["midHip"][:, 0] = x
        jt["midHip"][:, 1] = 0.0
        jt["midHip"][:, 2] = 0.95

        jt["lHip"][:, 0] = x
        jt["lHip"][:, 1] = 0.2
        jt["lHip"][:, 2] = 0.95

        jt["rHip"][:, 0] = x
        jt["rHip"][:, 1] = -0.2
        jt["rHip"][:, 2] = 0.95

        vel = compute_turning_velocity(jt, fps=fps)
        np.testing.assert_allclose(vel[3:-3], 0.0, atol=1e-4)

    def test_single_hip_occlusion_fallback(self):
        """Verify fallback calculation when rHip or lHip is missing."""
        fps = 25.0
        n_frames = 60
        target_deg_s = 60.0
        omega = np.radians(target_deg_s)

        t = np.arange(n_frames) / fps
        theta = omega * t

        jt = _create_mock_joint_trajectories(n_frames)
        half_w = 0.2
        jt["midHip"][:, :] = [0.0, 0.0, 0.9]
        jt["lHip"][:, 0] = half_w * np.cos(theta)
        jt["lHip"][:, 1] = half_w * np.sin(theta)
        jt["lHip"][:, 2] = 0.9

        jt["rHip"][:, 0] = -half_w * np.cos(theta)
        jt["rHip"][:, 1] = -half_w * np.sin(theta)
        jt["rHip"][:, 2] = 0.9

        # Simulate rHip occlusion during frames 10 to 20
        jt["rHip"][10:21] = np.nan
        # Simulate lHip occlusion during frames 35 to 45
        jt["lHip"][35:46] = np.nan

        vel = compute_turning_velocity(jt, fps=fps, smooth_window_frames=7)
        valid_vel = vel[5:-5]

        # Velocity should remain stable across occlusions
        self.assertAlmostEqual(float(np.mean(valid_vel)), target_deg_s, delta=2.5)

    def test_radians_output(self):
        """Verify output in radians per second."""
        fps = 25.0
        n_frames = 50
        target_deg_s = 90.0
        omega = np.radians(target_deg_s)

        t = np.arange(n_frames) / fps
        theta = omega * t

        jt = _create_mock_joint_trajectories(n_frames)
        jt["midHip"][:, :] = [0.0, 0.0, 1.0]
        jt["lHip"][:, 0] = 0.2 * np.cos(theta)
        jt["lHip"][:, 1] = 0.2 * np.sin(theta)
        jt["lHip"][:, 2] = 1.0
        jt["rHip"][:, 0] = -0.2 * np.cos(theta)
        jt["rHip"][:, 1] = -0.2 * np.sin(theta)
        jt["rHip"][:, 2] = 1.0

        vel_deg = compute_turning_velocity(jt, fps=fps, radians=False)
        vel_rad = compute_turning_velocity(jt, fps=fps, radians=True)

        np.testing.assert_allclose(np.radians(vel_deg), vel_rad, rtol=1e-5)
        self.assertAlmostEqual(float(np.mean(vel_rad[5:-5])), omega, delta=0.03)

    def test_smoothing_reduces_noise(self):
        """Verify that Savitzky-Golay smoothing reduces velocity variance from coordinate jitter."""
        np.random.seed(42)
        fps = 25.0
        n_frames = 100
        jt = _create_mock_joint_trajectories(n_frames)

        jt["midHip"][:, :] = [0.0, 0.0, 1.0]
        jt["lHip"][:, 0] = 0.0
        jt["lHip"][:, 1] = 0.2
        jt["lHip"][:, 2] = 1.0
        jt["rHip"][:, 0] = 0.0
        jt["rHip"][:, 1] = -0.2
        jt["rHip"][:, 2] = 1.0

        # Add Gaussian coordinate noise
        jt["lHip"][:, :2] += np.random.normal(0.0, 0.005, size=(n_frames, 2))
        jt["rHip"][:, :2] += np.random.normal(0.0, 0.005, size=(n_frames, 2))

        vel_raw = compute_turning_velocity(jt, fps=fps, smooth_window_frames=None)
        vel_smooth = compute_turning_velocity(jt, fps=fps, smooth_window_frames=9)

        var_raw = np.var(vel_raw[5:-5])
        var_smooth = np.var(vel_smooth[5:-5])

        self.assertLess(var_smooth, var_raw)

    def test_spine_axis_3d(self):
        """Simulate turning around a tilted 3D spine axis."""
        fps = 25.0
        n_frames = 60
        target_deg_s = 90.0
        omega = np.radians(target_deg_s)

        t = np.arange(n_frames) / fps
        theta = omega * t

        jt = _create_mock_joint_trajectories(n_frames)
        # Spine tilted forward by 45 degrees in x-z plane
        tilt = np.radians(45.0)
        spine_dir = np.array([np.sin(tilt), 0.0, np.cos(tilt)])

        jt["midHip"][:, :] = [0.0, 0.0, 0.8]
        jt["neck"][:, 0] = jt["midHip"][:, 0] + 0.6 * spine_dir[0]
        jt["neck"][:, 1] = jt["midHip"][:, 1] + 0.6 * spine_dir[1]
        jt["neck"][:, 2] = jt["midHip"][:, 2] + 0.6 * spine_dir[2]

        # Two orthogonal basis vectors forming a right-handed basis: b1 x b2 = spine_dir
        b1 = np.array([np.cos(tilt), 0.0, -np.sin(tilt)])  # perpendicular in x-z
        b2 = np.array([0.0, 1.0, 0.0])  # y-axis

        half_w = 0.2
        for i in range(n_frames):
            u = np.cos(theta[i]) * b1 + np.sin(theta[i]) * b2
            jt["lHip"][i] = jt["midHip"][i] + half_w * u
            jt["rHip"][i] = jt["midHip"][i] - half_w * u

        vel = compute_turning_velocity(jt, fps=fps, axis="spine", smooth_window_frames=7)
        valid_vel = vel[5:-5]
        self.assertAlmostEqual(float(np.mean(valid_vel)), target_deg_s, delta=2.5)

    def test_invalid_axis_and_edge_cases(self):
        """Invalid axis raises ValueError; empty input returns empty array."""
        jt = _create_mock_joint_trajectories(10)
        with self.assertRaises(ValueError):
            compute_turning_velocity(jt, axis="invalid_axis")

        empty_res = compute_turning_velocity({})
        self.assertEqual(len(empty_res), 0)

        empty_orient = compute_hip_orientation({})
        self.assertEqual(len(empty_orient), 0)

        # All NaNs
        nan_res = compute_turning_velocity(jt)
        self.assertEqual(len(nan_res), 10)
        self.assertTrue(np.all(np.isnan(nan_res)))


if __name__ == "__main__":
    unittest.main()
