"""
SkillCorner Body Pose — Joint, Skeleton, and Color Constants
============================================================
"""
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Joint Definitions and Anatomical Skeleton Connections (29 Joints)
# ---------------------------------------------------------------------------

ALL_JOINTS: List[str] = [
    "nose", "neck", "midHip",
    "lShoulder", "rShoulder",
    "lElbow", "rElbow",
    "lWrist", "rWrist",
    "lThumb", "rThumb",
    "lPinky", "rPinky",
    "lHip", "rHip",
    "lKnee", "rKnee",
    "lAnkle", "rAnkle",
    "lHeel", "rHeel",
    "lBigToe", "rBigToe",
    "lSmallToe", "rSmallToe",
    "lEye", "rEye",
    "lEar", "rEar",
]

# Anatomical connections forming the complete skeleton graph
CONNECTIONS: List[Tuple[str, str]] = [
    # Head & Craniofacial
    ("nose", "neck"),
    ("nose", "lEye"),
    ("nose", "rEye"),
    ("lEye", "lEar"),
    ("rEye", "rEar"),
    ("lEye", "rEye"),
    # Torso & Pelvis
    ("neck", "midHip"),
    ("neck", "lShoulder"),
    ("neck", "rShoulder"),
    ("lShoulder", "rShoulder"),
    ("midHip", "lHip"),
    ("midHip", "rHip"),
    ("lHip", "rHip"),
    ("lShoulder", "lHip"),
    ("rShoulder", "rHip"),
    # Left Arm & Hand
    ("lShoulder", "lElbow"),
    ("lElbow", "lWrist"),
    ("lWrist", "lThumb"),
    ("lWrist", "lPinky"),
    ("lThumb", "lPinky"),
    # Right Arm & Hand
    ("rShoulder", "rElbow"),
    ("rElbow", "rWrist"),
    ("rWrist", "rThumb"),
    ("rWrist", "rPinky"),
    ("rThumb", "rPinky"),
    # Left Leg & Foot
    ("lHip", "lKnee"),
    ("lKnee", "lAnkle"),
    ("lAnkle", "lHeel"),
    ("lAnkle", "lBigToe"),
    ("lAnkle", "lSmallToe"),
    ("lBigToe", "lSmallToe"),
    ("lBigToe", "lHeel"),
    ("lSmallToe", "lHeel"),
    # Right Leg & Foot
    ("rHip", "rKnee"),
    ("rKnee", "rAnkle"),
    ("rAnkle", "rHeel"),
    ("rAnkle", "rBigToe"),
    ("rAnkle", "rSmallToe"),
    ("rBigToe", "rSmallToe"),
    ("rBigToe", "rHeel"),
    ("rSmallToe", "rHeel"),
]

# Anatomical color scheme for depth and limb clarity
SIDE_COLORS: Dict[str, str] = {
    "left":   "#e74c3c",   # Coral Red for Left limbs
    "right":  "#3498db",   # Dodger Blue for Right limbs
    "center": "#9b59b6",   # Amethyst Purple for Spine/Torso
    "head":   "#f39c12",   # Amber/Gold for Craniofacial features
}
