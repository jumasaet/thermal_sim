"""Fixtures comunes. Los tests solo usan los módulos puros: no arrancan SimulationApp.

    ~/isaac_sim/python.sh -m pytest tests/
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pseudo_thermal.config import load_scene, load_thermal_camera  # noqa: E402

SCENE_YAML = ROOT / "configs" / "scenes" / "demo_industrial.yaml"
CAMERA_YAML = ROOT / "configs" / "thermal_camera.yaml"


@pytest.fixture
def scene():
    return load_scene(SCENE_YAML)


@pytest.fixture
def camera():
    return load_thermal_camera(CAMERA_YAML)
