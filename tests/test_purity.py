"""REGLA CRÍTICA (pseudo_thermal/__init__.py): los módulos puros no cargan omni/pxr/carb/isaacsim."""

import subprocess
import sys

from conftest import ROOT

PURE_MODULES = ("config", "cli", "processing", "visualization", "outputs", "ros2_publisher")
ISAAC_PACKAGES = ("omni", "pxr", "carb", "isaacsim")


def test_pure_modules_do_not_import_isaac():
    # En un intérprete limpio, para no depender de lo que haya importado pytest.
    code = (
        "import sys\n"
        "import pseudo_thermal\n"
        f"for name in {PURE_MODULES!r}:\n"
        "    __import__('pseudo_thermal.' + name)\n"
        f"loaded = sorted(m for m in sys.modules if m.split('.')[0] in {ISAAC_PACKAGES!r})\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
