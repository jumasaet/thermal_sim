"""config.py: los YAML de demo reproducen los valores del script original y los errores son claros."""

import copy

import pytest
import yaml

from conftest import CAMERA_YAML, SCENE_YAML
from pseudo_thermal.config import ConfigError, load_scene, load_thermal_camera


@pytest.fixture
def scene_data():
    return yaml.safe_load(SCENE_YAML.read_text(encoding="utf-8"))


@pytest.fixture
def camera_data():
    return yaml.safe_load(CAMERA_YAML.read_text(encoding="utf-8"))


def load_scene_from(tmp_path, data):
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_scene(path)


def load_camera_from(tmp_path, data):
    path = tmp_path / "camera.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_thermal_camera(path)


# ------------------------------------------------------------------ los YAML de demo = script original
def test_demo_scene_matches_legacy_values(scene):
    # THERMAL_MAP, T_MIN/T_MAX/T_DEFAULT/T_DEFAULT_SPREAD de scripts/live_pseudo_thermal_legacy.py
    assert scene.thermal_map() == {
        "/World/Floor": 20.0,
        "/World/Wall": 25.0,
        "/World/HotPipe": 80.0,
        "/World/Anomaly": 150.0,
        "/World/Robot": 35.0,
    }
    assert list(scene.thermal_map()) == ["/World/Floor", "/World/Wall", "/World/HotPipe", "/World/Anomaly", "/World/Robot"]
    assert scene.thermal.range_c == (15.0, 160.0)
    assert (scene.thermal.default_c, scene.thermal.default_spread_c) == (22.0, 2.0)
    assert [o.name for o in scene.objects] == ["Floor", "Wall", "HotPipe", "Anomaly", "Crate_01", "Crate_02", "Ball"]
    assert (scene.camera.eye, scene.camera.target, scene.camera.focal_length_mm) == ((4.5, 3.0, 2.2), (-2.5, 0.0, 1.0), 18.0)
    assert (scene.robot.speed_m_s, scene.robot.turn_speed_deg_s, scene.robot.mount) == (0.5, 30.0, (0.32, 0.0, 0.35))


def test_demo_camera_matches_legacy_values(camera):
    assert camera.resolution == (640, 480)
    assert camera.palette == "jet"
    assert camera.luminance_weight == 0.15
    assert (camera.blur.kernel, camera.blur.sigma) == (15, 3.0)
    assert (camera.noise.enabled, camera.noise.sigma_max) == (True, 4.0)


def test_object_without_temperature_is_none(scene):
    assert {o.name for o in scene.objects if o.temperature_c is None} == {"Crate_01", "Crate_02", "Ball"}


# ------------------------------------------------------------------ errores
def test_rejects_unknown_key_in_object(tmp_path, scene_data):
    scene_data["objects"][4]["colour"] = [1, 0, 0]
    with pytest.raises(ConfigError, match=r"objects\[4\] \(Crate_01\).*clave desconocida 'colour'"):
        load_scene_from(tmp_path, scene_data)


@pytest.mark.parametrize("section", [None, "thermal", "camera", "robot"])
def test_rejects_unknown_key_in_any_section(tmp_path, scene_data, section):
    target = scene_data if section is None else scene_data[section]
    target["temperatura"] = 1
    with pytest.raises(ConfigError, match="clave desconocida 'temperatura'"):
        load_scene_from(tmp_path, scene_data)


def test_rejects_unknown_key_in_camera_config(tmp_path, camera_data):
    camera_data["blur"]["radius"] = 3
    with pytest.raises(ConfigError, match=r"blur.*clave desconocida 'radius'"):
        load_camera_from(tmp_path, camera_data)


def test_rejects_field_of_another_type(tmp_path, scene_data):
    scene_data["objects"][0]["radius"] = 0.5  # Floor es un box
    with pytest.raises(ConfigError, match=r"\(Floor\).*'radius' no aplica a type: box"):
        load_scene_from(tmp_path, scene_data)


def test_rejects_duplicate_object_name(tmp_path, scene_data):
    scene_data["objects"].append(copy.deepcopy(scene_data["objects"][4]))
    with pytest.raises(ConfigError, match="nombre de objeto duplicado 'Crate_01'"):
        load_scene_from(tmp_path, scene_data)


def test_rejects_reserved_object_name(tmp_path, scene_data):
    scene_data["objects"][6]["name"] = "Robot"
    with pytest.raises(ConfigError, match="'Robot' está reservado"):
        load_scene_from(tmp_path, scene_data)


@pytest.mark.parametrize("field", ["name", "type", "position", "color", "size"])
def test_rejects_missing_object_field(tmp_path, scene_data, field):
    del scene_data["objects"][1][field]
    with pytest.raises(ConfigError, match=f"falta el campo obligatorio '{field}'"):
        load_scene_from(tmp_path, scene_data)


def test_rejects_missing_section(tmp_path, scene_data):
    del scene_data["thermal"]
    with pytest.raises(ConfigError, match="falta el campo obligatorio 'thermal'"):
        load_scene_from(tmp_path, scene_data)


@pytest.mark.parametrize(
    "path, value, message",
    [
        (("objects", 3, "temperature_c"), "caliente", "temperature_c: se esperaba un número, no 'caliente'"),
        (("objects", 3, "temperature_c"), True, "se esperaba un número"),
        (("objects", 3, "position"), [1.0, 2.0], "se esperaba una lista de 3 valores"),
        (("objects", 3, "type"), "cone", "valor 'cone' no válido"),
        (("objects", 3, "color"), [255, 0, 0], "debe ser >= 0 y <= 1"),
        (("objects", 2, "radius"), -0.1, "debe ser > 0"),
        (("thermal", "range_c"), [160.0, 15.0], "el mínimo debe ser menor que el máximo"),
        (("camera", "focal_length_mm"), "18mm", "se esperaba un número"),
        (("objects",), {"Floor": {}}, "se esperaba una lista"),
    ],
)
def test_rejects_wrong_type_or_value(tmp_path, scene_data, path, value, message):
    target = scene_data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfigError, match=message):
        load_scene_from(tmp_path, scene_data)


@pytest.mark.parametrize(
    "path, value, message",
    [
        (("resolution",), [640.0, 480], "se esperaba un entero"),
        (("resolution",), [16, 16], "mínimo 32x32"),
        (("palette",), "rainbow", "valor 'rainbow' no válido"),
        (("blur", "kernel"), 14, "entero impar"),
        (("noise", "enabled"), "yes please", "se esperaba true o false"),
        (("luminance_weight",), 1.5, "debe ser >= 0 y <= 1"),
    ],
)
def test_rejects_wrong_camera_value(tmp_path, camera_data, path, value, message):
    target = camera_data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfigError, match=message):
        load_camera_from(tmp_path, camera_data)


def test_error_names_the_file(tmp_path, scene_data):
    scene_data["objetos"] = []
    with pytest.raises(ConfigError, match=r"^scene\.yaml: clave desconocida 'objetos'"):
        load_scene_from(tmp_path, scene_data)


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(ConfigError, match="no se pudo leer"):
        load_scene(tmp_path / "no_existe.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("thermal: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML mal formado"):
        load_scene(bad)
