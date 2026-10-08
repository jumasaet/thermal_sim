"""PseudoThermalProcessor: temperatura -> gris, defaults, fondo, herencia y ruido con semilla."""

from dataclasses import replace

import numpy as np
import pytest

from pseudo_thermal.processing import PseudoThermalProcessor

SHAPE = (48, 64)  # alto x ancho


def make_processor(scene, camera, seed=None, thermal_map=None):
    return PseudoThermalProcessor(scene.thermal_map() if thermal_map is None else thermal_map, scene.thermal, camera, seed=seed)


def uniform_frame(inst_id: int, rgb: int = 0):
    """Todo el frame cubierto por una sola instancia, con un RGB plano."""
    ids = np.full(SHAPE, inst_id, dtype=np.uint32)
    rgba = np.full((*SHAPE, 4), rgb, dtype=np.uint8)
    return ids, rgba


# ------------------------------------------------------------------ temperatura explícita -> gris
@pytest.mark.parametrize(
    "temp_c, expected_gray",
    [(15.0, 0), (160.0, 255), (87.5, 127), (0.0, 0), (300.0, 255)],  # extremos, centro y saturación
)
def test_temperature_maps_to_gray_according_to_range(scene, camera, temp_c, expected_gray):
    assert scene.thermal.range_c == (15.0, 160.0)
    flat = replace(camera, luminance_weight=0.0)  # solo temperatura, sin sombreado
    proc = make_processor(scene, flat, thermal_map={"/World/X": temp_c})
    ids, rgba = uniform_frame(3, rgb=200)
    clean, t_max_visible = proc.render_clean(ids, {"3": "/World/X"}, rgba)
    assert t_max_visible == temp_c
    assert np.all(proc.finalize(clean, noise=False) == expected_gray)


@pytest.mark.parametrize("name", ["Floor", "Wall", "HotPipe", "Anomaly"])
def test_scene_object_with_temperature_maps_to_expected_gray(scene, camera, name):
    obj = next(o for o in scene.objects if o.name == name)
    t_min, t_max = scene.thermal.range_c
    luminance = 100.0  # RGB plano (100, 100, 100)
    base = (obj.temperature_c - t_min) * 255.0 / (t_max - t_min)
    expected = (1.0 - camera.luminance_weight) * base + camera.luminance_weight * luminance

    proc = make_processor(scene, camera)
    ids, rgba = uniform_frame(7, rgb=100)
    clean, t_max_visible = proc.render_clean(ids, {"7": obj.prim_path}, rgba)

    assert t_max_visible == obj.temperature_c
    np.testing.assert_allclose(clean, expected, atol=1e-3)  # frame uniforme: el blur no lo cambia
    gray = proc.finalize(clean, noise=False)
    assert gray.dtype == np.uint8
    assert np.all(gray == int(expected))


# ------------------------------------------------------------------ sin temperature_c -> default +- spread
def test_object_without_temperature_gets_stable_default_within_spread(scene, camera):
    thermal = scene.thermal
    proc = make_processor(scene, camera)
    temps = {}
    for name in ("Crate_01", "Crate_02", "Ball"):
        path = f"/World/{name}"
        assert path not in scene.thermal_map()
        temps[name] = proc.temperature_for_label(path)
        assert abs(temps[name] - thermal.default_c) <= thermal.default_spread_c
        assert proc.temperature_for_label(path) == temps[name]                         # segunda llamada
        assert make_processor(scene, camera).temperature_for_label(path) == temps[name]  # otro proceso/instancia
    assert len(set(temps.values())) == 3  # el offset por hash los distingue entre sí

    ids, _ = uniform_frame(9)
    first, _ = proc.temperature_image(ids, {"9": "/World/Crate_01"})
    second, _ = proc.temperature_image(ids, {"9": "/World/Crate_01"})
    np.testing.assert_array_equal(first, second)
    assert np.all(first == np.float32(temps["Crate_01"]))


# ------------------------------------------------------------------ fondo -> valor más frío
@pytest.mark.parametrize(
    "inst_id, id_to_labels",
    [
        (0, {"0": "BACKGROUND", "4": "/World/Anomaly"}),  # id 0
        (5, {"4": "/World/Anomaly"}),                      # id sin label
        (5, {"5": "INVALID"}),                             # label que no es un prim path
    ],
)
def test_background_is_coldest(scene, camera, inst_id, id_to_labels):
    proc = make_processor(scene, camera)
    ids, rgba = uniform_frame(inst_id, rgb=255)  # RGB brillante: la luminancia no debe colarse en el fondo

    temp_img, background = proc.temperature_image(ids, id_to_labels)
    assert background.all()
    assert np.all(temp_img == scene.thermal.range_c[0])

    clean, t_max_visible = proc.render_clean(ids, id_to_labels, rgba)
    assert np.isnan(t_max_visible)
    assert np.all(proc.finalize(clean, noise=False) == 0)


def test_background_is_colder_than_every_object(scene, camera):
    proc = make_processor(scene, camera)
    ids = np.zeros(SHAPE, dtype=np.uint32)
    ids[:, SHAPE[1] // 2 :] = 2  # mitad derecha: suelo (el objeto más frío de la escena)
    temp_img, background = proc.temperature_image(ids, {"2": "/World/Floor"})
    assert background[:, 0].all() and not background[:, -1].any()
    assert temp_img[background].max() < temp_img[~background].min()


# ------------------------------------------------------------------ prim hijo hereda del padre
def test_child_prim_inherits_parent_temperature(scene, camera):
    proc = make_processor(scene, camera)
    assert proc.temperature_for_label("/World/HotPipe") == 80.0
    assert proc.temperature_for_label("/World/HotPipe/Mesh") == 80.0
    assert proc.temperature_for_label("/World/HotPipe/Geom/Mesh_01") == 80.0
    assert proc.temperature_for_label("/World/Robot/Leg_FL") == scene.robot.temperature_c
    # Mismo prefijo de texto pero otro prim: no hereda.
    assert proc.temperature_for_label("/World/HotPipeline") != 80.0

    ids, rgba = uniform_frame(11)
    _, t_max_visible = proc.render_clean(ids, {"11": "/World/Anomaly/Mesh"}, rgba)
    assert t_max_visible == 150.0


def test_most_specific_parent_wins(scene, camera):
    proc = make_processor(scene, camera, thermal_map={"/World/X": 30.0, "/World/X/Hot": 90.0})
    assert proc.temperature_for_label("/World/X/Mesh") == 30.0
    assert proc.temperature_for_label("/World/X/Hot/Mesh") == 90.0


# ------------------------------------------------------------------ ruido con semilla
def test_finalize_is_reproducible_with_same_seed(scene, camera):
    ids, rgba = uniform_frame(4, rgb=80)
    labels = {"4": "/World/Wall"}

    def run(seed):
        proc = make_processor(scene, camera, seed=seed)
        clean, _ = proc.render_clean(ids, labels, rgba)
        return clean, [proc.finalize(clean, noise=True) for _ in range(3)]

    clean, first = run(1234)
    _, second = run(1234)
    _, other = run(4321)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[0], first[1])  # cada frame lleva ruido nuevo
    assert not np.array_equal(first[0], other[0])  # otra semilla, otro ruido
    assert not np.array_equal(first[0], make_processor(scene, camera).finalize(clean, noise=False))


def test_finalize_without_noise_does_not_depend_on_seed(scene, camera):
    ids, rgba = uniform_frame(4, rgb=80)
    results = []
    for seed in (1, 2, None):
        proc = make_processor(scene, camera, seed=seed)
        clean, _ = proc.render_clean(ids, {"4": "/World/Wall"}, rgba)
        results.append(proc.finalize(clean, noise=False))
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0], results[2])
