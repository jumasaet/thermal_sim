"""cli.py: precedencia YAML < flags y flags heredados del script original."""

import pytest

from pseudo_thermal import cli


def settings(*argv):
    args = cli.parse_args(list(argv))
    return args, *cli.load_settings(args)


def test_defaults_come_from_yaml(scene, camera):
    args, scene_cfg, camera_cfg = settings()
    assert scene_cfg == scene
    assert camera_cfg == camera
    assert args.seed is None and args.renderer == "RaytracedLighting"


def test_cli_flags_override_yaml(scene, camera):
    _, scene_cfg, camera_cfg = settings("--resolution", "800x600", "--palette", "whitehot", "--speed", "1.5", "--turn-speed", "45")
    assert camera_cfg.resolution == (800, 600)
    assert camera_cfg.palette == "whitehot"
    assert (scene_cfg.robot.speed_m_s, scene_cfg.robot.turn_speed_deg_s) == (1.5, 45.0)
    # Lo que no tiene flag no cambia.
    assert camera_cfg.blur == camera.blur and camera_cfg.noise == camera.noise
    assert scene_cfg.thermal == scene.thermal and scene_cfg.objects == scene.objects


def test_all_legacy_flags_still_parse():
    args = cli.parse_args(
        "--resolution 320x240 --palette hot --renderer raytracing --display png --png-path /tmp/x.png --autoplay "
        "--save-frames out --max-frames 5 --auto-track circular --ros2 --robot --speed 1 --turn-speed 10 --seed 7".split()
    )
    assert args.resolution == (320, 240) and args.renderer == "RaytracedLighting"
    assert (args.display, args.png_path, args.autoplay) == ("png", "/tmp/x.png", True)
    assert (args.save_frames, args.max_frames, args.auto_track) == ("out", 5, "circular")
    assert (args.ros2, args.robot, args.speed, args.turn_speed, args.seed) == (True, True, 1.0, 10.0, 7)


def test_invalid_scene_exits_with_clear_message(tmp_path):
    bad = tmp_path / "scene.yaml"
    bad.write_text("thermal: {range_c: [15, 160]}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Configuración inválida: scene.yaml: falta el campo obligatorio 'camera'"):
        settings("--scene", str(bad))


@pytest.mark.parametrize("argv", [["--max-frames", "-1"], ["--speed", "0"], ["--turn-speed", "-5"]])
def test_invalid_flag_values_are_rejected(argv):
    with pytest.raises(SystemExit):
        settings(*argv)
