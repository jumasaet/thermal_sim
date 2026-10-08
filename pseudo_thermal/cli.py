"""Línea de comandos y configuración efectiva (módulo puro).

Precedencia: valores de los YAML (--scene, --camera-config); los flags de CLI los sobrescriben.
Todo esto corre antes de SimulationApp, así que un flag o un YAML inválido falla al instante.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import replace
from pathlib import Path

from .config import MIN_RESOLUTION, PALETTES, ConfigError, SceneConfig, ThermalCameraConfig, load_scene, load_thermal_camera
from .ros2_publisher import ROS2_IR_TOPIC, ROS2_RGB_TOPIC

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCENE = PROJECT_DIR / "configs" / "scenes" / "demo_industrial.yaml"
DEFAULT_CAMERA_CONFIG = PROJECT_DIR / "configs" / "thermal_camera.yaml"

RENDERER_ALIASES = {
    "raytracing": "RaytracedLighting",  # "RayTracing" no es un modo válido en 6.1
    "raytracedlighting": "RaytracedLighting",
    "realtimepathtracing": "RealTimePathTracing",
    "pathtracing": "PathTracing",
}


def parse_resolution(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"resolución inválida '{text}', usa WxH (p.ej. 640x480)")
    w, h = int(m.group(1)), int(m.group(2))
    if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
        raise argparse.ArgumentTypeError(f"resolución demasiado pequeña (mínimo {MIN_RESOLUTION}x{MIN_RESOLUTION})")
    return w, h


def build_parser() -> argparse.ArgumentParser:
    # Los flags que sobrescriben un YAML tienen default None: None = "usa el valor del YAML".
    parser = argparse.ArgumentParser(description="Vista pseudo-térmica LWIR en vivo sobre Isaac Sim")
    parser.add_argument("--scene", metavar="PATH", default=str(DEFAULT_SCENE), help="YAML de escena (default configs/scenes/demo_industrial.yaml)")
    parser.add_argument("--camera-config", metavar="PATH", default=str(DEFAULT_CAMERA_CONFIG), help="YAML de la cámara térmica (default configs/thermal_camera.yaml)")
    parser.add_argument("--seed", type=int, default=None, metavar="N", help="semilla del ruido, para que dos corridas sean comparables (default: aleatoria)")
    parser.add_argument("--resolution", type=parse_resolution, default=None, help="WxH del render product (default: el de --camera-config)")
    parser.add_argument("--palette", choices=PALETTES, default=None, help="paleta inicial (default: la de --camera-config)")
    parser.add_argument("--renderer", default="RaytracedLighting", help="RaytracedLighting | RealTimePathTracing | PathTracing")
    parser.add_argument("--display", choices=("auto", "cv2", "png"), default="auto", help="auto: ventana cv2 si hay GUI, si no PNG")
    parser.add_argument("--png-path", default="/tmp/live_pseudo_thermal.png", help="salida en modo PNG")
    parser.add_argument("--autoplay", action="store_true", help="dar Play al timeline automáticamente")
    parser.add_argument("--save-frames", metavar="DIR", default=None, help="guardar cada frame procesado en DIR/frame_NNNNN.png (+ summary.json)")
    parser.add_argument("--max-frames", type=int, default=0, metavar="N", help="con --save-frames: cerrar tras N frames guardados (0 = ilimitado)")
    parser.add_argument(
        "--auto-track",
        choices=("linear", "circular"),
        default=None,
        help="mover la cámara en Play: recorre todo el trayecto en --max-frames frames (sin él, ida y vuelta continua)",
    )
    parser.add_argument("--ros2", action="store_true", help=f"publicar las imágenes en {ROS2_IR_TOPIC} y {ROS2_RGB_TOPIC} con rclpy")
    parser.add_argument(
        "--robot",
        action="store_true",
        help="montar la cámara en un robot inspector cinemático (WASD/flechas + Q/E con el viewport enfocado; con --auto-track sigue el trayecto)",
    )
    parser.add_argument("--speed", type=float, default=None, help="--robot: velocidad lineal y vertical en m/s (default: robot.speed_m_s de --scene)")
    parser.add_argument("--turn-speed", type=float, default=None, help="--robot: velocidad de giro en grados/s (default: robot.turn_speed_deg_s de --scene)")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Solo argparse (parse_known_args, para no chocar con args de Kit). ``renderer`` queda normalizado."""
    args, _unknown_args = build_parser().parse_known_args(argv)
    args.renderer = RENDERER_ALIASES.get(args.renderer.lower(), args.renderer)
    return args


def cv2_has_gui() -> bool:
    """False si OpenCV no se puede importar o está compilado sin backend GUI (opencv-python-headless).

    Import perezoso: se llama antes de SimulationApp, cuando cv2 aún puede no estar disponible.
    """
    try:
        import cv2
    except ImportError:
        return False
    m = re.search(r"^\s*GUI:\s*(\S+)", cv2.getBuildInformation(), re.MULTILINE)
    return not (m and m.group(1).upper() == "NONE")


def load_settings(args: argparse.Namespace) -> tuple[SceneConfig, ThermalCameraConfig]:
    """Valida los flags, carga los YAML y les aplica los flags. Sale con un mensaje claro si algo falla."""
    parser = build_parser()
    if args.max_frames < 0:
        parser.error("--max-frames debe ser >= 0")
    if (args.speed is not None and args.speed <= 0) or (args.turn_speed is not None and args.turn_speed <= 0):
        parser.error("--speed y --turn-speed deben ser > 0")

    try:
        scene = load_scene(args.scene)
        camera = load_thermal_camera(args.camera_config)
    except ConfigError as exc:
        sys.exit(f"[pseudo-thermal] Configuración inválida: {exc}")

    if args.resolution is not None:
        camera = replace(camera, resolution=args.resolution)
    if args.palette is not None:
        camera = replace(camera, palette=args.palette)
    if args.speed is not None:
        scene = replace(scene, robot=replace(scene.robot, speed_m_s=args.speed))
    if args.turn_speed is not None:
        scene = replace(scene, robot=replace(scene.robot, turn_speed_deg_s=args.turn_speed))

    if args.max_frames and not args.save_frames:
        if args.auto_track:
            print("[pseudo-thermal] Sin --save-frames, --max-frames solo fija la velocidad del --auto-track.")
        else:
            print("[pseudo-thermal] --max-frames se ignora sin --save-frames.")
    if args.display == "cv2" and not cv2_has_gui():
        sys.exit(
            "[pseudo-thermal] --display cv2 pedido pero OpenCV no tiene GUI (headless).\n"
            "  Instala:  ~/isaac_sim/python.sh -m pip install --no-deps opencv-python\n"
            "  o usa:    --display png"
        )
    return scene, camera
