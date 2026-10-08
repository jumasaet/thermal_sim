#!/usr/bin/env python3
"""
live_pseudo_thermal.py - Vista pseudo-térmica LWIR en tiempo real sobre Isaac Sim 6.1 (standalone).

Abre Isaac Sim con GUI, crea una escena simple (suelo, pared, tubo caliente, anomalía y
algunos objetos sin temperatura asignada), y mientras el timeline está en Play muestra en
una ventana OpenCV una imagen que imita una cámara térmica LWIR:

    instance_id_segmentation -> prim_path -> temperatura -> gris 0..255
    + 15 % de luminancia del RGB (volumen/sombras)
    + blur gaussiano (PSF) + ruido dependiente de la señal
    + paleta ironbow (INFERNO) o white-hot + barra de escala

No depende del repo irsim.

Uso:
    ~/isaac_sim/python.sh scripts/live_pseudo_thermal.py
    ~/isaac_sim/python.sh scripts/live_pseudo_thermal.py --resolution 800x600 --palette whitehot
    ~/isaac_sim/python.sh scripts/live_pseudo_thermal.py --autoplay --save-frames outputs/pseudo_thermal --max-frames 30
    ~/isaac_sim/python.sh scripts/live_pseudo_thermal.py --autoplay --save-frames outputs/pseudo_thermal \
        --max-frames 90 --auto-track linear      # -> outputs/pseudo_thermal_linear.mp4 si hay ffmpeg
    ~/isaac_sim/python.sh scripts/live_pseudo_thermal.py --autoplay --auto-track circular --ros2
        # publica /thermal_sim/ir/image_raw (bgr8) y /thermal_sim/rgb/image_raw (rgba8)
    ~/isaac_sim/python.sh scripts/live_pseudo_thermal.py --autoplay --robot --ros2
        # cámara montada en un robot inspector que se maneja con el teclado
    ~/isaac_sim/python.sh scripts/live_pseudo_thermal.py --autoplay --robot --auto-track linear \
        --save-frames outputs/pseudo_thermal --max-frames 90   # el robot recorre el trayecto solo

Controles (con la ventana "Pseudo-Thermal LWIR" enfocada):
    q / Esc  cerrar todo
    c        alternar ironbow <-> white-hot
    n        ruido on/off

Controles del robot (--robot sin --auto-track; con el viewport de Isaac Sim enfocado, NO la ventana cv2):
    W / ↑    avanzar          S / ↓    retroceder
    A / ←    girar izquierda  D / →    girar derecha
    Q        subir            E        bajar
El robot es puramente cinemático (un Xform que se traslada y rota; sin física ni colisiones).

Nota sobre OpenCV: Isaac Sim 6.1 solo trae `opencv-python-headless` (sin cv2.imshow).
Para tener ventana, instalar la variante con GUI en el Python de Isaac SIN tocar numpy:
    ~/isaac_sim/python.sh -m pip install --no-deps opencv-python
Si no hay GUI disponible, el script cae a modo PNG (--display png): escribe el frame en
--png-path y se puede ver con un visor que recargue el archivo (p.ej. `feh -R 0.2 <png>`).
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zlib

import numpy as np

# cv2 se importa ANTES de SimulationApp a propósito: en PYTHONPATH (setup_python_env.sh)
# kit/python/.../site-packages va antes que el prebundle headless de omni.pip.compute, así
# que si el usuario instaló opencv-python (con GUI) se carga esa. Una vez en sys.modules,
# las extensiones de Kit reutilizan el mismo módulo.
try:
    import cv2
except ImportError:  # se reintenta tras arrancar Kit
    cv2 = None


# --------------------------------------------------------------------------------------
# Configuración térmica
# --------------------------------------------------------------------------------------
THERMAL_MAP: dict[str, float] = {
    "/World/Floor": 20.0,
    "/World/Wall": 25.0,
    "/World/HotPipe": 80.0,
    "/World/Anomaly": 150.0,
    "/World/Robot": 35.0,  # solo existe con --robot; la electrónica calienta el cuerpo
}
T_MIN = 15.0
T_MAX = 160.0
T_DEFAULT = 22.0         # prims que no están en THERMAL_MAP
T_DEFAULT_SPREAD = 2.0   # +- °C, offset determinista por hash para distinguir objetos
######################################################################
LUMINANCE_WEIGHT = 0.15  # Porcentaje (0.0 a 1.0) de luces/sombras 3D mezcladas para dar volumen a la temperatura plana.
BLUR_KSIZE = (15, 15)    # Tamaño del área de desenfoque (siempre números impares). Mayor valor = el calor abarca más espacio.
BLUR_SIGMA = 3.0         # Intensidad del desenfoque gaussiano. Mayor valor = bordes de calor más suaves y difuminados.
NOISE_SIGMA_MAX = 4.0    # Cantidad de grano/ruido del sensor térmico (más visible en zonas oscuras/frías). 0.0 = sin ruido.
######################################################################
LUT_MAX_ID = 1 << 20     # por encima se usa np.unique en vez de una LUT densa

CAMERA_PATH = "/World/ThermalCamera"
CAMERA_EYE = (4.5, 3.0, 2.2)
CAMERA_TARGET = (-2.5, 0.0, 1.0)
WINDOW_NAME = "Pseudo-Thermal LWIR"
EMPTY_FRAMES_BEFORE_ORCHESTRATOR = 60

PALETTES = ("whitehot", "ironbow", "hot", "jet")

# --auto-track. La pared está en x=-4 y se extiende a lo largo de Y, así que el recorrido
# "linear" va paralelo a ella (eje Y): los objetos entran y salen del campo sin que la cámara
# atraviese la escena.
AUTO_TRACK_LOOP_FRAMES = 120            # sin --max-frames: ida y vuelta cada 120 frames
LINEAR_TRACK_X = 2.5
LINEAR_TRACK_Y = (-4.5, 4.5)
LINEAR_TRACK_Z = 2.0
LINEAR_TILT_DEG = -12.0                 # negativo = mira hacia abajo
ORBIT_CENTER = (-2.0, 0.0, 0.8)         # entre la anomalía y el tubo
ORBIT_RADIUS = 6.0
ORBIT_ARC_DEG = 90.0                    # centrado en +X (frente a la pared)
ORBIT_Z = 2.2
VIDEO_FPS = 15

# --robot. El origen de /World/Robot está en el suelo, bajo el centro del cuerpo; el frente es +X local.
ROBOT_PATH = "/World/Robot"
ROBOT_CAMERA_PATH = f"{ROBOT_PATH}/ThermalCamera"
ROBOT_BODY_SIZE = (0.6, 0.4, 0.25)      # largo (X) x ancho (Y) x alto (Z), m
ROBOT_BODY_Z = 0.3                      # centro del cuerpo sobre el suelo (cuadrúpedo pequeño)
ROBOT_LEG_RADIUS = 0.03
ROBOT_CAMERA_MOUNT = (0.32, 0.0, 0.35)  # 2 cm delante de la cara frontal (x=+0.30), mirando a +X
ROBOT_Z_MIN = 0.0                       # E no hunde las patas en el suelo
ROBOT_MAX_DT = 0.1                      # s; un frame lento no teletransporta el robot
KEY_HOLD_GRACE_S = 0.1                  # > un intervalo de auto-repeat de X11 (ver KeyboardTeleop)

# --ros2. Se usa el rclpy que trae Isaac (isaacsim.ros2.core/<distro>, compilado para Python 3.12)
# sin habilitar el bridge: el de /opt/ros/humble es cpython-310 y no carga en el Python de Isaac.
ROS2_DISTRO = os.environ.get("ROS_DISTRO") or "humble"
ROS2_BUNDLE_DIR = os.path.join(
    os.environ.get("ISAAC_PATH") or os.path.expanduser("~/isaac_sim"), "exts", "isaacsim.ros2.core", ROS2_DISTRO
)
ROS2_NODE_NAME = "thermal_sim"
ROS2_IR_TOPIC = "/thermal_sim/ir/image_raw"
ROS2_RGB_TOPIC = "/thermal_sim/rgb/image_raw"
ROS2_FRAME_ID = "thermal_camera"
ROS2_QOS_DEPTH = 10
_ROS2_REEXEC_MARK = "LIVE_PSEUDO_THERMAL_ROS2_REEXEC"

RENDERER_ALIASES = {
    "raytracing": "RaytracedLighting",  # "RayTracing" no es un modo válido en 6.1
    "raytracedlighting": "RaytracedLighting",
    "realtimepathtracing": "RealTimePathTracing",
    "pathtracing": "PathTracing",
}


# --------------------------------------------------------------------------------------
# CLI (antes de SimulationApp; parse_known_args para no chocar con args de Kit)
# --------------------------------------------------------------------------------------
def parse_resolution(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"resolución inválida '{text}', usa WxH (p.ej. 640x480)")
    w, h = int(m.group(1)), int(m.group(2))
    if w < 32 or h < 32:
        raise argparse.ArgumentTypeError("resolución demasiado pequeña (mínimo 32x32)")
    return w, h


parser = argparse.ArgumentParser(description="Vista pseudo-térmica LWIR en vivo sobre Isaac Sim")
parser.add_argument("--resolution", type=parse_resolution, default=(640, 480), help="WxH del render product (default 640x480)")
parser.add_argument("--palette", choices=PALETTES, default="jet", help="paleta inicial (default ironbow)") #"jet" for rainbow or "whitehot" for grayscale
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
parser.add_argument("--speed", type=float, default=0.5, help="--robot: velocidad lineal y vertical en m/s (default 0.5)")
parser.add_argument("--turn-speed", type=float, default=30.0, help="--robot: velocidad de giro en grados/s (default 30)")
args, _unknown_args = parser.parse_known_args()


def ensure_ros2_process_env() -> bool:
    """Deja el proceso listo para cargar el rclpy de Isaac; puede relanzar el script una vez.

    Las libs de <bundle>/lib no tienen RUNPATH y el loader solo lee LD_LIBRARY_PATH al arrancar
    el proceso, así que si falta hay que re-ejecutar el intérprete con el entorno corregido.
    Se hace aquí, antes de SimulationApp, para que el relanzamiento sea inmediato.
    """
    lib_dir = os.path.join(ROS2_BUNDLE_DIR, "lib")
    if not os.path.isdir(lib_dir):
        print(f"[pseudo-thermal] --ros2: no existe {lib_dir} (ROS_DISTRO={ROS2_DISTRO}); se continúa sin ROS2.")
        return False
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.environ.setdefault("ROS_DISTRO", ROS2_DISTRO)
    ld_paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    if lib_dir in ld_paths:
        return True
    if os.environ.get(_ROS2_REEXEC_MARK):
        print(f"[pseudo-thermal] --ros2: LD_LIBRARY_PATH sigue sin {lib_dir} tras relanzar; se continúa sin ROS2.")
        return False
    # Al final, como indica Isaac: el bundle trae su propio libssl/libcrypto/spdlog y no debe
    # tapar los de Kit. Si hay un Humble del sistema en el path, sus libs C (mismo ABI) ganan.
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(ld_paths + [lib_dir])
    os.environ[_ROS2_REEXEC_MARK] = "1"
    print("[pseudo-thermal] --ros2: relanzando con LD_LIBRARY_PATH para el rclpy de Isaac...", flush=True)
    os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])
    return False  # no se alcanza


ros2_enabled = args.ros2 and ensure_ros2_process_env()

if args.max_frames < 0:
    parser.error("--max-frames debe ser >= 0")
if args.speed <= 0 or args.turn_speed <= 0:
    parser.error("--speed y --turn-speed deben ser > 0")
if args.max_frames and not args.save_frames:
    if args.auto_track:
        print("[pseudo-thermal] Sin --save-frames, --max-frames solo fija la velocidad del --auto-track.")
    else:
        print("[pseudo-thermal] --max-frames se ignora sin --save-frames.")

RES_W, RES_H = args.resolution
renderer = RENDERER_ALIASES.get(args.renderer.lower(), args.renderer)


def cv2_has_gui() -> bool:
    """False si OpenCV está compilado sin backend GUI (opencv-python-headless)."""
    if cv2 is None:
        return False
    m = re.search(r"^\s*GUI:\s*(\S+)", cv2.getBuildInformation(), re.MULTILINE)
    return not (m and m.group(1).upper() == "NONE")


if args.display == "cv2" and not cv2_has_gui():
    sys.exit(
        "[pseudo-thermal] --display cv2 pedido pero OpenCV no tiene GUI (headless).\n"
        "  Instala:  ~/isaac_sim/python.sh -m pip install --no-deps opencv-python\n"
        "  o usa:    --display png"
    )


# --------------------------------------------------------------------------------------
# Arranque de Isaac Sim
# --------------------------------------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": False, "renderer": renderer})

import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: E402

_ext_manager = omni.kit.app.get_app().get_extension_manager()
_ext_manager.set_extension_enabled_immediate("omni.replicator.core", True)
import omni.replicator.core as rep  # noqa: E402

if cv2 is None:
    _ext_manager.set_extension_enabled_immediate("omni.pip.compute", True)
    try:
        import cv2  # noqa: F811
    except ImportError:
        simulation_app.close()
        sys.exit("[pseudo-thermal] No se encontró OpenCV. Instala: ~/isaac_sim/python.sh -m pip install --no-deps opencv-python")


# --------------------------------------------------------------------------------------
# Escena
# --------------------------------------------------------------------------------------
def make_omnipbr(stage, path: str, rgb, roughness: float = 0.6, metallic: float = 0.0) -> UsdShade.Material:
    """Material OmniPBR autorado con pxr puro (misma estructura que CreateMdlMaterialPrim)."""
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(out)
    material.CreateVolumeOutput("mdl").ConnectToSource(out)
    material.CreateDisplacementOutput("mdl").ConnectToSource(out)
    return material


def bind(prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def add_box(stage, path: str, center, size_xyz, material) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    xf = UsdGeom.XformCommonAPI(cube.GetPrim())
    xf.SetTranslate(Gf.Vec3d(*center))
    xf.SetScale(Gf.Vec3f(*size_xyz))
    bind(cube.GetPrim(), material)


def add_cylinder(stage, path: str, center, radius: float, height: float, axis: str, material) -> None:
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(axis)
    half = {"X": (height / 2, radius, radius), "Y": (radius, height / 2, radius), "Z": (radius, radius, height / 2)}[axis]
    cyl.CreateExtentAttr([Gf.Vec3f(*(-v for v in half)), Gf.Vec3f(*half)])
    UsdGeom.XformCommonAPI(cyl.GetPrim()).SetTranslate(Gf.Vec3d(*center))
    bind(cyl.GetPrim(), material)


def add_sphere(stage, path: str, center, radius: float, material) -> None:
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(radius)
    sph.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -radius), Gf.Vec3f(radius, radius, radius)])
    UsdGeom.XformCommonAPI(sph.GetPrim()).SetTranslate(Gf.Vec3d(*center))
    bind(sph.GetPrim(), material)


def add_look_at_camera(stage, path: str, eye, target, width: int, height: int) -> UsdGeom.Camera:
    cam = UsdGeom.Camera.Define(stage, path)
    h_aperture = 20.955
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateHorizontalApertureAttr(h_aperture)
    cam.CreateVerticalApertureAttr(h_aperture * height / width)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
    # SetLookAt da world->camera (cámara mira a -Z, up +Y); se invierte para camera->world.
    cam_to_world = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(cam_to_world.ExtractTranslation())
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(cam_to_world.ExtractRotationQuat())
    return cam


def build_scene(stage) -> None:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    looks = {
        "floor": make_omnipbr(stage, "/World/Looks/Floor", (0.35, 0.35, 0.37), roughness=0.9),
        "wall": make_omnipbr(stage, "/World/Looks/Wall", (0.72, 0.68, 0.58), roughness=0.8),
        "pipe": make_omnipbr(stage, "/World/Looks/Pipe", (0.72, 0.38, 0.20), roughness=0.35, metallic=0.6),
        "anomaly": make_omnipbr(stage, "/World/Looks/Anomaly", (0.85, 0.08, 0.08), roughness=0.5),
        "crate": make_omnipbr(stage, "/World/Looks/Crate", (0.55, 0.40, 0.22), roughness=0.8),
        "ball": make_omnipbr(stage, "/World/Looks/Ball", (0.15, 0.35, 0.75), roughness=0.4),
    }

    # Con temperatura explícita en THERMAL_MAP
    add_box(stage, "/World/Floor", (0.0, 0.0, -0.05), (20.0, 20.0, 0.1), looks["floor"])
    add_box(stage, "/World/Wall", (-4.0, 0.0, 2.0), (0.3, 12.0, 4.0), looks["wall"])
    add_cylinder(stage, "/World/HotPipe", (-3.5, 0.0, 1.6), radius=0.15, height=8.0, axis="Y", material=looks["pipe"])
    add_box(stage, "/World/Anomaly", (-1.5, -1.0, 0.175), (0.35, 0.35, 0.35), looks["anomaly"])

    # Sin entrada en THERMAL_MAP -> T_DEFAULT +- offset por hash
    add_box(stage, "/World/Crate_01", (-2.2, 1.6, 0.3), (0.6, 0.6, 0.6), looks["crate"])
    add_box(stage, "/World/Crate_02", (-0.8, 2.3, 0.3), (0.6, 0.6, 0.6), looks["crate"])
    add_sphere(stage, "/World/Ball", (0.3, -2.2, 0.3), 0.3, looks["ball"])

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(600.0)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.9, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(1.0)
    UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(Gf.Vec3f(40.0, 0.0, 30.0))

    add_look_at_camera(stage, CAMERA_PATH, eye=CAMERA_EYE, target=CAMERA_TARGET, width=RES_W, height=RES_H)


# --------------------------------------------------------------------------------------
# Robot inspector (--robot): visual, cinemático, sin física ni articulaciones
# --------------------------------------------------------------------------------------
def build_robot(stage) -> None:
    """Cuerpo + cuatro patas fijas bajo /World/Robot (todo hereda los 35 °C de THERMAL_MAP)."""
    UsdGeom.Xform.Define(stage, ROBOT_PATH)
    look = make_omnipbr(stage, "/World/Looks/Robot", (0.12, 0.12, 0.13), roughness=0.5, metallic=0.3)
    length, width, height = ROBOT_BODY_SIZE
    add_box(stage, f"{ROBOT_PATH}/Body", (0.0, 0.0, ROBOT_BODY_Z), ROBOT_BODY_SIZE, look)
    leg_height = ROBOT_BODY_Z - height / 2  # del suelo a la cara inferior del cuerpo
    for name, sx, sy in (("FL", 1, 1), ("FR", 1, -1), ("RL", -1, 1), ("RR", -1, -1)):
        center = (sx * (length / 2 - 0.08), sy * (width / 2 - 0.05), leg_height / 2)
        add_cylinder(stage, f"{ROBOT_PATH}/Leg_{name}", center, ROBOT_LEG_RADIUS, leg_height, "Z", look)


def mount_camera_on_robot(stage) -> None:
    """Re-parenta CAMERA_PATH bajo el robot (conserva la óptica) y la fija al frente mirando a +X local."""
    layer = stage.GetEditTarget().GetLayer()
    if not Sdf.CopySpec(layer, Sdf.Path(CAMERA_PATH), layer, Sdf.Path(ROBOT_CAMERA_PATH)):
        raise RuntimeError(f"no se pudo copiar {CAMERA_PATH} a {ROBOT_CAMERA_PATH}")
    stage.RemovePrim(CAMERA_PATH)
    # Misma pila translate + orient que add_look_at_camera, ahora en coordenadas del robot.
    eye = Gf.Vec3d(*ROBOT_CAMERA_MOUNT)
    cam_to_robot = Gf.Matrix4d().SetLookAt(eye, eye + Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 0, 1)).GetInverse()
    translate_op, orient_op = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT_CAMERA_PATH)).GetOrderedXformOps()
    translate_op.Set(eye)
    orient_op.Set(cam_to_robot.ExtractRotationQuat())


class InspectorRobot:
    """Pose de /World/Robot como translate (x, y, z) + rotateZ (yaw en grados). La cámara hija lo sigue."""

    def __init__(self, stage, position, yaw_deg: float):
        xformable = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT_PATH))
        xformable.ClearXformOpOrder()
        self._translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        self._yaw_op = xformable.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble)
        self.x, self.y, self.z = position
        self.yaw = yaw_deg
        self._apply()

    def _apply(self) -> None:
        self._translate_op.Set(Gf.Vec3d(self.x, self.y, self.z))
        self._yaw_op.Set(self.yaw)

    def drive(self, held: set[str], dt: float, speed: float, turn_speed: float) -> None:
        """Un paso de teleoperación: avance según el heading, giro en yaw y Q/E en Z."""
        forward = ("forward" in held) - ("back" in held)
        turn = ("left" in held) - ("right" in held)  # +yaw gira a la izquierda (stage +Z up)
        lift = ("up" in held) - ("down" in held)
        if not (forward or turn or lift):
            return
        self.yaw = (self.yaw + turn * turn_speed * dt + 180.0) % 360.0 - 180.0
        heading = math.radians(self.yaw)
        self.x += forward * speed * dt * math.cos(heading)
        self.y += forward * speed * dt * math.sin(heading)
        self.z = max(ROBOT_Z_MIN, self.z + lift * speed * dt)
        self._apply()

    def face(self, x: float, y: float, target) -> None:
        """Lleva el robot a (x, y), a su altura actual, con el frente hacia ``target`` (solo yaw)."""
        self.x, self.y = x, y
        self.yaw = math.degrees(math.atan2(target[1] - y, target[0] - x))
        self._apply()

    def hud_line(self) -> str:
        return f"robot x {self.x:+.2f} y {self.y:+.2f} z {self.z:+.2f} m | yaw {self.yaw:+.0f}"


class KeyboardTeleop:
    """Teclas mantenidas en la ventana de Kit (viewport enfocado), vía carb.input.

    X11 emite RELEASE->PRESS mientras una tecla sigue pulsada (auto-repeat): un RELEASE solo cuenta
    si no llega otro PRESS en KEY_HOLD_GRACE_S. Mismo criterio que isaacsim mobility_gen (inputs.py).
    """

    def __init__(self):
        import carb.input
        import omni.appwindow

        keys = carb.input.KeyboardInput
        self._event_type = carb.input.KeyboardEventType
        self._actions = {
            keys.W: "forward", keys.UP: "forward",
            keys.S: "back", keys.DOWN: "back",
            keys.A: "left", keys.LEFT: "left",
            keys.D: "right", keys.RIGHT: "right",
            keys.Q: "up",
            keys.E: "down",
        }
        self._down: dict = {}  # tecla -> None (pulsada) o perf_counter() del RELEASE pendiente
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, event, *args, **kwargs) -> bool:
        if event.input not in self._actions:
            return False  # que siga propagándose (atajos de Kit, campos de texto)
        if event.type in (self._event_type.KEY_PRESS, self._event_type.KEY_REPEAT):
            self._down[event.input] = None
        elif event.type == self._event_type.KEY_RELEASE and event.input in self._down:
            self._down[event.input] = time.perf_counter()
        return True

    def held(self) -> set[str]:
        now = time.perf_counter()
        for key, released in list(self._down.items()):
            if released is not None and now - released >= KEY_HOLD_GRACE_S:
                del self._down[key]
        return {self._actions[key] for key in self._down}

    def close(self) -> None:
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)


def frame_robot_in_viewport(viewport, robot: InspectorRobot) -> None:
    """Pone la cámara libre del viewport detrás y encima del robot, mirándolo (solo al arrancar)."""
    from omni.kit.viewport.utility.camera_state import ViewportCameraState

    heading = math.radians(robot.yaw)
    fx, fy = math.cos(heading), math.sin(heading)
    state = ViewportCameraState(viewport=viewport)
    state.set_position_world(Gf.Vec3d(robot.x - 3.0 * fx, robot.y - 3.0 * fy, robot.z + 2.0), True)
    state.set_target_world(Gf.Vec3d(robot.x + 1.0 * fx, robot.y + 1.0 * fy, robot.z + ROBOT_BODY_Z), True)


# --------------------------------------------------------------------------------------
# Procesamiento pseudo-térmico
# --------------------------------------------------------------------------------------
class PseudoThermalProcessor:
    def __init__(self, thermal_map: dict[str, float]):
        self._thermal_map = thermal_map
        self._keys_longest_first = sorted(thermal_map, key=len, reverse=True)
        self._label_cache: dict[str, float | None] = {}
        self._rng = np.random.default_rng()

    def temperature_for_label(self, label) -> float | None:
        """Temperatura de un prim path; None = background (no es un path USD)."""
        if not isinstance(label, str) or not label.startswith("/"):
            return None
        if label in self._label_cache:
            return self._label_cache[label]
        # El prim renderizado puede ser hijo del path configurado (p.ej. un Mesh bajo un Xform).
        for key in self._keys_longest_first:
            if label == key or label.startswith(key + "/"):
                temp = self._thermal_map[key]
                break
        else:
            # Hash del prim path (no del instance id) para que el offset sea estable entre frames.
            unit = zlib.crc32(label.encode("utf-8")) / 0xFFFFFFFF
            temp = T_DEFAULT + (2.0 * unit - 1.0) * T_DEFAULT_SPREAD
        self._label_cache[label] = temp
        return temp

    def temperature_image(self, ids: np.ndarray, id_to_labels: dict) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (temperatura °C float32 HxW, máscara de background bool HxW)."""
        entries: list[tuple[int, float]] = []
        for key, label in id_to_labels.items():
            try:
                inst_id = int(key)
            except (TypeError, ValueError):
                continue
            if inst_id == 0:
                continue
            temp = self.temperature_for_label(label)
            if temp is not None:
                entries.append((inst_id, temp))

        max_id = int(ids.max()) if ids.size else 0
        if max_id <= LUT_MAX_ID:
            lut = np.full(max_id + 1, np.nan, dtype=np.float32)
            for inst_id, temp in entries:
                if inst_id <= max_id:
                    lut[inst_id] = temp
            temp_img = lut[ids]
        else:
            by_id = dict(entries)
            uniq, inverse = np.unique(ids, return_inverse=True)
            values = np.array([by_id.get(int(u), np.nan) for u in uniq], dtype=np.float32)
            temp_img = values[inverse].reshape(ids.shape)

        background = np.isnan(temp_img)
        temp_img[background] = T_MIN
        return temp_img, background

    def render_clean(self, ids: np.ndarray, id_to_labels: dict, rgba: np.ndarray) -> tuple[np.ndarray, float]:
        """Imagen térmica sin ruido (float32 0..255) y la T máxima visible."""
        temp_img, background = self.temperature_image(ids, id_to_labels)
        t_max_visible = float(temp_img[~background].max()) if (~background).any() else float("nan")

        base = (temp_img - T_MIN) * (255.0 / (T_MAX - T_MIN))
        np.clip(base, 0.0, 255.0, out=base)

        rgb = rgba[..., :3].astype(np.float32)
        luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]  # 0..255
        luminance[background] = 0.0  # el cielo/fondo queda en el color más frío

        img = (1.0 - LUMINANCE_WEIGHT) * base + LUMINANCE_WEIGHT * luminance
        img = cv2.GaussianBlur(img, BLUR_KSIZE, BLUR_SIGMA)
        return img, t_max_visible

    def finalize(self, clean: np.ndarray, noise: bool) -> np.ndarray:
        img = clean
        if noise:
            sigma = NOISE_SIGMA_MAX * (1.0 - clean / 255.0)
            img = clean + self._rng.standard_normal(clean.shape, dtype=np.float32) * sigma
        return np.clip(img, 0.0, 255.0).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Composición de la imagen final (paleta + barra de escala + HUD)
# --------------------------------------------------------------------------------------
_colorbar_cache: dict[tuple[int, str], np.ndarray] = {}


def colorize(gray: np.ndarray, palette: str) -> np.ndarray:
    if palette == "ironbow":
        return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    elif palette == "hot":
        return cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    elif palette == "jet":
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    # Si la paleta es "whitehot" o no coincide con las anteriores, retorna escala de grises
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def colorbar(height: int, palette: str, width: int = 84) -> np.ndarray:
    key = (height, palette)
    if key in _colorbar_cache:
        return _colorbar_cache[key]
    bar = np.full((height, width, 3), 28, dtype=np.uint8)
    top, bottom = 14, height - 14
    n = bottom - top
    gradient = np.linspace(255, 0, n).astype(np.uint8)[:, None].repeat(18, axis=1)
    bar[top:bottom, 8:26] = colorize(gradient, palette)
    cv2.rectangle(bar, (7, top - 1), (26, bottom), (200, 200, 200), 1)
    for t in np.linspace(T_MIN, T_MAX, 6):
        y = int(round(bottom - 1 - (t - T_MIN) / (T_MAX - T_MIN) * (n - 1)))
        cv2.line(bar, (26, y), (31, y), (220, 220, 220), 1)
        cv2.putText(bar, f"{t:.0f}C", (34, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    _colorbar_cache[key] = bar
    return bar


def put_text(img: np.ndarray, text: str, org, scale: float = 0.5) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def compose_frame(gray: np.ndarray, palette: str, hud: list[str], banner: str | None) -> np.ndarray:
    frame = np.hstack([colorize(gray, palette), colorbar(gray.shape[0], palette)])
    for i, line in enumerate(hud):
        put_text(frame, line, (8, 20 + 18 * i))
    if banner:
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        org = ((gray.shape[1] - tw) // 2, (gray.shape[0] + th) // 2)
        put_text(frame, banner, org, scale=0.7)
    return frame


# --------------------------------------------------------------------------------------
# Salida: ventana cv2 o PNG en disco
# --------------------------------------------------------------------------------------
class CvWindowDisplay:
    interactive = True

    def __init__(self, name: str, size: tuple[int, int]):
        self._name = name
        self._shown = False
        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(name, *size)

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self._name, frame)
        self._shown = True

    def poll_key(self) -> int:
        key = cv2.waitKey(1)
        return -1 if key < 0 else key & 0xFF

    def closed_by_user(self) -> bool:
        if not self._shown:
            return False
        try:
            return cv2.getWindowProperty(self._name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def close(self) -> None:
        try:
            cv2.destroyWindow(self._name)
        except cv2.error:
            pass


class PngFileDisplay:
    interactive = False

    def __init__(self, path: str, max_hz: float = 10.0):
        self._path = os.path.abspath(path)
        root, ext = os.path.splitext(self._path)
        self._tmp = f"{root}.tmp{ext or '.png'}"
        self._min_dt = 1.0 / max_hz
        self._last = 0.0

    def show(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last < self._min_dt:
            return
        self._last = now
        if cv2.imwrite(self._tmp, frame):
            os.replace(self._tmp, self._path)  # reemplazo atómico: el visor nunca lee un PNG a medias

    def poll_key(self) -> int:
        return -1

    def closed_by_user(self) -> bool:
        return False

    def close(self) -> None:
        pass


class FrameSaver:
    """Guarda los frames procesados como DIR/frame_NNNNN.png y un summary.json al cerrar."""

    def __init__(self, out_dir: str, max_frames: int = 0):
        self.out_dir = os.path.abspath(out_dir)
        self.max_frames = max_frames
        self.count = 0
        self._play_time = 0.0   # suma de intervalos entre frames consecutivos en Play
        self._intervals = 0
        self._palettes_used: set[str] = set()
        self._started = time.strftime("%Y-%m-%dT%H:%M:%S")
        os.makedirs(self.out_dir, exist_ok=True)
        if any(f.startswith("frame_") and f.endswith(".png") for f in os.listdir(self.out_dir)):
            print(f"[pseudo-thermal] Aviso: {self.out_dir} ya tiene frame_*.png; se sobrescribirán.")
        print(f"[pseudo-thermal] Guardando frames en {self.out_dir}" + (f" (máx {max_frames})" if max_frames else ""))

    @property
    def done(self) -> bool:
        return bool(self.max_frames) and self.count >= self.max_frames

    def save(self, image: np.ndarray, palette: str, dt: float | None) -> None:
        """dt: segundos desde el frame procesado anterior (None tras Stop/Pausa)."""
        path = os.path.join(self.out_dir, f"frame_{self.count:05d}.png")
        if not cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            print(f"[pseudo-thermal] Error escribiendo {path}")
            return
        self.count += 1
        self._palettes_used.add(palette)
        if dt is not None:
            self._play_time += dt
            self._intervals += 1

    def write_summary(self) -> None:
        avg_fps = self._intervals / self._play_time if self._play_time > 0 else 0.0
        summary = {
            "num_frames": self.count,
            "avg_fps": round(avg_fps, 2),
            "thermal_map": THERMAL_MAP,
            "t_min": T_MIN,
            "t_max": T_MAX,
            "t_default": T_DEFAULT,
            "resolution": [RES_W, RES_H],
            "renderer": renderer,
            "palettes_used": sorted(self._palettes_used),
            "max_frames": self.max_frames,
            "auto_track": args.auto_track,
            "robot": args.robot,
            "started": self._started,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = os.path.join(self.out_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[pseudo-thermal] {self.count} frames guardados ({avg_fps:.1f} FPS promedio). Resumen: {path}")


def encode_mp4(frames_dir: str, num_frames: int, trajectory: str) -> str | None:
    """frames_dir/frame_%05d.png -> frames_dir/../pseudo_thermal_<trajectory>.mp4 si hay ffmpeg."""
    if shutil.which("ffmpeg") is None:
        print(f"[pseudo-thermal] ffmpeg no está en el PATH; los PNG quedan en {frames_dir}")
        return None
    out_path = os.path.join(os.path.dirname(frames_dir), f"pseudo_thermal_{trajectory}.mp4")
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(VIDEO_FPS),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-frames:v", str(num_frames),                 # ignora frame_*.png sobrantes de corridas anteriores
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",       # yuv420p exige ancho y alto pares
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_path,
    ]
    # Sin el LD_LIBRARY_PATH de Isaac, que puede hacer que el ffmpeg del sistema cargue libs ajenas.
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"[pseudo-thermal] ffmpeg falló ({result.returncode}): {result.stderr.strip()}")
        return None
    print(f"[pseudo-thermal] Video: {out_path}")
    return out_path


def create_display():
    width = RES_W + colorbar(RES_H, "ironbow").shape[1]
    if args.display in ("auto", "cv2") and cv2_has_gui():
        try:
            return CvWindowDisplay(WINDOW_NAME, (width, RES_H))
        except cv2.error as exc:
            if args.display == "cv2":
                raise
            print(f"[pseudo-thermal] No se pudo abrir ventana cv2 ({exc}); usando modo PNG.")
    elif args.display == "auto":
        print(
            "[pseudo-thermal] OpenCV sin GUI (headless). Modo PNG activo.\n"
            "  Para ventana en vivo: ~/isaac_sim/python.sh -m pip install --no-deps opencv-python"
        )
    display = PngFileDisplay(args.png_path)
    print(f"[pseudo-thermal] Escribiendo frames en {args.png_path}  (ver con: feh -R 0.2 {args.png_path})")
    return display


# --------------------------------------------------------------------------------------
# Movimiento automático de cámara (--auto-track)
# --------------------------------------------------------------------------------------
class CameraTrack:
    """Mueve la cámara frame a frame con un translate op + un orient op (look-at, horizonte nivelado).

    ``frames > 0``: el trayecto completo se recorre en ``frames`` frames y se queda al final.
    ``frames == 0``: ida y vuelta continua cada AUTO_TRACK_LOOP_FRAMES frames.
    Con ``robot`` se mueve el robot (que lleva la cámara) en vez de la cámara.
    """

    def __init__(self, stage, camera_path: str, trajectory: str, frames: int, robot: InspectorRobot | None = None):
        self._robot = robot
        self._xformable = None if robot is not None else UsdGeom.Xformable(stage.GetPrimAtPath(camera_path))
        self.trajectory = trajectory
        self.frames = frames

    def _ops(self):
        """translate + orient (double) del prim; si alguien cambió la pila de ops, la rehace."""
        ops = self._xformable.GetOrderedXformOps()
        if (
            [op.GetOpType() for op in ops] == [UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeOrient]
            and all(op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble for op in ops)
        ):
            return ops
        self._xformable.ClearXformOpOrder()
        return [
            self._xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "autotrack"),
            self._xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "autotrack"),
        ]

    def progress(self, index: int) -> float:
        """0..1 a lo largo del trayecto para el frame ``index``."""
        if self.frames > 0:
            return 0.0 if self.frames == 1 else min(index / (self.frames - 1), 1.0)
        u = (index / (AUTO_TRACK_LOOP_FRAMES - 1)) % 2.0
        return u if u <= 1.0 else 2.0 - u

    def pose(self, s: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """(ojo, punto mirado) en coordenadas del stage (+Z up)."""
        if self.trajectory == "linear":
            y = LINEAR_TRACK_Y[0] + s * (LINEAR_TRACK_Y[1] - LINEAR_TRACK_Y[0])
            eye = (LINEAR_TRACK_X, y, LINEAR_TRACK_Z)
            tilt = math.radians(LINEAR_TILT_DEG)
            return eye, (eye[0] - math.cos(tilt), y, eye[2] + math.sin(tilt))  # hacia -X (pared)
        cx, cy, _ = ORBIT_CENTER
        a = math.radians(-0.5 * ORBIT_ARC_DEG + s * ORBIT_ARC_DEG)
        return (cx + ORBIT_RADIUS * math.cos(a), cy + ORBIT_RADIUS * math.sin(a), ORBIT_Z), ORBIT_CENTER

    def place(self, index: int) -> None:
        eye, target = self.pose(self.progress(index))
        if self._robot is not None:
            # El robot recorre el trayecto en planta, a su propia altura, mirando al mismo punto.
            self._robot.face(eye[0], eye[1], target)
            return
        # SetLookAt con up=+Z construye right = forward x up y up' = right x forward: sin roll.
        cam_to_world = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
        translate_op, orient_op = self._ops()
        translate_op.Set(Gf.Vec3d(*eye))
        orient_op.Set(cam_to_world.ExtractRotationQuat())


# --------------------------------------------------------------------------------------
# Publicación ROS2 (--ros2), con rclpy directamente (sin el bridge de Isaac)
# --------------------------------------------------------------------------------------
class Ros2ImagePublisher:
    def __init__(self):
        # rclpy/sensor_msgs del bundle de Isaac (py3.12) delante de todo; fuera las rutas py3.10
        # de /opt/ros, que existen si se hizo source del ROS del sistema y cuyo rclpy no carga aquí.
        sys.path[:] = [p for p in sys.path if not (p.startswith("/opt/ros/") and "python3.10" in p)]
        bundle_py = os.path.join(ROS2_BUNDLE_DIR, "rclpy")
        if bundle_py not in sys.path:
            sys.path.insert(0, bundle_py)

        import rclpy
        from sensor_msgs.msg import Image

        self._rclpy = rclpy
        self._image_cls = Image
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(ROS2_NODE_NAME)
        self._ir_pub = self._node.create_publisher(Image, ROS2_IR_TOPIC, ROS2_QOS_DEPTH)
        self._rgb_pub = self._node.create_publisher(Image, ROS2_RGB_TOPIC, ROS2_QOS_DEPTH)
        print(
            f"[pseudo-thermal] ROS2 ({os.environ.get('ROS_DISTRO')}, {os.environ.get('RMW_IMPLEMENTATION')}): "
            f"nodo '{ROS2_NODE_NAME}' publicando {ROS2_IR_TOPIC} (bgr8) y {ROS2_RGB_TOPIC} (rgba8)"
        )

    def _image_msg(self, image: np.ndarray, encoding: str, stamp):
        msg = self._image_cls()
        msg.header.stamp = stamp
        msg.header.frame_id = ROS2_FRAME_ID
        msg.height, msg.width = int(image.shape[0]), int(image.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = msg.width * (image.shape[2] if image.ndim == 3 else 1)
        # array('B') entra por la vía rápida del setter; bytes/list validaría elemento a elemento.
        msg.data = array.array("B", np.ascontiguousarray(image).tobytes())
        return msg

    def publish(self, thermal_bgr: np.ndarray, rgba: np.ndarray) -> None:
        stamp = self._node.get_clock().now().to_msg()
        self._ir_pub.publish(self._image_msg(thermal_bgr, "bgr8", stamp))
        self._rgb_pub.publish(self._image_msg(rgba, "rgba8", stamp))
        self._rclpy.spin_once(self._node, timeout_sec=0)

    def close(self) -> None:
        try:
            self._node.destroy_node()
        finally:
            if self._rclpy.ok():
                self._rclpy.shutdown()


def create_ros2_publisher() -> Ros2ImagePublisher | None:
    try:
        return Ros2ImagePublisher()
    except Exception as exc:  # ImportError, fallo de rmw, libs no encontradas...
        lib_dir = os.path.join(ROS2_BUNDLE_DIR, "lib")
        print(
            f"[pseudo-thermal] --ros2 desactivado: no se pudo iniciar rclpy ({type(exc).__name__}: {exc}).\n"
            f"  El Python de Isaac es 3.12; el rclpy de /opt/ros/humble es 3.10 y no sirve aquí. Se usa el de\n"
            f"  {ROS2_BUNDLE_DIR}/rclpy, que necesita (el script lo intenta solo, relanzándose):\n"
            f"    export ROS_DISTRO={ROS2_DISTRO}\n"
            f"    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp\n"
            f"    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{lib_dir}\n"
            f"  El script sigue sin publicar."
        )
        return None


# --------------------------------------------------------------------------------------
# Lectura de anotadores
# --------------------------------------------------------------------------------------
def read_annotators(rgb_annot, seg_annot):
    """(ids HxW uint32, idToLabels, rgba HxWx4) o None si el render product aún no entrega datos."""
    try:
        rgba = rgb_annot.get_data()
        seg = seg_annot.get_data()
    except Exception:
        return None
    if rgba is None or not isinstance(seg, dict) or seg.get("data") is None:
        return None
    rgba = np.asarray(rgba)
    ids = np.asarray(seg["data"])
    id_to_labels = (seg.get("info") or {}).get("idToLabels") or {}
    if rgba.ndim != 3 or rgba.shape[:2] != (RES_H, RES_W) or ids.shape != (RES_H, RES_W):
        return None
    if not id_to_labels and not ids.any() and not rgba.any():
        return None  # buffers con la forma correcta pero aún sin renderizar
    return ids, id_to_labels, rgba


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    simulation_app.update()
    stage = usd_context.get_stage()
    build_scene(stage)

    camera_path = CAMERA_PATH
    robot = None
    if args.robot:
        build_robot(stage)
        mount_camera_on_robot(stage)
        camera_path = ROBOT_CAMERA_PATH
        # Donde estaba la cámara libre, en el suelo, con el frente hacia el punto que ella miraba.
        robot = InspectorRobot(stage, (CAMERA_EYE[0], CAMERA_EYE[1], 0.0), 0.0)
        robot.face(CAMERA_EYE[0], CAMERA_EYE[1], CAMERA_TARGET)

    # Sin --robot, el viewport usa la misma cámara: navegar el viewport (o moverla con el gizmo)
    # mueve también la vista pseudo-térmica. Con --robot se queda en su cámara libre y encuadra el
    # robot: navegar con la cámara montada la despegaría del soporte.
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            if robot is None:
                viewport.camera_path = CAMERA_PATH
            else:
                frame_robot_in_viewport(viewport, robot)
    except Exception as exc:
        if robot is None:
            print(f"[pseudo-thermal] No se pudo fijar la cámara del viewport ({exc}); selecciona {CAMERA_PATH} a mano.")
        else:
            print(f"[pseudo-thermal] No se pudo encuadrar el robot en el viewport ({exc}).")

    # Pose inicial del trayecto antes del warm-up, para que el primer render ya esté en él.
    track = CameraTrack(stage, camera_path, args.auto_track, args.max_frames, robot) if args.auto_track else None
    if track is not None:
        track.place(0)
        span = f"{track.frames} frames" if track.frames else f"ida y vuelta cada {AUTO_TRACK_LOOP_FRAMES} frames"
        mover = "robot" if robot is not None else "cámara"
        print(f"[pseudo-thermal] Auto-track {track.trajectory} ({mover}): {span}.")

    render_product = rep.create.render_product(camera_path, (RES_W, RES_H))
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    seg_annot = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation", init_params={"colorize": False})
    rgb_annot.attach([render_product])
    seg_annot.attach([render_product])

    for _ in range(10):  # deja compilar materiales y arrancar el render product
        simulation_app.update()

    processor = PseudoThermalProcessor(THERMAL_MAP)
    display = create_display()
    saver = FrameSaver(args.save_frames, args.max_frames) if args.save_frames else None
    ros2_pub = create_ros2_publisher() if ros2_enabled else None
    teleop = KeyboardTeleop() if robot is not None and track is None else None
    timeline = omni.timeline.get_timeline_interface()

    print(
        "\n[pseudo-thermal] Listo. Pulsa PLAY en Isaac Sim para ver la imagen pseudo-térmica.\n"
        f"  Resolución {RES_W}x{RES_H}, renderer {renderer}, paleta {args.palette}\n"
        + ("  Teclas en la ventana: q/Esc salir | c paleta | n ruido\n" if display.interactive else "")
        + (
            f"  Robot (viewport de Isaac enfocado): WASD/flechas mover-girar | Q/E subir/bajar"
            f" ({args.speed:g} m/s, {args.turn_speed:g} deg/s)\n"
            if teleop is not None
            else ""
        )
    )
    if args.autoplay:
        timeline.play()

    palette = args.palette
    noise_on = True
    last_clean: np.ndarray | None = None
    last_gray = np.zeros((RES_H, RES_W), dtype=np.uint8)
    last_t_max = float("nan")
    fps = 0.0
    proc_ms = 0.0
    last_frame_t: float | None = None
    empty_frames = 0
    orchestrator_started = False
    data_ok_reported = False
    track_index = 0  # frames procesados en Play; solo avanza cuando hay datos
    last_drive_t = time.perf_counter()

    try:
        while simulation_app.is_running():
            if teleop is not None:  # también en Stop/Pausa: el robot se ve moverse en el viewport
                now = time.perf_counter()
                robot.drive(teleop.held(), min(now - last_drive_t, ROBOT_MAX_DT), args.speed, args.turn_speed)
                last_drive_t = now
            if track is not None and timeline.is_playing():
                track.place(track_index)  # antes de update() para que este render use la nueva pose
            simulation_app.update()

            banner = None
            if timeline.is_playing():
                frame = read_annotators(rgb_annot, seg_annot)
                if frame is None:
                    empty_frames += 1
                    banner = "Esperando datos del render product..."
                    if not orchestrator_started and empty_frames >= EMPTY_FRAMES_BEFORE_ORCHESTRATOR:
                        print(f"[pseudo-thermal] {empty_frames} frames sin datos: llamando rep.orchestrator.run()")
                        rep.orchestrator.run()
                        orchestrator_started = True
                else:
                    if not data_ok_reported:
                        how = "tras rep.orchestrator.run()" if orchestrator_started else "sin rep.orchestrator.run()"
                        print(f"[pseudo-thermal] Anotadores entregando datos ({how}).")
                        data_ok_reported = True
                    empty_frames = 0
                    ids, id_to_labels, rgba = frame
                    t0 = time.perf_counter()
                    last_clean, last_t_max = processor.render_clean(ids, id_to_labels, rgba)
                    last_gray = processor.finalize(last_clean, noise_on)
                    now = time.perf_counter()
                    proc_ms = 0.9 * proc_ms + 0.1 * (now - t0) * 1000.0 if proc_ms else (now - t0) * 1000.0
                    dt = None if last_frame_t is None else now - last_frame_t
                    if dt is not None:
                        inst_fps = 1.0 / max(dt, 1e-6)
                        fps = 0.9 * fps + 0.1 * inst_fps if fps else inst_fps
                    last_frame_t = now
                    track_index += 1
                    # Imagen térmica con la paleta activa, sin HUD ni barra de escala.
                    thermal_bgr = colorize(last_gray, palette)
                    if ros2_pub is not None:
                        ros2_pub.publish(thermal_bgr, rgba)
                    if saver is not None:
                        saver.save(thermal_bgr, palette, dt)
                        if saver.done:
                            print(f"[pseudo-thermal] --max-frames {saver.max_frames} alcanzado; cerrando.")
                            break
            else:
                last_frame_t = None
                banner = "STOP - pulsa Play en Isaac Sim" if timeline.is_stopped() else "PAUSA"

            hud = [
                f"{fps:5.1f} FPS | proc {proc_ms:4.1f} ms",
                f"{palette} | ruido {'on' if noise_on else 'off'}",
            ]
            if not np.isnan(last_t_max):
                hud.append(f"Tmax {last_t_max:.1f}C")
            if robot is not None:
                if teleop is not None:
                    hud.append("WASD: mover | QE: subir/bajar")
                hud.append(robot.hud_line())
            display.show(compose_frame(last_gray, palette, hud, banner))

            key = display.poll_key()
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                palette = PALETTES[(PALETTES.index(palette) + 1) % len(PALETTES)]
            elif key == ord("n"):
                noise_on = not noise_on
                if last_clean is not None and not timeline.is_playing():
                    last_gray = processor.finalize(last_clean, noise_on)
            if display.closed_by_user():
                break
    except KeyboardInterrupt:
        pass
    finally:
        if saver is not None:
            saver.write_summary()
            if saver.count > 0:
                encode_mp4(saver.out_dir, saver.count, args.auto_track or "static")
        if ros2_pub is not None:
            ros2_pub.close()
        if teleop is not None:
            teleop.close()
        display.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
