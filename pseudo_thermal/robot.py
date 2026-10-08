"""Robot inspector, cámara montada, teclado y --auto-track (módulo de Isaac: importar solo tras
crear SimulationApp).

El robot es visual y puramente cinemático: un Xform que se traslada y rota, sin física,
colisiones ni articulaciones. Su origen está en el suelo, bajo el centro del cuerpo; el frente
es +X local.
"""

from __future__ import annotations

import math
import time

from pxr import Gf, Sdf, UsdGeom

from .config import ROBOT_PATH, SceneConfig
from .scene_builder import CAMERA_PATH, LOOKS_PATH, add_box, add_cylinder, make_omnipbr

ROBOT_CAMERA_PATH = f"{ROBOT_PATH}/ThermalCamera"
ROBOT_BODY_SIZE = (0.6, 0.4, 0.25)      # largo (X) x ancho (Y) x alto (Z), m
ROBOT_BODY_Z = 0.3                      # centro del cuerpo sobre el suelo (cuadrúpedo pequeño)
ROBOT_LEG_RADIUS = 0.03
ROBOT_Z_MIN = 0.0                       # E no hunde las patas en el suelo
ROBOT_MAX_DT = 0.1                      # s; un frame lento no teletransporta el robot
KEY_HOLD_GRACE_S = 0.1                  # > un intervalo de auto-repeat de X11 (ver KeyboardTeleop)

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


def build_robot(stage) -> None:
    """Cuerpo + cuatro patas fijas bajo ROBOT_PATH (todo hereda robot.temperature_c del YAML)."""
    UsdGeom.Xform.Define(stage, ROBOT_PATH)
    look = make_omnipbr(stage, f"{LOOKS_PATH}/Robot", (0.12, 0.12, 0.13), roughness=0.5, metallic=0.3)
    length, width, height = ROBOT_BODY_SIZE
    add_box(stage, f"{ROBOT_PATH}/Body", (0.0, 0.0, ROBOT_BODY_Z), ROBOT_BODY_SIZE, look)
    leg_height = ROBOT_BODY_Z - height / 2  # del suelo a la cara inferior del cuerpo
    for name, sx, sy in (("FL", 1, 1), ("FR", 1, -1), ("RL", -1, 1), ("RR", -1, -1)):
        center = (sx * (length / 2 - 0.08), sy * (width / 2 - 0.05), leg_height / 2)
        add_cylinder(stage, f"{ROBOT_PATH}/Leg_{name}", center, ROBOT_LEG_RADIUS, leg_height, "Z", look)


def mount_camera_on_robot(stage, mount) -> None:
    """Re-parenta CAMERA_PATH bajo el robot (conserva la óptica) y la fija en ``mount`` mirando a +X local."""
    layer = stage.GetEditTarget().GetLayer()
    if not Sdf.CopySpec(layer, Sdf.Path(CAMERA_PATH), layer, Sdf.Path(ROBOT_CAMERA_PATH)):
        raise RuntimeError(f"no se pudo copiar {CAMERA_PATH} a {ROBOT_CAMERA_PATH}")
    stage.RemovePrim(CAMERA_PATH)
    # Misma pila translate + orient que add_look_at_camera, ahora en coordenadas del robot.
    eye = Gf.Vec3d(*mount)
    cam_to_robot = Gf.Matrix4d().SetLookAt(eye, eye + Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 0, 1)).GetInverse()
    translate_op, orient_op = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT_CAMERA_PATH)).GetOrderedXformOps()
    translate_op.Set(eye)
    orient_op.Set(cam_to_robot.ExtractRotationQuat())


class InspectorRobot:
    """Pose de ROBOT_PATH como translate (x, y, z) + rotateZ (yaw en grados). La cámara hija lo sigue."""

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


def spawn_robot(stage, scene: SceneConfig) -> InspectorRobot:
    """Crea el robot, le monta la cámara de la escena (queda en ROBOT_CAMERA_PATH) y lo coloca."""
    build_robot(stage)
    mount_camera_on_robot(stage, scene.robot.mount)
    # Donde estaba la cámara libre, en el suelo, con el frente hacia el punto que ella miraba.
    eye, target = scene.camera.eye, scene.camera.target
    robot = InspectorRobot(stage, (eye[0], eye[1], 0.0), 0.0)
    robot.face(eye[0], eye[1], target)
    return robot


class KeyboardTeleop:
    """Teclas mantenidas en la ventana de Kit (viewport enfocado), vía carb.input.

    X11 emite RELEASE->PRESS mientras una tecla sigue pulsada (auto-repeat): un RELEASE solo cuenta
    si no llega otro PRESS en KEY_HOLD_GRACE_S. Mismo criterio que isaacsim mobility_gen (inputs.py).
    """

    def __init__(self, speed: float, turn_speed: float):
        """``speed``: m/s (lineal y vertical); ``turn_speed``: grados/s."""
        import carb.input
        import omni.appwindow

        keys = carb.input.KeyboardInput
        self.speed = speed
        self.turn_speed = turn_speed
        self._last_drive_t: float | None = None
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

    def drive(self, robot: InspectorRobot) -> None:
        """Mueve el robot según las teclas mantenidas y el tiempo desde la llamada anterior."""
        now = time.perf_counter()
        dt = 0.0 if self._last_drive_t is None else min(now - self._last_drive_t, ROBOT_MAX_DT)
        self._last_drive_t = now
        robot.drive(self.held(), dt, self.speed, self.turn_speed)

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


def setup_viewport(robot: InspectorRobot | None) -> None:
    """Sin robot, el viewport usa la cámara térmica: navegar el viewport (o moverla con el gizmo)
    mueve también la vista pseudo-térmica. Con robot se queda en su cámara libre y encuadra el
    robot: navegar con la cámara montada la despegaría del soporte.
    """
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

    def describe(self) -> str:
        span = f"{self.frames} frames" if self.frames else f"ida y vuelta cada {AUTO_TRACK_LOOP_FRAMES} frames"
        mover = "robot" if self._robot is not None else "cámara"
        return f"Auto-track {self.trajectory} ({mover}): {span}."

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
