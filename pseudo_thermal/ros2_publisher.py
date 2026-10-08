"""Publicación ROS2 (--ros2) con rclpy directamente, sin el bridge de Isaac (módulo puro).

Se usa el rclpy que trae Isaac (isaacsim.ros2.core/<distro>, compilado para Python 3.12): el de
/opt/ros/humble es cpython-310 y no carga en el Python de Isaac. Aquí solo se usa su ruta en
disco; rclpy se importa al crear el nodo, no al importar este módulo.
"""

from __future__ import annotations

import array
import os
import sys

import numpy as np

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


def ensure_ros2_process_env() -> bool:
    """Deja el proceso listo para cargar el rclpy de Isaac; puede relanzar el script una vez.

    Las libs de <bundle>/lib no tienen RUNPATH y el loader solo lee LD_LIBRARY_PATH al arrancar
    el proceso, así que si falta hay que re-ejecutar el intérprete con el entorno corregido.
    Se llama antes de SimulationApp, para que el relanzamiento sea inmediato.
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
