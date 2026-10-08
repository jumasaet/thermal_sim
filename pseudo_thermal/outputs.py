"""Salidas: ventana cv2 o PNG en disco, guardado de frames, summary.json y MP4 (módulo puro)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import cv2
import numpy as np

from .cli import cv2_has_gui
from .visualization import COLORBAR_WIDTH

WINDOW_NAME = "Pseudo-Thermal LWIR"
VIDEO_FPS = 15


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


def create_display(mode: str, png_path: str, resolution: tuple[int, int]):
    """``mode``: auto (ventana cv2 si hay GUI, si no PNG) | cv2 | png."""
    res_w, res_h = resolution
    if mode in ("auto", "cv2") and cv2_has_gui():
        try:
            return CvWindowDisplay(WINDOW_NAME, (res_w + COLORBAR_WIDTH, res_h))
        except cv2.error as exc:
            if mode == "cv2":
                raise
            print(f"[pseudo-thermal] No se pudo abrir ventana cv2 ({exc}); usando modo PNG.")
    elif mode == "auto":
        print(
            "[pseudo-thermal] OpenCV sin GUI (headless). Modo PNG activo.\n"
            "  Para ventana en vivo: ~/isaac_sim/python.sh -m pip install --no-deps opencv-python"
        )
    display = PngFileDisplay(png_path)
    print(f"[pseudo-thermal] Escribiendo frames en {png_path}  (ver con: feh -R 0.2 {png_path})")
    return display


class FrameSaver:
    """Guarda los frames procesados como DIR/frame_NNNNN.png y un summary.json al cerrar."""

    def __init__(self, out_dir: str, max_frames: int, run_info: dict):
        """``run_info``: datos fijos de la corrida que van al summary.json (ver ``run_info``)."""
        self.out_dir = os.path.abspath(out_dir)
        self.max_frames = max_frames
        self.count = 0
        self._run_info = run_info
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
        info = self._run_info
        summary = {
            "num_frames": self.count,
            "avg_fps": round(avg_fps, 2),
            "thermal_map": info["thermal_map"],
            "t_min": info["t_min"],
            "t_max": info["t_max"],
            "t_default": info["t_default"],
            "resolution": info["resolution"],
            "renderer": info["renderer"],
            "palettes_used": sorted(self._palettes_used),
            "max_frames": self.max_frames,
            "auto_track": info["auto_track"],
            "robot": info["robot"],
            "started": self._started,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = os.path.join(self.out_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[pseudo-thermal] {self.count} frames guardados ({avg_fps:.1f} FPS promedio). Resumen: {path}")


def run_info(scene, camera, renderer: str, auto_track: str | None, robot: bool) -> dict:
    """Datos fijos de la corrida para el summary.json, a partir de SceneConfig y ThermalCameraConfig."""
    return {
        "thermal_map": scene.thermal_map(),
        "t_min": scene.thermal.range_c[0],
        "t_max": scene.thermal.range_c[1],
        "t_default": scene.thermal.default_c,
        "resolution": list(camera.resolution),
        "renderer": renderer,
        "auto_track": auto_track,
        "robot": robot,
    }


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
