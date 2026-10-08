"""Composición de la imagen final: paleta + barra de escala + HUD (módulo puro)."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .config import PALETTES

COLORBAR_WIDTH = 84  # px

_colorbar_cache: dict[tuple, np.ndarray] = {}


def colorize(gray: np.ndarray, palette: str) -> np.ndarray:
    if palette == "ironbow":
        return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    elif palette == "hot":
        return cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    elif palette == "jet":
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Si la paleta es "whitehot" o no coincide con las anteriores, retorna escala de grises
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def next_palette(palette: str) -> str:
    return PALETTES[(PALETTES.index(palette) + 1) % len(PALETTES)]


def colorbar(height: int, palette: str, range_c: tuple[float, float], width: int = COLORBAR_WIDTH) -> np.ndarray:
    """Barra de escala vertical con 6 marcas entre range_c[0] y range_c[1] (°C)."""
    key = (height, palette, range_c, width)
    if key in _colorbar_cache:
        return _colorbar_cache[key]
    t_min, t_max = range_c
    bar = np.full((height, width, 3), 28, dtype=np.uint8)
    top, bottom = 14, height - 14
    n = bottom - top
    gradient = np.linspace(255, 0, n).astype(np.uint8)[:, None].repeat(18, axis=1)
    bar[top:bottom, 8:26] = colorize(gradient, palette)
    cv2.rectangle(bar, (7, top - 1), (26, bottom), (200, 200, 200), 1)
    for t in np.linspace(t_min, t_max, 6):
        y = int(round(bottom - 1 - (t - t_min) / (t_max - t_min) * (n - 1)))
        cv2.line(bar, (26, y), (31, y), (220, 220, 220), 1)
        cv2.putText(bar, f"{t:.0f}C", (34, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    _colorbar_cache[key] = bar
    return bar


def put_text(img: np.ndarray, text: str, org, scale: float = 0.5) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def compose_frame(
    gray: np.ndarray, palette: str, range_c: tuple[float, float], hud: list[str], banner: str | None
) -> np.ndarray:
    """Imagen que se muestra: térmica con paleta + barra de escala, con el HUD y el aviso encima."""
    frame = np.hstack([colorize(gray, palette), colorbar(gray.shape[0], palette, range_c)])
    for i, line in enumerate(hud):
        put_text(frame, line, (8, 20 + 18 * i))
    if banner:
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        org = ((gray.shape[1] - tw) // 2, (gray.shape[0] + th) // 2)
        put_text(frame, banner, org, scale=0.7)
    return frame


class FrameStats:
    """FPS y tiempo de procesado suavizados (media exponencial) para el HUD."""

    def __init__(self):
        self.fps = 0.0
        self.proc_ms = 0.0
        self._last_frame_t: float | None = None

    def frame_done(self, started: float) -> float | None:
        """Tras procesar un frame que empezó en ``started`` (perf_counter).

        Devuelve los segundos desde el frame procesado anterior (None tras Stop/Pausa).
        """
        now = time.perf_counter()
        elapsed_ms = (now - started) * 1000.0
        self.proc_ms = 0.9 * self.proc_ms + 0.1 * elapsed_ms if self.proc_ms else elapsed_ms
        dt = None if self._last_frame_t is None else now - self._last_frame_t
        if dt is not None:
            inst_fps = 1.0 / max(dt, 1e-6)
            self.fps = 0.9 * self.fps + 0.1 * inst_fps if self.fps else inst_fps
        self._last_frame_t = now
        return dt

    def not_playing(self) -> None:
        """En Stop/Pausa: el siguiente frame no cuenta un intervalo."""
        self._last_frame_t = None


def hud_lines(
    stats: FrameStats, palette: str, noise_on: bool, t_max_visible: float, robot_line: str | None, teleop: bool
) -> list[str]:
    hud = [
        f"{stats.fps:5.1f} FPS | proc {stats.proc_ms:4.1f} ms",
        f"{palette} | ruido {'on' if noise_on else 'off'}",
    ]
    if not np.isnan(t_max_visible):
        hud.append(f"Tmax {t_max_visible:.1f}C")
    if robot_line is not None:
        if teleop:
            hud.append("WASD: mover | QE: subir/bajar")
        hud.append(robot_line)
    return hud
