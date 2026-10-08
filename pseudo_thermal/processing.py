"""Procesamiento pseudo-térmico: segmentación por instancia + RGB -> gris 0..255 (módulo puro).

    instance id -> prim path -> temperatura (°C) -> gris según thermal.range_c
    + luminance_weight de luminancia del RGB + blur gaussiano           (render_clean)
    + ruido gaussiano dependiente de la señal, recorte a uint8          (finalize)
"""

from __future__ import annotations

import zlib

import cv2
import numpy as np

from .config import ThermalCameraConfig, ThermalConfig

LUT_MAX_ID = 1 << 20  # por encima se usa np.unique en vez de una LUT densa


class PseudoThermalProcessor:
    def __init__(
        self,
        thermal_map: dict[str, float],
        thermal: ThermalConfig,
        camera: ThermalCameraConfig,
        seed: int | None = None,
    ):
        """``thermal_map``: prim path -> °C. ``seed``: semilla del ruido (None = distinta en cada corrida)."""
        self._thermal_map = thermal_map
        self._keys_longest_first = sorted(thermal_map, key=len, reverse=True)
        self._label_cache: dict[str, float | None] = {}
        self._rng = np.random.default_rng(seed)
        self._t_min, self._t_max = thermal.range_c
        self._t_default = thermal.default_c
        self._t_default_spread = thermal.default_spread_c
        self._luminance_weight = camera.luminance_weight
        self._blur_ksize = (camera.blur.kernel, camera.blur.kernel)
        self._blur_sigma = camera.blur.sigma
        self._noise_sigma_max = camera.noise.sigma_max

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
            temp = self._t_default + (2.0 * unit - 1.0) * self._t_default_spread
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
        temp_img[background] = self._t_min
        return temp_img, background

    def render_clean(self, ids: np.ndarray, id_to_labels: dict, rgba: np.ndarray) -> tuple[np.ndarray, float]:
        """Imagen térmica sin ruido (float32 0..255) y la T máxima visible."""
        temp_img, background = self.temperature_image(ids, id_to_labels)
        t_max_visible = float(temp_img[~background].max()) if (~background).any() else float("nan")

        base = (temp_img - self._t_min) * (255.0 / (self._t_max - self._t_min))
        np.clip(base, 0.0, 255.0, out=base)

        rgb = rgba[..., :3].astype(np.float32)
        luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]  # 0..255
        luminance[background] = 0.0  # el cielo/fondo queda en el color más frío

        img = (1.0 - self._luminance_weight) * base + self._luminance_weight * luminance
        img = cv2.GaussianBlur(img, self._blur_ksize, self._blur_sigma)
        return img, t_max_visible

    def finalize(self, clean: np.ndarray, noise: bool) -> np.ndarray:
        """Añade el ruido (si ``noise``) y recorta a uint8. Cada llamada con ruido avanza el generador."""
        img = clean
        if noise:
            sigma = self._noise_sigma_max * (1.0 - clean / 255.0)
            img = clean + self._rng.standard_normal(clean.shape, dtype=np.float32) * sigma
        return np.clip(img, 0.0, 255.0).astype(np.uint8)
