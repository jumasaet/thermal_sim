"""Render product + anotadores rgb e instance_id_segmentation (módulo de Isaac: importar solo
tras crear SimulationApp).
"""

from __future__ import annotations

import numpy as np
import omni.kit.app

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
import omni.replicator.core as rep  # noqa: E402

EMPTY_FRAMES_BEFORE_ORCHESTRATOR = 60


def read_annotators(rgb_annot, seg_annot, resolution: tuple[int, int]):
    """(ids HxW uint32, idToLabels, rgba HxWx4) o None si el render product aún no entrega datos."""
    res_w, res_h = resolution
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
    if rgba.ndim != 3 or rgba.shape[:2] != (res_h, res_w) or ids.shape != (res_h, res_w):
        return None
    if not id_to_labels and not ids.any() and not rgba.any():
        return None  # buffers con la forma correcta pero aún sin renderizar
    return ids, id_to_labels, rgba


class ThermalSensors:
    """Render product de la cámara con los dos anotadores que necesita el procesado."""

    def __init__(self, camera_path: str, resolution: tuple[int, int]):
        self._resolution = resolution
        self._render_product = rep.create.render_product(camera_path, resolution)
        self._rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._seg_annot = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation", init_params={"colorize": False})
        self._rgb_annot.attach([self._render_product])
        self._seg_annot.attach([self._render_product])
        self._empty_frames = 0
        self._orchestrator_started = False
        self._data_ok_reported = False

    def read(self):
        """Frame actual como en read_annotators, o None si aún no hay datos.

        Llamar una vez por update() en Play. Si pasan EMPTY_FRAMES_BEFORE_ORCHESTRATOR frames
        seguidos sin datos, arranca rep.orchestrator (una sola vez).
        """
        frame = read_annotators(self._rgb_annot, self._seg_annot, self._resolution)
        if frame is None:
            self._empty_frames += 1
            if not self._orchestrator_started and self._empty_frames >= EMPTY_FRAMES_BEFORE_ORCHESTRATOR:
                print(f"[pseudo-thermal] {self._empty_frames} frames sin datos: llamando rep.orchestrator.run()")
                rep.orchestrator.run()
                self._orchestrator_started = True
            return None
        if not self._data_ok_reported:
            how = "tras rep.orchestrator.run()" if self._orchestrator_started else "sin rep.orchestrator.run()"
            print(f"[pseudo-thermal] Anotadores entregando datos ({how}).")
            self._data_ok_reported = True
        self._empty_frames = 0
        return frame
