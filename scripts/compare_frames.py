#!/usr/bin/env python3
"""
compare_frames.py DIR_A DIR_B - compara dos corridas de --save-frames frame a frame.

Empareja DIR_A/frame_NNNNN.png con DIR_B/frame_NNNNN.png y reporta, por frame, la diferencia
absoluta media y máxima (0..255, sobre todos los píxeles y canales), más un resumen global.

    ~/isaac_sim/python.sh scripts/compare_frames.py outputs/compare_legacy/frames outputs/compare_refactor/frames

Las diferencias solo son "niveles de gris" si las dos corridas usaron --palette whitehot: con
una paleta de color (jet, ironbow, hot) un nivel de gris se convierte en varios niveles por canal.

Código de salida: 0 si todos los frames tienen pareja y tamaño iguales y la media global es
menor que --max-mean (default 1.0); 1 en caso contrario.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

FRAME_RE = re.compile(r"frame_(\d{5})\.png")


def frames_in(directory: Path) -> dict[int, Path]:
    if not directory.is_dir():
        sys.exit(f"compare_frames: no existe el directorio {directory}")
    frames = {int(m.group(1)): p for p in directory.iterdir() if (m := FRAME_RE.fullmatch(p.name))}
    if not frames:
        sys.exit(f"compare_frames: {directory} no tiene frame_NNNNN.png")
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara frame_NNNNN.png de dos directorios por pares")
    parser.add_argument("dir_a", type=Path, metavar="DIR_A")
    parser.add_argument("dir_b", type=Path, metavar="DIR_B")
    parser.add_argument("--max-mean", type=float, default=1.0, help="media global máxima aceptable (default 1.0)")
    args = parser.parse_args()

    frames_a, frames_b = frames_in(args.dir_a), frames_in(args.dir_b)
    common = sorted(frames_a.keys() & frames_b.keys())
    only_a = sorted(frames_a.keys() - frames_b.keys())
    only_b = sorted(frames_b.keys() - frames_a.keys())

    print(f"A: {args.dir_a}  ({len(frames_a)} frames)")
    print(f"B: {args.dir_b}  ({len(frames_b)} frames)")
    print(f"\n{'frame':>7}  {'media':>8}  {'máx':>5}")

    ok = not (only_a or only_b)
    total_abs = 0.0
    total_values = 0
    worst_max, worst_max_frame = 0, None
    worst_mean, worst_mean_frame = 0.0, None
    for index in common:
        image_a = cv2.imread(str(frames_a[index]), cv2.IMREAD_UNCHANGED)
        image_b = cv2.imread(str(frames_b[index]), cv2.IMREAD_UNCHANGED)
        if image_a is None or image_b is None:
            print(f"{index:7d}  no se pudo leer {'A' if image_a is None else 'B'}")
            ok = False
            continue
        if image_a.shape != image_b.shape:
            print(f"{index:7d}  tamaño distinto: A {image_a.shape} vs B {image_b.shape}")
            ok = False
            continue
        diff = np.abs(image_a.astype(np.int16) - image_b.astype(np.int16))
        frame_mean, frame_max = float(diff.mean()), int(diff.max())
        print(f"{index:7d}  {frame_mean:8.4f}  {frame_max:5d}")
        total_abs += float(diff.sum())
        total_values += diff.size
        if frame_max > worst_max or worst_max_frame is None:
            worst_max, worst_max_frame = frame_max, index
        if frame_mean > worst_mean or worst_mean_frame is None:
            worst_mean, worst_mean_frame = frame_mean, index

    for name, missing in (("A", only_a), ("B", only_b)):
        if missing:
            print(f"\nSolo en {name}: {', '.join(f'frame_{i:05d}.png' for i in missing)}")
    if not total_values:
        print("\nNo se pudo comparar ningún frame.")
        return 1

    global_mean = total_abs / total_values
    ok = ok and global_mean < args.max_mean
    print(
        f"\n{len(common)} frames comparados"
        f" | media global {global_mean:.4f}"
        f" | peor media {worst_mean:.4f} (frame {worst_mean_frame})"
        f" | máximo {worst_max} (frame {worst_max_frame})"
    )
    print(f"{'OK' if ok else 'DIFERENTE'}: media global {'<' if global_mean < args.max_mean else '>='} {args.max_mean:g}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
