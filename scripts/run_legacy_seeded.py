#!/usr/bin/env python3
"""
run_legacy_seeded.py --seed N [flags del legacy...] - ejecuta live_pseudo_thermal_legacy.py SIN
modificarlo, pero con el ruido sembrado, para poder compararlo con run_pseudo_thermal.py --seed N.

El legacy crea su generador con np.random.default_rng() (semilla aleatoria) y no tiene flag para
fijarla ni para apagar el ruido. Aquí se sustituye np.random.default_rng para que, llamado sin
semilla, use N; después se ejecuta el script original tal cual, como __main__.

    ~/isaac_sim/python.sh scripts/run_legacy_seeded.py --seed 0 --autoplay --auto-track linear \
        --max-frames 20 --palette whitehot --save-frames outputs/compare_legacy/frames

Solo sirve para la comparación (scripts/compare_frames.py); la demo usa el legacy directamente.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys

import numpy as np

LEGACY_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_pseudo_thermal_legacy.py")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False, description="Legacy con el ruido sembrado")
    parser.add_argument("--seed", type=int, required=True)
    own, legacy_args = parser.parse_known_args()

    unseeded_default_rng = np.random.default_rng
    np.random.default_rng = lambda seed=None: unseeded_default_rng(own.seed if seed is None else seed)

    sys.argv = [LEGACY_SCRIPT, *legacy_args]
    runpy.run_path(LEGACY_SCRIPT, run_name="__main__")


if __name__ == "__main__":
    main()
