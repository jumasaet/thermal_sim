"""Vista pseudo-térmica LWIR en tiempo real sobre Isaac Sim 6.1 (standalone).

    instance_id_segmentation -> prim_path -> temperatura -> gris 0..255
    + luminancia del RGB (volumen/sombras) + blur gaussiano (PSF) + ruido dependiente de la señal
    + paleta + barra de escala

La entrada es ``run_pseudo_thermal.py``; este paquete solo contiene los módulos que usa.

REGLA CRÍTICA DE IMPORTS
------------------------
Módulos "puros" (NumPy / cv2 / PyYAML / stdlib). NO importan omni, pxr, carb ni isaacsim en
ningún nivel, así que se pueden importar y testear sin SimulationApp:

    config          carga y validación de los YAML (dataclasses)
    cli             argparse, precedencia YAML < flags, comprobación de GUI de OpenCV
    processing      PseudoThermalProcessor
    visualization   paletas, barra de escala, HUD, compose_frame
    outputs         displays (cv2/PNG), guardado de frames, summary.json, MP4
    ros2_publisher  nodo rclpy y publishers (rclpy se importa al crear el nodo)

Módulos de Isaac. Solo se importan en ``run_pseudo_thermal.py`` DESPUÉS de crear SimulationApp
(antes de eso omni/pxr/carb no existen):

    scene_builder   escena USD desde el YAML (pxr)
    sensors         render product + anotadores (omni.replicator)
    robot           robot inspector, cámara montada, teclado (carb.input), auto-track

Este ``__init__`` no importa nada a propósito: ``import pseudo_thermal`` tiene que seguir siendo
seguro antes de SimulationApp. ``tests/test_purity.py`` vigila la regla.
"""
