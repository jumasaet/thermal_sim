#!/usr/bin/env python3
"""
run_pseudo_thermal.py - Vista pseudo-térmica LWIR en tiempo real sobre Isaac Sim 6.1 (standalone).

Abre Isaac Sim con GUI, construye la escena de un YAML y, mientras el timeline está en Play,
muestra una imagen que imita una cámara térmica LWIR (ver pseudo_thermal/__init__.py).

Uso:
    ~/isaac_sim/python.sh run_pseudo_thermal.py
    ~/isaac_sim/python.sh run_pseudo_thermal.py --scene configs/scenes/demo_industrial.yaml --palette whitehot
    ~/isaac_sim/python.sh run_pseudo_thermal.py --autoplay --save-frames outputs/pseudo_thermal \
        --max-frames 90 --auto-track linear      # -> outputs/pseudo_thermal_linear.mp4 si hay ffmpeg
    ~/isaac_sim/python.sh run_pseudo_thermal.py --autoplay --auto-track circular --ros2
    ~/isaac_sim/python.sh run_pseudo_thermal.py --autoplay --robot --ros2
    ~/isaac_sim/python.sh run_pseudo_thermal.py --help

Controles (con la ventana "Pseudo-Thermal LWIR" enfocada):
    q / Esc  cerrar todo        c  cambiar de paleta        n  ruido on/off

Controles del robot (--robot sin --auto-track; con el viewport de Isaac Sim enfocado, NO la ventana cv2):
    W / ↑  avanzar    S / ↓  retroceder    A / ←  girar izquierda    D / →  girar derecha
    Q  subir          E  bajar

Nota sobre OpenCV: Isaac Sim 6.1 solo trae `opencv-python-headless` (sin cv2.imshow). Sin GUI el
script cae a modo PNG (escribe el frame en --png-path; ver con `feh -R 0.2 <png>`). Para tener
ventana:  ~/isaac_sim/python.sh -m pip install --no-deps opencv-python
"""

from __future__ import annotations

import sys
import time

import numpy as np

from pseudo_thermal import cli, ros2_publisher

# cv2 se importa ANTES de SimulationApp a propósito: en PYTHONPATH (setup_python_env.sh)
# kit/python/.../site-packages va antes que el prebundle headless de omni.pip.compute, así
# que si el usuario instaló opencv-python (con GUI) se carga esa. Una vez en sys.modules,
# las extensiones de Kit reutilizan el mismo módulo.
try:
    import cv2  # noqa: F401
except ImportError:  # se reintenta tras arrancar Kit (ver import_cv2_from_kit)
    cv2 = None


def main() -> None:
    # 1. Argumentos y configuración: YAML de escena y de cámara; los flags de CLI los sobrescriben.
    args = cli.parse_args()
    ros2_enabled = args.ros2 and ros2_publisher.ensure_ros2_process_env()  # puede relanzar el proceso
    scene_cfg, camera_cfg = cli.load_settings(args)
    res_w, res_h = camera_cfg.resolution

    # 2. Isaac Sim. A partir de aquí existen omni/pxr/carb y se pueden importar los módulos de Isaac.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False, "renderer": args.renderer})

    import omni.timeline
    import omni.usd

    if cv2 is None:
        import_cv2_from_kit(simulation_app)
    from pseudo_thermal import outputs, visualization
    from pseudo_thermal.processing import PseudoThermalProcessor
    from pseudo_thermal.robot import ROBOT_CAMERA_PATH, CameraTrack, KeyboardTeleop, setup_viewport, spawn_robot
    from pseudo_thermal.scene_builder import CAMERA_PATH, build_scene
    from pseudo_thermal.sensors import ThermalSensors

    # 3. Escena, robot (opcional) y trayecto automático (opcional).
    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    simulation_app.update()
    stage = usd_context.get_stage()
    build_scene(stage, scene_cfg, camera_cfg.resolution)

    camera_path = CAMERA_PATH
    robot = None
    if args.robot:
        robot = spawn_robot(stage, scene_cfg)
        camera_path = ROBOT_CAMERA_PATH
    setup_viewport(robot)

    # Pose inicial del trayecto antes del warm-up, para que el primer render ya esté en él.
    track = CameraTrack(stage, camera_path, args.auto_track, args.max_frames, robot) if args.auto_track else None
    if track is not None:
        track.place(0)
        print(f"[pseudo-thermal] {track.describe()}")

    # 4. Sensores, procesado y salidas.
    sensors = ThermalSensors(camera_path, camera_cfg.resolution)
    for _ in range(10):  # deja compilar materiales y arrancar el render product
        simulation_app.update()

    processor = PseudoThermalProcessor(scene_cfg.thermal_map(), scene_cfg.thermal, camera_cfg, seed=args.seed)
    display = outputs.create_display(args.display, args.png_path, camera_cfg.resolution)
    saver = None
    if args.save_frames:
        run_info = outputs.run_info(scene_cfg, camera_cfg, args.renderer, args.auto_track, args.robot)
        saver = outputs.FrameSaver(args.save_frames, args.max_frames, run_info)
    ros2_pub = ros2_publisher.create_ros2_publisher() if ros2_enabled else None
    teleop = None
    if robot is not None and track is None:
        teleop = KeyboardTeleop(scene_cfg.robot.speed_m_s, scene_cfg.robot.turn_speed_deg_s)
    timeline = omni.timeline.get_timeline_interface()

    print_ready(args, camera_cfg, display, teleop)
    if args.autoplay:
        timeline.play()

    # 5. Loop: un render por vuelta; en Play se procesa, publica y guarda el frame.
    palette = camera_cfg.palette
    noise_on = camera_cfg.noise.enabled
    last_clean: np.ndarray | None = None
    last_gray = np.zeros((res_h, res_w), dtype=np.uint8)
    last_t_max = float("nan")
    stats = visualization.FrameStats()
    track_index = 0  # frames procesados en Play; solo avanza cuando hay datos

    try:
        while simulation_app.is_running():
            if teleop is not None:  # también en Stop/Pausa: el robot se ve moverse en el viewport
                teleop.drive(robot)
            if track is not None and timeline.is_playing():
                track.place(track_index)  # antes de update() para que este render use la nueva pose
            simulation_app.update()

            banner = None
            if timeline.is_playing():
                frame = sensors.read()
                if frame is None:
                    banner = "Esperando datos del render product..."
                else:
                    ids, id_to_labels, rgba = frame
                    t0 = time.perf_counter()
                    last_clean, last_t_max = processor.render_clean(ids, id_to_labels, rgba)
                    last_gray = processor.finalize(last_clean, noise_on)
                    dt = stats.frame_done(t0)
                    track_index += 1
                    # Imagen térmica con la paleta activa, sin HUD ni barra de escala.
                    thermal_bgr = visualization.colorize(last_gray, palette)
                    if ros2_pub is not None:
                        ros2_pub.publish(thermal_bgr, rgba)
                    if saver is not None:
                        saver.save(thermal_bgr, palette, dt)
                        if saver.done:
                            print(f"[pseudo-thermal] --max-frames {saver.max_frames} alcanzado; cerrando.")
                            break
            else:
                stats.not_playing()
                banner = "STOP - pulsa Play en Isaac Sim" if timeline.is_stopped() else "PAUSA"

            robot_line = robot.hud_line() if robot is not None else None
            hud = visualization.hud_lines(stats, palette, noise_on, last_t_max, robot_line, teleop is not None)
            display.show(visualization.compose_frame(last_gray, palette, scene_cfg.thermal.range_c, hud, banner))

            key = display.poll_key()
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                palette = visualization.next_palette(palette)
            elif key == ord("n"):
                noise_on = not noise_on
                if last_clean is not None and not timeline.is_playing():
                    last_gray = processor.finalize(last_clean, noise_on)
            if display.closed_by_user():
                break
    except KeyboardInterrupt:
        pass
    finally:
        # 6. Cierre: resumen + MP4, ROS2, teclado, ventana e Isaac.
        if saver is not None:
            saver.write_summary()
            if saver.count > 0:
                outputs.encode_mp4(saver.out_dir, saver.count, args.auto_track or "static")
        if ros2_pub is not None:
            ros2_pub.close()
        if teleop is not None:
            teleop.close()
        display.close()
        simulation_app.close()


def import_cv2_from_kit(simulation_app) -> None:
    """cv2 no se pudo importar antes de Kit: se habilita el prebundle de omni.pip.compute y se reintenta."""
    import omni.kit.app

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.pip.compute", True)
    try:
        import cv2  # noqa: F401
    except ImportError:
        simulation_app.close()
        sys.exit("[pseudo-thermal] No se encontró OpenCV. Instala: ~/isaac_sim/python.sh -m pip install --no-deps opencv-python")


def print_ready(args, camera_cfg, display, teleop) -> None:
    res_w, res_h = camera_cfg.resolution
    print(
        "\n[pseudo-thermal] Listo. Pulsa PLAY en Isaac Sim para ver la imagen pseudo-térmica.\n"
        f"  Resolución {res_w}x{res_h}, renderer {args.renderer}, paleta {camera_cfg.palette}\n"
        + ("  Teclas en la ventana: q/Esc salir | c paleta | n ruido\n" if display.interactive else "")
        + (
            f"  Robot (viewport de Isaac enfocado): WASD/flechas mover-girar | Q/E subir/bajar"
            f" ({teleop.speed:g} m/s, {teleop.turn_speed:g} deg/s)\n"
            if teleop is not None
            else ""
        )
    )


if __name__ == "__main__":
    main()
