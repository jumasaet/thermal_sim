# Pseudo-Thermal LWIR Simulator

Simulador pseudo-térmico en tiempo real sobre NVIDIA Isaac Sim 6.1.0.
Genera imágenes que imitan una cámara térmica LWIR procesando en 2D la
segmentación por instancia y el RGB del viewport.

Parte del WP4 de GITCC/EMPLOI — simulador multimodal para robot
cuadrúpedo de inspección industrial.

## Requisitos

- NVIDIA Isaac Sim 6.1.0 (`~/isaac_sim/`)
- Ubuntu 22.04, RTX 3090
- ROS2 Humble (solo para publicación de topics)
- ffmpeg (solo para codificar MP4)

No hay dependencias de Python que instalar: NumPy, OpenCV, PyYAML y pytest
ya vienen en el Python de Isaac (`~/isaac_sim/python.sh`).

## Setup rápido

```bash
cd ~/multimodal_sim/thermal_sim

# Hacer ejecutable el launcher (una sola vez)
chmod +x launch_thermal.sh
```

## Estructura

```
run_pseudo_thermal.py            entrada: args → (relanzamiento ROS2) → SimulationApp → construir → loop
pseudo_thermal/
    config.py                    carga y valida los YAML (dataclasses)
    cli.py                       argparse y precedencia YAML < flags
    processing.py                PseudoThermalProcessor: ids + RGB → gris 0..255
    visualization.py             paletas, barra de escala, HUD, compose_frame
    outputs.py                   ventana cv2 / PNG, frames, summary.json, MP4
    ros2_publisher.py            nodo rclpy y publishers
    scene_builder.py             escena USD desde el YAML              (Isaac)
    sensors.py                   render product + anotadores           (Isaac)
    robot.py                     robot, cámara montada, teclado, auto-track  (Isaac)
configs/
    scenes/demo_industrial.yaml  objetos, temperaturas, luces, cámara, robot
    thermal_camera.yaml          resolución, paleta, luminancia, blur, ruido
tests/                           tests de los módulos puros (sin Isaac)
scripts/
    compare_frames.py            compara dos corridas de --save-frames
    run_legacy_seeded.py         corre el legacy con el ruido sembrado (solo para comparar)
    live_pseudo_thermal_legacy.py  script original de un solo archivo, sin modificar
launch_thermal.sh                lanza run_pseudo_thermal.py con ROS2
```

**Regla de imports.** Los módulos marcados `(Isaac)` importan `pxr`/`omni`/`carb`
y solo se pueden importar después de crear `SimulationApp`; lo hace
`run_pseudo_thermal.py`. El resto son puros (NumPy, cv2, PyYAML) y no deben
importar nada de Isaac: por eso se pueden testear sin abrir el simulador.
`tests/test_purity.py` falla si alguien rompe la regla.

## Modos de uso

### 1. Grabar con cámara en movimiento (demo)

```bash
~/isaac_sim/python.sh run_pseudo_thermal.py \
    --autoplay \
    --save-frames outputs/pseudo_thermal \
    --max-frames 90 \
    --auto-track linear
```

Trayectorias disponibles: `linear` (paralela a la pared) o `circular`
(arco de 90° alrededor de la escena). Si hay ffmpeg, al cerrar se codifica
`outputs/pseudo_thermal_linear.mp4`.

### 2. Grabar cámara estática

```bash
~/isaac_sim/python.sh run_pseudo_thermal.py \
    --autoplay \
    --save-frames outputs/pseudo_thermal \
    --max-frames 30
```

### 3. Publicar por ROS2 (visualización en RViz)

**Terminal 1 — Isaac Sim** (NO hacer `source /opt/ros/humble/setup.bash` aquí):

```bash
cd ~/multimodal_sim/thermal_sim
./launch_thermal.sh
# o con opciones:
./launch_thermal.sh --auto-track circular --save-frames outputs/demo --max-frames 90
```

**Terminal 2 — ROS2** (esta SÍ lleva source):

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=15
rviz2
# o ver un topic específico:
rqt_image_view /thermal_sim/ir/image_raw
```

Topics publicados:

| Topic | Encoding | Contenido |
|-------|----------|-----------|
| `/thermal_sim/ir/image_raw` | bgr8 | Imagen pseudo-térmica con paleta |
| `/thermal_sim/rgb/image_raw` | rgba8 | Imagen RGB del viewport |

### 4. Modo interactivo con GUI (sin grabar)

```bash
~/isaac_sim/python.sh run_pseudo_thermal.py
```

Abre Isaac Sim, pulsa Play manualmente, y mueve la cámara con el gizmo.
La imagen térmica se actualiza en `/tmp/live_pseudo_thermal.png`
(o en una ventana, si OpenCV tiene GUI; ver Notas).

### 5. Robot inspector

```bash
# Teclado: clic en el viewport de Isaac y WASD/flechas para mover, Q/E para subir/bajar
~/isaac_sim/python.sh run_pseudo_thermal.py --autoplay --robot

# El robot recorre el trayecto solo
~/isaac_sim/python.sh run_pseudo_thermal.py --autoplay --robot --auto-track linear \
    --save-frames outputs/robot_demo --max-frames 90
```

Con la ventana `Pseudo-Thermal LWIR` enfocada (solo si OpenCV tiene GUI):
`q`/`Esc` cierra, `c` cambia de paleta, `n` activa o desactiva el ruido.

## Referencia de flags

| Flag | Default | Descripción |
|------|---------|-------------|
| `--scene PATH` | `configs/scenes/demo_industrial.yaml` | YAML de escena |
| `--camera-config PATH` | `configs/thermal_camera.yaml` | YAML de la cámara térmica |
| `--seed N` | aleatoria | Semilla del ruido (dos corridas con la misma semilla llevan el mismo ruido) |
| `--autoplay` | off | Da Play al timeline automáticamente |
| `--save-frames DIR` | off | Guarda frames como `DIR/frame_NNNNN.png` + `summary.json` |
| `--max-frames N` | 0 (ilimitado) | Para después de N frames y cierra |
| `--auto-track linear\|circular` | off | Mueve la cámara (o el robot) automáticamente |
| `--ros2` | off | Publica topics ROS2 (usar con `launch_thermal.sh`) |
| `--robot` | off | Monta la cámara en el robot inspector |
| `--speed M_S` | YAML de escena (0.5) | `--robot`: velocidad lineal y vertical, m/s |
| `--turn-speed DEG_S` | YAML de escena (30) | `--robot`: velocidad de giro, grados/s |
| `--resolution WxH` | YAML de cámara (640x480) | Resolución del render product |
| `--palette whitehot\|ironbow\|hot\|jet` | YAML de cámara (jet) | Paleta inicial |
| `--display auto\|cv2\|png` | auto | Método de visualización |
| `--png-path PATH` | `/tmp/live_pseudo_thermal.png` | Salida en modo PNG |
| `--renderer` | RaytracedLighting | Renderer de Isaac |

**Precedencia:** valen los valores de los YAML; un flag de CLI, si se pasa,
los sobrescribe para esa corrida.

## Editar la escena desde el YAML

Todo lo que antes eran constantes del script está en dos archivos, con cada
campo comentado (qué es y en qué unidades):

- [configs/scenes/demo_industrial.yaml](configs/scenes/demo_industrial.yaml):
  modelo térmico, objetos, luces, cámara y robot.
- [configs/thermal_camera.yaml](configs/thermal_camera.yaml): resolución,
  paleta, peso de la luminancia, blur y ruido.

Para una escena propia, copia el YAML de demo y pásalo con `--scene`:

```bash
cp configs/scenes/demo_industrial.yaml configs/scenes/mi_escena.yaml
~/isaac_sim/python.sh run_pseudo_thermal.py --scene configs/scenes/mi_escena.yaml
```

### Objetos y temperaturas

Cada entrada de `objects` crea el prim `/World/<name>`:

```yaml
objects:
  - name: Tank              # único; prim /World/Tank
    type: cylinder          # box | cylinder | sphere
    position: [1.0, 1.0, 0.5]   # centro, m
    radius: 0.4             # cylinder y sphere, m
    height: 1.0             # cylinder, m
    axis: Z                 # cylinder: X | Y | Z
    color: [0.2, 0.6, 0.3]  # RGB 0..1
    roughness: 0.5          # opcional (0.6)
    metallic: 0.0           # opcional (0.0)
    temperature_c: 60.0     # opcional, °C
```

Un `box` lleva `size: [x, y, z]` en vez de `radius`/`height`/`axis`.

- **Cambiar una temperatura:** edita `temperature_c` del objeto.
- **Objeto a temperatura ambiente:** quita `temperature_c`. Recibe
  `thermal.default_c` ± `thermal.default_spread_c` (offset fijo por hash del
  prim path, así dos objetos sin temperatura se distinguen entre sí).
- **Rango de visualización:** `thermal.range_c: [mín, máx]` es lo que se mapea
  a gris 0..255 y a la barra de escala. El fondo queda siempre en el mínimo.
- Los prims hijos heredan la temperatura del padre (`/World/Tank/Mesh` usa la
  de `/World/Tank`).

Valores de la escena de demo:

| Objeto | Path USD | Temperatura |
|--------|----------|-------------|
| Suelo | `/World/Floor` | 20 °C |
| Pared | `/World/Wall` | 25 °C |
| Tubo caliente | `/World/HotPipe` | 80 °C |
| Anomalía | `/World/Anomaly` | 150 °C |
| Robot (`--robot`) | `/World/Robot` | 35 °C |
| Cajas y bola | `/World/Crate_01`, `/World/Crate_02`, `/World/Ball` | 22 °C ± 2 °C |

Rango de visualización: 15–160 °C.

### Errores de configuración

Los YAML se validan antes de abrir Isaac Sim. Una clave desconocida, un campo
que falta, un tipo incorrecto o un nombre de objeto repetido paran el
programa con el archivo y el campo en el mensaje:

```
[pseudo-thermal] Configuración inválida: mi_escena.yaml > objects[4] (Crate_01): clave desconocida 'colour' (válidas: name, type, ...)
[pseudo-thermal] Configuración inválida: mi_escena.yaml > objects: nombre de objeto duplicado 'Crate_01'
```

Los nombres `Looks`, `Lights`, `Robot` y `ThermalCamera` están reservados.

## Tests

Cubren los módulos puros (configuración, procesado térmico, CLI) y no abren
Isaac Sim; tardan un par de segundos:

```bash
cd ~/multimodal_sim/thermal_sim
~/isaac_sim/python.sh -m pytest tests/
```

`pytest.ini` desactiva los plugins de pytest de ROS, que no cargan en el
Python 3.12 de Isaac si el terminal tiene hecho `source /opt/ros/humble/setup.bash`.

## Comparar con el script original

`scripts/live_pseudo_thermal_legacy.py` es el script original de un solo
archivo, sin modificar. Para comprobar que el paquete produce la misma imagen
se graban los mismos 20 frames con cada uno y se comparan. `--palette whitehot`
hace que la diferencia se lea en niveles de gris, y la semilla que el ruido
sea el mismo (el legacy no tiene `--seed`; `run_legacy_seeded.py` lo ejecuta
tal cual, fijándole la semilla desde fuera).

```bash
cd ~/multimodal_sim/thermal_sim

# 1. Legacy
~/isaac_sim/python.sh scripts/run_legacy_seeded.py --seed 0 \
    --autoplay --auto-track linear --max-frames 20 --palette whitehot \
    --save-frames outputs/compare_legacy/frames

# 2. Paquete nuevo
~/isaac_sim/python.sh run_pseudo_thermal.py --seed 0 \
    --autoplay --auto-track linear --max-frames 20 --palette whitehot \
    --save-frames outputs/compare_refactor/frames

# 3. Comparación (media y máximo de la diferencia absoluta por frame)
~/isaac_sim/python.sh scripts/compare_frames.py \
    outputs/compare_legacy/frames outputs/compare_refactor/frames
```

Se espera una diferencia media global por debajo de 1 nivel de gris: lo único
que puede variar entre dos corridas es la luminancia del render RTX, que pesa
un 15 % en la imagen. `compare_frames.py` termina con código 0 si se cumple.

## Notas importantes

- **ROS2 y Isaac no comparten terminal.** Las libs de spdlog del
  sistema (`/opt/ros/humble/lib`) chocan con las de Isaac. Usar
  `launch_thermal.sh` o setear manualmente:
```bash
  export LD_LIBRARY_PATH=/home/jumasaet/isaac_sim/exts/isaacsim.ros2.core/humble/lib
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export ROS_DOMAIN_ID=15
```
- **OpenCV headless.** Isaac trae `opencv-python-headless`. Para ventana
  `cv2.imshow`:
```bash
  ~/isaac_sim/python.sh -m pip install --no-deps opencv-python
```
- Los frames guardados no llevan overlay de texto (FPS, Tmax) para
  que sirvan como dato limpio.
- `summary.json` se escribe al cerrar con estadísticas de la corrida.
