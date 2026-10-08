# Comandos — Pseudo-Thermal Simulator

Todos desde ~/multimodal_sim/thermal_sim/
```bash
cd ~/multimodal_sim/thermal_sim/
```
## Modo interactivo (robot + teclado)
```bash
~/isaac_sim/python.sh run_pseudo_thermal.py --autoplay --robot
```
## Grabar video (robot se mueve solo)
```bash
~/isaac_sim/python.sh run_pseudo_thermal.py \
    --autoplay --robot --auto-track linear \
    --save-frames outputs/demo --max-frames 90
```
## Con ROS2 (terminal nueva, SIN source de ROS)
```bash
export LD_LIBRARY_PATH=/home/jumasaet/isaac_sim/exts/isaacsim.ros2.core/humble/lib
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=15
~/isaac_sim/python.sh run_pseudo_thermal.py --autoplay --robot --ros2
```
## Ver topics ROS2 (otra terminal, CON source de ROS)
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=15
rviz2
```
## Editar escena y temperaturas

configs/scenes/demo_industrial.yaml

## Editar blur, ruido, paleta

configs/thermal_camera.yaml

## Tests
```bash
~/isaac_sim/python.sh -m pytest tests/
```