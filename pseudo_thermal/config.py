"""Carga y validación de los YAML de escena y de cámara térmica (módulo puro, ver __init__).

    load_scene(path)           -> SceneConfig          configs/scenes/*.yaml
    load_thermal_camera(path)  -> ThermalCameraConfig  configs/thermal_camera.yaml

Cualquier problema (clave desconocida, campo faltante, tipo incorrecto, nombre duplicado) lanza
ConfigError con el archivo y la ruta del campo, p.ej.:

    demo_industrial.yaml > objects[4] (Crate_01): clave desconocida 'colour' (válidas: name, ...)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

Vec3 = tuple[float, float, float]

PALETTES = ("whitehot", "ironbow", "hot", "jet")
OBJECT_TYPES = ("box", "cylinder", "sphere")
LIGHT_TYPES = ("dome", "distant")
AXES = ("X", "Y", "Z")
WORLD_PATH = "/World"
ROBOT_PATH = f"{WORLD_PATH}/Robot"
RESERVED_NAMES = ("Looks", "Lights", "Robot", "ThermalCamera")  # prims que crea el propio simulador
MIN_RESOLUTION = 32


class ConfigError(ValueError):
    """YAML inválido; el mensaje indica archivo, sección y campo."""


# --------------------------------------------------------------------------------------
# Dataclasses (lo que consume el resto del paquete)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ThermalConfig:
    range_c: tuple[float, float]  # (T mín, T máx) que se mapea a gris 0..255
    default_c: float              # objetos sin temperature_c
    default_spread_c: float       # +- °C, offset determinista por hash del prim path


@dataclass(frozen=True)
class ObjectConfig:
    name: str
    type: str                     # box | cylinder | sphere
    position: Vec3                # centro, m
    color: Vec3                   # RGB 0..1
    roughness: float
    metallic: float
    temperature_c: float | None   # None -> default_c +- default_spread_c
    size: Vec3 | None = None      # box: (x, y, z) m
    radius: float | None = None   # cylinder, sphere: m
    height: float | None = None   # cylinder: m
    axis: str | None = None       # cylinder: X | Y | Z

    @property
    def prim_path(self) -> str:
        return f"{WORLD_PATH}/{self.name}"


@dataclass(frozen=True)
class LightConfig:
    name: str
    type: str                     # dome | distant
    intensity: float
    color: Vec3 | None            # None -> no se autora (blanco)
    angle_deg: float | None       # solo distant; None -> no se autora
    rotation_xyz_deg: Vec3 | None


@dataclass(frozen=True)
class CameraConfig:
    eye: Vec3
    target: Vec3
    focal_length_mm: float


@dataclass(frozen=True)
class RobotConfig:
    temperature_c: float
    speed_m_s: float
    turn_speed_deg_s: float
    mount: Vec3                   # cámara respecto al origen del robot, m


@dataclass(frozen=True)
class SceneConfig:
    thermal: ThermalConfig
    objects: tuple[ObjectConfig, ...]
    lights: tuple[LightConfig, ...]
    camera: CameraConfig
    robot: RobotConfig

    def thermal_map(self) -> dict[str, float]:
        """prim path -> °C: objetos con temperature_c, más el robot (se monte o no con --robot)."""
        temps = {obj.prim_path: obj.temperature_c for obj in self.objects if obj.temperature_c is not None}
        temps[ROBOT_PATH] = self.robot.temperature_c
        return temps


@dataclass(frozen=True)
class BlurConfig:
    kernel: int                   # lado del kernel gaussiano, px (impar)
    sigma: float


@dataclass(frozen=True)
class NoiseConfig:
    enabled: bool
    sigma_max: float              # niveles de gris, en las zonas más frías


@dataclass(frozen=True)
class ThermalCameraConfig:
    resolution: tuple[int, int]   # (ancho, alto) px
    palette: str
    luminance_weight: float
    blur: BlurConfig
    noise: NoiseConfig


# --------------------------------------------------------------------------------------
# Lectura tipada de campos
# --------------------------------------------------------------------------------------
_REQUIRED = object()


def _require(condition: bool, where: str, message: str) -> None:
    if not condition:
        raise ConfigError(f"{where}: {message}")


class _Section:
    """Un bloque 'clave: valor' del YAML: rechaza claves desconocidas y lee campos tipados."""

    def __init__(self, data, where: str, allowed: tuple[str, ...]):
        _require(isinstance(data, dict), where, f"se esperaba un bloque 'clave: valor', no {data!r}")
        for key in data:
            _require(key in allowed, where, f"clave desconocida {key!r} (válidas: {', '.join(allowed)})")
        self._data, self.where = data, where

    def get(self, key: str, parse, default=_REQUIRED):
        """Campo ``key`` pasado por ``parse``; ausente o null -> ``default`` (o error si es obligatorio)."""
        value = self._data.get(key)
        if value is None:
            _require(default is not _REQUIRED, self.where, f"falta el campo obligatorio '{key}'")
            return default
        return parse(value, f"{self.where} > {key}")

    def section(self, key: str, allowed: tuple[str, ...]) -> _Section:
        return self.get(key, lambda value, where: _Section(value, where, allowed))


def _number(value, where: str) -> float:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    _require(ok, where, f"se esperaba un número, no {value!r}")
    return float(value)


def _integer(value, where: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), where, f"se esperaba un entero, no {value!r}")
    return value


def _boolean(value, where: str) -> bool:
    _require(isinstance(value, bool), where, f"se esperaba true o false, no {value!r}")
    return value


def _text(value, where: str) -> str:
    _require(isinstance(value, str), where, f"se esperaba un texto, no {value!r}")
    return value


def _sequence(value, where: str) -> list:
    _require(isinstance(value, list), where, f"se esperaba una lista, no {value!r}")
    return value


def _choice(*options: str):
    def parse(value, where: str) -> str:
        _require(value in options, where, f"valor {value!r} no válido (opciones: {', '.join(options)})")
        return value

    return parse


def _vector(n: int, item=_number):
    def parse(value, where: str) -> tuple:
        ok = isinstance(value, list) and len(value) == n
        _require(ok, where, f"se esperaba una lista de {n} valores, no {value!r}")
        return tuple(item(v, f"{where}[{i}]") for i, v in enumerate(value))

    return parse


def _in_range(lo: float, hi: float = math.inf, *, open_lo: bool = False, parse=_number):
    """Número (o cada componente de un vector) dentro de [lo, hi]; con open_lo, (lo, hi]."""

    def check(value, where: str):
        value = parse(value, where)
        for v in value if isinstance(value, tuple) else (value,):
            ok = (v > lo if open_lo else v >= lo) and v <= hi
            bounds = f"{'>' if open_lo else '>='} {lo:g}" + ("" if hi == math.inf else f" y <= {hi:g}")
            _require(ok, where, f"debe ser {bounds}, no {v!r}")
        return value

    return check


_vec3 = _vector(3)
_positive = _in_range(0.0, open_lo=True)
_unit = _in_range(0.0, 1.0)
_color = _in_range(0.0, 1.0, parse=_vec3)


def _name(value, where: str) -> str:
    name = _text(value, where)
    _require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is not None, where,
             f"{name!r} no sirve como nombre de prim USD (letras, dígitos y '_', sin empezar por dígito)")
    _require(name not in RESERVED_NAMES, where, f"{name!r} está reservado (lo crea el simulador en {WORLD_PATH}/{name})")
    return name


def _named_items(root: _Section, key: str, parse_item, what: str) -> tuple:
    """Lista de bloques con 'name' único (cada nombre es un prim)."""
    items = tuple(parse_item(item, f"{root.where} > {key}[{i}]") for i, item in enumerate(root.get(key, _sequence)))
    seen: set[str] = set()
    for item in items:
        _require(item.name not in seen, f"{root.where} > {key}", f"nombre de {what} duplicado {item.name!r}")
        seen.add(item.name)
    return items


# --------------------------------------------------------------------------------------
# Escena
# --------------------------------------------------------------------------------------
_OBJECT_COMMON = ("name", "type", "position", "color", "roughness", "metallic", "temperature_c")
_OBJECT_GEOMETRY = {"box": ("size",), "cylinder": ("radius", "height", "axis"), "sphere": ("radius",)}
_GEOMETRY_PARSERS = {
    "size": _in_range(0.0, open_lo=True, parse=_vec3),
    "radius": _positive,
    "height": _positive,
    "axis": _choice(*AXES),
}
_LIGHT_FIELDS = {"dome": ("color", "rotation_xyz_deg"), "distant": ("color", "angle_deg", "rotation_xyz_deg")}


def _typed_section(item, where: str, common: tuple[str, ...], by_type: dict[str, tuple[str, ...]]):
    """(sección, name, type) de un bloque cuyas claves válidas dependen de su 'type'."""
    every_key = common + tuple(dict.fromkeys(k for keys in by_type.values() for k in keys))
    if isinstance(item, dict) and isinstance(item.get("name"), str):
        where = f"{where} ({item['name']})"  # los errores nombran el objeto, no solo su índice
    section = _Section(item, where, every_key)
    name = section.get("name", _name)
    kind = section.get("type", _choice(*by_type))
    allowed = common + by_type[kind]
    for key in item:
        _require(key in allowed, where, f"'{key}' no aplica a type: {kind} (válidas: {', '.join(allowed)})")
    return section, name, kind


def _parse_object(item, where: str) -> ObjectConfig:
    section, name, kind = _typed_section(item, where, _OBJECT_COMMON, _OBJECT_GEOMETRY)
    return ObjectConfig(
        name=name,
        type=kind,
        position=section.get("position", _vec3),
        color=section.get("color", _color),
        roughness=section.get("roughness", _unit, 0.6),
        metallic=section.get("metallic", _unit, 0.0),
        temperature_c=section.get("temperature_c", _number, None),
        **{key: section.get(key, _GEOMETRY_PARSERS[key]) for key in _OBJECT_GEOMETRY[kind]},
    )


def _parse_light(item, where: str) -> LightConfig:
    section, name, kind = _typed_section(item, where, ("name", "type", "intensity"), _LIGHT_FIELDS)
    return LightConfig(
        name=name,
        type=kind,
        intensity=section.get("intensity", _in_range(0.0)),
        color=section.get("color", _color, None),
        angle_deg=section.get("angle_deg", _in_range(0.0), None),
        rotation_xyz_deg=section.get("rotation_xyz_deg", _vec3, None),
    )


def parse_scene(data, source: str = "<escena>") -> SceneConfig:
    root = _Section(data, source, ("thermal", "objects", "lights", "camera", "robot"))

    thermal = root.section("thermal", ("range_c", "default_c", "default_spread_c"))
    range_c = thermal.get("range_c", _vector(2))
    _require(range_c[0] < range_c[1], f"{thermal.where} > range_c", f"el mínimo debe ser menor que el máximo, no {list(range_c)}")

    camera = root.section("camera", ("eye", "target", "focal_length_mm"))
    eye, target = camera.get("eye", _vec3), camera.get("target", _vec3)
    _require(eye != target, camera.where, "eye y target no pueden ser el mismo punto")

    robot = root.section("robot", ("temperature_c", "speed_m_s", "turn_speed_deg_s", "mount"))
    return SceneConfig(
        thermal=ThermalConfig(
            range_c=range_c,
            default_c=thermal.get("default_c", _number),
            default_spread_c=thermal.get("default_spread_c", _in_range(0.0)),
        ),
        objects=_named_items(root, "objects", _parse_object, "objeto"),
        lights=_named_items(root, "lights", _parse_light, "luz"),
        camera=CameraConfig(eye=eye, target=target, focal_length_mm=camera.get("focal_length_mm", _positive)),
        robot=RobotConfig(
            temperature_c=robot.get("temperature_c", _number),
            speed_m_s=robot.get("speed_m_s", _positive),
            turn_speed_deg_s=robot.get("turn_speed_deg_s", _positive),
            mount=robot.get("mount", _vec3),
        ),
    )


# --------------------------------------------------------------------------------------
# Cámara térmica
# --------------------------------------------------------------------------------------
def _odd(value, where: str) -> int:
    kernel = _integer(value, where)
    _require(kernel >= 1 and kernel % 2 == 1, where, f"debe ser un entero impar >= 1, no {kernel}")
    return kernel


def parse_thermal_camera(data, source: str = "<cámara>") -> ThermalCameraConfig:
    root = _Section(data, source, ("resolution", "palette", "luminance_weight", "blur", "noise"))
    resolution = root.get("resolution", _vector(2, _integer))
    _require(min(resolution) >= MIN_RESOLUTION, f"{source} > resolution",
             f"mínimo {MIN_RESOLUTION}x{MIN_RESOLUTION}, no {list(resolution)}")
    blur = root.section("blur", ("kernel", "sigma"))
    noise = root.section("noise", ("enabled", "sigma_max"))
    return ThermalCameraConfig(
        resolution=resolution,
        palette=root.get("palette", _choice(*PALETTES)),
        luminance_weight=root.get("luminance_weight", _unit),
        blur=BlurConfig(kernel=blur.get("kernel", _odd), sigma=blur.get("sigma", _in_range(0.0))),
        noise=NoiseConfig(enabled=noise.get("enabled", _boolean), sigma_max=noise.get("sigma_max", _in_range(0.0))),
    )


# --------------------------------------------------------------------------------------
# Archivos
# --------------------------------------------------------------------------------------
def _load_yaml(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"no se pudo leer {path}: {exc.strerror or exc}") from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: YAML mal formado: {exc}") from exc


def load_scene(path) -> SceneConfig:
    return parse_scene(_load_yaml(path), Path(path).name)


def load_thermal_camera(path) -> ThermalCameraConfig:
    return parse_thermal_camera(_load_yaml(path), Path(path).name)
