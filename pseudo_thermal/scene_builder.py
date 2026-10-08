"""Escena USD a partir de SceneConfig (módulo de Isaac: importar solo tras crear SimulationApp).

Cada objeto del YAML es un prim /World/<name> con su material OmniPBR en /World/Looks/<name>;
las luces van en /World/Lights/<name> y la cámara en CAMERA_PATH.
"""

from __future__ import annotations

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from .config import WORLD_PATH, LightConfig, ObjectConfig, SceneConfig

CAMERA_PATH = f"{WORLD_PATH}/ThermalCamera"
LOOKS_PATH = f"{WORLD_PATH}/Looks"
LIGHTS_PATH = f"{WORLD_PATH}/Lights"
CAMERA_H_APERTURE_MM = 20.955
CAMERA_CLIPPING_M = (0.05, 1000.0)


def make_omnipbr(stage, path: str, rgb, roughness: float = 0.6, metallic: float = 0.0) -> UsdShade.Material:
    """Material OmniPBR autorado con pxr puro (misma estructura que CreateMdlMaterialPrim)."""
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(out)
    material.CreateVolumeOutput("mdl").ConnectToSource(out)
    material.CreateDisplacementOutput("mdl").ConnectToSource(out)
    return material


def bind(prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def add_box(stage, path: str, center, size_xyz, material) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    xf = UsdGeom.XformCommonAPI(cube.GetPrim())
    xf.SetTranslate(Gf.Vec3d(*center))
    xf.SetScale(Gf.Vec3f(*size_xyz))
    bind(cube.GetPrim(), material)


def add_cylinder(stage, path: str, center, radius: float, height: float, axis: str, material) -> None:
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(axis)
    half = {"X": (height / 2, radius, radius), "Y": (radius, height / 2, radius), "Z": (radius, radius, height / 2)}[axis]
    cyl.CreateExtentAttr([Gf.Vec3f(*(-v for v in half)), Gf.Vec3f(*half)])
    UsdGeom.XformCommonAPI(cyl.GetPrim()).SetTranslate(Gf.Vec3d(*center))
    bind(cyl.GetPrim(), material)


def add_sphere(stage, path: str, center, radius: float, material) -> None:
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(radius)
    sph.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -radius), Gf.Vec3f(radius, radius, radius)])
    UsdGeom.XformCommonAPI(sph.GetPrim()).SetTranslate(Gf.Vec3d(*center))
    bind(sph.GetPrim(), material)


def add_object(stage, obj: ObjectConfig, material) -> None:
    if obj.type == "box":
        add_box(stage, obj.prim_path, obj.position, obj.size, material)
    elif obj.type == "cylinder":
        add_cylinder(stage, obj.prim_path, obj.position, obj.radius, obj.height, obj.axis, material)
    else:
        add_sphere(stage, obj.prim_path, obj.position, obj.radius, material)


def add_light(stage, light: LightConfig) -> None:
    """Solo se autoran los atributos presentes en el YAML; el resto queda en el default de USD."""
    schema = UsdLux.DomeLight if light.type == "dome" else UsdLux.DistantLight
    prim = schema.Define(stage, f"{LIGHTS_PATH}/{light.name}")
    prim.CreateIntensityAttr(light.intensity)
    if light.color is not None:
        prim.CreateColorAttr(Gf.Vec3f(*light.color))
    if light.angle_deg is not None:
        prim.CreateAngleAttr(light.angle_deg)
    if light.rotation_xyz_deg is not None:
        UsdGeom.XformCommonAPI(prim.GetPrim()).SetRotate(Gf.Vec3f(*light.rotation_xyz_deg))


def add_look_at_camera(stage, path: str, eye, target, focal_length_mm: float, width: int, height: int) -> UsdGeom.Camera:
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal_length_mm)
    cam.CreateHorizontalApertureAttr(CAMERA_H_APERTURE_MM)
    cam.CreateVerticalApertureAttr(CAMERA_H_APERTURE_MM * height / width)
    cam.CreateClippingRangeAttr(Gf.Vec2f(*CAMERA_CLIPPING_M))
    # SetLookAt da world->camera (cámara mira a -Z, up +Y); se invierte para camera->world.
    cam_to_world = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(cam_to_world.ExtractTranslation())
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(cam_to_world.ExtractRotationQuat())
    return cam


def build_scene(stage, scene: SceneConfig, resolution: tuple[int, int]) -> None:
    """Stage +Z up en metros con los objetos, las luces y la cámara térmica de ``scene``."""
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, WORLD_PATH)
    stage.SetDefaultPrim(world.GetPrim())

    looks = {
        obj.name: make_omnipbr(stage, f"{LOOKS_PATH}/{obj.name}", obj.color, obj.roughness, obj.metallic)
        for obj in scene.objects
    }
    for obj in scene.objects:
        add_object(stage, obj, looks[obj.name])
    for light in scene.lights:
        add_light(stage, light)

    width, height = resolution
    add_look_at_camera(stage, CAMERA_PATH, scene.camera.eye, scene.camera.target, scene.camera.focal_length_mm, width, height)
