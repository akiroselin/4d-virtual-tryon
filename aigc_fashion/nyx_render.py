from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .body_mesh_avatar import BodyMeshAvatar
from .data import GarmentMesh


@dataclass(frozen=True)
class NyxBackendStatus:
    available: bool
    platform: str
    machine: str
    gs_nyx_available: bool
    plugin_available: bool
    requirement: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_nyx_backend() -> NyxBackendStatus:
    gs_nyx_available = importlib.util.find_spec("gs_nyx") is not None
    plugin_available = importlib.util.find_spec("gs_nyx_plugin") is not None
    operating_system = platform.system()
    machine = platform.machine()
    supported_host = operating_system in {"Linux", "Windows"} and machine in {
        "x86_64",
        "AMD64",
    }
    available = supported_host and gs_nyx_available and plugin_available
    if not supported_host:
        reason = "Nyx 0.1.1 provides x86-64 Linux/Windows CUDA wheels, not Apple Silicon binaries"
    elif not gs_nyx_available or not plugin_available:
        reason = "install gs-nyx-plugin and its native gs_nyx CUDA wheel"
    else:
        reason = "Nyx Python plugin and native renderer are importable"
    return NyxBackendStatus(
        available=available,
        platform=operating_system,
        machine=machine,
        gs_nyx_available=gs_nyx_available,
        plugin_available=plugin_available,
        requirement="NVIDIA compute capability >= 7.0, CUDA >= 12.9, driver >= 575",
        reason=reason,
    )


def export_nyx_scene_assets(
    out_dir: str | Path,
    avatar: BodyMeshAvatar,
    garment: GarmentMesh,
    garment_vertices: np.ndarray,
    texture_image: str | Path,
) -> dict[str, Any]:
    """Export Z-up PBR GLBs consumed unchanged by the Genesis Nyx plugin."""

    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    texture_path = Path(texture_image)
    texture = Image.open(texture_path).convert("RGB")
    is_pleated_skirt = any("pleated_skirt" in label for label in garment.panel_labels)
    print_atlas_preview = texture.copy() if is_pleated_skirt else _build_print_atlas(texture)
    # Nyx/glTF samples V from the opposite image-space origin used by PIL.
    # Flip the embedded pixels once and leave mesh UVs untouched. This is more
    # robust than relying on viewer-specific texture-origin conversion.
    print_atlas = print_atlas_preview.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    print_atlas_path = out / "tshirt_base_color.png"
    print_atlas_preview.save(print_atlas_path)
    gltf_texture_path = out / "tshirt_base_color_gltf.png"
    print_atlas.save(gltf_texture_path)

    body_vertices = _to_genesis_z_up(np.asarray(avatar.mesh_vertices), avatar)
    cloth_vertices = _to_genesis_z_up(np.asarray(garment_vertices), avatar)
    body_faces = np.asarray(avatar.mesh_faces, dtype=np.int32)
    cloth_faces = np.asarray(garment.faces, dtype=np.int32)

    body_material = PBRMaterial(
        name="anny_skin",
        baseColorFactor=[235, 199, 176, 255],
        metallicFactor=0.0,
        roughnessFactor=0.78,
        doubleSided=False,
    )
    body = trimesh.Trimesh(body_vertices, body_faces, process=False)
    body.fix_normals()
    body.visual = TextureVisuals(
        uv=np.zeros((len(body_vertices), 2), dtype=np.float32),
        material=body_material,
    )
    body_path = out / "anny_body_pbr.glb"
    body_path.write_bytes(body.export(file_type="glb"))

    labels = np.asarray(garment.panel_labels)
    front_print_faces = (
        np.ones(len(cloth_faces), dtype=bool)
        if is_pleated_skirt
        else np.all(labels[cloth_faces] == "front_front_body", axis=1)
    )
    uv = (
        np.asarray(garment.uvs, dtype=np.float32).copy()
        if garment.uvs is not None
        else _normalized_xy(garment.material_vertices[:, :2])
    )
    # glTF/Nyx consume the authored UV convention directly. The PIL debug
    # renderer flips image-space Y while rasterizing; repeating that flip here
    # turns the embedded print upside down in GLB viewers.
    white_material = PBRMaterial(
        name="wool_plaid_base" if is_pleated_skirt else "cotton_jersey_white",
        baseColorFactor=[74, 66, 58, 255] if is_pleated_skirt else [244, 244, 241, 255],
        metallicFactor=0.0,
        roughnessFactor=0.82 if is_pleated_skirt else 0.88,
        doubleSided=True,
    )
    print_material = PBRMaterial(
        name="wool_plaid_texture" if is_pleated_skirt else "cotton_jersey_print",
        baseColorFactor=[255, 255, 255, 255],
        baseColorTexture=print_atlas,
        metallicFactor=0.0,
        roughnessFactor=0.82 if is_pleated_skirt else 0.88,
        doubleSided=True,
    )
    cloth_scene = trimesh.Scene()
    if np.any(~front_print_faces):
        cloth_scene.add_geometry(
            _submesh_with_material(
                cloth_vertices,
                cloth_faces[~front_print_faces],
                uv,
                white_material,
            ),
            node_name="cotton_shell_base",
            geom_name="cotton_shell_base",
        )
    if np.any(front_print_faces):
        cloth_scene.add_geometry(
            _submesh_with_material(
                cloth_vertices,
                cloth_faces[front_print_faces],
                uv,
                print_material,
            ),
            node_name="cotton_shell_front_print",
            geom_name="cotton_shell_front_print",
        )
    garment_path = out / (
        "continuous_pleated_skirt_pbr.glb" if is_pleated_skirt else "continuous_tshirt_pbr.glb"
    )
    garment_path.write_bytes(cloth_scene.export(file_type="glb"))

    manifest = {
        "renderer": "Genesis Nyx FastPathTracer",
        "backend": probe_nyx_backend().to_dict(),
        "coordinate_system": "Genesis Z-up; source (x, up_y, front_z) maps to (x, -front_z, up_y)",
        "camera_view": "front",
        "body_glb": str(body_path.resolve()),
        "garment_glb": str(garment_path.resolve()),
        "base_color_texture": str(print_atlas_path.resolve()),
        "embedded_gltf_texture": str(gltf_texture_path.resolve()),
        "garment_material": {
            "model": "Disney-style PBR",
            "metallic": 0.0,
            "roughness": 0.82 if is_pleated_skirt else 0.88,
            "double_sided": True,
            "single_white_base_across_collar_body_and_sleeves": not is_pleated_skirt,
            "image_conditioned_plaid": is_pleated_skirt,
        },
        "topology": garment.summary(),
    }
    manifest_path = out / "nyx_scene_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def render_nyx_scene(
    assets: dict[str, Any],
    output_path: str | Path,
    *,
    resolution: tuple[int, int] = (1024, 1024),
    spp: int = 128,
) -> Path:
    """Render exported assets with the real Nyx CUDA backend."""

    status = probe_nyx_backend()
    if not status.available:
        raise RuntimeError(f"Nyx backend is unavailable: {status.reason}")

    import genesis as gs
    import gs_nyx.nyx_py_renderer as npr
    import gs_nyx.nyx_py_sdk as nps
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

    gs.init()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    for asset_key in ("body_glb", "garment_glb"):
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=assets[asset_key],
                fixed=True,
                collision=False,
                decimate=False,
                convexify=False,
                file_meshes_are_zup=True,
            )
        )
    scene.add_entity(
        morph=gs.morphs.Plane(
            pos=(0.0, 0.0, -0.005),
            plane_size=(4.0, 4.0),
        ),
        surface=gs.surfaces.BSDF(
            color=(0.64, 0.66, 0.68),
            metallic=0.0,
            roughness=0.92,
        ),
    )

    environment = nps.EnvironmentMapAsset()
    environment.tint = nps.float3(0.32, 0.34, 0.38)
    environment.multiplier = 0.8
    lights = [
        {
            "type": "directional",
            "dir": (0.28, 0.50, -0.82),
            "color": (1.0, 0.96, 0.92),
            "intensity": 3.0,
        },
        {
            "type": "point",
            "pos": (-0.9, -1.3, 1.55),
            "color": (0.80, 0.88, 1.0),
            "intensity": 18.0,
            "range": 4.0,
        },
    ]
    camera = scene.add_sensor(
        NyxCameraOptions(
            res=resolution,
            pos=(0.0, -2.65, 0.78),
            lookat=(0.0, 0.0, 0.70),
            fov=27.0,
            spp=spp,
            denoise=True,
            render_mode=npr.ERenderMode.FastPathTracer,
            env_maps=(environment,),
            lights=lights,
        )
    )
    scene.build(n_envs=1)
    scene.step()
    rgb = camera.read().rgb[0].cpu().numpy()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output)
    return output


def _to_genesis_z_up(vertices: np.ndarray, avatar: BodyMeshAvatar) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    floor_y = float(np.min(avatar.mesh_vertices[:, 1]))
    return np.column_stack(
        (
            vertices[:, 0],
            -(vertices[:, 2] - float(avatar.base_z)),
            vertices[:, 1] - floor_y,
        )
    ).astype(np.float32)


def _submesh_with_material(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    material: Any,
):
    import trimesh
    from trimesh.visual.texture import TextureVisuals

    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    submesh = trimesh.Trimesh(
        vertices=vertices[used],
        faces=remap[faces],
        process=False,
    )
    submesh.visual = TextureVisuals(uv=uv[used], material=material)
    return submesh


def _build_print_atlas(source: Image.Image, size: int = 1024) -> Image.Image:
    source_array = np.asarray(source.convert("RGB"), dtype=np.int16)
    chroma = source_array.max(axis=2) - source_array.min(axis=2)
    warm = (
        (source_array[:, :, 0] > 105)
        & (source_array[:, :, 0] >= source_array[:, :, 1] * 0.92)
        & (source_array[:, :, 1] >= source_array[:, :, 2] * 0.72)
    )
    subject = (chroma > 18) & warm
    ys, xs = np.nonzero(subject)
    if xs.size:
        pad_x = max(8, int(0.10 * (xs.max() - xs.min() + 1)))
        pad_y = max(8, int(0.08 * (ys.max() - ys.min() + 1)))
        box = (
            max(0, int(xs.min()) - pad_x),
            max(0, int(ys.min()) - pad_y),
            min(source.width, int(xs.max()) + pad_x + 1),
            min(source.height, int(ys.max()) + pad_y + 1),
        )
    else:
        box = (0, 0, source.width, source.height)
    crop = source.crop(box).convert("RGB")
    crop_array = np.asarray(crop, dtype=np.int16)
    crop_chroma = crop_array.max(axis=2) - crop_array.min(axis=2)
    crop_mean = crop_array.mean(axis=2)
    crop_warm = (
        (crop_array[:, :, 0] >= crop_array[:, :, 1] * 0.94)
        & (crop_array[:, :, 1] >= crop_array[:, :, 2] * 0.74)
    )
    alpha = np.where(
        ((crop_chroma > 13) & crop_warm) | (crop_mean < 175),
        255,
        0,
    ).astype(np.uint8)
    alpha_image = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(radius=2.0))

    atlas = Image.new("RGB", (size, size), (244, 244, 241))
    destination = (
        int(0.20 * size),
        int(0.22 * size),
        int(0.80 * size),
        int(0.70 * size),
    )
    destination_size = (destination[2] - destination[0], destination[3] - destination[1])
    crop = crop.resize(destination_size, Image.Resampling.LANCZOS)
    alpha_image = alpha_image.resize(destination_size, Image.Resampling.LANCZOS)
    atlas.paste(crop, destination[:2], alpha_image)
    return atlas


def _normalized_xy(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float32)
    minimum = coordinates.min(axis=0, keepdims=True)
    maximum = coordinates.max(axis=0, keepdims=True)
    return (coordinates - minimum) / np.maximum(maximum - minimum, 1.0e-8)
