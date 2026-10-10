from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .avatar import SmplAvatarProxy
from .data import GarmentMesh

_TEXTURE_BBOX_CACHE: dict[int, tuple[int, int, int, int]] = {}
_FRONT_BODY_BBOX_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_DISPLAY_EDGE_CACHE: dict[int, np.ndarray] = {}
_AVATAR_LAYER_CACHE: dict[tuple[object, ...], Image.Image] = {}
_TEXTURE_ARRAY_CACHE: dict[int, np.ndarray] = {}


def front_print_uv_window(
    mesh: GarmentMesh,
    texture_image: Image.Image | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a metric-correct print window in the front panel UV chart.

    The panel UV is a material coordinate, not a square pixel grid. Correct
    the window using the rest-panel x/y metric so a portrait print keeps its
    aspect ratio after it is placed on the garment.
    """

    uv = np.asarray(mesh.uvs, dtype=np.float32)
    labels = np.asarray(mesh.panel_labels)
    front = labels == "front_front_body"
    if uv.shape != (len(mesh.vertices), 2) or not np.any(front):
        return (
            np.asarray([0.20, 0.28], dtype=np.float32),
            np.asarray([0.80, 0.78], dtype=np.float32),
        )
    material = np.asarray(mesh.material_vertices, dtype=np.float32)
    u_span = max(float(np.ptp(uv[front, 0])), 1.0e-6)
    v_span = max(float(np.ptp(uv[front, 1])), 1.0e-6)
    x_per_u = max(float(np.ptp(material[front, 0])) / u_span, 1.0e-6)
    y_per_v = max(float(np.ptp(material[front, 1])) / v_span, 1.0e-6)
    metric_ratio = y_per_v / x_per_u
    if texture_image is not None:
        x0, y0, x1, y1 = _chromatic_bbox(texture_image)
        source_aspect = max(float(x1 - x0), 1.0) / max(float(y1 - y0), 1.0)
    else:
        source_aspect = 0.744
    v_min, v_max = 0.28, 0.78
    target_width = (v_max - v_min) * source_aspect * metric_ratio
    width = min(0.60, max(0.24, target_width))
    u_center = 0.50
    return (
        np.asarray([u_center - 0.5 * width, v_min], dtype=np.float32),
        np.asarray([u_center + 0.5 * width, v_max], dtype=np.float32),
    )


@dataclass(frozen=True)
class GarmentRenderLayer:
    vertices: np.ndarray
    mesh: GarmentMesh
    texture_image: Image.Image | None = None
    closed_sleeve_visibility: bool = False


def render_layered_rollout(
    path: Path,
    layer_rollouts: list[tuple[list[np.ndarray], GarmentMesh, Image.Image | None, bool]],
    caption: str,
    *,
    avatar: SmplAvatarProxy | None = None,
    duration_ms: int = 90,
    method_label: str = "layered intrinsic cloth",
    avatar_vertex_rollout: list[np.ndarray] | None = None,
    view_yaw_rad: float = 0.0,
) -> Path:
    """Render independently simulated garments in one body-aligned scene.

    Layer order is back-to-front at garment overlaps. Rollouts with different
    frame counts are synchronized by normalized physical progress.
    """

    if not layer_rollouts or any(not rollout for rollout, *_rest in layer_rollouts):
        raise ValueError("at least one non-empty garment rollout is required")
    frame_count = max(len(rollout) for rollout, *_rest in layer_rollouts)
    if avatar_vertex_rollout is not None and len(avatar_vertex_rollout) != frame_count:
        raise ValueError("avatar_vertex_rollout must match the rendered frame count")
    images: list[Image.Image] = []
    for frame_index in range(frame_count):
        progress = frame_index / max(frame_count - 1, 1)
        layers = []
        for rollout, mesh, texture, closed_sleeve in layer_rollouts:
            source_index = int(round(progress * max(len(rollout) - 1, 0)))
            layers.append(
                GarmentRenderLayer(
                    np.asarray(rollout[source_index], dtype=np.float32),
                    mesh,
                    texture,
                    closed_sleeve,
                )
            )
        images.append(
            render_layered_frame(
                layers,
                frame_index,
                caption,
                avatar=avatar,
                avatar_vertices=(
                    None if avatar_vertex_rollout is None else avatar_vertex_rollout[frame_index]
                ),
                method_label=method_label,
                view_yaw_rad=view_yaw_rad,
            )
        )
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return path


def render_layered_frame(
    layers: list[GarmentRenderLayer],
    step: int,
    caption: str,
    *,
    avatar: SmplAvatarProxy | None = None,
    size: int = 760,
    method_label: str = "layered intrinsic cloth",
    avatar_vertices: np.ndarray | None = None,
    view_yaw_rad: float = 0.0,
    show_shell_boundary_edges: bool = False,
) -> Image.Image:
    if not layers:
        raise ValueError("at least one garment layer is required")
    for layer in layers:
        if layer.vertices.shape != (len(layer.mesh.vertices), 3):
            raise ValueError("layer vertices must match its garment mesh")

    viewed_layers = [
        GarmentRenderLayer(
            _rotate_view_y(layer.vertices, view_yaw_rad),
            layer.mesh,
            layer.texture_image,
            layer.closed_sleeve_visibility,
        )
        for layer in layers
    ]
    viewed_avatar_vertices = avatar_vertices
    if viewed_avatar_vertices is None and avatar is not None:
        viewed_avatar_vertices = getattr(avatar, "mesh_vertices", None)
    if viewed_avatar_vertices is not None:
        viewed_avatar_vertices = _rotate_view_y(
            np.asarray(viewed_avatar_vertices, dtype=np.float32), view_yaw_rad
        )

    image = Image.new("RGB", (size, size), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    combined = np.concatenate([layer.vertices for layer in viewed_layers], axis=0)
    scene_reference = _scene_reference_points(
        combined, avatar, avatar_vertices=viewed_avatar_vertices
    )
    z_min = float(combined[:, 2].min())
    z_span = max(float(combined[:, 2].max() - z_min), 1.0e-6)
    prepared: list[tuple[GarmentRenderLayer, np.ndarray, np.ndarray]] = []
    for layer in viewed_layers:
        canvas = _world_to_canvas(
            layer.vertices,
            size=size,
            reference=scene_reference,
        )
        labels = layer.mesh.panel_labels
        back_vertex = np.asarray(
            [label.startswith("back_") for label in labels], dtype=bool
        )
        sleeve_vertex = np.asarray(["sleeve" in label for label in labels], dtype=bool)
        back_face = np.sum(back_vertex[layer.mesh.faces], axis=1) >= 2
        sleeve_face = np.any(sleeve_vertex[layer.mesh.faces], axis=1)
        behind_body = back_face & (
            ~sleeve_face if layer.closed_sleeve_visibility else True
        )
        prepared.append((layer, canvas, behind_body))

    for layer, canvas, behind_body in prepared:
        _draw_garment_faces(
            image,
            draw,
            layer.vertices,
            layer.mesh.faces,
            canvas,
            layer.vertices[:, 2],
            z_min,
            z_span,
            layer.texture_image,
            layer.mesh,
            np.flatnonzero(behind_body),
        )
    if avatar is not None:
        if viewed_avatar_vertices is not None and getattr(avatar, "mesh_faces", None) is not None:
            _draw_avatar_mesh(
                image,
                viewed_avatar_vertices,
                np.asarray(avatar.mesh_faces),
                scene_reference,
                size,
            )
        else:
            _draw_avatar_proxy(image, draw, avatar, scene_reference, size)
    for layer, canvas, behind_body in prepared:
        _draw_garment_faces(
            image,
            draw,
            layer.vertices,
            layer.mesh.faces,
            canvas,
            layer.vertices[:, 2],
            z_min,
            z_span,
            layer.texture_image,
            layer.mesh,
            np.flatnonzero(~behind_body),
        )
        if show_shell_boundary_edges:
            for first, second in _display_edges(layer.mesh):
                draw.line(
                    [tuple(canvas[int(first)]), tuple(canvas[int(second)])],
                    fill=(116, 145, 154),
                    width=1,
                )

    draw.text((24, 22), f"AIGC fashion {method_label} - frame {step:03d}", fill=(28, 34, 42))
    if avatar is not None:
        view_label = (
            "FRONT VIEW (+z)"
            if abs(view_yaw_rad) < 1.0e-6
            else f"3/4 VIEW ({np.degrees(view_yaw_rad):.0f} deg)"
        )
        draw.text((size - 154, 22), view_label, fill=(54, 68, 82))
    draw.text((24, 48), caption[:110], fill=(80, 88, 98))
    return image


def _rotate_view_y(points: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Express world points in a camera frame yawed around the vertical axis."""

    points = np.asarray(points, dtype=np.float32)
    if abs(yaw_rad) < 1.0e-8:
        return points.copy()
    cosine = float(np.cos(yaw_rad))
    sine = float(np.sin(yaw_rad))
    viewed = points.copy()
    viewed[:, 0] = cosine * points[:, 0] + sine * points[:, 2]
    viewed[:, 2] = -sine * points[:, 0] + cosine * points[:, 2]
    return viewed


def render_rollout(
    path: Path,
    frames: list[np.ndarray],
    mesh: GarmentMesh,
    caption: str,
    avatar: SmplAvatarProxy | None = None,
    texture_image: Image.Image | None = None,
    duration_ms: int = 80,
    method_label: str = "intrinsic cloth",
    semantic_sleeve_occlusion: bool = False,
) -> Path:
    images = [
        render_frame(
            frame,
            mesh.faces,
            mesh.pinned_vertices,
            i,
            caption,
            avatar=avatar,
            texture_image=texture_image,
            mesh=mesh,
            method_label=method_label,
            semantic_sleeve_occlusion=semantic_sleeve_occlusion,
        )
        for i, frame in enumerate(frames)
    ]
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return path


def render_frame(
    vertices: np.ndarray,
    faces: np.ndarray,
    pinned: np.ndarray,
    step: int,
    caption: str,
    avatar: SmplAvatarProxy | None = None,
    texture_image: Image.Image | None = None,
    mesh: GarmentMesh | None = None,
    size: int = 760,
    method_label: str = "intrinsic cloth",
    semantic_sleeve_occlusion: bool = False,
) -> Image.Image:
    image = Image.new("RGB", (size, size), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    scene_reference = _scene_reference_points(vertices, avatar)
    canvas = _world_to_canvas(vertices, size=size, reference=scene_reference)
    z = vertices[:, 2]
    z_min = float(z.min())
    z_span = max(float(z.max() - z_min), 1e-6)
    if mesh is not None and avatar is not None:
        is_back_vertex = np.asarray([label.startswith("back_") for label in mesh.panel_labels], dtype=bool)
        is_sleeve_vertex = np.asarray(["sleeve" in label for label in mesh.panel_labels], dtype=bool)
        # Body/limb occlusion is a semantic visibility decision. A face-center
        # depth test misclassifies triangles that straddle an arm at the armpit,
        # producing large triangular skin holes in an otherwise closed sleeve.
        is_back_face = np.sum(is_back_vertex[faces], axis=1) >= 2
        # The armhole transition is a continuous shell: even a triangle with
        # one sleeve-labelled vertex can bridge the torso and sleeve tube.
        # Treating it as a torso back face lets the avatar erase that triangle
        # and creates a false underarm opening in the composited render.
        is_sleeve_face = np.any(is_sleeve_vertex[faces], axis=1)
        behind_body = is_back_face & (~is_sleeve_face if semantic_sleeve_occlusion else True)
        _draw_garment_faces(
            image, draw, vertices, faces, canvas, z, z_min, z_span, texture_image, mesh, np.flatnonzero(behind_body)
        )
        for a, b in _display_edges(mesh):
            edge_back = bool(is_back_vertex[int(a)] and is_back_vertex[int(b)])
            edge_sleeve = bool(is_sleeve_vertex[int(a)] or is_sleeve_vertex[int(b)])
            if edge_back and not (semantic_sleeve_occlusion and edge_sleeve):
                draw.line([tuple(canvas[int(a)]), tuple(canvas[int(b)])], fill=(116, 145, 154), width=1)
        _draw_avatar_proxy(image, draw, avatar, scene_reference, size)
        _draw_garment_faces(
            image, draw, vertices, faces, canvas, z, z_min, z_span, texture_image, mesh, np.flatnonzero(~behind_body)
        )
        for a, b in _display_edges(mesh):
            edge_back = bool(is_back_vertex[int(a)] and is_back_vertex[int(b)])
            edge_sleeve = bool(is_sleeve_vertex[int(a)] or is_sleeve_vertex[int(b)])
            if not edge_back or (semantic_sleeve_occlusion and edge_sleeve):
                draw.line([tuple(canvas[int(a)]), tuple(canvas[int(b)])], fill=(116, 145, 154), width=1)
    else:
        if avatar is not None:
            _draw_avatar_proxy(image, draw, avatar, scene_reference, size)
        _draw_garment_faces(
            image, draw, vertices, faces, canvas, z, z_min, z_span, texture_image, mesh, np.arange(faces.shape[0])
        )
        if mesh is not None:
            for a, b in _display_edges(mesh):
                draw.line([tuple(canvas[int(a)]), tuple(canvas[int(b)])], fill=(116, 145, 154), width=1)

    draw.text((24, 22), f"AIGC fashion {method_label} - frame {step:03d}", fill=(28, 34, 42))
    if avatar is not None:
        draw.text((size - 142, 22), "FRONT VIEW (+z)", fill=(54, 68, 82))
    draw.text((24, 48), caption[:110], fill=(80, 88, 98))
    return image


def _avatar_mid_depth(avatar: SmplAvatarProxy, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    front = np.asarray(avatar.support_height_np(xy), dtype=np.float32)
    back = np.asarray(avatar.back_height_np(xy), dtype=np.float32)
    valid = np.isfinite(front) & np.isfinite(back) & (np.abs(front) < 1.0e4) & (np.abs(back) < 1.0e4)
    midpoint = np.full(front.shape, float(avatar.base_z), dtype=np.float32)
    midpoint[valid] = 0.5 * (front[valid] + back[valid])
    return midpoint, valid


def _draw_garment_faces(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    vertices: np.ndarray,
    faces: np.ndarray,
    canvas: np.ndarray,
    z: np.ndarray,
    z_min: float,
    z_span: float,
    texture_image: Image.Image | None,
    mesh: GarmentMesh | None,
    face_indices: np.ndarray,
) -> None:
    if face_indices.size == 0:
        return
    skirt_vertex_shade: np.ndarray | None = None
    if texture_image is not None and mesh is not None and any(
        "pleated_skirt" in label for label in mesh.panel_labels
    ):
        normals = _smooth_vertex_normals(vertices, faces)
        light = np.asarray([-0.35, 0.20, 0.92], dtype=np.float32)
        light /= np.linalg.norm(light)
        skirt_vertex_shade = 0.58 + 0.48 * np.clip(normals @ light, 0.0, 1.0)
    order = face_indices[np.argsort(z[faces[face_indices]].mean(axis=1))]
    for face_idx in order:
        tri = faces[int(face_idx)]
        height = float((z[tri].mean() - z_min) / z_span)
        if texture_image is not None and mesh is not None:
            labels = [mesh.panel_labels[int(index)] for index in tri]
            if all("pleated_skirt" in label for label in labels) and mesh.uvs is not None:
                assert skirt_vertex_shade is not None
                _draw_uv_textured_face(
                    image,
                    canvas[tri],
                    np.asarray(mesh.uvs[tri], dtype=np.float32),
                    skirt_vertex_shade[tri],
                    texture_image,
                )
                continue
            # Midpoint refinement creates a few boundary triangles whose
            # third vertex inherits the back-panel label. They are still part
            # of the front surface and must carry the continuous print UV;
            # otherwise the image is cut into white triangular gaps.
            front_body_count = sum(label == "front_front_body" for label in labels)
            if front_body_count >= 2:
                _draw_textured_front_face(image, canvas[tri], tri, height, texture_image, mesh)
                continue
        fill = _face_fill(vertices, tri, height, texture_image, mesh)
        draw.polygon([tuple(canvas[i]) for i in tri], fill=fill)


def _draw_uv_textured_face(
    image: Image.Image,
    canvas_triangle: np.ndarray,
    vertex_uv: np.ndarray,
    vertex_shade: np.ndarray,
    texture_image: Image.Image,
) -> None:
    xy = np.asarray(canvas_triangle, dtype=np.float64)
    x_min = max(0, int(np.floor(np.min(xy[:, 0]))))
    x_max = min(image.width - 1, int(np.ceil(np.max(xy[:, 0]))))
    y_min = max(0, int(np.floor(np.min(xy[:, 1]))))
    y_max = min(image.height - 1, int(np.ceil(np.max(xy[:, 1]))))
    if x_max < x_min or y_max < y_min:
        return
    x0, y0 = xy[0]
    x1, y1 = xy[1]
    x2, y2 = xy[2]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(denominator)) < 1.0e-8:
        return
    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5,
        np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5,
    )
    w0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
    w1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1.0e-4) & (w1 >= -1.0e-4) & (w2 >= -1.0e-4)
    if not np.any(inside):
        return
    uv = w0[..., None] * vertex_uv[0] + w1[..., None] * vertex_uv[1] + w2[..., None] * vertex_uv[2]
    texture = _texture_array(texture_image)
    sample_x = np.clip(uv[..., 0], 0.0, 1.0) * (texture_image.width - 1)
    sample_y = np.clip(1.0 - uv[..., 1], 0.0, 1.0) * (texture_image.height - 1)
    sampled = _bilinear_texture_sample(texture, sample_x[inside], sample_y[inside])
    shade = (
        w0[..., None] * vertex_shade[0]
        + w1[..., None] * vertex_shade[1]
        + w2[..., None] * vertex_shade[2]
    )
    sampled *= shade[inside]
    patch = np.zeros((*inside.shape, 3), dtype=np.uint8)
    patch[inside] = np.clip(sampled, 0, 255).astype(np.uint8)
    alpha = np.zeros(inside.shape, dtype=np.uint8)
    alpha[inside] = 255
    image.paste(
        Image.fromarray(patch),
        (x_min, y_min),
        Image.fromarray(alpha),
    )


def _smooth_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return area-weighted normals for continuous cloth shading."""

    points = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int32)
    corners = points[triangles]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(points)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1.0e-10
    normals[valid] /= lengths[valid]
    normals[~valid] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return normals


def _draw_textured_front_face(
    image: Image.Image,
    canvas_triangle: np.ndarray,
    tri: np.ndarray,
    height: float,
    texture_image: Image.Image,
    mesh: GarmentMesh,
) -> None:
    vertex_uv = _front_print_vertex_uvs(mesh, tri, texture_image)
    if vertex_uv is None:
        return

    xy = np.asarray(canvas_triangle, dtype=np.float64)
    x_min = max(0, int(np.floor(np.min(xy[:, 0]))))
    x_max = min(image.width - 1, int(np.ceil(np.max(xy[:, 0]))))
    y_min = max(0, int(np.floor(np.min(xy[:, 1]))))
    y_max = min(image.height - 1, int(np.ceil(np.max(xy[:, 1]))))
    if x_max < x_min or y_max < y_min:
        return

    x0, y0 = xy[0]
    x1, y1 = xy[1]
    x2, y2 = xy[2]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(denominator)) < 1e-8:
        return

    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5,
        np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5,
    )
    weight0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
    weight1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
    weight2 = 1.0 - weight0 - weight1
    inside = (weight0 >= -1e-4) & (weight1 >= -1e-4) & (weight2 >= -1e-4)
    if not np.any(inside):
        return

    uv = (
        weight0[..., None] * vertex_uv[0]
        + weight1[..., None] * vertex_uv[1]
        + weight2[..., None] * vertex_uv[2]
    )
    cloth_value = int(np.clip(232 + 18 * height, 0, 255))
    patch = np.full((*inside.shape, 3), cloth_value, dtype=np.uint8)
    print_mask = inside & np.all((uv >= 0.0) & (uv <= 1.0), axis=2)
    if np.any(print_mask):
        texture = _texture_array(texture_image)
        bx0, by0, bx1, by1 = _chromatic_bbox(texture_image)
        sample_x = bx0 + uv[..., 0] * max(bx1 - bx0, 1)
        sample_y = by0 + uv[..., 1] * max(by1 - by0, 1)
        sampled = _bilinear_texture_sample(texture, sample_x[print_mask], sample_y[print_mask])
        cloth_shade = 0.90 + 0.10 * height
        patch[print_mask] = np.clip(sampled * cloth_shade, 0, 255).astype(np.uint8)

    alpha = np.zeros(inside.shape, dtype=np.uint8)
    alpha[inside] = 255
    patch_image = Image.fromarray(patch, mode="RGB")
    mask_image = Image.fromarray(alpha, mode="L")
    image.paste(patch_image, (x_min, y_min), mask_image)


def _front_print_vertex_uvs(
    mesh: GarmentMesh,
    tri: np.ndarray,
    texture_image: Image.Image | None = None,
) -> np.ndarray | None:
    if mesh.uvs is not None:
        body_uv = np.asarray(mesh.uvs[tri], dtype=np.float32)
    else:
        bbox = _front_body_bbox(mesh)
        if bbox is None:
            return None
        mins, maxs = bbox
        body_uv = (mesh.material_vertices[tri, :2] - mins) / np.maximum(maxs - mins, 1e-6)
    print_min, print_max = front_print_uv_window(mesh, texture_image)
    uv = (body_uv - print_min) / np.maximum(print_max - print_min, 1e-6)
    uv[:, 1] = 1.0 - uv[:, 1]
    return uv


def _texture_array(image: Image.Image) -> np.ndarray:
    key = id(image)
    cached = _TEXTURE_ARRAY_CACHE.get(key)
    if cached is None:
        cached = np.asarray(image.convert("RGB"), dtype=np.float32)
        _TEXTURE_ARRAY_CACHE[key] = cached
    return cached


def _bilinear_texture_sample(texture: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, texture.shape[1] - 1.0)
    y = np.clip(y, 0.0, texture.shape[0] - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, texture.shape[1] - 1)
    y1 = np.minimum(y0 + 1, texture.shape[0] - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = texture[y0, x0] * (1.0 - wx) + texture[y0, x1] * wx
    bottom = texture[y1, x0] * (1.0 - wx) + texture[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def _face_fill(
    vertices: np.ndarray,
    tri: np.ndarray,
    height: float,
    texture_image: Image.Image | None,
    mesh: GarmentMesh | None,
) -> tuple[int, int, int]:
    if texture_image is None or mesh is None:
        return (42, int(128 + 72 * height), 188)

    labels = [mesh.panel_labels[int(idx)] for idx in tri]
    if all("pleated_skirt" in label for label in labels):
        uv = (
            np.asarray(mesh.uvs[tri].mean(axis=0), dtype=np.float32)
            if mesh.uvs is not None
            else np.asarray([0.5, 0.5], dtype=np.float32)
        )
        px = int(np.clip(uv[0], 0.0, 1.0) * (texture_image.width - 1))
        py = int(np.clip(1.0 - uv[1], 0.0, 1.0) * (texture_image.height - 1))
        color = np.asarray(texture_image.getpixel((px, py))[:3], dtype=np.float32)
        # Depth modulation exposes the geometric pleats without baking the
        # source photograph's lighting and fold shadows into the material.
        color *= 0.72 + 0.40 * height
        return tuple(int(channel) for channel in np.clip(color, 0, 255))
    if not all(label.startswith("front_front_body") for label in labels):
        shade = int(222 + 24 * height)
        return (shade, shade, shade)

    uv = _front_print_uv(vertices, mesh, tri, texture_image)
    if uv is None:
        shade = int(232 + 18 * height)
        return (shade, shade, shade)

    x0, y0, x1, y1 = _chromatic_bbox(texture_image)
    px = min(texture_image.width - 1, max(0, int(x0 + uv[0] * max(1, x1 - x0))))
    py = min(texture_image.height - 1, max(0, int(y0 + uv[1] * max(1, y1 - y0))))
    color = np.asarray(texture_image.getpixel((px, py))[:3], dtype=np.float32)
    cloth_shade = 0.86 + 0.18 * height
    color = np.clip(color * cloth_shade, 0, 255)
    return tuple(int(channel) for channel in color)


def _soft_edge(fill: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(max(208, int(channel * 0.96)) for channel in fill)


def _display_edges(mesh: GarmentMesh) -> np.ndarray:
    key = id(mesh)
    cached = _DISPLAY_EDGE_CACHE.get(key)
    if cached is not None:
        return cached

    counts: dict[tuple[int, int], int] = {}
    for a, b, c in mesh.faces:
        for edge in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            edge = tuple(sorted(edge))
            counts[edge] = counts.get(edge, 0) + 1
    boundary_edges = {edge for edge, count in counts.items() if count == 1}
    edges = np.asarray(sorted(boundary_edges), dtype=np.int32)
    _DISPLAY_EDGE_CACHE[key] = edges
    return edges


def _front_print_uv(
    vertices: np.ndarray,
    mesh: GarmentMesh,
    tri: np.ndarray,
    texture_image: Image.Image | None = None,
) -> np.ndarray | None:
    if mesh.uvs is not None:
        body_uv = mesh.uvs[tri].mean(axis=0)
    else:
        bbox = _front_body_bbox(mesh)
        if bbox is None:
            return None
        mins, maxs = bbox
        center = mesh.material_vertices[tri, :2].mean(axis=0)
        body_uv = (center - mins) / np.maximum(maxs - mins, 1e-6)
    print_min, print_max = front_print_uv_window(mesh, texture_image)
    if np.any(body_uv < print_min) or np.any(body_uv > print_max):
        return None
    uv = (body_uv - print_min) / np.maximum(print_max - print_min, 1e-6)
    return np.asarray([uv[0], 1.0 - uv[1]], dtype=np.float32)


def _front_body_bbox(mesh: GarmentMesh) -> tuple[np.ndarray, np.ndarray] | None:
    key = id(mesh)
    cached = _FRONT_BODY_BBOX_CACHE.get(key)
    if cached is not None:
        return cached
    body_indices = np.asarray(
        [idx for idx, label in enumerate(mesh.panel_labels) if label == "front_front_body"],
        dtype=np.int32,
    )
    if body_indices.size == 0:
        return None
    xy = mesh.material_vertices[body_indices, :2]
    bbox = (xy.min(axis=0), xy.max(axis=0))
    _FRONT_BODY_BBOX_CACHE[key] = bbox
    return bbox


def _chromatic_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    key = id(image)
    cached = _TEXTURE_BBOX_CACHE.get(key)
    if cached is not None:
        return cached
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    warm = (arr[:, :, 0] > arr[:, :, 1] * 0.85) & (arr[:, :, 1] > arr[:, :, 2] * 0.75)
    mask = (chroma > 28) & warm
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        bbox = (0, 0, image.width - 1, image.height - 1)
        _TEXTURE_BBOX_CACHE[key] = bbox
        return bbox
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad_x = int((x1 - x0 + 1) * 0.12)
    pad_y = int((y1 - y0 + 1) * 0.10)
    bbox = (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(image.width - 1, x1 + pad_x),
        min(image.height - 1, y1 + pad_y),
    )
    _TEXTURE_BBOX_CACHE[key] = bbox
    return bbox


def _scene_reference_points(
    vertices: np.ndarray,
    avatar: SmplAvatarProxy | None,
    *,
    avatar_vertices: np.ndarray | None = None,
) -> np.ndarray:
    if avatar is None:
        return vertices
    body_vertices = (
        avatar_vertices
        if avatar_vertices is not None
        else getattr(avatar, "mesh_vertices", None)
    )
    if body_vertices is None:
        return vertices
    body_vertices = np.asarray(body_vertices, dtype=np.float32)
    y_min = float(vertices[:, 1].min() - 0.24)
    y_max = float(body_vertices[:, 1].max() + 0.02)
    visible = (body_vertices[:, 1] >= y_min) & (body_vertices[:, 1] <= y_max)
    if not np.any(visible):
        return vertices
    return np.concatenate([vertices, body_vertices[visible]], axis=0)


def _world_to_canvas(points: np.ndarray, size: int, pad: int = 74, reference: np.ndarray | None = None) -> np.ndarray:
    reference_points = points if reference is None else reference
    xy = points[:, :2].copy()
    reference_xy = reference_points[:, :2].copy()
    mins = reference_xy.min(axis=0)
    maxs = reference_xy.max(axis=0)
    center = (mins + maxs) * 0.5
    span = max(float((maxs - mins).max()), 1e-6)
    mins = center - span * 0.58
    maxs = center + span * 0.58
    uv = (xy - mins) / np.maximum(maxs - mins, 1e-6)
    uv[:, 1] = 1.0 - uv[:, 1]
    return pad + uv * (size - 2 * pad)


def _draw_avatar_proxy(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    avatar: SmplAvatarProxy,
    reference: np.ndarray,
    size: int,
) -> None:
    mesh_vertices = getattr(avatar, "mesh_vertices", None)
    mesh_faces = getattr(avatar, "mesh_faces", None)
    if mesh_vertices is not None and mesh_faces is not None:
        _draw_avatar_mesh(image, np.asarray(mesh_vertices), np.asarray(mesh_faces), reference, size)
        return

    y_values = np.linspace(avatar.center_y - avatar.torso_half_height, avatar.center_y + avatar.torso_half_height, 80)
    left = []
    right = []
    for y in y_values:
        y_norm = np.asarray([(y - avatar.center_y) / max(avatar.torso_half_height, 1e-6)])
        width = float(avatar._half_width_np(y_norm)[0])
        left.append([-width, y, avatar.base_z])
        right.append([width, y, avatar.base_z])
    outline = np.asarray(left + right[::-1], dtype=np.float32)
    canvas = _world_to_canvas(outline, size=size, reference=reference)
    draw.polygon([tuple(point) for point in canvas], fill=(232, 226, 216), outline=(172, 154, 136))
    shoulder_y = avatar.center_y + 0.78 * avatar.torso_half_height
    shoulder = np.asarray(
        [
            [-avatar.shoulder_half_width, shoulder_y, avatar.base_z],
            [avatar.shoulder_half_width, shoulder_y, avatar.base_z],
        ],
        dtype=np.float32,
    )
    shoulder_canvas = _world_to_canvas(shoulder, size=size, reference=reference)
    draw.line([tuple(point) for point in shoulder_canvas], fill=(150, 132, 112), width=3)


def _draw_avatar_mesh(
    image: Image.Image,
    body_vertices: np.ndarray,
    body_faces: np.ndarray,
    reference: np.ndarray,
    size: int,
) -> None:
    reference_xy = reference[:, :2]
    center = 0.5 * (reference_xy.min(axis=0) + reference_xy.max(axis=0))
    span = float(np.max(reference_xy.max(axis=0) - reference_xy.min(axis=0)))
    cache_key = (
        id(body_vertices),
        id(body_faces),
        size,
        *(float(value) for value in np.round(center, 5)),
        round(span, 5),
    )
    cached = _AVATAR_LAYER_CACHE.get(cache_key)
    if cached is not None:
        image.paste(cached, (0, 0), cached)
        return

    ref_min = reference[:, 1].min() - 0.16
    ref_max = reference[:, 1].max() + 0.04
    full_triangles = body_vertices[body_faces]
    signed_volume = np.einsum(
        "ij,ij->i",
        full_triangles[:, 0],
        np.cross(full_triangles[:, 1], full_triangles[:, 2]),
    ).sum() / 6.0
    visible = (body_vertices[:, 1] >= ref_min) & (body_vertices[:, 1] <= ref_max)
    face_mask = visible[body_faces].any(axis=1)
    faces = body_faces[face_mask]
    if faces.size == 0:
        return
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    canvas = _world_to_canvas(body_vertices, size=size, reference=reference)
    z = body_vertices[:, 2]
    z_min = float(z.min())
    z_span = max(float(z.max() - z_min), 1e-6)
    triangles = body_vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normal_length = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-10)
    normals /= normal_length
    if signed_volume < 0.0:
        normals *= -1.0
    light = np.asarray([-0.32, 0.42, 0.85], dtype=np.float32)
    light /= np.linalg.norm(light)
    lambert = np.clip(normals @ light, 0.0, 1.0)
    depth = (z[faces].mean(axis=1) - z_min) / z_span
    brightness = np.clip(0.76 + 0.18 * lambert + 0.06 * depth, 0.0, 1.0)
    skin = np.asarray([239.0, 214.0, 194.0], dtype=np.float32)
    for face_idx in np.argsort(z[faces].mean(axis=1)):
        tri = faces[face_idx]
        color = np.clip(skin * brightness[face_idx], 0, 255).astype(np.uint8)
        draw.polygon(
            [tuple(canvas[i]) for i in tri],
            fill=(int(color[0]), int(color[1]), int(color[2]), 255),
        )
    _AVATAR_LAYER_CACHE[cache_key] = layer
    image.paste(layer, (0, 0), layer)


def _draw_avatar_silhouette(draw: ImageDraw.ImageDraw, avatar: SmplAvatarProxy, reference: np.ndarray, size: int) -> None:
    body_vertices = getattr(avatar, "mesh_vertices", None)
    if body_vertices is None:
        return
    body_vertices = np.asarray(body_vertices, dtype=np.float32)
    y_min = float(reference[:, 1].min())
    y_max = float(reference[:, 1].max())
    y_values = np.linspace(y_min, y_max, 96)
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for y in y_values:
        band = np.abs(body_vertices[:, 1] - y) < 0.018
        if not np.any(band):
            continue
        pts = body_vertices[band]
        left.append(pts[np.argmin(pts[:, 0])])
        right.append(pts[np.argmax(pts[:, 0])])
    if len(left) < 4 or len(right) < 4:
        return
    left_canvas = _world_to_canvas(np.asarray(left), size=size, reference=reference)
    right_canvas = _world_to_canvas(np.asarray(right), size=size, reference=reference)
    draw.line([tuple(point) for point in left_canvas], fill=(116, 92, 76), width=2)
    draw.line([tuple(point) for point in right_canvas], fill=(116, 92, 76), width=2)
