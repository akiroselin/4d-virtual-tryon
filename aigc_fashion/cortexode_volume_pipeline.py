from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import trimesh

from .data import GarmentMesh
from .topology_correction import (
    ExternalCortexODETopologyBackend,
    SurfaceDomain,
    TopologyCorrectionStage,
    TopologyDisposition,
    analyze_triangle_mesh,
)


class GarmentUNet3D(nn.Module):
    """3D U-Net matching the segmentation stage used by CortexODE."""

    def __init__(self, input_channels: int = 1, output_channels: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(64, 128, 3, stride=2, padding=1)
        self.conv5 = nn.Conv3d(128, 128, 3, stride=2, padding=1)
        self.decode4 = nn.Conv3d(256, 64, 3, padding=1)
        self.decode3 = nn.Conv3d(128, 32, 3, padding=1)
        self.decode2 = nn.Conv3d(64, 16, 3, padding=1)
        self.decode1 = nn.Conv3d(32, 16, 3, padding=1)
        self.output1 = nn.Conv3d(16, 16, 3, padding=1)
        self.output2 = nn.Conv3d(16, output_channels, 3, padding=1)

    def forward(self, volume: Tensor) -> Tensor:
        x1 = F.leaky_relu(self.conv1(volume), 0.2)
        x2 = F.leaky_relu(self.conv2(x1), 0.2)
        x3 = F.leaky_relu(self.conv3(x2), 0.2)
        x4 = F.leaky_relu(self.conv4(x3), 0.2)
        encoded = F.leaky_relu(self.conv5(x4), 0.2)
        decoded = F.interpolate(encoded, scale_factor=2, mode="trilinear", align_corners=True)
        decoded = F.leaky_relu(self.decode4(torch.cat((decoded, x4), dim=1)), 0.2)
        decoded = F.interpolate(decoded, scale_factor=2, mode="trilinear", align_corners=True)
        decoded = F.leaky_relu(self.decode3(torch.cat((decoded, x3), dim=1)), 0.2)
        decoded = F.interpolate(decoded, scale_factor=2, mode="trilinear", align_corners=True)
        decoded = F.leaky_relu(self.decode2(torch.cat((decoded, x2), dim=1)), 0.2)
        decoded = F.interpolate(decoded, scale_factor=2, mode="trilinear", align_corners=True)
        decoded = F.leaky_relu(self.decode1(torch.cat((decoded, x1), dim=1)), 0.2)
        return self.output2(F.leaky_relu(self.output1(decoded), 0.2))


@dataclass(frozen=True)
class CortexODEVolumeConfig:
    resolution: int = 48
    unet_steps: int = 36
    unet_learning_rate: float = 2.0e-3
    sdf_sigma_voxels: float = 0.6
    topology_threshold_voxels: float = 4.0
    marching_level_voxels: float = 0.8
    world_padding_m: float = 0.035
    seed: int = 113

    def __post_init__(self) -> None:
        if self.resolution < 32 or self.resolution % 16:
            raise ValueError("3D U-Net resolution must be at least 32 and divisible by 16")
        if self.unet_steps <= 0 or self.unet_learning_rate <= 0.0:
            raise ValueError("3D U-Net optimization settings must be positive")


@dataclass(frozen=True)
class CortexODEVolumeReport:
    pipeline: str
    unet_mode: str
    resolution: int
    unet_steps: int
    segmentation_dice: float
    raw_foreground_voxels: int
    corrected_foreground_voxels: int
    closed_proxy_topology: dict[str, Any]
    semantic_opening_boundary_loops: int
    semantic_openings: tuple[str, ...]
    semantic_stitched_links: tuple[str, ...]
    topology_backend: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CortexODEVolumeEvidence:
    sdf: np.ndarray
    probability: np.ndarray
    world_min: np.ndarray
    world_max: np.ndarray
    closed_vertices: np.ndarray
    closed_faces: np.ndarray


class MultiscaleSDFSampler(nn.Module):
    """Differentiable CortexODE-style local volume sampler.

    Two scales each provide SDF plus its three spatial derivatives, yielding the
    eight body-context channels expected by ``PhysicsConditionedCortexODE``.
    """

    def __init__(self, evidence: CortexODEVolumeEvidence) -> None:
        super().__init__()
        sdf = np.asarray(evidence.sdf, dtype=np.float32)
        spacing = (np.asarray(evidence.world_max) - evidence.world_min) / (np.asarray(sdf.shape) - 1)
        gradients = np.gradient(sdf, *spacing, edge_order=1)
        features = np.stack((sdf, *gradients), axis=0)
        features = torch.as_tensor(features).unsqueeze(0)
        self.register_buffer("fine", features)
        self.register_buffer("coarse", F.avg_pool3d(features, kernel_size=2, stride=2))
        self.register_buffer("world_min", torch.as_tensor(evidence.world_min, dtype=torch.float32))
        self.register_buffer("world_max", torch.as_tensor(evidence.world_max, dtype=torch.float32))

    def forward(self, points: Tensor) -> Tensor:
        normalized = 2.0 * (points - self.world_min) / (self.world_max - self.world_min) - 1.0
        # grid_sample's coordinate order is W,H,D; our volume axes are x,y,z.
        grid = normalized[..., [2, 1, 0]].reshape(1, -1, 1, 1, 3)
        samples = []
        for volume in (self.fine, self.coarse):
            value = F.grid_sample(
                volume,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            samples.append(value[0, :, :, 0, 0].T)
        return torch.cat(samples, dim=-1)


def build_cortexode_volume_evidence(
    garment: GarmentMesh,
    *,
    cortexode_checkout: str | Path,
    config: CortexODEVolumeConfig = CortexODEVolumeConfig(),
) -> tuple[CortexODEVolumeEvidence, CortexODEVolumeReport]:
    """Run 3D U-Net -> SDF -> Fast Marching -> Marching Cubes.

    The current PoC distills one scene's image/category/body occupancy into the
    U-Net. It executes the real architecture but is not a pretrained garment
    segmentation model; dataset training is a separate model-delivery task.
    """

    torch.manual_seed(config.seed)
    semantic_topology = semantic_topology_from_panel_analysis(garment)
    closed_vertices, closed_faces = cap_garment_boundaries(garment)
    closed_mesh = trimesh.Trimesh(closed_vertices, closed_faces, process=True)
    closed_mesh.fix_normals()
    if not closed_mesh.is_watertight:
        raise RuntimeError("semantic garment capping did not produce a watertight proxy")

    world_min = closed_mesh.bounds[0] - config.world_padding_m
    world_max = closed_mesh.bounds[1] + config.world_padding_m
    axes = [np.linspace(world_min[i], world_max[i], config.resolution) for i in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    signed = trimesh.proximity.signed_distance(closed_mesh, points).reshape(
        (config.resolution,) * 3
    )
    target_segmentation = signed >= 0.0
    degraded = gaussian_filter(target_segmentation.astype(np.float32), sigma=1.15)
    degraded += 0.025 * np.random.default_rng(config.seed).standard_normal(degraded.shape)
    input_volume = torch.as_tensor(degraded, dtype=torch.float32)[None, None]
    target_volume = torch.as_tensor(target_segmentation.astype(np.int64))[None]

    unet = GarmentUNet3D()
    optimizer = torch.optim.Adam(unet.parameters(), lr=config.unet_learning_rate)
    for _ in range(config.unet_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = unet(input_volume)
        loss = F.cross_entropy(logits, target_volume)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = unet(input_volume)
        probability = torch.softmax(logits, dim=1)[0, 1].numpy()
        predicted = torch.argmax(logits, dim=1)[0].numpy().astype(bool)
    predicted = _largest_component(predicted)
    intersection = np.count_nonzero(predicted & target_segmentation)
    dice = 2.0 * intersection / max(
        np.count_nonzero(predicted) + np.count_nonzero(target_segmentation), 1
    )

    sdf = distance_transform_edt(~predicted) - distance_transform_edt(predicted)
    sdf = gaussian_filter(sdf.astype(np.float64), sigma=config.sdf_sigma_voxels)
    backend = ExternalCortexODETopologyBackend(cortexode_checkout)
    corrected = TopologyCorrectionStage(backend).correct_closed_sdf(
        sdf,
        threshold=config.topology_threshold_voxels,
        domain=SurfaceDomain.CLOSED_LEVEL_SET,
    )
    vertices_voxel, faces, _normals, _values = marching_cubes(
        -corrected,
        level=-config.marching_level_voxels,
        method="lorensen",
    )
    spacing = (world_max - world_min) / (config.resolution - 1)
    vertices_world = world_min + vertices_voxel * spacing
    topology = analyze_triangle_mesh(vertices_world, faces.astype(np.int32))
    gate = TopologyCorrectionStage().validate_extracted_mesh(
        vertices_world,
        faces.astype(np.int32),
        domain=SurfaceDomain.CLOSED_LEVEL_SET,
    )
    if gate.disposition is not TopologyDisposition.VERIFIED:
        raise RuntimeError(f"Fast Marching output failed the genus-0 gate: {gate.reason}")
    corrected_foreground = corrected <= config.marching_level_voxels
    evidence = CortexODEVolumeEvidence(
        sdf=corrected.astype(np.float32),
        probability=probability.astype(np.float32),
        world_min=world_min.astype(np.float32),
        world_max=world_max.astype(np.float32),
        closed_vertices=vertices_world.astype(np.float32),
        closed_faces=faces.astype(np.int32),
    )
    open_topology = analyze_triangle_mesh(garment.vertices, garment.faces)
    report = CortexODEVolumeReport(
        pipeline="3D U-Net -> signed distance transform -> Fast Marching topology -> Marching Cubes",
        unet_mode="per-scene occupancy distillation; pretrained garment weights not yet supplied",
        resolution=config.resolution,
        unet_steps=config.unet_steps,
        segmentation_dice=float(dice),
        raw_foreground_voxels=int(np.count_nonzero(predicted)),
        corrected_foreground_voxels=int(np.count_nonzero(corrected_foreground)),
        closed_proxy_topology=asdict(topology),
        semantic_opening_boundary_loops=int(open_topology.boundary_loops or 0),
        semantic_openings=tuple(item["name"] for item in semantic_topology["openings"]),
        semantic_stitched_links=tuple(
            item["name"] for item in semantic_topology["stitched_links"]
        ),
        topology_backend="official CortexODE Bazin fast-marching implementation loaded in place",
    )
    return evidence, report


def shell_initialization_from_volume(
    garment_template: GarmentMesh,
    evidence: CortexODEVolumeEvidence,
    *,
    smoothing_iterations: int = 6,
    projection_blend: float = 0.18,
) -> GarmentMesh:
    """Transfer the corrected closed proxy back to a semantic open shell graph."""

    from .cortexode_intrinsic_shell import build_coarse_intrinsic_initial_shell

    smoothed = build_coarse_intrinsic_initial_shell(
        garment_template,
        iterations=smoothing_iterations,
        strength=0.30,
    )
    closed = trimesh.Trimesh(evidence.closed_vertices, evidence.closed_faces, process=False)
    if not 0.0 <= projection_blend <= 1.0:
        raise ValueError("projection_blend must lie in [0, 1]")
    closest, _distance, _triangle = trimesh.proximity.closest_point(closed, smoothed.vertices)
    # Capped regions are temporary topology devices. Keep semantic openings on
    # their template boundary rather than snapping them onto a cap interior.
    boundary = _boundary_vertex_mask(len(smoothed.vertices), smoothed.faces)
    closest[boundary] = smoothed.vertices[boundary]
    closest = smoothed.vertices + projection_blend * (closest - smoothed.vertices)
    return GarmentMesh(
        vertices=np.asarray(closest, dtype=np.float32),
        faces=smoothed.faces.copy(),
        pinned_vertices=smoothed.pinned_vertices.copy(),
        panel_labels=list(smoothed.panel_labels),
        seam_edges=list(smoothed.seam_edges),
        rest_vertices=np.asarray(closest, dtype=np.float32).copy(),
        uvs=None if smoothed.uvs is None else smoothed.uvs.copy(),
        seam_interfaces=list(smoothed.seam_interfaces),
    )


def cap_garment_boundaries(garment: GarmentMesh) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(garment.vertices, dtype=np.float64).tolist()
    faces = np.asarray(garment.faces, dtype=np.int32).tolist()
    for loop in _boundary_loops(garment.faces):
        center_id = len(vertices)
        vertices.append(np.mean(np.asarray(garment.vertices)[loop], axis=0).tolist())
        for index, first in enumerate(loop):
            second = loop[(index + 1) % len(loop)]
            faces.append([int(second), int(first), center_id])
    mesh = trimesh.Trimesh(np.asarray(vertices), np.asarray(faces), process=False)
    trimesh.repair.fix_normals(mesh, multibody=True)
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)


def semantic_topology_from_panel_analysis(garment: GarmentMesh) -> dict[str, Any]:
    """Compile panel semantics into free openings and welded shell links."""

    loops = _boundary_loops(garment.faces)
    labels = np.asarray(garment.panel_labels, dtype=object)
    centers = [np.mean(garment.vertices[loop], axis=0) for loop in loops]
    lower_labels = [str(value).lower() for value in garment.panel_labels]
    is_skirt = any("skirt" in value for value in lower_labels)
    is_tshirt = any("body" in value or "sleeve" in value for value in lower_labels)

    opening_names: dict[int, str] = {}
    if is_skirt and len(loops) == 2:
        order = sorted(range(len(loops)), key=lambda index: centers[index][1], reverse=True)
        opening_names[order[0]] = "waist"
        opening_names[order[1]] = "hem"
    elif is_tshirt:
        torso_loops = []
        for index, loop in enumerate(loops):
            loop_labels = [str(value).lower() for value in labels[loop]]
            if sum("sleeve" in value for value in loop_labels) >= 0.45 * len(loop):
                opening_names[index] = "right_cuff" if centers[index][0] > 0.0 else "left_cuff"
            else:
                torso_loops.append(index)
        if torso_loops:
            ordered = sorted(torso_loops, key=lambda index: centers[index][1], reverse=True)
            opening_names[ordered[0]] = "neck"
            if len(ordered) > 1:
                opening_names[ordered[-1]] = "hem"
    for index in range(len(loops)):
        opening_names.setdefault(index, f"opening_{index}")

    stitched_components = _cross_panel_edge_components(garment)
    stitched_links = []
    for component_index, component in enumerate(stitched_components):
        edge_array = np.asarray(component, dtype=np.int32)
        vertex_ids = np.unique(edge_array)
        panel_names = sorted(set(str(value) for value in labels[vertex_ids]))
        center = np.mean(garment.vertices[vertex_ids], axis=0)
        name = _semantic_link_name(panel_names, center, component_index)
        stitched_links.append(
            {
                "name": name,
                "panels": panel_names,
                "edges": edge_array.tolist(),
                "edge_count": int(len(edge_array)),
                "physics_compilation": (
                    "welded structural rest-length edges + Kirchhoff-Love cross-link hinges"
                ),
            }
        )
    return {
        "source": "panel labels + garment family + body coordinate axes",
        "openings": [
            {
                "name": opening_names[index],
                "vertices": [int(value) for value in loop],
                "vertex_count": len(loop),
                "physics_compilation": "free boundary with Piola traction condition",
            }
            for index, loop in enumerate(loops)
        ],
        "stitched_links": stitched_links,
        "continuous_shell": not bool(garment.seam_edges),
        "runtime_seam_gap_constraints": bool(garment.seam_edges),
    }


def _largest_component(mask: np.ndarray) -> np.ndarray:
    components, count = label(mask, structure=np.ones((3, 3, 3), dtype=np.int8))
    if count == 0:
        raise RuntimeError("3D U-Net predicted no garment foreground")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == int(np.argmax(sizes))


def _boundary_loops(faces: np.ndarray) -> list[list[int]]:
    counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(first), int(second))))
            counts[edge] = counts.get(edge, 0) + 1
    adjacency: dict[int, list[int]] = {}
    for (first, second), count in counts.items():
        if count == 1:
            adjacency.setdefault(first, []).append(second)
            adjacency.setdefault(second, []).append(first)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("garment boundary is not a collection of simple loops")
    loops: list[list[int]] = []
    remaining = set(adjacency)
    while remaining:
        start = min(remaining)
        loop = [start]
        previous = -1
        current = start
        while True:
            candidates = [value for value in adjacency[current] if value != previous]
            following = candidates[0]
            if following == start:
                break
            loop.append(following)
            previous, current = current, following
        remaining.difference_update(loop)
        loops.append(loop)
    return loops


def _boundary_vertex_mask(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    mask = np.zeros(vertex_count, dtype=bool)
    for loop in _boundary_loops(faces):
        mask[np.asarray(loop, dtype=np.int64)] = True
    return mask


def _cross_panel_edge_components(garment: GarmentMesh) -> list[list[tuple[int, int]]]:
    labels = garment.panel_labels
    edges: set[tuple[int, int]] = set()
    for face in np.asarray(garment.faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            a, b = sorted((int(first), int(second)))
            if labels[a] != labels[b]:
                edges.add((a, b))
    vertex_edges: dict[int, set[tuple[int, int]]] = {}
    for edge in edges:
        vertex_edges.setdefault(edge[0], set()).add(edge)
        vertex_edges.setdefault(edge[1], set()).add(edge)
    components: list[list[tuple[int, int]]] = []
    remaining = set(edges)
    while remaining:
        first = remaining.pop()
        component = {first}
        stack = [first[0], first[1]]
        while stack:
            vertex = stack.pop()
            for edge in vertex_edges.get(vertex, ()):
                if edge in remaining:
                    remaining.remove(edge)
                    component.add(edge)
                    stack.extend(edge)
        components.append(sorted(component))
    return components


def _semantic_link_name(panel_names: list[str], center: np.ndarray, index: int) -> str:
    joined = " ".join(name.lower() for name in panel_names)
    side = "right" if center[0] > 0.0 else "left"
    if "skirt" in joined:
        return f"{side}_side_seam"
    if "sleeve" in joined and "body" in joined:
        return f"{side}_sleeve_armhole"
    if "front_body" in joined and "back_body" in joined:
        return f"{side}_shoulder_or_side_seam"
    return f"topology_link_{index}"
