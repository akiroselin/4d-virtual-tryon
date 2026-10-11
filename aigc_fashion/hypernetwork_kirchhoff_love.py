from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .biquaternion_phase import membrane_strain_to_complex

from .physics_conditioned_flow import SpectralLinear


WARP_BRANCH = 0
WEFT_BRANCH = 1
SHEAR_BRANCH = 2
AREA_BRANCH = 3
BENDING_BRANCH = 4
CONSTITUTIVE_BRANCHES = 5


@dataclass(frozen=True)
class KirchhoffLoveShellGeometry:
    """Immutable material-space operators for a multi-chart thin shell."""

    vertex_count: int
    panel_count: int
    vertex_material_coordinates: Tensor
    faces: Tensor
    inverse_material_matrices: Tensor
    rest_areas: Tensor
    face_material_coordinates: Tensor
    face_panel_features: Tensor
    hinges: Tensor
    hinge_face_pairs: Tensor
    rest_hinge_angles: Tensor
    hinge_weights: Tensor
    hinge_material_coordinates: Tensor
    hinge_panel_features: Tensor
    free_boundary_edges: Tensor
    free_boundary_face_ids: Tensor
    free_boundary_material_normals: Tensor
    free_boundary_lengths: Tensor
    vertex_areas: Tensor
    rejected_face_count: int

    def to(self, reference: Tensor) -> KirchhoffLoveShellGeometry:
        floating_names = {
            "inverse_material_matrices",
            "rest_areas",
            "face_material_coordinates",
            "face_panel_features",
            "hinge_weights",
            "rest_hinge_angles",
            "hinge_material_coordinates",
            "hinge_panel_features",
            "free_boundary_material_normals",
            "free_boundary_lengths",
            "vertex_areas",
            "vertex_material_coordinates",
        }
        integer_names = {
            "faces",
            "hinges",
            "hinge_face_pairs",
            "free_boundary_edges",
            "free_boundary_face_ids",
        }
        values: dict[str, object] = {
            "vertex_count": self.vertex_count,
            "panel_count": self.panel_count,
            "rejected_face_count": self.rejected_face_count,
        }
        for name in floating_names:
            values[name] = getattr(self, name).to(
                device=reference.device,
                dtype=reference.dtype,
            )
        for name in integer_names:
            values[name] = getattr(self, name).to(device=reference.device)
        return KirchhoffLoveShellGeometry(**values)


def build_kirchhoff_love_shell_geometry(
    material_coordinates: Tensor,
    faces: Tensor,
    panel_ids: Tensor,
    seam_pairs: Tensor | None = None,
    *,
    minimum_triangle_quality: float = 0.02,
    reference_positions: Tensor | None = None,
) -> KirchhoffLoveShellGeometry:
    """Build discrete Kirchhoff-Love operators from an atlas or rest surface.

    ``material_coordinates`` condition the material hypernetwork. By default
    they also define the membrane metric. A welded implicit shell can instead
    pass ``reference_positions``; every triangle then receives a local tangent
    chart derived from its 3D rest surface, avoiding a distorted global UV map.
    """

    if material_coordinates.ndim != 2 or material_coordinates.shape[1] != 2:
        raise ValueError("material_coordinates must have shape (V, 2)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if panel_ids.shape != (len(material_coordinates),):
        raise ValueError("panel_ids must have shape (V,)")
    if reference_positions is not None and reference_positions.shape != (
        len(material_coordinates),
        3,
    ):
        raise ValueError("reference_positions must have shape (V, 3)")
    if minimum_triangle_quality <= 0.0:
        raise ValueError("minimum_triangle_quality must be positive")
    if len(material_coordinates) == 0 or len(faces) == 0:
        raise ValueError("the shell requires vertices and faces")

    coordinates = material_coordinates.detach().cpu().to(torch.float64)
    triangles = faces.detach().cpu().to(torch.long)
    panels = panel_ids.detach().cpu().to(torch.long)
    reference = (
        reference_positions.detach().cpu().to(torch.float64)
        if reference_positions is not None
        else None
    )
    if int(triangles.min()) < 0 or int(triangles.max()) >= len(coordinates):
        raise ValueError("faces contain an out-of-range vertex index")
    panel_count = int(panels.max().item()) + 1
    normalized_coordinates = _normalise_material_charts(coordinates, panels, panel_count)

    if reference is None:
        local_face_coordinates = coordinates[triangles]
        edge_01 = local_face_coordinates[:, 1] - local_face_coordinates[:, 0]
        edge_02 = local_face_coordinates[:, 2] - local_face_coordinates[:, 0]
        edge_12 = local_face_coordinates[:, 2] - local_face_coordinates[:, 1]
        metric_edge_squared = torch.stack(
            (
                edge_01.square().sum(dim=-1),
                edge_02.square().sum(dim=-1),
                edge_12.square().sum(dim=-1),
            ),
            dim=-1,
        )
    else:
        rest_triangles = reference[triangles]
        rest_01 = rest_triangles[:, 1] - rest_triangles[:, 0]
        rest_02 = rest_triangles[:, 2] - rest_triangles[:, 0]
        rest_12 = rest_triangles[:, 2] - rest_triangles[:, 1]
        length_01 = torch.linalg.vector_norm(rest_01, dim=-1).clamp_min(1.0e-16)
        tangent_01 = rest_01 / length_01[:, None]
        coordinate_02_x = (rest_02 * tangent_01).sum(dim=-1)
        coordinate_02_y = torch.sqrt(
            (rest_02.square().sum(dim=-1) - coordinate_02_x.square()).clamp_min(0.0)
        )
        zeros = torch.zeros_like(length_01)
        local_face_coordinates = torch.stack(
            (
                torch.stack((zeros, zeros), dim=-1),
                torch.stack((length_01, zeros), dim=-1),
                torch.stack((coordinate_02_x, coordinate_02_y), dim=-1),
            ),
            dim=1,
        )
        edge_01 = local_face_coordinates[:, 1] - local_face_coordinates[:, 0]
        edge_02 = local_face_coordinates[:, 2] - local_face_coordinates[:, 0]
        edge_12 = local_face_coordinates[:, 2] - local_face_coordinates[:, 1]
        metric_edge_squared = torch.stack(
            (
                rest_01.square().sum(dim=-1),
                rest_02.square().sum(dim=-1),
                rest_12.square().sum(dim=-1),
            ),
            dim=-1,
        )
    material_matrices = torch.stack((edge_01, edge_02), dim=-1)
    determinants = torch.linalg.det(material_matrices)
    maximum_edge_squared = metric_edge_squared.max(dim=-1).values
    quality = determinants.abs() / maximum_edge_squared.clamp_min(1.0e-16)
    same_panel = (
        (panels[triangles[:, 0]] == panels[triangles[:, 1]])
        & (panels[triangles[:, 0]] == panels[triangles[:, 2]])
    )
    valid = (quality >= minimum_triangle_quality) & same_panel
    valid_face_ids = torch.nonzero(valid, as_tuple=False).flatten()
    if len(valid_face_ids) == 0:
        raise ValueError("no non-degenerate material triangles remain")

    physical_faces = triangles[valid_face_ids]
    physical_local_coordinates = local_face_coordinates[valid_face_ids]
    material_matrices = material_matrices[valid_face_ids]
    inverse_material_matrices = torch.linalg.inv(material_matrices)
    rest_areas = 0.5 * determinants[valid_face_ids].abs()
    face_panel_ids = panels[physical_faces[:, 0]]
    face_panel_features = F.one_hot(face_panel_ids, panel_count).to(torch.float64)
    face_material_coordinates = normalized_coordinates[physical_faces].mean(dim=1)

    full_edge_counts: dict[tuple[int, int], int] = {}
    for face in triangles.tolist():
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(a), int(b))))
            full_edge_counts[edge] = full_edge_counts.get(edge, 0) + 1

    edge_incidence: dict[tuple[int, int], list[tuple[int, int]]] = {}
    oriented_edges: dict[tuple[int, int], tuple[int, int]] = {}
    for local_face_id, face in enumerate(physical_faces.tolist()):
        for a, b, opposite in (
            (face[0], face[1], face[2]),
            (face[1], face[2], face[0]),
            (face[2], face[0], face[1]),
        ):
            edge = tuple(sorted((int(a), int(b))))
            edge_incidence.setdefault(edge, []).append((local_face_id, int(opposite)))
            oriented_edges.setdefault(edge, (int(a), int(b)))

    hinges: list[list[int]] = []
    hinge_face_pairs: list[list[int]] = []
    hinge_weights: list[float] = []
    hinge_material_coordinates: list[Tensor] = []
    hinge_panel_ids: list[int] = []
    for edge, incidence in edge_incidence.items():
        if len(incidence) != 2:
            continue
        (face_a, opposite_a), (face_b, opposite_b) = incidence
        if int(face_panel_ids[face_a]) != int(face_panel_ids[face_b]):
            continue
        a, b = edge
        metric_source = coordinates if reference is None else reference
        rest_edge_squared = float((metric_source[b] - metric_source[a]).square().sum())
        dual_area = float(rest_areas[face_a] + rest_areas[face_b])
        hinges.append([a, b, opposite_a, opposite_b])
        hinge_face_pairs.append([face_a, face_b])
        hinge_weights.append(rest_edge_squared / max(dual_area, 1.0e-12))
        hinge_material_coordinates.append(
            0.5 * (normalized_coordinates[a] + normalized_coordinates[b])
        )
        hinge_panel_ids.append(int(face_panel_ids[face_a]))

    stitched_edges = _stitched_boundary_edges(
        seam_pairs,
        {edge for edge, count in full_edge_counts.items() if count == 1},
    )
    free_edges: list[list[int]] = []
    free_face_ids: list[int] = []
    free_normals: list[Tensor] = []
    free_lengths: list[float] = []
    for edge, incidence in edge_incidence.items():
        if full_edge_counts.get(edge, 0) != 1 or edge in stitched_edges:
            continue
        face_id, _opposite = incidence[0]
        a, b = oriented_edges[edge]
        face_vertices = physical_faces[face_id]
        local_a = int(torch.nonzero(face_vertices == a, as_tuple=False)[0, 0])
        local_b = int(torch.nonzero(face_vertices == b, as_tuple=False)[0, 0])
        face_coordinates = physical_local_coordinates[face_id]
        tangent = face_coordinates[local_b] - face_coordinates[local_a]
        length = float(torch.linalg.vector_norm(tangent))
        if length <= 1.0e-12:
            continue
        normal = torch.stack((tangent[1], -tangent[0])) / length
        midpoint = 0.5 * (
            face_coordinates[local_a] + face_coordinates[local_b]
        )
        centroid = face_coordinates.mean(dim=0)
        if float(torch.dot(normal, centroid - midpoint)) > 0.0:
            normal = -normal
        free_edges.append([a, b])
        free_face_ids.append(face_id)
        free_normals.append(normal)
        free_lengths.append(length)

    vertex_areas = torch.zeros(len(coordinates), dtype=torch.float64)
    for corner in range(3):
        vertex_areas.index_add_(0, physical_faces[:, corner], rest_areas / 3.0)

    hinge_indices = _long_rows(hinges, 4)
    hinge_face_indices = _long_rows(hinge_face_pairs, 2)
    if len(hinge_indices):
        rest_embedding = (
            reference
            if reference is not None
            else torch.cat(
                (coordinates, torch.zeros((len(coordinates), 1), dtype=coordinates.dtype)),
                dim=-1,
            )
        )
        rest_hinge_angles = _signed_dihedral_angles(
            rest_embedding,
            physical_faces,
            hinge_indices,
            hinge_face_indices,
        )
    else:
        rest_hinge_angles = torch.empty(0, dtype=torch.float64)

    return KirchhoffLoveShellGeometry(
        vertex_count=len(coordinates),
        panel_count=panel_count,
        vertex_material_coordinates=normalized_coordinates.to(torch.float32),
        faces=physical_faces,
        inverse_material_matrices=inverse_material_matrices.to(torch.float32),
        rest_areas=rest_areas.to(torch.float32),
        face_material_coordinates=face_material_coordinates.to(torch.float32),
        face_panel_features=face_panel_features.to(torch.float32),
        hinges=hinge_indices,
        hinge_face_pairs=hinge_face_indices,
        rest_hinge_angles=rest_hinge_angles.to(torch.float32),
        hinge_weights=torch.as_tensor(hinge_weights, dtype=torch.float32),
        hinge_material_coordinates=_float_rows(hinge_material_coordinates, 2),
        hinge_panel_features=(
            F.one_hot(torch.as_tensor(hinge_panel_ids), panel_count).to(torch.float32)
            if hinge_panel_ids
            else torch.empty((0, panel_count), dtype=torch.float32)
        ),
        free_boundary_edges=_long_rows(free_edges, 2),
        free_boundary_face_ids=torch.as_tensor(free_face_ids, dtype=torch.long),
        free_boundary_material_normals=_float_rows(free_normals, 2),
        free_boundary_lengths=torch.as_tensor(free_lengths, dtype=torch.float32),
        vertex_areas=vertex_areas.to(torch.float32),
        rejected_face_count=int((~valid).sum().item()),
    )


@dataclass(frozen=True)
class LocalNODEParameters:
    first_weight: Tensor
    first_bias: Tensor
    hidden_weight: Tensor
    hidden_bias: Tensor
    output_weight: Tensor
    output_bias: Tensor
    stiffness: Tensor


class MaterialPointHyperNetwork(nn.Module):
    """Map each material point to a complete local constitutive NODE."""

    def __init__(
        self,
        panel_count: int,
        *,
        latent_dim: int = 12,
        hidden_dim: int = 32,
        node_hidden_dim: int = 4,
        spectral_cap: float = 0.9,
        generated_spectral_cap: float = 0.75,
    ) -> None:
        super().__init__()
        if min(panel_count, latent_dim, hidden_dim, node_hidden_dim) <= 0:
            raise ValueError("hypernetwork dimensions must be positive")
        self.panel_count = panel_count
        self.latent_dim = latent_dim
        self.node_hidden_dim = node_hidden_dim
        self.generated_spectral_cap = float(generated_spectral_cap)
        coordinate_feature_dim = 10
        input_dim = latent_dim + coordinate_feature_dim + panel_count
        parameters_per_branch = (
            node_hidden_dim * 2
            + node_hidden_dim
            + node_hidden_dim * node_hidden_dim
            + node_hidden_dim
            + node_hidden_dim
            + 1
            + 1
        )
        self.parameters_per_branch = parameters_per_branch
        self.input_layer = SpectralLinear(
            input_dim,
            hidden_dim,
            max_singular_value=spectral_cap,
        )
        self.hidden_layer = SpectralLinear(
            hidden_dim,
            hidden_dim,
            max_singular_value=spectral_cap,
        )
        self.output_layer = SpectralLinear(
            hidden_dim,
            CONSTITUTIVE_BRANCHES * parameters_per_branch,
            max_singular_value=spectral_cap,
        )
        nn.init.normal_(self.output_layer.weight, std=1.0e-3)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
        self.register_buffer(
            "minimum_stiffness",
            torch.tensor([20.0, 20.0, 2.0, 20.0, 5.0e-7]),
        )
        self.register_buffer(
            "maximum_stiffness",
            torch.tensor([500.0, 500.0, 120.0, 500.0, 2.0e-4]),
        )

    def forward(
        self,
        material_coordinates: Tensor,
        panel_features: Tensor,
        material_latent: Tensor,
    ) -> LocalNODEParameters:
        if material_coordinates.ndim != 2 or material_coordinates.shape[1] != 2:
            raise ValueError("material_coordinates must have shape (N, 2)")
        if panel_features.shape != (len(material_coordinates), self.panel_count):
            raise ValueError(
                f"panel_features must have shape (N, {self.panel_count})"
            )
        latent = _batch_latent(material_latent, len(material_coordinates), self.latent_dim)
        coordinate_features = _coordinate_fourier_features(material_coordinates)
        features = torch.cat((latent, coordinate_features, panel_features), dim=-1)
        hidden = torch.tanh(self.input_layer(features))
        hidden = torch.tanh(self.hidden_layer(hidden))
        raw = self.output_layer(hidden).reshape(
            len(material_coordinates),
            CONSTITUTIVE_BRANCHES,
            self.parameters_per_branch,
        )

        width = self.node_hidden_dim
        cursor = 0

        def take(count: int) -> Tensor:
            nonlocal cursor
            value = raw[:, :, cursor : cursor + count]
            cursor += count
            return value

        first_weight = take(width * 2).reshape(-1, CONSTITUTIVE_BRANCHES, width, 2)
        first_bias = take(width)
        hidden_weight = take(width * width).reshape(
            -1,
            CONSTITUTIVE_BRANCHES,
            width,
            width,
        )
        hidden_bias = take(width)
        output_weight = take(width).reshape(-1, CONSTITUTIVE_BRANCHES, 1, width)
        output_bias = take(1)
        raw_stiffness = take(1).squeeze(-1)
        if cursor != self.parameters_per_branch:
            raise RuntimeError("constitutive NODE parameter layout is inconsistent")

        first_weight = _cap_generated_weights(
            0.5 * torch.tanh(first_weight),
            self.generated_spectral_cap,
        )
        hidden_weight = _cap_generated_weights(
            0.5 * torch.tanh(hidden_weight),
            self.generated_spectral_cap,
        )
        output_weight = _cap_generated_weights(
            0.5 * torch.tanh(output_weight),
            self.generated_spectral_cap,
        )
        minimum = self.minimum_stiffness.to(raw)
        maximum = self.maximum_stiffness.to(raw)
        log_stiffness = torch.log(minimum) + torch.sigmoid(raw_stiffness) * (
            torch.log(maximum) - torch.log(minimum)
        )
        return LocalNODEParameters(
            first_weight=first_weight,
            first_bias=0.25 * torch.tanh(first_bias),
            hidden_weight=hidden_weight,
            hidden_bias=0.25 * torch.tanh(hidden_bias),
            output_weight=output_weight,
            output_bias=0.25 * torch.tanh(output_bias),
            stiffness=torch.exp(log_stiffness),
        )


class MonotoneConstitutiveNODE(nn.Module):
    """One-dimensional NODE flow with a zero-anchored monotone response."""

    def __init__(self, *, integration_steps: int = 4, rate_limit: float = 0.5) -> None:
        super().__init__()
        if integration_steps <= 0 or rate_limit <= 0.0:
            raise ValueError("integration_steps and rate_limit must be positive")
        self.integration_steps = integration_steps
        self.rate_limit = float(rate_limit)

    def response(
        self,
        strains: Tensor,
        parameters: LocalNODEParameters,
        branches: Sequence[int] | Tensor,
    ) -> Tensor:
        if strains.ndim != 2:
            raise ValueError("strains must have shape (N, K)")
        branch_ids = torch.as_tensor(branches, dtype=torch.long, device=strains.device)
        if branch_ids.shape != (strains.shape[1],):
            raise ValueError("one constitutive branch is required per strain channel")
        if torch.any((branch_ids < 0) | (branch_ids >= CONSTITUTIVE_BRANCHES)):
            raise ValueError("constitutive branch index is out of range")
        selected = _select_node_branches(parameters, branch_ids)
        state = torch.stack((strains, torch.zeros_like(strains)), dim=-1)
        step = 1.0 / self.integration_steps
        time = 0.0
        for _ in range(self.integration_steps):
            k1 = self._vector_field(state, time, selected)
            k2 = self._vector_field(state + 0.5 * step * k1, time + 0.5 * step, selected)
            k3 = self._vector_field(state + 0.5 * step * k2, time + 0.5 * step, selected)
            k4 = self._vector_field(state + step * k3, time + step, selected)
            state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            time += step
        anchored_flow = state[:, :, 0] - state[:, :, 1]
        return selected.stiffness * anchored_flow

    def energy(
        self,
        strains: Tensor,
        parameters: LocalNODEParameters,
        branches: Sequence[int] | Tensor,
    ) -> Tensor:
        # Three-point Gauss-Legendre quadrature of integral_0^strain response(s) ds.
        points = strains.new_tensor([0.1127016654, 0.5, 0.8872983346])
        weights = strains.new_tensor([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])
        integral = torch.zeros_like(strains)
        for point, weight in zip(points, weights):
            integral = integral + weight * self.response(point * strains, parameters, branches)
        return strains * integral

    def _vector_field(
        self,
        state: Tensor,
        time: float,
        parameters: LocalNODEParameters,
    ) -> Tensor:
        time_channel = torch.full_like(state, time)
        inputs = torch.stack((state, time_channel), dim=-1)
        hidden = torch.tanh(
            torch.einsum("nkpi,nkhi->nkph", inputs, parameters.first_weight)
            + parameters.first_bias[:, :, None, :]
        )
        hidden = torch.tanh(
            torch.einsum("nkpi,nkhi->nkph", hidden, parameters.hidden_weight)
            + parameters.hidden_bias[:, :, None, :]
        )
        output = torch.einsum(
            "nkph,nkqh->nkpq",
            hidden,
            parameters.output_weight,
        ).squeeze(-1)
        output = output + parameters.output_bias.squeeze(-1)[:, :, None]
        return self.rate_limit * torch.tanh(output)


@dataclass(frozen=True)
class KirchhoffLoveShellOutput:
    deformation_gradient: Tensor
    metric: Tensor
    principal_stretches: Tensor
    intrinsic_edge_stretches: Tensor
    membrane_strains: Tensor
    complex_membrane_trace: Tensor
    complex_membrane_deviator: Tensor
    membrane_resultants: Tensor
    first_piola: Tensor
    membrane_energy: Tensor
    strain_limit_utilization: Tensor
    strain_limit_energy: Tensor
    bending_strain: Tensor
    bending_energy: Tensor
    membrane_internal_force: Tensor
    strain_limit_internal_force: Tensor
    bending_internal_force: Tensor
    free_boundary_traction: Tensor
    face_stiffness: Tensor
    hinge_stiffness: Tensor


def _hypernode_intrinsic_edge_strain_barrier(
    intrinsic_edge_stretches: Tensor,
    rest_areas: Tensor,
    face_stiffness: Tensor,
    *,
    maximum_tensile_strain: float | None,
    maximum_compressive_strain: float | None,
    onset_fraction: float,
    barrier_scale: float,
) -> tuple[Tensor, Tensor]:
    """Return a discrete intrinsic-metric barrier generated by Hyper-NODE stiffness.

    A welded shell has no trustworthy global warp/weft chart, so the three
    immutable rest-edge stretches provide its parameterization-free material
    metric. Shear remains a separate constitutive NODE branch.
    """

    supplied = (
        maximum_tensile_strain is not None,
        maximum_compressive_strain is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError("tensile and compressive strain limits must be supplied together")
    if not all(supplied):
        return (
            intrinsic_edge_stretches.new_zeros(intrinsic_edge_stretches.shape),
            rest_areas.new_zeros(rest_areas.shape),
        )
    assert maximum_tensile_strain is not None
    assert maximum_compressive_strain is not None
    if not 0.0 < maximum_tensile_strain < 1.0:
        raise ValueError("maximum_tensile_strain must lie in (0, 1)")
    if not 0.0 < maximum_compressive_strain < 1.0:
        raise ValueError("maximum_compressive_strain must lie in (0, 1)")
    if not 0.0 < onset_fraction < 1.0 or barrier_scale < 0.0:
        raise ValueError("strain barrier onset and scale are invalid")

    tensile_usage = F.relu(intrinsic_edge_stretches - 1.0) / maximum_tensile_strain
    compressive_usage = F.relu(1.0 - intrinsic_edge_stretches) / maximum_compressive_strain
    utilization = torch.maximum(tensile_usage, compressive_usage)
    activation = F.relu(utilization - onset_fraction)
    # A finite denominator keeps the force numerically usable after an external
    # impulse crosses the envelope; such crossings are handled by the emergency
    # metric recovery in the integrator.
    remaining = (1.0 - utilization).clamp_min(0.025)
    barrier_shape = (activation / remaining).square()
    violation_shape = (F.relu(utilization - 1.0) / 0.025).square()
    local_membrane_scale = face_stiffness[:, (WARP_BRANCH, WEFT_BRANCH)].mean(dim=-1)
    barrier_density = barrier_scale * local_membrane_scale * (
        barrier_shape + violation_shape
    ).sum(dim=-1)
    return utilization, rest_areas * barrier_density


class HyperNetworkKirchhoffLoveShell(nn.Module):
    """Data-driven orthotropic constitutive field on KL shell kinematics."""

    def __init__(
        self,
        panel_count: int,
        *,
        latent_dim: int = 12,
        hyper_hidden_dim: int = 32,
        node_hidden_dim: int = 4,
        node_steps: int = 4,
        spectral_cap: float = 0.9,
    ) -> None:
        super().__init__()
        self.hypernetwork = MaterialPointHyperNetwork(
            panel_count,
            latent_dim=latent_dim,
            hidden_dim=hyper_hidden_dim,
            node_hidden_dim=node_hidden_dim,
            spectral_cap=spectral_cap,
        )
        self.constitutive_node = MonotoneConstitutiveNODE(integration_steps=node_steps)

    def forward(
        self,
        positions: Tensor,
        geometry: KirchhoffLoveShellGeometry,
        material_latent: Tensor,
        *,
        compute_internal_forces: bool = True,
        compute_bending_forces: bool = True,
        create_graph: bool = True,
        maximum_tensile_strain: float | None = None,
        maximum_compressive_strain: float | None = None,
        strain_limit_onset_fraction: float = 0.8,
        strain_limit_barrier_scale: float = 0.02,
    ) -> KirchhoffLoveShellOutput:
        if positions.shape != (geometry.vertex_count, 3):
            raise ValueError(
                f"positions must have shape ({geometry.vertex_count}, 3)"
            )
        shell = geometry.to(positions)
        triangles = positions[shell.faces]
        spatial_matrices = torch.stack(
            (triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            dim=-1,
        )
        deformation_gradient = spatial_matrices @ shell.inverse_material_matrices
        metric = deformation_gradient.transpose(-1, -2) @ deformation_gradient
        # The closed-form 2x2 spectrum uses a tiny discriminant regularizer.
        # ``eigvalsh`` has undefined second derivatives at the isotropic rest
        # metric, which poisons differentiable force rollouts with NaNs.
        metric_trace = metric[:, 0, 0] + metric[:, 1, 1]
        metric_discriminant = torch.sqrt(
            (metric[:, 0, 0] - metric[:, 1, 1]).square()
            + 4.0 * metric[:, 0, 1].square()
            + 1.0e-12
        )
        metric_eigenvalues = torch.stack(
            (
                0.5 * (metric_trace - metric_discriminant),
                0.5 * (metric_trace + metric_discriminant),
            ),
            dim=-1,
        )
        principal_stretches = torch.sqrt(metric_eigenvalues.clamp_min(1.0e-10))
        rest_material_matrices = torch.linalg.inv(shell.inverse_material_matrices)
        rest_face_edges = torch.stack(
            (
                rest_material_matrices[:, :, 0],
                rest_material_matrices[:, :, 1],
                rest_material_matrices[:, :, 1] - rest_material_matrices[:, :, 0],
            ),
            dim=1,
        )
        current_face_edges = torch.stack(
            (
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 1],
            ),
            dim=1,
        )
        intrinsic_edge_stretches = (
            torch.linalg.vector_norm(current_face_edges, dim=-1)
            / torch.linalg.vector_norm(rest_face_edges, dim=-1).clamp_min(1.0e-10)
        )
        determinant = torch.linalg.det(metric).clamp_min(1.0e-10)
        area_ratio = torch.sqrt(determinant)
        raw_membrane_strains = torch.stack(
            (
                0.5 * (metric[:, 0, 0] - 1.0),
                0.5 * (metric[:, 1, 1] - 1.0),
                metric[:, 0, 1],
                area_ratio - 1.0,
            ),
            dim=-1,
        )
        complex_membrane = membrane_strain_to_complex(raw_membrane_strains)
        membrane_strains = torch.stack(
            (
                0.5 * (complex_membrane.trace + complex_membrane.deviator.real),
                0.5 * (complex_membrane.trace - complex_membrane.deviator.real),
                complex_membrane.deviator.imag,
                raw_membrane_strains[:, 3],
            ),
            dim=-1,
        )
        face_parameters = self.hypernetwork(
            shell.face_material_coordinates,
            shell.face_panel_features,
            material_latent,
        )
        membrane_resultants = self.constitutive_node.response(
            membrane_strains,
            face_parameters,
            (WARP_BRANCH, WEFT_BRANCH, SHEAR_BRANCH, AREA_BRANCH),
        )
        membrane_density = self.constitutive_node.energy(
            membrane_strains,
            face_parameters,
            (WARP_BRANCH, WEFT_BRANCH, SHEAR_BRANCH, AREA_BRANCH),
        ).sum(dim=-1)
        membrane_energy = shell.rest_areas * membrane_density
        strain_limit_utilization, strain_limit_energy = (
            _hypernode_intrinsic_edge_strain_barrier(
                intrinsic_edge_stretches,
                shell.rest_areas,
                face_parameters.stiffness,
                maximum_tensile_strain=maximum_tensile_strain,
                maximum_compressive_strain=maximum_compressive_strain,
                onset_fraction=strain_limit_onset_fraction,
                barrier_scale=strain_limit_barrier_scale,
            )
        )
        first_piola = _surface_first_piola(
            deformation_gradient,
            metric,
            area_ratio,
            membrane_resultants,
        )
        membrane_internal_force = _membrane_nodal_forces(
            first_piola,
            shell,
            positions,
        )
        if (
            compute_internal_forces
            and maximum_tensile_strain is not None
            and positions.requires_grad
        ):
            strain_limit_internal_force = -torch.autograd.grad(
                strain_limit_energy.sum(),
                positions,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
        else:
            strain_limit_internal_force = torch.zeros_like(positions)
        if compute_internal_forces:
            membrane_internal_force = (
                membrane_internal_force + strain_limit_internal_force
            )
        else:
            membrane_internal_force = torch.zeros_like(positions)

        bending_strain, bending_energy, hinge_stiffness = self._bending_energy(
            positions,
            shell,
            material_latent,
        )
        if compute_internal_forces and compute_bending_forces and len(shell.hinges):
            if not positions.requires_grad:
                raise ValueError("positions must require gradients for bending forces")
            bending_internal_force = -torch.autograd.grad(
                bending_energy.sum(),
                positions,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
        else:
            bending_internal_force = torch.zeros_like(positions)

        if len(shell.free_boundary_edges):
            boundary_piola = first_piola[shell.free_boundary_face_ids]
            free_boundary_traction = torch.einsum(
                "bij,bj->bi",
                boundary_piola,
                shell.free_boundary_material_normals,
            )
        else:
            free_boundary_traction = positions.new_empty((0, 3))
        return KirchhoffLoveShellOutput(
            deformation_gradient=deformation_gradient,
            metric=metric,
            principal_stretches=principal_stretches,
            intrinsic_edge_stretches=intrinsic_edge_stretches,
            membrane_strains=membrane_strains,
            complex_membrane_trace=complex_membrane.trace,
            complex_membrane_deviator=complex_membrane.deviator,
            membrane_resultants=membrane_resultants,
            first_piola=first_piola,
            membrane_energy=membrane_energy,
            strain_limit_utilization=strain_limit_utilization,
            strain_limit_energy=strain_limit_energy,
            bending_strain=bending_strain,
            bending_energy=bending_energy,
            membrane_internal_force=membrane_internal_force,
            strain_limit_internal_force=strain_limit_internal_force,
            bending_internal_force=bending_internal_force,
            free_boundary_traction=free_boundary_traction,
            face_stiffness=face_parameters.stiffness,
            hinge_stiffness=hinge_stiffness,
        )

    def _bending_energy(
        self,
        positions: Tensor,
        shell: KirchhoffLoveShellGeometry,
        material_latent: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if len(shell.hinges) == 0:
            empty = positions.new_empty((0,))
            return empty, empty, positions.new_empty((0, CONSTITUTIVE_BRANCHES))
        face_triangles = positions[shell.faces]
        normals = torch.cross(
            face_triangles[:, 1] - face_triangles[:, 0],
            face_triangles[:, 2] - face_triangles[:, 0],
            dim=-1,
        )
        normals = F.normalize(normals, dim=-1, eps=1.0e-8)
        current_angles = _signed_dihedral_angles(
            positions,
            shell.faces,
            shell.hinges,
            shell.hinge_face_pairs,
        )
        angle_delta = current_angles - shell.rest_hinge_angles
        bending_strain = torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta))
        hinge_parameters = self.hypernetwork(
            shell.hinge_material_coordinates,
            shell.hinge_panel_features,
            material_latent,
        )
        bending_density = self.constitutive_node.energy(
            bending_strain[:, None],
            hinge_parameters,
            (BENDING_BRANCH,),
        ).squeeze(-1)
        bending_energy = shell.hinge_weights * bending_density
        return bending_strain, bending_energy, hinge_parameters.stiffness


@dataclass(frozen=True)
class ShellDynamicsLoss:
    total: Tensor
    shell_energy: Tensor
    equilibrium: Tensor
    traction: Tensor
    contact: Tensor
    acceleration: Tensor
    output: KirchhoffLoveShellOutput


def kirchhoff_love_dynamics_loss(
    model: HyperNetworkKirchhoffLoveShell,
    geometry: KirchhoffLoveShellGeometry,
    previous_positions: Tensor,
    current_positions: Tensor,
    next_positions: Tensor,
    material_latent: Tensor,
    *,
    dt: float,
    contact_closest: Tensor | None = None,
    contact_normals: Tensor | None = None,
    prescribed_free_boundary_traction: Tensor | None = None,
    pinned_vertices: Tensor | None = None,
    areal_density_kg_m2: float = 0.18,
    gravity_m_s2: tuple[float, float, float] = (0.0, -9.81, 0.0),
    contact_clearance_m: float = 0.003,
    contact_stiffness_n_m: float = 0.4,
    equilibrium_scale_n: float = 2.0e-3,
    traction_scale_n_m: float = 10.0,
    maximum_tensile_strain: float | None = None,
    maximum_compressive_strain: float | None = None,
    strain_limit_onset_fraction: float = 0.8,
    strain_limit_barrier_scale: float = 0.02,
) -> ShellDynamicsLoss:
    """Dynamic shell balance plus the free-edge Neumann condition."""

    if dt <= 0.0 or areal_density_kg_m2 <= 0.0:
        raise ValueError("dt and areal_density_kg_m2 must be positive")
    if equilibrium_scale_n <= 0.0 or traction_scale_n_m <= 0.0:
        raise ValueError("loss normalisation scales must be positive")
    if previous_positions.shape != current_positions.shape or next_positions.shape != current_positions.shape:
        raise ValueError("the three trajectory samples must have matching shapes")
    output = model(
        current_positions,
        geometry,
        material_latent,
        compute_bending_forces=True,
        create_graph=True,
        maximum_tensile_strain=maximum_tensile_strain,
        maximum_compressive_strain=maximum_compressive_strain,
        strain_limit_onset_fraction=strain_limit_onset_fraction,
        strain_limit_barrier_scale=strain_limit_barrier_scale,
    )
    shell = geometry.to(current_positions)
    mass = areal_density_kg_m2 * shell.vertex_areas
    acceleration = (next_positions - 2.0 * current_positions + previous_positions) / (dt * dt)
    gravity = current_positions.new_tensor(gravity_m_s2)
    external_force = mass[:, None] * gravity
    predicted_traction = output.free_boundary_traction
    if prescribed_free_boundary_traction is None:
        target_traction = torch.zeros_like(predicted_traction)
    else:
        target_traction = prescribed_free_boundary_traction.to(current_positions)
        if target_traction.shape == (3,):
            target_traction = target_traction[None, :].expand_as(predicted_traction)
        if target_traction.shape != predicted_traction.shape:
            raise ValueError(
                "prescribed_free_boundary_traction must have shape (3,) or (B, 3)"
            )
    if len(shell.free_boundary_edges):
        boundary_edge_force = shell.free_boundary_lengths[:, None] * target_traction
        external_force = external_force.clone()
        external_force.index_add_(
            0,
            shell.free_boundary_edges[:, 0],
            0.5 * boundary_edge_force,
        )
        external_force.index_add_(
            0,
            shell.free_boundary_edges[:, 1],
            0.5 * boundary_edge_force,
        )
    contact_loss = current_positions.new_zeros(())
    if contact_closest is not None or contact_normals is not None:
        if contact_closest is None or contact_normals is None:
            raise ValueError("contact_closest and contact_normals must be provided together")
        if contact_closest.shape != current_positions.shape or contact_normals.shape != current_positions.shape:
            raise ValueError("contact tensors must have shape (V, 3)")
        normals = F.normalize(contact_normals.to(current_positions), dim=-1, eps=1.0e-8)
        gap = ((current_positions - contact_closest.to(current_positions)) * normals).sum(dim=-1)
        compression = F.relu(contact_clearance_m - gap)
        contact_force = contact_stiffness_n_m * compression[:, None] * normals
        external_force = external_force + contact_force
        contact_loss = (compression / max(contact_clearance_m, 1.0e-6)).square().mean()

    residual = mass[:, None] * acceleration - (
        output.membrane_internal_force
        + output.bending_internal_force
        + external_force
    )
    active = mass > 0.0
    if pinned_vertices is not None and len(pinned_vertices):
        active = active.clone()
        active[pinned_vertices.to(device=active.device, dtype=torch.long)] = False
    equilibrium = _pseudo_huber(residual[active] / equilibrium_scale_n).mean()
    traction = (
        _pseudo_huber((predicted_traction - target_traction) / traction_scale_n_m).mean()
        if len(predicted_traction)
        else current_positions.new_zeros(())
    )
    reference_area = shell.rest_areas.sum().clamp_min(1.0e-8)
    shell_energy = (
        output.membrane_energy.sum()
        + output.strain_limit_energy.sum()
        + output.bending_energy.sum()
    ) / reference_area
    shell_energy = torch.log1p(shell_energy / 100.0)
    total = shell_energy + equilibrium + traction + 0.25 * contact_loss
    return ShellDynamicsLoss(
        total=total,
        shell_energy=shell_energy,
        equilibrium=equilibrium,
        traction=traction,
        contact=contact_loss,
        acceleration=acceleration,
        output=output,
    )


def _normalise_material_charts(
    coordinates: Tensor,
    panel_ids: Tensor,
    panel_count: int,
) -> Tensor:
    result = torch.zeros_like(coordinates)
    for panel in range(panel_count):
        mask = panel_ids == panel
        if not torch.any(mask):
            continue
        values = coordinates[mask]
        lower = values.min(dim=0).values
        span = (values.max(dim=0).values - lower).clamp_min(1.0e-8)
        result[mask] = 2.0 * (values - lower) / span - 1.0
    return result


def _stitched_boundary_edges(
    seam_pairs: Tensor | None,
    boundary_edges: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    if seam_pairs is None or seam_pairs.numel() == 0:
        return set()
    pairs = seam_pairs.detach().cpu().to(torch.long).reshape(-1, 2)
    mates: dict[int, set[int]] = {}
    for a, b in pairs.tolist():
        mates.setdefault(int(a), set()).add(int(b))
        mates.setdefault(int(b), set()).add(int(a))
    stitched: set[tuple[int, int]] = set()
    for edge in boundary_edges:
        a, b = edge
        for mate_a in mates.get(a, ()):
            if any(tuple(sorted((mate_a, mate_b))) in boundary_edges for mate_b in mates.get(b, ())):
                stitched.add(edge)
                break
    return stitched


def _coordinate_fourier_features(coordinates: Tensor) -> Tensor:
    return torch.cat(
        (
            coordinates,
            torch.sin(pi * coordinates),
            torch.cos(pi * coordinates),
            torch.sin(2.0 * pi * coordinates),
            torch.cos(2.0 * pi * coordinates),
        ),
        dim=-1,
    )


def _batch_latent(latent: Tensor, batch_size: int, latent_dim: int) -> Tensor:
    if latent.shape == (latent_dim,):
        return latent[None, :].expand(batch_size, -1)
    if latent.shape == (1, latent_dim):
        return latent.expand(batch_size, -1)
    if latent.shape == (batch_size, latent_dim):
        return latent
    raise ValueError(f"material_latent must have shape ({latent_dim},) or (N, {latent_dim})")


def _cap_generated_weights(weights: Tensor, cap: float) -> Tensor:
    # Frobenius norm is a differentiable upper bound on the spectral norm:
    # ||W||_2 <= ||W||_F. It preserves the Lipschitz guarantee without an
    # expensive batched SVD for every material point and physical substep.
    frobenius = torch.linalg.vector_norm(weights, dim=(-2, -1))
    scale = torch.clamp(frobenius / cap, min=1.0)
    return weights / scale[..., None, None]


def _select_node_branches(
    parameters: LocalNODEParameters,
    branches: Tensor,
) -> LocalNODEParameters:
    return LocalNODEParameters(
        first_weight=parameters.first_weight.index_select(1, branches),
        first_bias=parameters.first_bias.index_select(1, branches),
        hidden_weight=parameters.hidden_weight.index_select(1, branches),
        hidden_bias=parameters.hidden_bias.index_select(1, branches),
        output_weight=parameters.output_weight.index_select(1, branches),
        output_bias=parameters.output_bias.index_select(1, branches),
        stiffness=parameters.stiffness.index_select(1, branches),
    )


def _surface_first_piola(
    deformation_gradient: Tensor,
    metric: Tensor,
    area_ratio: Tensor,
    resultants: Tensor,
) -> Tensor:
    warp = deformation_gradient[:, :, 0]
    weft = deformation_gradient[:, :, 1]
    first_piola = torch.stack(
        (
            resultants[:, WARP_BRANCH, None] * warp
            + resultants[:, SHEAR_BRANCH, None] * weft,
            resultants[:, WEFT_BRANCH, None] * weft
            + resultants[:, SHEAR_BRANCH, None] * warp,
        ),
        dim=-1,
    )
    identity = torch.eye(2, device=metric.device, dtype=metric.dtype)[None, :, :]
    inverse_metric = torch.linalg.inv(metric + 1.0e-7 * identity)
    area_gradient = area_ratio[:, None, None] * deformation_gradient @ inverse_metric
    return first_piola + resultants[:, AREA_BRANCH, None, None] * area_gradient


def _membrane_nodal_forces(
    first_piola: Tensor,
    shell: KirchhoffLoveShellGeometry,
    positions: Tensor,
) -> Tensor:
    energy_gradient = (
        shell.rest_areas[:, None, None]
        * (first_piola @ shell.inverse_material_matrices.transpose(-1, -2))
    )
    force = torch.zeros_like(positions)
    force.index_add_(0, shell.faces[:, 1], -energy_gradient[:, :, 0])
    force.index_add_(0, shell.faces[:, 2], -energy_gradient[:, :, 1])
    force.index_add_(
        0,
        shell.faces[:, 0],
        energy_gradient[:, :, 0] + energy_gradient[:, :, 1],
    )
    return force


def _signed_dihedral_angles(
    positions: Tensor,
    faces: Tensor,
    hinges: Tensor,
    hinge_face_pairs: Tensor,
) -> Tensor:
    if len(hinges) == 0:
        return positions.new_empty((0,))
    triangles = positions[faces]
    normals = F.normalize(
        torch.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=-1,
        ),
        dim=-1,
        eps=1.0e-10,
    )
    paired = normals[hinge_face_pairs]
    edge_tangent = F.normalize(
        positions[hinges[:, 1]] - positions[hinges[:, 0]],
        dim=-1,
        eps=1.0e-10,
    )
    sine = (
        edge_tangent * torch.cross(paired[:, 0], paired[:, 1], dim=-1)
    ).sum(dim=-1)
    cosine = (paired[:, 0] * paired[:, 1]).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.atan2(sine, cosine)


def _long_rows(values: list[list[int]], width: int) -> Tensor:
    if not values:
        return torch.empty((0, width), dtype=torch.long)
    return torch.as_tensor(values, dtype=torch.long).reshape(-1, width)


def _float_rows(values: list[Tensor], width: int) -> Tensor:
    if not values:
        return torch.empty((0, width), dtype=torch.float32)
    return torch.stack(values, dim=0).to(torch.float32).reshape(-1, width)


def _pseudo_huber(value: Tensor) -> Tensor:
    return torch.sqrt(1.0 + value.square()) - 1.0
