from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
from torch import Tensor

from .hypernetwork_kirchhoff_love import (
    KirchhoffLoveShellGeometry,
    KirchhoffLoveShellOutput,
)
from .reference_map_bundle import ReferenceMapFiberBundle


@dataclass(frozen=True)
class VertexMassMeasure:
    """Atomic mass and momentum measure on intrinsic shell vertices."""

    material_coordinates: Tensor
    sheet_ids: Tensor
    positions: Tensor
    momenta: Tensor
    masses: Tensor

    def __post_init__(self) -> None:
        count = len(self.positions)
        if self.positions.shape != (count, 3) or self.momenta.shape != (count, 3):
            raise ValueError("vertex measure positions and momenta must have shape (V,3)")
        if self.material_coordinates.shape != (count, 2):
            raise ValueError("vertex material coordinates must have shape (V,2)")
        if self.sheet_ids.shape != (count,) or torch.any(self.sheet_ids < 0):
            raise ValueError("vertex sheet identities must be non-negative with shape (V,)")
        if self.masses.shape != (count,) or torch.any(self.masses <= 0.0):
            raise ValueError("vertex measure masses must be positive with shape (V,)")

    @property
    def velocities(self) -> Tensor:
        return self.momenta / self.masses[:, None]

    def total_mass(self) -> Tensor:
        return self.masses.sum()

    def total_momentum(self) -> Tensor:
        return self.momenta.sum(dim=0)

    def center_of_mass(self) -> Tensor:
        return (self.masses[:, None] * self.positions).sum(dim=0) / self.total_mass()


@dataclass(frozen=True)
class FaceDeformationMeasure:
    """Rest-area quadrature measure carrying membrane and frame fibres."""

    material_coordinates: Tensor
    sheet_ids: Tensor
    positions: Tensor
    rest_area_weights: Tensor
    deformation_gradient: Tensor
    metric: Tensor
    orientation_quaternion: Tensor

    def __post_init__(self) -> None:
        count = len(self.positions)
        expected = {
            "material_coordinates": (count, 2),
            "sheet_ids": (count,),
            "positions": (count, 3),
            "rest_area_weights": (count,),
            "deformation_gradient": (count, 3, 2),
            "metric": (count, 2, 2),
            "orientation_quaternion": (count, 4),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"face measure {name} must have shape {shape}")
        if torch.any(self.sheet_ids < 0) or torch.any(self.rest_area_weights <= 0.0):
            raise ValueError("face sheet identities and rest-area weights are invalid")


@dataclass(frozen=True)
class HingeCurvatureMeasure:
    """Hinge quadrature measure carrying discrete KL curvature strain."""

    material_coordinates: Tensor
    sheet_ids: Tensor
    positions: Tensor
    weights: Tensor
    bending_strain: Tensor

    def __post_init__(self) -> None:
        count = len(self.positions)
        expected = {
            "material_coordinates": (count, 2),
            "sheet_ids": (count,),
            "positions": (count, 3),
            "weights": (count,),
            "bending_strain": (count,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"hinge measure {name} must have shape {shape}")
        if torch.any(self.sheet_ids < 0) or torch.any(self.weights <= 0.0):
            raise ValueError("hinge sheet identities and weights are invalid")


@dataclass(frozen=True)
class BundleValuedMeasureState:
    """Graded empirical measure on material, spatial, and constitutive fibres."""

    garment_id: str
    hierarchy_level: int
    vertex: VertexMassMeasure
    face: FaceDeformationMeasure
    hinge: HingeCurvatureMeasure
    material_latent: Tensor
    time: float

    def __post_init__(self) -> None:
        if not self.garment_id:
            raise ValueError("bundle measure garment_id must not be empty")
        if self.hierarchy_level < 0:
            raise ValueError("bundle measure hierarchy level must be non-negative")
        if self.material_latent.ndim not in {1, 2}:
            raise ValueError("bundle measure material latent must be rank one or two")


@dataclass(frozen=True)
class BundleMeasureHamiltonian:
    total: Tensor
    kinetic: Tensor
    membrane: Tensor
    bending: Tensor
    gravity: Tensor
    seam: Tensor
    contact: Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach())
            for name in (
                "total",
                "kinetic",
                "membrane",
                "bending",
                "gravity",
                "seam",
                "contact",
            )
        }


@dataclass(frozen=True)
class BundleMeasureContinuityResidual:
    mass: Tensor
    material_coordinates: Tensor
    position_transport: Tensor
    momentum_balance: Tensor

    def squared_norm(self) -> Tensor:
        return (
            self.mass.square().mean()
            + self.material_coordinates.square().mean()
            + self.position_transport.square().mean()
            + self.momentum_balance.square().mean()
        )


@dataclass(frozen=True)
class BundlePortHamiltonianTrainingConfig:
    """Dimensioned normalisation and weights for measure-level training."""

    force_scale_n: float = 2.0e-3
    velocity_scale_m_s: float = 0.1
    power_scale_w: float = 2.0e-3
    continuity_weight: float = 1.0
    energy_balance_weight: float = 0.1
    passivity_weight: float = 0.1
    dissipation_weight: float = 0.1

    def __post_init__(self) -> None:
        if min(self.force_scale_n, self.velocity_scale_m_s, self.power_scale_w) <= 0.0:
            raise ValueError("bundle training force, velocity, and power scales must be positive")
        if min(
            self.continuity_weight,
            self.energy_balance_weight,
            self.passivity_weight,
            self.dissipation_weight,
        ) < 0.0:
            raise ValueError("bundle training loss weights must be non-negative")


@dataclass(frozen=True)
class BundlePortHamiltonianTrainingLoss:
    total: Tensor
    continuity: Tensor
    mass_conservation: Tensor
    material_advection: Tensor
    position_transport: Tensor
    momentum_balance: Tensor
    energy_balance: Tensor
    passivity: Tensor
    dissipation: Tensor
    hamiltonian_rate_w: Tensor
    supplied_power_w: Tensor
    dissipative_power_w: Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach())
            for name in (
                "total",
                "continuity",
                "mass_conservation",
                "material_advection",
                "position_transport",
                "momentum_balance",
                "energy_balance",
                "passivity",
                "dissipation",
                "hamiltonian_rate_w",
                "supplied_power_w",
                "dissipative_power_w",
            )
        }


@dataclass(frozen=True)
class BundleMeasureReport:
    representation: str
    garment_id: str
    hierarchy_level: int
    vertex_atoms: int
    face_atoms: int
    hinge_atoms: int
    sheet_count: int
    total_mass_kg: float
    reference_area_m2: float
    total_momentum_kg_m_s: tuple[float, float, float]
    material_map_immutable: bool
    mass_drift_kg: float
    maximum_material_coordinate_drift: float
    hamiltonian_j: dict[str, float]
    transport_equation: str
    contact_reset: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BundleMeasureBuilder:
    """Lift an intrinsic triangle shell into an exact graded empirical measure."""

    def __init__(
        self,
        reference_map: ReferenceMapFiberBundle,
        geometry: KirchhoffLoveShellGeometry,
        *,
        areal_density_kg_m2: float,
    ) -> None:
        if areal_density_kg_m2 <= 0.0:
            raise ValueError("bundle measure areal density must be positive")
        if not torch.equal(reference_map.faces.cpu(), geometry.faces.cpu()):
            raise ValueError("reference map and shell geometry faces must agree")
        self.reference_map = reference_map
        self.geometry = geometry
        self.areal_density_kg_m2 = float(areal_density_kg_m2)
        self.vertex_sheet_ids = _vertex_sheet_ids(
            geometry.vertex_count,
            geometry.faces,
            reference_map.face_sheet_ids,
        )

    def vertex_measure(self, positions: Tensor, velocities: Tensor) -> VertexMassMeasure:
        if positions.shape != (self.geometry.vertex_count, 3):
            raise ValueError("bundle measure positions do not match shell geometry")
        if velocities.shape != positions.shape:
            raise ValueError("bundle measure velocities must match positions")
        shell = self.geometry.to(positions)
        masses = self.areal_density_kg_m2 * shell.vertex_areas.clamp_min(1.0e-8)
        return VertexMassMeasure(
            shell.vertex_material_coordinates,
            self.vertex_sheet_ids.to(device=positions.device),
            positions,
            masses[:, None] * velocities,
            masses,
        )

    def lift(
        self,
        positions: Tensor,
        velocities: Tensor,
        material_latent: Tensor,
        *,
        time: float,
        garment_id: str = "cloth",
        hierarchy_level: int = 2,
        shell_output: KirchhoffLoveShellOutput | None = None,
    ) -> BundleValuedMeasureState:
        shell = self.geometry.to(positions)
        vertex = self.vertex_measure(positions, velocities)
        fibre = self.reference_map.sample(positions)
        if shell_output is not None:
            if shell_output.deformation_gradient.shape != fibre.deformation_gradient.shape:
                raise ValueError("shell output does not match bundle face measure")
            deformation_gradient = shell_output.deformation_gradient
            metric = shell_output.metric
            bending_strain = shell_output.bending_strain
        else:
            deformation_gradient = fibre.deformation_gradient
            metric = fibre.metric
            bending_strain = _bending_strain(positions, shell)
        face = FaceDeformationMeasure(
            shell.face_material_coordinates,
            self.reference_map.face_sheet_ids.to(device=positions.device),
            positions[shell.faces].mean(dim=1),
            shell.rest_areas,
            deformation_gradient,
            metric,
            fibre.orientation_quaternion,
        )
        hinge_sheet_ids = (
            self.reference_map.face_sheet_ids.to(device=positions.device)[
                shell.hinge_face_pairs[:, 0]
            ]
            if len(shell.hinges)
            else torch.empty(0, dtype=torch.long, device=positions.device)
        )
        hinge = HingeCurvatureMeasure(
            shell.hinge_material_coordinates,
            hinge_sheet_ids,
            (
                0.5 * (positions[shell.hinges[:, 0]] + positions[shell.hinges[:, 1]])
                if len(shell.hinges)
                else positions.new_empty((0, 3))
            ),
            shell.hinge_weights,
            bending_strain,
        )
        return BundleValuedMeasureState(
            garment_id,
            hierarchy_level,
            vertex,
            face,
            hinge,
            material_latent,
            float(time),
        )

    def report(
        self,
        state: BundleValuedMeasureState,
        hamiltonian: BundleMeasureHamiltonian,
        *,
        reference_state: BundleValuedMeasureState | None = None,
    ) -> BundleMeasureReport:
        reference = state if reference_state is None else reference_state
        momentum = state.vertex.total_momentum().detach().cpu().tolist()
        sheet_ids = torch.cat((state.vertex.sheet_ids, state.face.sheet_ids))
        return BundleMeasureReport(
            representation=(
                "graded bundle-valued empirical measure: vertex mass/momentum + "
                "face deformation/frame + hinge curvature"
            ),
            garment_id=state.garment_id,
            hierarchy_level=state.hierarchy_level,
            vertex_atoms=len(state.vertex.positions),
            face_atoms=len(state.face.positions),
            hinge_atoms=len(state.hinge.positions),
            sheet_count=int(sheet_ids.max().item()) + 1 if len(sheet_ids) else 0,
            total_mass_kg=float(state.vertex.total_mass().detach()),
            reference_area_m2=float(state.face.rest_area_weights.sum().detach()),
            total_momentum_kg_m_s=tuple(float(value) for value in momentum),
            material_map_immutable=True,
            mass_drift_kg=float(
                (state.vertex.total_mass() - reference.vertex.total_mass()).detach().abs()
            ),
            maximum_material_coordinate_drift=float(
                (
                    state.vertex.material_coordinates
                    - reference.vertex.material_coordinates.to(
                        state.vertex.material_coordinates
                    )
                )
                .detach()
                .abs()
                .max()
            ),
            hamiltonian_j=hamiltonian.to_dict(),
            transport_equation=(
                "dmu + Div_E(mu[(J-R)delta_H/delta_mu + G_u u]dt "
                "+ mu G_C dlambda)=0"
            ),
            contact_reset="mu_plus = (R_contact)_# mu_minus on the momentum fibre",
        )


def assemble_bundle_measure_hamiltonian(
    state: BundleValuedMeasureState,
    membrane_energy: Tensor,
    bending_energy: Tensor,
    *,
    constitutive_scale: float = 1.0,
    gravity_m_s2: tuple[float, float, float] = (0.0, -9.81, 0.0),
    seam_energy: Tensor | None = None,
    contact_energy: Tensor | None = None,
) -> BundleMeasureHamiltonian:
    if membrane_energy.shape != (len(state.face.positions),):
        raise ValueError("one membrane energy is required per face atom")
    if bending_energy.shape != (len(state.hinge.positions),):
        raise ValueError("one bending energy is required per hinge atom")
    if constitutive_scale < 0.0:
        raise ValueError("constitutive energy scale must be non-negative")
    vertex = state.vertex
    kinetic = 0.5 * (vertex.momenta.square().sum(dim=-1) / vertex.masses).sum()
    membrane = float(constitutive_scale) * membrane_energy.sum()
    bending = float(constitutive_scale) * bending_energy.sum()
    gravity_vector = vertex.positions.new_tensor(gravity_m_s2)
    gravity = -(vertex.masses[:, None] * gravity_vector * vertex.positions).sum()
    seam = vertex.positions.new_zeros(()) if seam_energy is None else seam_energy
    contact = vertex.positions.new_zeros(()) if contact_energy is None else contact_energy
    if seam.ndim != 0 or contact.ndim != 0:
        raise ValueError("bundle seam and contact energies must be scalar")
    total = kinetic + membrane + bending + gravity + seam + contact
    return BundleMeasureHamiltonian(
        total, kinetic, membrane, bending, gravity, seam, contact
    )


def symplectic_euler_measure_step(
    measure: VertexMassMeasure,
    force: Tensor,
    dt: float,
) -> VertexMassMeasure:
    """Canonical kick-drift transport of empirical mass atoms."""

    if force.shape != measure.positions.shape:
        raise ValueError("bundle measure force must match vertex positions")
    if dt <= 0.0:
        raise ValueError("bundle measure time step must be positive")
    momenta = measure.momenta + float(dt) * force
    positions = measure.positions + float(dt) * momenta / measure.masses[:, None]
    return VertexMassMeasure(
        measure.material_coordinates,
        measure.sheet_ids,
        positions,
        momenta,
        measure.masses,
    )


def momentum_fibre_push_forward(
    state: BundleValuedMeasureState,
    velocities: Tensor,
    *,
    time: float | None = None,
) -> BundleValuedMeasureState:
    """Push the measure through a contact reset acting only on momentum."""

    if velocities.shape != state.vertex.positions.shape:
        raise ValueError("push-forward velocities must match vertex atoms")
    vertex = vertex_momentum_fibre_push_forward(state.vertex, velocities)
    return replace(state, vertex=vertex, time=state.time if time is None else float(time))


def vertex_momentum_fibre_push_forward(
    measure: VertexMassMeasure,
    velocities: Tensor,
) -> VertexMassMeasure:
    if velocities.shape != measure.positions.shape:
        raise ValueError("push-forward velocities must match vertex atoms")
    return replace(measure, momenta=measure.masses[:, None] * velocities)


def bundle_measure_continuity_residual(
    previous: VertexMassMeasure,
    current: VertexMassMeasure,
    force: Tensor,
    dt: float,
) -> BundleMeasureContinuityResidual:
    if len(previous.positions) != len(current.positions) or force.shape != current.positions.shape:
        raise ValueError("bundle continuity states and force must align")
    if dt <= 0.0:
        raise ValueError("bundle continuity time step must be positive")
    return BundleMeasureContinuityResidual(
        mass=(current.masses - previous.masses) / float(dt),
        material_coordinates=(
            current.material_coordinates - previous.material_coordinates
        )
        / float(dt),
        position_transport=(current.positions - previous.positions) / float(dt)
        - current.velocities,
        momentum_balance=(current.momenta - previous.momenta) / float(dt) - force,
    )


def bundle_measure_port_hamiltonian_loss(
    current: BundleValuedMeasureState,
    following: BundleValuedMeasureState,
    force: Tensor,
    current_hamiltonian: BundleMeasureHamiltonian,
    following_hamiltonian: BundleMeasureHamiltonian,
    *,
    dt: float,
    supplied_power: Tensor | None = None,
    dissipative_power: Tensor | None = None,
    hierarchy_weight: Tensor | None = None,
    config: BundlePortHamiltonianTrainingConfig = BundlePortHamiltonianTrainingConfig(),
) -> BundlePortHamiltonianTrainingLoss:
    """Train empirical shell transport and open-system energy consistency.

    ``current`` is supported at ``q_k`` with the incoming half-step momentum,
    while ``following`` is supported at ``q_(k+1)`` with the outgoing half-step
    momentum. Conservative storage stays in ``H``. Control/constraint power is
    supplied explicitly, and irreversible ports must have non-positive power.
    """

    if current.garment_id != following.garment_id:
        raise ValueError("bundle training states must belong to the same garment")
    if dt <= 0.0:
        raise ValueError("bundle training time step must be positive")
    residual = bundle_measure_continuity_residual(
        current.vertex,
        following.vertex,
        force,
        dt,
    )
    mass = current.vertex.masses
    weight = (
        torch.ones_like(mass)
        if hierarchy_weight is None
        else hierarchy_weight.to(mass)
    )
    if weight.shape != mass.shape or torch.any(weight < 0.0):
        raise ValueError("bundle hierarchy weight must be non-negative with shape (V,)")
    normalizer = weight.sum().clamp_min(1.0)
    mass_rate_scale = mass.mean().clamp_min(1.0e-12) / float(dt)
    mass_conservation = (
        weight * (residual.mass / mass_rate_scale).square()
    ).sum() / normalizer
    material_advection = (
        weight[:, None] * (float(dt) * residual.material_coordinates).square()
    ).sum() / (2.0 * normalizer)
    position_transport = (
        weight
        * (
            residual.position_transport / config.velocity_scale_m_s
        ).square().sum(dim=-1)
    ).sum() / normalizer
    scaled_momentum = residual.momentum_balance / config.force_scale_n
    momentum_balance = (
        weight * (torch.sqrt(1.0 + scaled_momentum.square().sum(dim=-1)) - 1.0)
    ).sum() / normalizer
    continuity = (
        mass_conservation
        + material_advection
        + position_transport
        + momentum_balance
    )

    supplied = _scalar_power(
        supplied_power,
        current.vertex.positions,
        name="supplied power",
    )
    dissipative = _scalar_power(
        dissipative_power,
        current.vertex.positions,
        name="dissipative power",
    )
    hamiltonian_rate = (
        following_hamiltonian.total - current_hamiltonian.total
    ) / float(dt)
    normalized_balance = (
        hamiltonian_rate - supplied - dissipative
    ) / config.power_scale_w
    energy_balance = torch.sqrt(1.0 + normalized_balance.square()) - 1.0
    passivity_excess = torch.relu(
        (hamiltonian_rate - supplied) / config.power_scale_w
    )
    passivity = torch.sqrt(1.0 + passivity_excess.square()) - 1.0
    dissipation_excess = torch.relu(dissipative / config.power_scale_w)
    dissipation = torch.sqrt(1.0 + dissipation_excess.square()) - 1.0
    total = (
        config.continuity_weight * continuity
        + config.energy_balance_weight * energy_balance
        + config.passivity_weight * passivity
        + config.dissipation_weight * dissipation
    )
    return BundlePortHamiltonianTrainingLoss(
        total,
        continuity,
        mass_conservation,
        material_advection,
        position_transport,
        momentum_balance,
        energy_balance,
        passivity,
        dissipation,
        hamiltonian_rate,
        supplied,
        dissipative,
    )


def _vertex_sheet_ids(
    vertex_count: int,
    faces: Tensor,
    face_sheet_ids: Tensor,
) -> Tensor:
    faces_cpu = faces.detach().cpu().to(torch.long)
    sheets_cpu = face_sheet_ids.detach().cpu().to(torch.long)
    minimum = torch.full((vertex_count,), torch.iinfo(torch.long).max, dtype=torch.long)
    maximum = torch.full((vertex_count,), -1, dtype=torch.long)
    for corner in range(3):
        vertex_ids = faces_cpu[:, corner]
        minimum.scatter_reduce_(0, vertex_ids, sheets_cpu, reduce="amin", include_self=True)
        maximum.scatter_reduce_(0, vertex_ids, sheets_cpu, reduce="amax", include_self=True)
    if torch.any(maximum < 0):
        raise ValueError("every bundle-measure vertex must belong to a face")
    # A welded shell vertex may lie on the overlap of two panel charts. The
    # vertex mass atom is unique, so use the minimum incident chart as a stable
    # representative; face atoms retain every original panel identity.
    return minimum


def _scalar_power(
    value: Tensor | None,
    reference: Tensor,
    *,
    name: str,
) -> Tensor:
    result = reference.new_zeros(()) if value is None else value.to(reference)
    if result.ndim != 0 or torch.is_complex(result):
        raise ValueError(f"bundle {name} must be a real scalar")
    return result


def _bending_strain(positions: Tensor, shell: KirchhoffLoveShellGeometry) -> Tensor:
    if not len(shell.hinges):
        return positions.new_empty((0,))
    triangles = positions[shell.faces]
    normals = torch.nn.functional.normalize(
        torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=-1,
        ),
        dim=-1,
        eps=1.0e-10,
    )
    paired = normals[shell.hinge_face_pairs]
    edge_tangent = torch.nn.functional.normalize(
        positions[shell.hinges[:, 1]] - positions[shell.hinges[:, 0]],
        dim=-1,
        eps=1.0e-10,
    )
    sine = (
        edge_tangent
        * torch.linalg.cross(paired[:, 0], paired[:, 1], dim=-1)
    ).sum(dim=-1)
    cosine = (paired[:, 0] * paired[:, 1]).sum(dim=-1).clamp(-1.0, 1.0)
    delta = torch.atan2(sine, cosine) - shell.rest_hinge_angles
    return torch.atan2(torch.sin(delta), torch.cos(delta))
