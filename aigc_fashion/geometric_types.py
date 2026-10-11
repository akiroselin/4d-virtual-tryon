from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class GeometryContractError(ValueError):
    """Raised when a learned spatial module violates its declared symmetry."""


class ModuleRole(str, Enum):
    SHARED_LEARNED = "shared_learned"
    PER_SCENE_FIELD = "per_scene_field"
    NUMERICAL_SOLVER = "numerical_solver"
    OBSERVATION = "observation"


class SymmetryGroup(str, Enum):
    NONE = "none"
    E3 = "E(3)"
    SE3_LEFT = "SE(3)-left"
    SO2_GAUGE = "SO(2)-gauge"
    INTRINSIC_ISOMETRY = "intrinsic-isometry"
    E3_TIME_TRANSLATION = "E(3)xR-time-translation"
    PRODUCT_PHASE = "E(3)xT*SE(3)xSO(2)-phase"


class GeometricFieldKind(str, Enum):
    UNTYPED_CARTESIAN = "untyped-cartesian-components"
    SCALAR_0E = "scalar-0e"
    PSEUDOSCALAR_0O = "pseudoscalar-0o"
    POLAR_VECTOR_1O = "polar-vector-1o"
    AXIAL_VECTOR_1E = "axial-vector-1e"
    AFFINE_POSITION_1O = "affine-position-1o"
    SE3_POSE = "SE3-pose"
    SE3_TANGENT = "SE3-tangent"
    SE3_COTANGENT = "SE3-cotangent"
    SO2_CHARGE = "SO2-charge"
    INTRINSIC_SCALAR = "intrinsic-scalar"
    MATERIAL_COORDINATE = "material-coordinate"
    TIME_SCALAR = "time-scalar"
    CANONICAL_MOMENTUM_1O = "canonical-momentum-1o"


class GeometryPrimitive(str, Enum):
    RELATIVE_DISPLACEMENT = "relative-displacement"
    RELATIVE_TIME = "relative-time"
    GROUP_COMPOSITION = "group-composition"
    LAPLACE_BELTRAMI = "laplace-beltrami"
    PARALLEL_TRANSPORT = "parallel-transport"
    RIEMANNIAN_EXP_LOG = "riemannian-exp-log"
    SYMPLECTIC_FORM = "symplectic-form"
    COTANGENT_LIFT = "cotangent-lift"
    CLIFFORD_PRODUCT = "clifford-product"


class ArchitectureFamily(str, Enum):
    EGNN = "EGNN"
    PAINN = "PaiNN"
    SEGNN_E3NN = "SEGNN/e3nn"
    GEOMETRIC_CLIFFORD = "geometric-Clifford"
    DIFFUSION_NET = "DiffusionNet"
    SE3_MESSAGE_PASSING = "SE3-message-passing"
    RIEMANNIAN_SCORE = "Riemannian-score"
    CONSERVATIVE_GNO = "conservative-GNO"
    SPATIOTEMPORAL_EQUIVARIANT_ATTENTION = "spatiotemporal-equivariant-attention"
    GAUGE_EQUIVARIANT_CNN = "gauge-equivariant-CNN"
    E3_ANCHOR_NDF = "E3-anchor-NDF"
    SCENE_OPTIMIZED_NDF = "scene-optimized-NDF"
    SCENE_OPTIMIZED_FLOW = "scene-optimized-flow"
    NUMERICAL_PDE = "numerical-PDE"
    PORT_HAMILTONIAN_GRAPH = "port-Hamiltonian-graph"


class GeometricTask(str, Enum):
    PARTICLE_DYNAMICS = "particle-dynamics"
    SOFT_SURFACE = "soft-surface"
    RIGID_ARTICULATION = "rigid-articulation"
    FLUID_MULTIPHYSICS = "fluid-multiphysics"
    SCENE_GRAPH_4D = "scene-graph-4D"
    TACTILE_FIELD = "tactile-field"
    GENERATIVE_POSE = "generative-pose"
    MIXED_MULTIVECTOR = "mixed-multivector"


@dataclass(frozen=True)
class GeometricFieldSpec:
    name: str
    kind: GeometricFieldKind
    channels: int = 1
    charge: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise GeometryContractError("field name must not be empty")
        if self.channels <= 0:
            raise GeometryContractError(f"{self.name}: channels must be positive")
        if self.kind is GeometricFieldKind.SO2_CHARGE:
            if (
                not isinstance(self.charge, int)
                or isinstance(self.charge, bool)
                or self.charge == 0
            ):
                raise GeometryContractError(
                    f"{self.name}: an SO(2) field requires a non-zero integer charge"
                )
        elif self.charge is not None:
            raise GeometryContractError(
                f"{self.name}: charge is only valid for an SO(2) field"
            )


@dataclass(frozen=True)
class GeometricModuleContract:
    name: str
    role: ModuleRole
    symmetry: SymmetryGroup
    family: ArchitectureFamily
    inputs: tuple[GeometricFieldSpec, ...]
    outputs: tuple[GeometricFieldSpec, ...]
    primitives: frozenset[GeometryPrimitive] = frozenset()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise GeometryContractError("module name must not be empty")
        if not self.inputs or not self.outputs:
            raise GeometryContractError(f"{self.name}: inputs and outputs are required")
        names = [field.name for field in (*self.inputs, *self.outputs)]
        if len(names) != len(set(names)):
            raise GeometryContractError(f"{self.name}: field names must be unique")
        self._validate_shared_learned_contract()

    def _validate_shared_learned_contract(self) -> None:
        learned_role = self.role in {
            ModuleRole.SHARED_LEARNED,
            ModuleRole.PER_SCENE_FIELD,
        }
        if not learned_role:
            return
        all_fields = (*self.inputs, *self.outputs)
        if any(field.kind is GeometricFieldKind.UNTYPED_CARTESIAN for field in all_fields):
            raise GeometryContractError(
                f"{self.name}: learned modules cannot consume or emit untyped xyz"
            )
        spatial_outputs = {
            GeometricFieldKind.POLAR_VECTOR_1O,
            GeometricFieldKind.AXIAL_VECTOR_1E,
            GeometricFieldKind.AFFINE_POSITION_1O,
            GeometricFieldKind.SE3_POSE,
            GeometricFieldKind.SE3_TANGENT,
            GeometricFieldKind.SE3_COTANGENT,
            GeometricFieldKind.SO2_CHARGE,
        }
        if self.role is ModuleRole.SHARED_LEARNED and self.symmetry is SymmetryGroup.NONE and any(
            field.kind in spatial_outputs for field in self.outputs
        ):
            raise GeometryContractError(
                f"{self.name}: a shared spatial output requires a non-trivial group action"
            )

        has_position = any(
            field.kind is GeometricFieldKind.AFFINE_POSITION_1O for field in self.inputs
        )
        if self.symmetry in {SymmetryGroup.E3, SymmetryGroup.E3_TIME_TRANSLATION}:
            if has_position and GeometryPrimitive.RELATIVE_DISPLACEMENT not in self.primitives:
                raise GeometryContractError(
                    f"{self.name}: E(3) position inputs must be reduced to relative displacement"
                )
        if self.symmetry is SymmetryGroup.E3_TIME_TRANSLATION:
            has_time = any(
                field.kind is GeometricFieldKind.TIME_SCALAR for field in self.inputs
            )
            if has_time and GeometryPrimitive.RELATIVE_TIME not in self.primitives:
                raise GeometryContractError(
                    f"{self.name}: time-shift symmetry requires relative time"
                )
        if self.symmetry is SymmetryGroup.SE3_LEFT:
            if GeometryPrimitive.GROUP_COMPOSITION not in self.primitives:
                raise GeometryContractError(
                    f"{self.name}: SE(3) symmetry requires relative group composition"
                )
        if self.symmetry is SymmetryGroup.SO2_GAUGE:
            if GeometryPrimitive.PARALLEL_TRANSPORT not in self.primitives:
                raise GeometryContractError(
                    f"{self.name}: gauge features require parallel transport"
                )
            if not any(field.kind is GeometricFieldKind.SO2_CHARGE for field in all_fields):
                raise GeometryContractError(
                    f"{self.name}: an SO(2)-gauge module must declare charged fields"
                )
        if self.symmetry is SymmetryGroup.INTRINSIC_ISOMETRY:
            if GeometryPrimitive.LAPLACE_BELTRAMI not in self.primitives:
                raise GeometryContractError(
                    f"{self.name}: intrinsic symmetry requires a Laplace-Beltrami operator"
                )


class GeometricArchitectureManifest:
    """Registry and build gate for shared geometric-learning modules."""

    def __init__(self, contracts: Iterable[GeometricModuleContract] = ()) -> None:
        self._contracts: dict[str, GeometricModuleContract] = {}
        for contract in contracts:
            self.register(contract)

    @property
    def contracts(self) -> tuple[GeometricModuleContract, ...]:
        return tuple(self._contracts.values())

    def register(self, contract: GeometricModuleContract) -> None:
        if contract.name in self._contracts:
            raise GeometryContractError(f"duplicate geometric module: {contract.name}")
        self._contracts[contract.name] = contract

    def require(self, names: Iterable[str]) -> None:
        missing = sorted(set(names) - self._contracts.keys())
        if missing:
            raise GeometryContractError(
                "missing geometric contracts: " + ", ".join(missing)
            )

    def families(self) -> frozenset[ArchitectureFamily]:
        return frozenset(contract.family for contract in self._contracts.values())


@dataclass(frozen=True)
class ArchitectureRecommendation:
    primary: ArchitectureFamily
    alternatives: tuple[ArchitectureFamily, ...]
    reason: str


def recommend_geometric_architecture(
    task: GeometricTask,
    *,
    needs_higher_order_irreps: bool = False,
) -> ArchitectureRecommendation:
    """Choose the cheapest geometric family that expresses the required fields."""

    if task is GeometricTask.PARTICLE_DYNAMICS:
        if needs_higher_order_irreps:
            return ArchitectureRecommendation(
                ArchitectureFamily.SEGNN_E3NN,
                (ArchitectureFamily.PAINN, ArchitectureFamily.EGNN),
                "Higher-order tensor coupling justifies explicit irreps and tensor products.",
            )
        return ArchitectureRecommendation(
            ArchitectureFamily.EGNN,
            (ArchitectureFamily.PAINN, ArchitectureFamily.SEGNN_E3NN),
            "Scalar/vector particle dynamics usually needs only low-order E(3) paths.",
        )
    routes = {
        GeometricTask.SOFT_SURFACE: ArchitectureRecommendation(
            ArchitectureFamily.DIFFUSION_NET,
            (ArchitectureFamily.SEGNN_E3NN,),
            "Material-surface propagation should use intrinsic operators.",
        ),
        GeometricTask.RIGID_ARTICULATION: ArchitectureRecommendation(
            ArchitectureFamily.SE3_MESSAGE_PASSING,
            (ArchitectureFamily.GEOMETRIC_CLIFFORD,),
            "Relative group elements encode joints without choosing a world frame.",
        ),
        GeometricTask.FLUID_MULTIPHYSICS: ArchitectureRecommendation(
            ArchitectureFamily.CONSERVATIVE_GNO,
            (ArchitectureFamily.SEGNN_E3NN,),
            "Irregular multiscale PDE states need conservative integral kernels.",
        ),
        GeometricTask.SCENE_GRAPH_4D: ArchitectureRecommendation(
            ArchitectureFamily.SPATIOTEMPORAL_EQUIVARIANT_ATTENTION,
            (ArchitectureFamily.SEGNN_E3NN,),
            "Hierarchy, relative space, and relative time belong in one graph operator.",
        ),
        GeometricTask.TACTILE_FIELD: ArchitectureRecommendation(
            ArchitectureFamily.GAUGE_EQUIVARIANT_CNN,
            (),
            "Tangential directions require local-frame transport, not a canonical frame.",
        ),
        GeometricTask.GENERATIVE_POSE: ArchitectureRecommendation(
            ArchitectureFamily.RIEMANNIAN_SCORE,
            (ArchitectureFamily.SE3_MESSAGE_PASSING,),
            "Pose noise and scores must remain on the SE(3) manifold.",
        ),
        GeometricTask.MIXED_MULTIVECTOR: ArchitectureRecommendation(
            ArchitectureFamily.GEOMETRIC_CLIFFORD,
            (ArchitectureFamily.SEGNN_E3NN,),
            "Mixed points, directions, oriented planes, and rotors justify multivectors.",
        ),
    }
    return routes[task]


def build_genesis_gl_manifest() -> GeometricArchitectureManifest:
    """Return contracts for the implemented Genesis-GL replacement modules."""

    scalar = GeometricFieldKind.SCALAR_0E
    position = GeometricFieldKind.AFFINE_POSITION_1O
    vector = GeometricFieldKind.POLAR_VECTOR_1O
    contracts = (
        GeometricModuleContract(
            "particle_cloth_prior",
            ModuleRole.SHARED_LEARNED,
            SymmetryGroup.E3,
            ArchitectureFamily.EGNN,
            (
                GeometricFieldSpec("particle_position", position),
                GeometricFieldSpec("particle_velocity", vector),
                GeometricFieldSpec("material_invariants", scalar, channels=5),
            ),
            (GeometricFieldSpec("internal_force", vector),),
            frozenset({GeometryPrimitive.RELATIVE_DISPLACEMENT}),
        ),
        GeometricModuleContract(
            "softbody_intrinsic_diffusion",
            ModuleRole.SHARED_LEARNED,
            SymmetryGroup.INTRINSIC_ISOMETRY,
            ArchitectureFamily.DIFFUSION_NET,
            (GeometricFieldSpec("surface_features", GeometricFieldKind.INTRINSIC_SCALAR),),
            (GeometricFieldSpec("diffused_features", GeometricFieldKind.INTRINSIC_SCALAR),),
            frozenset({GeometryPrimitive.LAPLACE_BELTRAMI}),
        ),
        GeometricModuleContract(
            "se3_generative_score",
            ModuleRole.SHARED_LEARNED,
            SymmetryGroup.SE3_LEFT,
            ArchitectureFamily.RIEMANNIAN_SCORE,
            (GeometricFieldSpec("noisy_pose", GeometricFieldKind.SE3_POSE),),
            (GeometricFieldSpec("body_score", GeometricFieldKind.SE3_TANGENT),),
            frozenset(
                {GeometryPrimitive.GROUP_COMPOSITION, GeometryPrimitive.RIEMANNIAN_EXP_LOG}
            ),
        ),
        GeometricModuleContract(
            "tactile_gauge_transport",
            ModuleRole.SHARED_LEARNED,
            SymmetryGroup.SO2_GAUGE,
            ArchitectureFamily.GAUGE_EQUIVARIANT_CNN,
            (GeometricFieldSpec("taxel_direction", GeometricFieldKind.SO2_CHARGE, charge=1),),
            (GeometricFieldSpec("contact_feature", GeometricFieldKind.SO2_CHARGE, charge=1),),
            frozenset({GeometryPrimitive.PARALLEL_TRANSPORT}),
        ),
        GeometricModuleContract(
            "body_contact_gauge_response",
            ModuleRole.SHARED_LEARNED,
            SymmetryGroup.SO2_GAUGE,
            ArchitectureFamily.GAUGE_EQUIVARIANT_CNN,
            (
                GeometricFieldSpec(
                    "contact_tangent_velocity",
                    GeometricFieldKind.SO2_CHARGE,
                    charge=1,
                ),
                GeometricFieldSpec("contact_sdf_gap", scalar),
            ),
            (GeometricFieldSpec("contact_friction_scale", scalar),),
            frozenset({GeometryPrimitive.PARALLEL_TRANSPORT}),
        ),
        GeometricModuleContract(
            "rigid_joint_messages",
            ModuleRole.SHARED_LEARNED,
            SymmetryGroup.SE3_LEFT,
            ArchitectureFamily.SE3_MESSAGE_PASSING,
            (
                GeometricFieldSpec("link_pose", GeometricFieldKind.SE3_POSE),
                GeometricFieldSpec("joint_invariants", scalar),
            ),
            (GeometricFieldSpec("joint_wrench", GeometricFieldKind.SE3_COTANGENT),),
            frozenset({GeometryPrimitive.GROUP_COMPOSITION}),
        ),
        GeometricModuleContract(
            "fluid_graph_operator",
            ModuleRole.SHARED_LEARNED,
            SymmetryGroup.E3,
            ArchitectureFamily.CONSERVATIVE_GNO,
            (
                GeometricFieldSpec("fluid_position", position),
                GeometricFieldSpec("fluid_scalar_state", scalar),
                GeometricFieldSpec("fluid_vector_state", vector),
            ),
            (
                GeometricFieldSpec("scalar_flux_rate", scalar),
                GeometricFieldSpec("vector_flux_rate", vector),
            ),
            frozenset({GeometryPrimitive.RELATIVE_DISPLACEMENT}),
        ),
        GeometricModuleContract(
            "scene_graph_attention",
            ModuleRole.SHARED_LEARNED,
            SymmetryGroup.E3_TIME_TRANSLATION,
            ArchitectureFamily.SPATIOTEMPORAL_EQUIVARIANT_ATTENTION,
            (
                GeometricFieldSpec("object_position", position),
                GeometricFieldSpec("object_time", GeometricFieldKind.TIME_SCALAR),
                GeometricFieldSpec("object_invariants", scalar),
            ),
            (
                GeometricFieldSpec("scene_scalars", scalar),
                GeometricFieldSpec("scene_vectors", vector),
            ),
            frozenset(
                {GeometryPrimitive.RELATIVE_DISPLACEMENT, GeometryPrimitive.RELATIVE_TIME}
            ),
        ),
    )
    return GeometricArchitectureManifest(contracts)


def build_whole_system_geometry_manifest() -> GeometricArchitectureManifest:
    """Combine shared Genesis-GL contracts with coordinate-bearing system roles."""

    manifest = build_genesis_gl_manifest()
    untyped = GeometricFieldKind.UNTYPED_CARTESIAN
    manifest.register(
        GeometricModuleContract(
            "genesis_numerical_physics",
            ModuleRole.NUMERICAL_SOLVER,
            SymmetryGroup.NONE,
            ArchitectureFamily.NUMERICAL_PDE,
            (GeometricFieldSpec("solver_coordinates", untyped),),
            (GeometricFieldSpec("integrated_state", untyped),),
        )
    )
    manifest.register(
        GeometricModuleContract(
            "geometric_phase_graph",
            ModuleRole.NUMERICAL_SOLVER,
            SymmetryGroup.PRODUCT_PHASE,
            ArchitectureFamily.PORT_HAMILTONIAN_GRAPH,
            (
                GeometricFieldSpec("cloth_phase_position", GeometricFieldKind.AFFINE_POSITION_1O),
                GeometricFieldSpec(
                    "cloth_phase_momentum",
                    GeometricFieldKind.CANONICAL_MOMENTUM_1O,
                ),
                GeometricFieldSpec("rigid_phase_pose", GeometricFieldKind.SE3_POSE),
                GeometricFieldSpec("rigid_phase_wrench", GeometricFieldKind.SE3_COTANGENT),
                GeometricFieldSpec(
                    "contact_phase_tangent",
                    GeometricFieldKind.SO2_CHARGE,
                    charge=1,
                ),
            ),
            (
                GeometricFieldSpec("next_phase_position", GeometricFieldKind.AFFINE_POSITION_1O),
                GeometricFieldSpec(
                    "next_phase_momentum",
                    GeometricFieldKind.CANONICAL_MOMENTUM_1O,
                ),
            ),
            frozenset(
                {
                    GeometryPrimitive.SYMPLECTIC_FORM,
                    GeometryPrimitive.COTANGENT_LIFT,
                    GeometryPrimitive.PARALLEL_TRANSPORT,
                }
            ),
        )
    )
    manifest.register(
        GeometricModuleContract(
            "neuralclothsim_scene_ndf",
            ModuleRole.PER_SCENE_FIELD,
            SymmetryGroup.E3,
            ArchitectureFamily.E3_ANCHOR_NDF,
            (
                GeometricFieldSpec("material_time_query", GeometricFieldKind.MATERIAL_COORDINATE),
                GeometricFieldSpec("physical_time", GeometricFieldKind.TIME_SCALAR),
                GeometricFieldSpec("cloth_anchor_position", GeometricFieldKind.AFFINE_POSITION_1O),
                GeometricFieldSpec("body_anchor_position", GeometricFieldKind.AFFINE_POSITION_1O),
                GeometricFieldSpec("body_anchor_normal", GeometricFieldKind.POLAR_VECTOR_1O),
                GeometricFieldSpec("typed_vector_bank", GeometricFieldKind.POLAR_VECTOR_1O),
            ),
            (
                GeometricFieldSpec("scene_surface_position", GeometricFieldKind.AFFINE_POSITION_1O),
                GeometricFieldSpec("scene_surface_velocity", GeometricFieldKind.POLAR_VECTOR_1O),
            ),
            frozenset({GeometryPrimitive.RELATIVE_DISPLACEMENT}),
        )
    )
    manifest.register(
        GeometricModuleContract(
            "cortexode_scene_flow",
            ModuleRole.PER_SCENE_FIELD,
            SymmetryGroup.E3,
            ArchitectureFamily.SCENE_OPTIMIZED_FLOW,
            (
                GeometricFieldSpec("flow_state", GeometricFieldKind.AFFINE_POSITION_1O),
                GeometricFieldSpec("body_sdf", GeometricFieldKind.SCALAR_0E),
                GeometricFieldSpec("body_sdf_gradient", GeometricFieldKind.POLAR_VECTOR_1O),
                GeometricFieldSpec(
                    "flow_material_coordinate",
                    GeometricFieldKind.MATERIAL_COORDINATE,
                ),
            ),
            (GeometricFieldSpec("flow_velocity", GeometricFieldKind.POLAR_VECTOR_1O),),
            frozenset({GeometryPrimitive.RELATIVE_DISPLACEMENT}),
        )
    )
    return manifest
