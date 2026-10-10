from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.geometric_types import (  # noqa: E402
    ArchitectureFamily,
    GeometricArchitectureManifest,
    GeometricFieldKind,
    GeometricFieldSpec,
    GeometricModuleContract,
    GeometricTask,
    GeometryContractError,
    GeometryPrimitive,
    ModuleRole,
    SymmetryGroup,
    build_genesis_gl_manifest,
    build_whole_system_geometry_manifest,
    recommend_geometric_architecture,
)


def test_shared_model_rejects_untyped_xyz() -> None:
    with pytest.raises(GeometryContractError, match="untyped xyz"):
        GeometricModuleContract(
            "raw_xyz_mlp",
            ModuleRole.SHARED_LEARNED,
            SymmetryGroup.E3,
            ArchitectureFamily.EGNN,
            (GeometricFieldSpec("xyz", GeometricFieldKind.UNTYPED_CARTESIAN),),
            (GeometricFieldSpec("force", GeometricFieldKind.POLAR_VECTOR_1O),),
            frozenset({GeometryPrimitive.RELATIVE_DISPLACEMENT}),
        )


def test_e3_and_time_contracts_require_relative_geometry() -> None:
    position = GeometricFieldSpec("position", GeometricFieldKind.AFFINE_POSITION_1O)
    output = GeometricFieldSpec("velocity", GeometricFieldKind.POLAR_VECTOR_1O)
    with pytest.raises(GeometryContractError, match="relative displacement"):
        GeometricModuleContract(
            "absolute_position_network",
            ModuleRole.SHARED_LEARNED,
            SymmetryGroup.E3,
            ArchitectureFamily.EGNN,
            (position,),
            (output,),
        )

    with pytest.raises(GeometryContractError, match="relative time"):
        GeometricModuleContract(
            "absolute_time_attention",
            ModuleRole.SHARED_LEARNED,
            SymmetryGroup.E3_TIME_TRANSLATION,
            ArchitectureFamily.SPATIOTEMPORAL_EQUIVARIANT_ATTENTION,
            (position, GeometricFieldSpec("time", GeometricFieldKind.TIME_SCALAR)),
            (output,),
            frozenset({GeometryPrimitive.RELATIVE_DISPLACEMENT}),
        )


def test_only_numerical_solver_keeps_untyped_coordinate_access() -> None:
    numerical = GeometricModuleContract(
        "genesis_mpm_kernel",
        ModuleRole.NUMERICAL_SOLVER,
        SymmetryGroup.NONE,
        ArchitectureFamily.NUMERICAL_PDE,
        (GeometricFieldSpec("particle_xyz", GeometricFieldKind.UNTYPED_CARTESIAN),),
        (GeometricFieldSpec("next_xyz", GeometricFieldKind.UNTYPED_CARTESIAN),),
    )
    with pytest.raises(GeometryContractError, match="untyped xyz"):
        GeometricModuleContract(
            "single_scene_ndf",
            ModuleRole.PER_SCENE_FIELD,
            SymmetryGroup.NONE,
            ArchitectureFamily.SCENE_OPTIMIZED_NDF,
            (GeometricFieldSpec("query_xyz", GeometricFieldKind.UNTYPED_CARTESIAN),),
            (GeometricFieldSpec("surface_xyz", GeometricFieldKind.UNTYPED_CARTESIAN),),
        )
    manifest = GeometricArchitectureManifest((numerical,))
    assert len(manifest.contracts) == 1


def test_project_manifest_covers_every_shared_replacement() -> None:
    manifest = build_genesis_gl_manifest()
    manifest.require(
        {
            "particle_cloth_prior",
            "softbody_intrinsic_diffusion",
            "se3_generative_score",
            "tactile_gauge_transport",
            "body_contact_gauge_response",
            "rigid_joint_messages",
            "fluid_graph_operator",
            "scene_graph_attention",
        }
    )
    assert {
        ArchitectureFamily.EGNN,
        ArchitectureFamily.DIFFUSION_NET,
        ArchitectureFamily.RIEMANNIAN_SCORE,
        ArchitectureFamily.GAUGE_EQUIVARIANT_CNN,
        ArchitectureFamily.SE3_MESSAGE_PASSING,
        ArchitectureFamily.CONSERVATIVE_GNO,
        ArchitectureFamily.SPATIOTEMPORAL_EQUIVARIANT_ATTENTION,
    } <= manifest.families()


def test_whole_system_manifest_keeps_coordinate_exceptions_explicit() -> None:
    manifest = build_whole_system_geometry_manifest()
    manifest.require(
        {
            "genesis_numerical_physics",
            "geometric_phase_graph",
            "neuralclothsim_scene_ndf",
            "cortexode_scene_flow",
        }
    )
    roles = {contract.name: contract.role for contract in manifest.contracts}
    contracts = {contract.name: contract for contract in manifest.contracts}
    assert roles["genesis_numerical_physics"] is ModuleRole.NUMERICAL_SOLVER
    assert roles["geometric_phase_graph"] is ModuleRole.NUMERICAL_SOLVER
    assert roles["neuralclothsim_scene_ndf"] is ModuleRole.PER_SCENE_FIELD
    assert roles["cortexode_scene_flow"] is ModuleRole.PER_SCENE_FIELD
    assert contracts["neuralclothsim_scene_ndf"].symmetry is SymmetryGroup.E3
    assert contracts["neuralclothsim_scene_ndf"].family is ArchitectureFamily.E3_ANCHOR_NDF
    assert contracts["cortexode_scene_flow"].symmetry is SymmetryGroup.E3
    assert contracts["geometric_phase_graph"].symmetry is SymmetryGroup.PRODUCT_PHASE
    assert (
        contracts["geometric_phase_graph"].family
        is ArchitectureFamily.PORT_HAMILTONIAN_GRAPH
    )
    for name, contract in contracts.items():
        if contract.role in {ModuleRole.SHARED_LEARNED, ModuleRole.PER_SCENE_FIELD}:
            assert all(
                field.kind is not GeometricFieldKind.UNTYPED_CARTESIAN
                for field in (*contract.inputs, *contract.outputs)
            ), name


def test_router_defaults_to_low_order_and_uses_clifford_selectively() -> None:
    particle = recommend_geometric_architecture(GeometricTask.PARTICLE_DYNAMICS)
    tensor_particle = recommend_geometric_architecture(
        GeometricTask.PARTICLE_DYNAMICS,
        needs_higher_order_irreps=True,
    )
    multivector = recommend_geometric_architecture(GeometricTask.MIXED_MULTIVECTOR)
    assert particle.primary is ArchitectureFamily.EGNN
    assert tensor_particle.primary is ArchitectureFamily.SEGNN_E3NN
    assert multivector.primary is ArchitectureFamily.GEOMETRIC_CLIFFORD


def test_so2_fields_require_a_charge_and_duplicate_modules_are_rejected() -> None:
    with pytest.raises(GeometryContractError, match="non-zero integer charge"):
        GeometricFieldSpec("direction", GeometricFieldKind.SO2_CHARGE)
    with pytest.raises(GeometryContractError, match="non-zero integer charge"):
        GeometricFieldSpec("fractional_direction", GeometricFieldKind.SO2_CHARGE, charge=1.5)

    manifest = build_genesis_gl_manifest()
    with pytest.raises(GeometryContractError, match="duplicate"):
        manifest.register(manifest.contracts[0])
