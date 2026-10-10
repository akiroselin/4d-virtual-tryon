from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import copy
import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor
import torch.nn.functional as F

from .body_mesh_avatar import BodyMeshAvatar
from .bundle_measure_port_hamiltonian import (
    BundleMeasureBuilder,
    assemble_bundle_measure_hamiltonian,
    bundle_measure_port_hamiltonian_loss,
)
from .behavioral_phase import BehavioralPhaseTrajectory
from .biquaternion_phase import BiquaternionVertexPhase
from .complex_scene_hamiltonian import (
    KahlerClothPhase,
    assemble_complex_scene_hamiltonian,
)
from .cloth4d_geometric_bridge import (
    Cloth4DGeometricConditioner,
    Cloth4DGeometricConditioning,
)
from .cortexode_intrinsic_shell import (
    CortexODEIntrinsicShellConfig,
    reconstruct_intrinsic_shell_cortexode,
)
from .cortexode_volume_pipeline import (
    CortexODEVolumeConfig,
    build_cortexode_volume_evidence,
    semantic_topology_from_panel_analysis,
    shell_initialization_from_volume,
)
from .data import AvatarSpec, GarmentMesh, MaterialSpec
from .dynamic_events import (
    ActiveSide,
    EventSurface,
    EventSurfaceKind,
    ZeroCrossingEventDetector,
)
from .e3_anchor_field import E3AnchorNeuralClothField
from .genesis_tactile_adapter import adapt_genesis_taxel_field
from .geometric_phase_graph import IntrinsicClothPhaseState
from .learned_panel_mesh import garment_mesh_from_sewformer_spec
from .metric_projection import material_strain_limits
from .neuralclothsim_dynamics import (
    BodyContactSample,
    NeuralCloth4DDynamics,
    NeuralCloth4DDynamicsConfig,
)
from .implicit_shell import ImplicitShellBuildReport, build_body_sdf_tshirt_shell
from .image_pattern_provider import probe_design2garmentcode
from .hypernetwork_kirchhoff_love import (
    HyperNetworkKirchhoffLoveShell,
    KirchhoffLoveShellGeometry,
    build_kirchhoff_love_shell_geometry,
    kirchhoff_love_dynamics_loss,
)
from .patterns import export_obj
from .physics import (
    SimulationConfig,
    _project_avatar_constraints,
    project_distance_constraints,
    simulate_intrinsic_cloth,
)
from .physics_conditioned_flow import SceneMaterialLatent
from .pleated_skirt import (
    PleatedSkirtBuildReport,
    build_body_conditioned_pleated_skirt_shell,
    build_pleated_skirt_construction,
    export_pleated_skirt_construction,
    infer_pleated_skirt_spec,
)
from .reference_map_bundle import ReferenceMapFiberBundle
from .render import render_frame, render_rollout
from .variational_phase_pinn import (
    VariationalPhasePINNConfig,
    assemble_cloth_hamiltonian,
    compile_cloth_hamiltonian_contract,
    variational_phase_pinn_residual,
)


@dataclass(frozen=True)
class AnnyCloth4DConfig:
    resolution: int = 23
    dt: float = 0.006
    teacher_steps: int = 12
    teacher_edge_iterations: int = 48
    teacher_substeps_per_frame: int = 1
    training_steps: int = 180
    minimum_training_steps: int = 80
    query_batch_size: int = 80
    sampled_edges: int = 28
    sampled_seams: int = 18
    learning_rate: float = 2.5e-3
    target_rmse_m: float = 0.0018
    cloth_anchors_per_panel: int = 12
    body_anchor_count: int = 48
    body_neighbors_per_query: int = 14
    spectral_basis_size: int = 32
    intrinsic_channels: int = 6
    scene_feature_dim: int = 16
    behavior_profile: BehavioralPhaseTrajectory = field(
        default_factory=BehavioralPhaseTrajectory
    )
    material_latent_dim: int = 12
    hidden_dim: int = 24
    inference_chunk_size: int = 192
    shell_physics_interval: int = 6
    shell_physics_warmup_steps: int = 0
    shell_energy_weight: float = 0.02
    shell_equilibrium_weight: float = 1.0e-4
    shell_traction_weight: float = 5.0e-4
    variational_pinn_weight: float = 2.5e-5
    bundle_measure_weight: float = 2.5e-5
    implicit_shell_target_faces: int = 1400
    cortexode_volume_resolution: int = 32
    cortexode_unet_steps: int = 20
    cortexode_optimization_steps: int = 80
    cortexode_rk4_steps: int = 4
    dynamic_substeps_per_frame: int = 3
    dynamic_refit_steps: int = 60
    dynamic_refit_minimum_steps: int = 30
    dynamic_maximum_acceleration_m_s2: float = 100.0
    dynamic_initial_shell_acceleration_target_m_s2: float = 20.0
    dynamic_residual_force_scale: float = 0.01
    dynamic_residual_acceleration_limit_m_s2: float = 5.0
    dynamic_metric_projection_adaptive_iterations: int = 256
    seed: int = 73

    @property
    def frame_dt(self) -> float:
        return self.dt * self.teacher_substeps_per_frame

    def __post_init__(self) -> None:
        if isinstance(self.behavior_profile, dict):
            object.__setattr__(
                self,
                "behavior_profile",
                BehavioralPhaseTrajectory(**self.behavior_profile),
            )
        if not isinstance(self.behavior_profile, BehavioralPhaseTrajectory):
            raise ValueError("behavior_profile must be a BehavioralPhaseTrajectory")
        integer_values = (
            self.resolution,
            self.teacher_steps,
            self.teacher_edge_iterations,
            self.teacher_substeps_per_frame,
            self.training_steps,
            self.minimum_training_steps,
            self.query_batch_size,
            self.sampled_edges,
            self.sampled_seams,
            self.cloth_anchors_per_panel,
            self.body_anchor_count,
            self.body_neighbors_per_query,
            self.spectral_basis_size,
            self.intrinsic_channels,
            self.scene_feature_dim,
            self.material_latent_dim,
            self.hidden_dim,
            self.inference_chunk_size,
            self.shell_physics_interval,
            self.implicit_shell_target_faces,
            self.cortexode_volume_resolution,
            self.cortexode_unet_steps,
            self.cortexode_optimization_steps,
            self.cortexode_rk4_steps,
            self.dynamic_substeps_per_frame,
            self.dynamic_refit_steps,
            self.dynamic_refit_minimum_steps,
            self.dynamic_metric_projection_adaptive_iterations,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("simulation dimensions and iteration counts must be positive")
        if self.dt <= 0.0 or self.learning_rate <= 0.0 or self.target_rmse_m <= 0.0:
            raise ValueError("dt, learning_rate, and target_rmse_m must be positive")
        if self.shell_physics_warmup_steps < 0:
            raise ValueError("shell_physics_warmup_steps must be non-negative")
        if min(
            self.shell_energy_weight,
            self.shell_equilibrium_weight,
            self.shell_traction_weight,
            self.variational_pinn_weight,
            self.bundle_measure_weight,
        ) < 0.0:
            raise ValueError("shell physics loss weights must be non-negative")
        if self.minimum_training_steps > self.training_steps:
            raise ValueError("minimum_training_steps cannot exceed training_steps")
        if self.dynamic_refit_minimum_steps > self.dynamic_refit_steps:
            raise ValueError("dynamic_refit_minimum_steps cannot exceed dynamic_refit_steps")
        if min(
            self.dynamic_maximum_acceleration_m_s2,
            self.dynamic_initial_shell_acceleration_target_m_s2,
            self.dynamic_residual_acceleration_limit_m_s2,
        ) <= 0.0 or self.dynamic_residual_force_scale < 0.0:
            raise ValueError(
                "dynamic acceleration settings must be positive and residual scale non-negative"
            )
        if self.intrinsic_channels < 6 or self.scene_feature_dim < 8:
            raise ValueError("the Anny runtime requires at least 6 intrinsic and 8 scene channels")


@dataclass(frozen=True)
class GenesisIPCBackendStatus:
    requested_backend: str
    repository: str
    repository_commit: str | None
    genesis_source_available: bool
    pyuipc_available: bool
    platform: str
    executable_here: bool
    reason: str
    fallback: str
    stock_pbd_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _PreparedGeometry:
    material_coordinates: Tensor
    panel_ids: Tensor
    panel_names: list[str]
    surface_features: Tensor
    laplace_eigenvalues: Tensor
    laplace_eigenvectors: Tensor
    vertex_mass: Tensor
    edge_index: Tensor
    edge_pairs: Tensor
    rest_lengths: Tensor
    cotangent_weights: Tensor
    seam_pairs: Tensor
    boundary_weight: Tensor
    cloth_anchor_ids: Tensor
    body_anchor_positions: Tensor
    body_anchor_normals: Tensor
    body_anchor_mask: Tensor
    local_query_vectors: Tensor
    tactile_edge_index: Tensor
    contact_closest: Tensor
    contact_normals: Tensor
    panel_adjacency: tuple[tuple[int, int], ...]
    shell: KirchhoffLoveShellGeometry


@dataclass(frozen=True)
class _FieldBundle:
    field: E3AnchorNeuralClothField
    conditioner: Cloth4DGeometricConditioner
    material_latent: SceneMaterialLatent
    shell_model: HyperNetworkKirchhoffLoveShell


def probe_genesis_ipc_backend(genesis_repo: str | Path) -> GenesisIPCBackendStatus:
    repo = Path(genesis_repo).expanduser().resolve()
    source_available = (repo / "genesis" / "engine" / "materials" / "FEM" / "cloth.py").exists()
    commit: str | None = None
    if (repo / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            commit = result.stdout.strip() or None
    pyuipc_available = importlib.util.find_spec("uipc") is not None
    executable = source_available and pyuipc_available
    if not source_available:
        reason = "Genesis FEM.Cloth source is not present"
    elif not pyuipc_available:
        reason = (
            "Genesis FEM.Cloth requires pyuipc; no compatible module is installed "
            f"for {platform.system()} {platform.machine()}"
        )
    else:
        reason = "Genesis FEM.Cloth plus IPC is importable"
    return GenesisIPCBackendStatus(
        requested_backend="Genesis FEM.Cloth + IPC",
        repository=str(repo),
        repository_commit=commit,
        genesis_source_available=source_available,
        pyuipc_available=pyuipc_available,
        platform=f"{platform.system()} {platform.machine()}",
        executable_here=executable,
        reason=reason,
        fallback="canonical-metric intrinsic shell trajectory distillation",
    )


def run_anny_cloth4d_simulation(
    *,
    out_dir: str | Path,
    body_mesh: str | Path,
    pattern_spec: str | Path,
    texture_image: str | Path,
    garment_image: str | Path | None = None,
    warm_start_obj: str | Path | None,
    genesis_repo: str | Path,
    design2garmentcode_repo: str | Path | None = None,
    simulation_topology: str = "implicit_shell",
    shell_reconstruction_mode: str = "direct",
    placement_mode: str = "arrangement_points",
    body_source_up_axis: str = "z",
    body_source_front_axis: str = "-y",
    avatar_spec: AvatarSpec | None = None,
    teacher_rollout: str | Path | None = None,
    warm_start_checkpoint: str | Path | None = None,
    garment_kind: str = "tshirt",
    material_spec: MaterialSpec | None = None,
    config: AnnyCloth4DConfig = AnnyCloth4DConfig(),
) -> dict[str, Any]:
    """Optimize one E(3)-equivariant 4D T-shirt field on an Anny scene.

    The stock Genesis PBD cloth path is deliberately never invoked. When the
    official Genesis FEM.Cloth/IPC dependency is unavailable, the run records
    that boundary and uses the existing intrinsic shell as a detached trajectory
    teacher. The optimized field and every geometric conditioning branch remain
    the same, so a Linux IPC trajectory can replace the teacher array directly.
    """

    started = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame_dir = out / "frames"
    frame_dir.mkdir(exist_ok=True)
    texture_path = Path(texture_image)
    if not texture_path.exists():
        raise FileNotFoundError(f"texture image not found: {texture_path}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    avatar_spec = avatar_spec or AvatarSpec(
        chest_cm=64.0,
        waist_cm=56.0,
        shoulder_cm=30.0,
        height_cm=128.0,
        pose="a-pose",
    )
    avatar = BodyMeshAvatar.from_obj(
        body_mesh,
        avatar_spec,
        source_x_axis="x",
        source_up_axis=body_source_up_axis,
        source_front_axis=body_source_front_axis,
    )
    if garment_kind not in {"tshirt", "pleated_skirt"}:
        raise ValueError("garment_kind must be tshirt or pleated_skirt")
    if simulation_topology not in {"implicit_shell", "seam_atlas", "pleated_skirt_shell"}:
        raise ValueError(
            "simulation_topology must be implicit_shell, seam_atlas, or pleated_skirt_shell"
        )
    if garment_kind == "pleated_skirt" and simulation_topology != "pleated_skirt_shell":
        raise ValueError("pleated_skirt requires pleated_skirt_shell topology")
    if shell_reconstruction_mode not in {"direct", "cortexode"}:
        raise ValueError("shell_reconstruction_mode must be direct or cortexode")

    material_spec = material_spec or MaterialSpec()
    implicit_report: ImplicitShellBuildReport | None = None
    skirt_report: PleatedSkirtBuildReport | None = None
    manufacturing_construction: dict[str, Any] | None = None
    manufacturing_artifacts: dict[str, str] = {}
    cortexode_volume_report: dict[str, Any] | None = None
    cortexode_shell_report: dict[str, Any] | None = None
    cortexode_artifacts: dict[str, str] = {}
    semantic_sewing_topology: dict[str, Any] | None = None
    if simulation_topology == "pleated_skirt_shell":
        skirt_spec = infer_pleated_skirt_spec(garment_image or texture_path)
        dressed_mesh, skirt_report = build_body_conditioned_pleated_skirt_shell(
            avatar,
            skirt_spec,
        )
        construction = build_pleated_skirt_construction(dressed_mesh, skirt_spec)
        manufacturing_construction = construction.to_dict()
        manufacturing_artifacts = export_pleated_skirt_construction(
            out / "manufacturing",
            construction,
        )
        flat_mesh = dressed_mesh
        effective_placement_mode = "waist_hip_thigh_arrangement_points"
        avatar._last_placement_mode = effective_placement_mode
        avatar._last_arrangement_assignments = {
            "continuous_pleated_skirt_shell": (
                "waist_front",
                "waist_back",
                "waist_side_L",
                "waist_side_R",
                "hip_front",
                "hip_back",
                "hip_side_L",
                "hip_side_R",
                "thigh_L",
                "thigh_R",
            )
        }
        avatar._last_arrangement_diagnostics = dict(skirt_report.collision)
    else:
        flat_mesh = garment_mesh_from_sewformer_spec(pattern_spec, resolution=config.resolution)
    if simulation_topology == "implicit_shell":
        dressed_mesh, implicit_report = build_body_sdf_tshirt_shell(
            avatar,
            flat_mesh,
            target_faces=config.implicit_shell_target_faces,
        )
        effective_placement_mode = "implicit_body_sdf_shell"
        avatar._last_placement_mode = effective_placement_mode
        avatar._last_arrangement_assignments = {
            "continuous_tshirt_shell": tuple(
                point.name for point in avatar.arrangement.points
            )
        }
        avatar._last_arrangement_diagnostics = dict(implicit_report.collision)
    elif simulation_topology == "seam_atlas":
        dressed_mesh = avatar.dress_mesh(flat_mesh, placement_mode=placement_mode)
        effective_placement_mode = placement_mode
    if shell_reconstruction_mode == "cortexode":
        cortexode_checkout = Path(genesis_repo).resolve().parent / "cortexode"
        evidence, volume_report = build_cortexode_volume_evidence(
            dressed_mesh,
            cortexode_checkout=cortexode_checkout,
            config=CortexODEVolumeConfig(
                resolution=config.cortexode_volume_resolution,
                unet_steps=config.cortexode_unet_steps,
                seed=config.seed + 40,
            ),
        )
        cortexode_initial = shell_initialization_from_volume(
            dressed_mesh,
            evidence,
            smoothing_iterations=4,
        )
        target_shell = dressed_mesh
        dressed_mesh, shell_report, cortexode_trajectory = reconstruct_intrinsic_shell_cortexode(
            target_shell,
            avatar=avatar,
            initial=cortexode_initial,
            volume_evidence=evidence,
            material=material_spec,
            config=CortexODEIntrinsicShellConfig(
                optimization_steps=config.cortexode_optimization_steps,
                rk4_steps=config.cortexode_rk4_steps,
                seed=config.seed + 41,
            ),
        )
        if not shell_report.topology_preserved or shell_report.minimum_orientation_jacobian <= 0.0:
            raise RuntimeError("CortexODE shell failed the diffeomorphic topology gate")
        volume_path = out / "cortexode_volume_evidence.npz"
        np.savez_compressed(
            volume_path,
            sdf=evidence.sdf,
            probability=evidence.probability,
            world_min=evidence.world_min,
            world_max=evidence.world_max,
        )
        trajectory_path = out / "cortexode_shell_trajectory.npy"
        np.save(trajectory_path, cortexode_trajectory)
        initial_path = out / "cortexode_initial_shell.obj"
        reconstructed_path = out / "cortexode_reconstructed_shell.obj"
        closed_proxy_path = out / "cortexode_closed_proxy.obj"
        export_obj(initial_path, cortexode_initial)
        export_obj(reconstructed_path, dressed_mesh)
        export_obj(
            closed_proxy_path,
            GarmentMesh(
                vertices=evidence.closed_vertices,
                faces=evidence.closed_faces,
                pinned_vertices=np.empty(0, dtype=np.int32),
                panel_labels=["closed_proxy"] * len(evidence.closed_vertices),
                seam_edges=[],
            ),
        )
        cortexode_volume_report = volume_report.to_dict()
        cortexode_shell_report = shell_report.to_dict()
        semantic_sewing_topology = semantic_topology_from_panel_analysis(target_shell)
        semantic_topology_path = out / "semantic_sewing_topology.json"
        semantic_topology_path.write_text(
            json.dumps(semantic_sewing_topology, indent=2), encoding="utf-8"
        )
        cortexode_artifacts = {
            "volume_evidence": str(volume_path),
            "closed_proxy": str(closed_proxy_path),
            "semantic_open_initial_shell": str(initial_path),
            "reconstructed_intrinsic_shell": str(reconstructed_path),
            "ode_trajectory": str(trajectory_path),
            "semantic_sewing_topology": str(semantic_topology_path),
        }
        effective_placement_mode = f"{effective_placement_mode}+cortexode_volume_flow"
    arrangement_path = out / "avatar_arrangement.json"
    arrangement_payload = avatar.arrangement.to_dict()
    arrangement_payload.update(
        {
            "simulation_topology": simulation_topology,
            "shell_reconstruction_mode": shell_reconstruction_mode,
            "shell_reconstruction_mode": shell_reconstruction_mode,
            "placement_mode": effective_placement_mode,
            "implicit_shell": (
                implicit_report.to_dict() if implicit_report is not None else None
            ),
            "pleated_skirt_shell": (
                skirt_report.to_dict() if skirt_report is not None else None
            ),
            "cortexode_volume": cortexode_volume_report,
            "cortexode_intrinsic_shell": cortexode_shell_report,
            "semantic_sewing_topology": semantic_sewing_topology,
            "cortexode_volume": cortexode_volume_report,
            "cortexode_intrinsic_shell": cortexode_shell_report,
            "runtime": avatar.summary()["arrangement"],
        }
    )
    arrangement_path.write_text(
        json.dumps(arrangement_payload, indent=2),
        encoding="utf-8",
    )
    if warm_start_obj is not None:
        warm_vertices, warm_faces = _load_complete_obj(warm_start_obj)
        if warm_vertices.shape != dressed_mesh.vertices.shape or not np.array_equal(
            warm_faces,
            dressed_mesh.faces,
        ):
            raise ValueError("warm-start OBJ does not match the selected garment topology")
        dressed_mesh = _mesh_with_vertices(dressed_mesh, warm_vertices)

    backend = probe_genesis_ipc_backend(genesis_repo)
    d2g_repo = (
        Path(design2garmentcode_repo)
        if design2garmentcode_repo is not None
        else Path(genesis_repo).parent / "design2garmentcode"
    )
    pattern_provider = probe_design2garmentcode(d2g_repo)
    bootstrap_frames, teacher_metrics = _load_or_generate_teacher(
        teacher_rollout,
        dressed_mesh,
        avatar,
        config,
        material_spec,
    )
    teacher_path = out / "physics_teacher_rollout.npy"
    np.save(teacher_path, bootstrap_frames)
    canonical = bootstrap_frames[0].copy()
    field_mesh = _mesh_with_vertices(dressed_mesh, canonical)
    geometry = _prepare_geometry(
        field_mesh,
        avatar,
        canonical,
        config,
        reference_surface_metric=simulation_topology in {"implicit_shell", "pleated_skirt_shell"},
    )
    intrinsic_rest_obj = out / "intrinsic_rest_shell.obj"
    export_obj(intrinsic_rest_obj, field_mesh)
    intrinsic_operators_path = out / "intrinsic_operators.npz"
    np.savez_compressed(
        intrinsic_operators_path,
        faces=field_mesh.faces,
        edge_pairs=geometry.edge_pairs.detach().cpu().numpy(),
        rest_lengths=geometry.rest_lengths.detach().cpu().numpy(),
        cotangent_weights=geometry.cotangent_weights.detach().cpu().numpy(),
        vertex_mass=geometry.vertex_mass.detach().cpu().numpy(),
        laplace_eigenvalues=geometry.laplace_eigenvalues.detach().cpu().numpy(),
        laplace_eigenvectors=geometry.laplace_eigenvectors.detach().cpu().numpy(),
        rest_hinge_angles=geometry.shell.rest_hinge_angles.detach().cpu().numpy(),
        hinges=geometry.shell.hinges.detach().cpu().numpy(),
    )
    hamiltonian_contract_path = out / "phase_hamiltonian_contract.json"
    hamiltonian_contract_path.write_text(
        json.dumps(
            compile_cloth_hamiltonian_contract(
                has_contact=True,
                has_seams=bool(len(geometry.seam_pairs)),
            ).to_dict(),
            indent=2,
        ),
        encoding="utf-8",
    )

    bundle = _build_field_bundle(geometry, config)
    warm_start_report = _load_pretrained_field_bundle(
        bundle,
        geometry,
        warm_start_checkpoint,
        config,
    )
    bootstrap_history = _fit_scene_field(
        bundle, geometry, bootstrap_frames, material_spec, config
    )
    dynamic_rollout = _simulate_neuralcloth4d_dynamics(
        bundle,
        geometry,
        field_mesh,
        avatar,
        material_spec,
        bootstrap_frames,
        config,
    )
    teacher_frames = dynamic_rollout.positions.detach().cpu().numpy().astype(np.float32)
    dynamic_velocities = dynamic_rollout.velocities.detach().cpu().numpy().astype(np.float32)
    dynamic_path = out / "neuralclothsim4d_physical_rollout.npy"
    dynamic_velocity_path = out / "neuralclothsim4d_velocities.npy"
    dynamic_report_path = out / "neuralclothsim4d_dynamics.json"
    np.save(dynamic_path, teacher_frames)
    np.save(dynamic_velocity_path, dynamic_velocities)
    dynamic_report_path.write_text(
        json.dumps(dynamic_rollout.report.to_dict(), indent=2),
        encoding="utf-8",
    )
    refit_config = replace(
        config,
        training_steps=config.dynamic_refit_steps,
        minimum_training_steps=config.dynamic_refit_minimum_steps,
        shell_physics_warmup_steps=min(
            config.shell_physics_warmup_steps,
            max(config.dynamic_refit_minimum_steps // 2, 0),
        ),
    )
    dynamic_history = _fit_scene_field(
        bundle, geometry, teacher_frames, material_spec, refit_config
    )
    history = [
        {**record, "phase": "bootstrap_constitutive_calibration"}
        for record in bootstrap_history
    ] + [
        {**record, "phase": "physical_time_4d_refit"}
        for record in dynamic_history
    ]
    raw_frames = _infer_rollout(bundle, geometry, teacher_frames, config)
    constrained_frames = _apply_hard_constraints(
        raw_frames,
        teacher_frames,
        field_mesh,
        avatar,
        geometry,
    )

    raw_path = out / "e3_anchor_raw_rollout.npy"
    constrained_path = out / "e3_anchor_constrained_rollout.npy"
    np.save(raw_path, raw_frames)
    np.save(constrained_path, constrained_frames)
    texture = Image.open(texture_path).convert("RGB")
    gif_path = out / "e3_anchor_4d_rollout.gif"
    final_path = out / "e3_anchor_4d_final.png"
    teacher_gif_path = out / "physics_teacher_rollout.gif"
    teacher_final_path = out / "physics_teacher_final.png"
    comparison_gif = out / "teacher_vs_e3_anchor_4d.gif"
    comparison_final = out / "teacher_vs_e3_anchor_4d_final.png"
    closed_sleeve_visibility = any(
        "sleeve" in label for label in field_mesh.panel_labels
    )
    if simulation_topology == "pleated_skirt_shell":
        caption = "Anny child / image-conditioned box-pleated skirt / Hyper-NODE Kirchhoff-Love 4D field"
    elif simulation_topology == "implicit_shell":
        caption = "Anny child / continuous implicit T-shirt shell / Hyper-NODE Kirchhoff-Love 4D field"
    else:
        caption = "Anny child / learned T-shirt panels / Hyper-NODE Kirchhoff-Love 4D field"
    render_rollout(
        teacher_gif_path,
        [frame for frame in teacher_frames],
        field_mesh,
        caption,
        avatar=avatar,
        texture_image=texture,
        duration_ms=110,
        method_label="NeuralClothSim physical-time rollout",
        semantic_sleeve_occlusion=closed_sleeve_visibility,
    )
    render_frame(
        teacher_frames[-1],
        field_mesh.faces,
        field_mesh.pinned_vertices,
        len(teacher_frames) - 1,
        caption,
        avatar=avatar,
        texture_image=texture,
        mesh=field_mesh,
        method_label="NeuralClothSim physical-time rollout",
        semantic_sleeve_occlusion=closed_sleeve_visibility,
    ).save(teacher_final_path)
    render_rollout(
        gif_path,
        [frame for frame in constrained_frames],
        field_mesh,
        caption,
        avatar=avatar,
        texture_image=texture,
        duration_ms=110,
        method_label="E3 + Hyper-NODE KL shell",
        semantic_sleeve_occlusion=closed_sleeve_visibility,
    )
    render_frame(
        constrained_frames[-1],
        field_mesh.faces,
        field_mesh.pinned_vertices,
        len(constrained_frames) - 1,
        caption,
        avatar=avatar,
        texture_image=texture,
        mesh=field_mesh,
        method_label="E3 + Hyper-NODE KL shell",
        semantic_sleeve_occlusion=closed_sleeve_visibility,
    ).save(final_path)
    _render_comparison(
        teacher_frames,
        constrained_frames,
        field_mesh,
        avatar,
        texture,
        comparison_gif,
        comparison_final,
        caption,
    )

    for frame_index, vertices in enumerate(constrained_frames):
        export_obj(frame_dir / f"frame_{frame_index:03d}.obj", _mesh_with_vertices(field_mesh, vertices))
    final_obj = out / "e3_anchor_4d_final.obj"
    export_obj(final_obj, _mesh_with_vertices(field_mesh, constrained_frames[-1]))

    checkpoint = out / "e3_anchor_4d_checkpoint.pt"
    torch.save(
        {
            "field": bundle.field.state_dict(),
            "conditioner": bundle.conditioner.state_dict(),
            "material_latent": bundle.material_latent.state_dict(),
            "hypernetwork_kirchhoff_love": bundle.shell_model.state_dict(),
            "config": asdict(config),
            "panel_names": geometry.panel_names,
            "cloth_anchor_ids": geometry.cloth_anchor_ids,
            "body_anchor_positions": geometry.body_anchor_positions,
            "body_anchor_normals": geometry.body_anchor_normals,
        },
        checkpoint,
    )

    genesis_artifacts = _export_genesis_ipc_inputs(
        out / "genesis_ipc_input",
        field_mesh,
        canonical,
        avatar,
        backend,
        config,
    )
    metrics = _rollout_metrics(
        raw_frames,
        constrained_frames,
        teacher_frames,
        field_mesh,
        avatar,
        geometry,
        bundle,
        config,
    )
    events = _detect_body_sdf_events(constrained_frames, field_mesh, avatar, config.frame_dt)
    elapsed = time.perf_counter() - started
    summary = {
        "status": "completed",
        "real_world_units": "meters",
        "avatar": avatar_spec.__dict__,
        "avatar_collision_proxy": avatar.summary(),
        "garment": {
            **field_mesh.summary(),
            "pattern_spec": (
                str(Path(pattern_spec)) if garment_kind == "tshirt" else None
            ),
            "source_image": (
                str(Path(garment_image)) if garment_image is not None else str(texture_path)
            ),
            "warm_start_obj": str(Path(warm_start_obj)) if warm_start_obj is not None else None,
            "simulation_topology": simulation_topology,
            "placement_mode": effective_placement_mode,
            "front_back_shells": True,
            "learned_panel_topology": garment_kind == "tshirt",
            "pattern_topology_source": (
                "learned sewing-pattern specification"
                if garment_kind == "tshirt"
                else "deterministic garment-family prior; external image model not executed"
            ),
            "manufacturing_panel_atlas": flat_mesh.summary(),
            "manufacturing_construction": manufacturing_construction,
            "image_pattern_provider": pattern_provider.to_dict(),
            "implicit_shell": (
                implicit_report.to_dict() if implicit_report is not None else None
            ),
            "pleated_skirt_shell": (
                skirt_report.to_dict() if skirt_report is not None else None
            ),
            "image_inferred_garment_kind": garment_kind,
            "material": asdict(material_spec),
        },
        "geometric_4d_field": {
            "representation": "cloth/body-anchor E(3)-equivariant vector field",
            "per_scene_optimization": True,
            "exact_query_reference": True,
            "cloth_anchors": int(len(geometry.cloth_anchor_ids)),
            "body_anchors": int(len(geometry.body_anchor_positions)),
            "panels": geometry.panel_names,
            "particle_prior": "E(3) 1x1o relative-vector message passing",
            "surface_prior": "DiffusionNet-style Laplace-Beltrami spectral diffusion",
            "scene_reasoning": "hierarchy-aware spatiotemporal equivariant attention",
            "behavior_hierarchy": {
                "path": "person/root -> body -> garment -> panel/contact",
                "profile": asdict(config.behavior_profile),
                "role": "external behavioral phase context, never a cloth material parameter",
            },
            "tactile_prior": "SO(2) gauge transport with parallel-transport connection",
            "lipschitz_path": "SpectralLinear on every learned affine path",
        },
        "physics_prior": {
            "genesis": backend.to_dict(),
            "teacher_used_this_run": (
                "Genesis FEM.Cloth + IPC bootstrap followed by NeuralClothSim physical-time rollout"
                if backend.executable_here and teacher_rollout is not None
                else "intrinsic bootstrap followed by NeuralClothSim physical-time rollout"
            ),
            "teacher_metrics": teacher_metrics,
            "dynamic_rollout": dynamic_rollout.report.to_dict(),
            "stock_genesis_pbd_used": False,
            "hard_constraints": (
                "body SDF, sampled triangle contact, immutable continuous-shell rest metric; "
                "no runtime seam constraints"
                if simulation_topology in {"implicit_shell", "pleated_skirt_shell"}
                else "body SDF, sampled triangle contact, staged seam closure, and immutable "
                "2D sewing-pattern metric"
            ),
            "semantic_link_compilation": (
                "panel-derived stitched links are welded structural edges and cross-link "
                "Kirchhoff-Love hinges; semantic openings remain free-traction boundaries"
                if semantic_sewing_topology is not None
                else None
            ),
            "constitutive_model": (
                "material-point Hyper-network NODE adapted to Kirchhoff-Love "
                "membrane and discrete rest-curvature bending"
            ),
            "inverse_physics_losses": (
                "dynamic nodal equilibrium, free-edge Piola traction, body contact"
            ),
            "variational_phase_pinn": (
                "discrete Hamilton-d'Alembert stationarity, work-energy balance, "
                "SDF contact KKT, and friction/damping dissipation"
            ),
            "bundle_measure_port_hamiltonian_training": {
                "state": "graded mu_vertex + mu_face + mu_hinge empirical measure",
                "continuity": (
                    "mass/material-label conservation, position transport, and "
                    "canonical momentum balance"
                ),
                "energy": (
                    "full KL measure-Hamiltonian rate equals supplied constraint/control "
                    "power plus non-positive dissipative power"
                ),
                "contact": (
                    "smooth contact is a constraint port during training; localized "
                    "runtime events remain momentum-fibre push-forwards"
                ),
                "weight": config.bundle_measure_weight,
            },
            "complex_scene_hamiltonian": (
                "mass-scaled local Kahler cloth phase, real H(z,z_bar), and "
                "hierarchy-aware node/relation/constraint energy ledger"
            ),
            "hamiltonian_contract": str(hamiltonian_contract_path),
            "physical_time_authority": (
                "Hyper-NODE Kirchhoff-Love internal force + gravity/damping/contact/seams; "
                "bounded E(3) residual; metric projection after every substep"
            ),
            "material_law_coordinates": (
                "learned atlas coordinates for material conditioning plus per-face 3D "
                "reference-surface tangent charts"
                if simulation_topology in {"implicit_shell", "pleated_skirt_shell"}
                else "2D learned sewing-pattern charts"
            ),
            "rejected_degenerate_material_faces": geometry.shell.rejected_face_count,
        },
        "dynamic_topology_events": {
            "detector": (
                "physical substep CCD root localization/restart plus explicit "
                "body-SDF zero-crossing diagnostics"
            ),
            "physical_rollout_contact_events": list(
                dynamic_rollout.report.contact_events
            ),
            "events": events,
        },
        "training": {
            "warm_start": warm_start_report,
            "requested_steps": config.training_steps,
            "completed_steps": len(history),
            "final": history[-1] if history else {},
            "physics_updates": sum(
                int(record.get("dynamic_equilibrium", 0.0) > 0.0) for record in history
            ),
            "last_physics": next(
                (
                    record
                    for record in reversed(history)
                    if record.get("dynamic_equilibrium", 0.0) > 0.0
                ),
                {},
            ),
            "history": history,
        },
        "metrics": metrics,
        "runtime_seconds": elapsed,
        "artifacts": {
            "gif": str(gif_path),
            "final_png": str(final_path),
            "physics_teacher_gif": str(teacher_gif_path),
            "physics_teacher_final_png": str(teacher_final_path),
            "final_obj": str(final_obj),
            "frame_directory": str(frame_dir),
            "comparison_gif": str(comparison_gif),
            "comparison_final_png": str(comparison_final),
            "teacher_rollout": str(dynamic_path),
            "bootstrap_teacher_rollout": str(teacher_path),
            "neuralclothsim4d_physical_rollout": str(dynamic_path),
            "neuralclothsim4d_velocities": str(dynamic_velocity_path),
            "neuralclothsim4d_dynamics_report": str(dynamic_report_path),
            "raw_field_rollout": str(raw_path),
            "constrained_field_rollout": str(constrained_path),
            "checkpoint": str(checkpoint),
            "intrinsic_rest_shell": str(intrinsic_rest_obj),
            "intrinsic_operators": str(intrinsic_operators_path),
            "avatar_arrangement": str(arrangement_path),
            "manufacturing": manufacturing_artifacts,
            "cortexode": cortexode_artifacts,
            "genesis_ipc_input": genesis_artifacts,
        },
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_pretrained_field_bundle(
    bundle: _FieldBundle,
    geometry: _PreparedGeometry,
    checkpoint: str | Path | None,
    config: AnnyCloth4DConfig,
) -> dict[str, Any]:
    if checkpoint is None:
        return {"used": False, "reason": "random initialization"}
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"warm-start checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "field",
        "conditioner",
        "material_latent",
        "hypernetwork_kirchhoff_love",
        "config",
        "panel_names",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"warm-start checkpoint is missing keys: {missing}")
    source_config = payload["config"]
    compatibility_fields = (
        "intrinsic_channels",
        "scene_feature_dim",
        "material_latent_dim",
        "hidden_dim",
    )
    mismatches = {
        name: (source_config.get(name), getattr(config, name))
        for name in compatibility_fields
        if source_config.get(name) != getattr(config, name)
    }
    if mismatches:
        raise ValueError(f"warm-start checkpoint architecture mismatch: {mismatches}")
    if list(payload["panel_names"]) != geometry.panel_names:
        raise ValueError("warm-start checkpoint panel topology does not match this shell")
    bundle.field.load_state_dict(payload["field"], strict=True)
    conditioner_migration = _load_conditioner_checkpoint_state(
        bundle.conditioner, payload["conditioner"]
    )
    bundle.material_latent.load_state_dict(payload["material_latent"], strict=True)
    bundle.shell_model.load_state_dict(payload["hypernetwork_kirchhoff_love"], strict=True)
    return {
        "used": True,
        "checkpoint": str(path),
        "source_scene_feature_dim": int(source_config["scene_feature_dim"]),
        "mode": "pretrained trajectory field -> variational phase PINN fine-tune",
        "conditioner_state_migration": conditioner_migration,
    }


def _load_conditioner_checkpoint_state(
    conditioner: Cloth4DGeometricConditioner,
    checkpoint_state: dict[str, Tensor],
) -> dict[str, Any]:
    """Load both legacy one-hop and current hierarchy-aware scene attention."""

    state = dict(checkpoint_state)
    legacy_scene_keys = [
        key
        for key in state
        if key.startswith("scene.") and not key.startswith("scene.layers.")
    ]
    if not legacy_scene_keys:
        conditioner.load_state_dict(state, strict=True)
        return {"applied": False, "mode": "native hierarchy-aware checkpoint"}

    target = conditioner.state_dict()
    layer_ids = sorted(
        {
            int(key.split(".")[2])
            for key in target
            if key.startswith("scene.layers.")
        }
    )
    if not layer_ids:
        raise ValueError("legacy scene checkpoint cannot be loaded into this conditioner")
    for key in legacy_scene_keys:
        value = state.pop(key)
        suffix = key.removeprefix("scene.")
        for layer_id in layer_ids:
            destination = f"scene.layers.{layer_id}.{suffix}"
            if destination not in target or target[destination].shape != value.shape:
                raise ValueError(
                    f"legacy scene parameter is incompatible with {destination}: {key}"
                )
            state[destination] = value.clone()
    conditioner.load_state_dict(state, strict=True)
    return {
        "applied": True,
        "mode": "legacy one-hop attention copied into hierarchy-aware stack",
        "source_layers": 1,
        "target_layers": len(layer_ids),
        "migrated_parameters": len(legacy_scene_keys),
    }


def _load_or_generate_teacher(
    teacher_rollout: str | Path | None,
    mesh: GarmentMesh,
    avatar: BodyMeshAvatar,
    config: AnnyCloth4DConfig,
    material: MaterialSpec,
) -> tuple[np.ndarray, dict[str, float]]:
    if teacher_rollout is not None:
        frames = np.asarray(np.load(teacher_rollout), dtype=np.float32)
        if frames.ndim != 3 or frames.shape[1:] != mesh.vertices.shape:
            raise ValueError("teacher rollout must have shape (T, V, 3)")
        return frames, {"source": "provided trajectory", "frames": float(len(frames))}

    canonical_metric_mesh = GarmentMesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        pinned_vertices=mesh.pinned_vertices.copy(),
        panel_labels=list(mesh.panel_labels),
        seam_edges=list(mesh.seam_edges),
        rest_vertices=mesh.material_vertices.copy(),
        uvs=None if mesh.uvs is None else mesh.uvs.copy(),
        seam_interfaces=list(mesh.seam_interfaces),
    )
    total_steps = config.teacher_steps * config.teacher_substeps_per_frame
    is_lower_garment = any("pleated_skirt" in label for label in mesh.panel_labels)
    waist_y = float(
        mesh.vertices[mesh.pinned_vertices, 1].max()
        if len(mesh.pinned_vertices)
        else mesh.vertices[:, 1].max()
    )
    sim_config = SimulationConfig(
        steps=total_steps,
        dt=config.dt,
        frame_interval=config.teacher_substeps_per_frame,
        edge_projection_iterations=config.teacher_edge_iterations,
        stretch_tolerance=0.010 if is_lower_garment else 0.015,
        bend_smoothing=0.0 if is_lower_garment else 0.006,
        gravity_y=-9.81,
        max_velocity=0.3,
        attachment_mode="waist_halfspace" if is_lower_garment else "fixed",
        # Keep the waist attached until a measured waistband/hip retention and
        # friction gate is available. Releasing a merely image-inferred waist
        # lets the garment slide off and is less physical than an avatar-bound
        # wearing constraint.
        attachment_release_step=None,
        attachment_halfspace_y=(waist_y if is_lower_garment else None),
    )
    frames, _final, metrics = simulate_intrinsic_cloth(
        canonical_metric_mesh,
        material,
        sim_config,
        avatar=avatar,
    )
    return np.asarray(frames, dtype=np.float32), metrics


def _simulate_neuralcloth4d_dynamics(
    bundle: _FieldBundle,
    geometry: _PreparedGeometry,
    mesh: GarmentMesh,
    avatar: BodyMeshAvatar,
    material: MaterialSpec,
    bootstrap_frames: np.ndarray,
    config: AnnyCloth4DConfig,
):
    """Generate the authoritative physical-time trajectory for the 4D field."""

    areal_density = float(np.clip(0.06 * material.density, 0.08, 0.65))
    dynamics = NeuralCloth4DDynamics(
        bundle.shell_model,
        geometry.shell,
        rest_positions=torch.as_tensor(mesh.material_vertices, dtype=torch.float32),
        edge_index=geometry.edge_index,
        rest_lengths=geometry.rest_lengths,
        cotangent_weights=geometry.cotangent_weights,
        pinned_vertices=torch.as_tensor(mesh.pinned_vertices, dtype=torch.long),
        seam_pairs=geometry.seam_pairs,
        material=material,
        residual_model=bundle.conditioner.particle,
        config=NeuralCloth4DDynamicsConfig(
            dt=config.frame_dt / config.dynamic_substeps_per_frame,
            substeps_per_frame=config.dynamic_substeps_per_frame,
            areal_density_kg_m2=areal_density,
            maximum_acceleration_m_s2=config.dynamic_maximum_acceleration_m_s2,
            initial_shell_acceleration_target_m_s2=(
                config.dynamic_initial_shell_acceleration_target_m_s2
            ),
            residual_force_scale=config.dynamic_residual_force_scale,
            residual_acceleration_limit_m_s2=(
                config.dynamic_residual_acceleration_limit_m_s2
            ),
            metric_projection_iterations=max(64, config.teacher_edge_iterations),
            metric_projection_adaptive_iterations=(
                config.dynamic_metric_projection_adaptive_iterations
            ),
        ),
    )
    # The intrinsic/PBD pass is only a canonical dressing warm-up. Start the
    # authoritative physical rollout from its settled endpoint without
    # importing PBD velocity into the Hyper-NODE dynamics.
    initial = torch.as_tensor(bootstrap_frames[-1], dtype=torch.float32)
    initial_velocity = torch.zeros_like(initial)

    def contact_query(positions: Tensor, _physical_time: float) -> BodyContactSample:
        points = positions.detach().cpu().numpy()
        closest, _distance, triangle_ids = avatar._proximity.on_surface(points)
        normals = np.asarray(
            avatar._collision_mesh.face_normals[triangle_ids],
            dtype=np.float32,
        ).copy()
        direction = points - closest
        flipped = np.sum(direction * normals, axis=1) < 0.0
        normals[flipped] *= -1.0
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-8)
        return BodyContactSample(
            torch.as_tensor(closest, device=positions.device, dtype=positions.dtype),
            torch.as_tensor(normals, device=positions.device, dtype=positions.dtype),
            torch.zeros_like(positions),
        )

    return dynamics.rollout(
        initial,
        bundle.material_latent(),
        frame_count=len(bootstrap_frames),
        initial_velocities=initial_velocity,
        body_contact_query=contact_query,
        differentiable=False,
    )


def _load_complete_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    loaded = trimesh.load(Path(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return np.asarray(loaded.vertices, dtype=np.float32), np.asarray(loaded.faces, dtype=np.int32)


def _mesh_with_vertices(mesh: GarmentMesh, vertices: np.ndarray) -> GarmentMesh:
    return GarmentMesh(
        vertices=np.asarray(vertices, dtype=np.float32).copy(),
        faces=mesh.faces.copy(),
        pinned_vertices=mesh.pinned_vertices.copy(),
        panel_labels=list(mesh.panel_labels),
        seam_edges=list(mesh.seam_edges),
        rest_vertices=None if mesh.rest_vertices is None else mesh.rest_vertices.copy(),
        uvs=None if mesh.uvs is None else mesh.uvs.copy(),
        seam_interfaces=list(mesh.seam_interfaces),
    )


def _prepare_geometry(
    mesh: GarmentMesh,
    avatar: BodyMeshAvatar,
    canonical: np.ndarray,
    config: AnnyCloth4DConfig,
    *,
    reference_surface_metric: bool = False,
) -> _PreparedGeometry:
    material_chart_labels = (
        ["implicit_tshirt_shell"] * len(mesh.vertices)
        if reference_surface_metric
        else [_material_chart_label(label) for label in mesh.panel_labels]
    )
    panel_names = sorted(set(material_chart_labels))
    panel_lookup = {name: index for index, name in enumerate(panel_names)}
    panel_ids_np = np.asarray(
        [panel_lookup[label] for label in material_chart_labels],
        dtype=np.int64,
    )
    material = np.asarray(
        mesh.uvs
        if reference_surface_metric and mesh.uvs is not None
        else mesh.material_vertices[:, :2],
        dtype=np.float32,
    )
    eigenvalues, eigenvectors, mass, edge_pairs, cotangent = _cotangent_spectrum(
        mesh.material_vertices,
        mesh.faces,
        config.spectral_basis_size,
    )
    rest_lengths = np.linalg.norm(
        mesh.material_vertices[edge_pairs[:, 1]]
        - mesh.material_vertices[edge_pairs[:, 0]],
        axis=1,
    ).astype(np.float32)
    valid = rest_lengths > 1.0e-7
    edge_pairs = edge_pairs[valid]
    cotangent = cotangent[valid]
    rest_lengths = rest_lengths[valid]

    boundary = np.ones((len(canonical), 1), dtype=np.float32)
    boundary[mesh.pinned_vertices] = 0.0
    surface_features = _surface_scalar_features(
        material,
        panel_ids_np,
        panel_names,
        boundary[:, 0],
        avatar,
        canonical,
        config.intrinsic_channels,
        semantic_labels=mesh.panel_labels,
    )
    cloth_anchor_ids = _panel_farthest_samples(
        material,
        panel_ids_np,
        len(panel_names),
        config.cloth_anchors_per_panel,
    )
    body_positions, body_normals = _sample_body_anchors(
        avatar,
        canonical,
        config.body_anchor_count,
    )
    body_mask = _nearest_anchor_mask(
        canonical,
        body_positions,
        config.body_neighbors_per_query,
    )
    tactile_edges = _knn_edges(body_positions, neighbors=4)
    local_query_vectors = _panel_local_vector_basis(
        material,
        panel_ids_np,
        canonical,
    )

    closest, _distance, triangle_ids = avatar._proximity.on_surface(canonical)
    contact_normals = np.asarray(avatar._collision_mesh.face_normals[triangle_ids], dtype=np.float32)
    direction = canonical - closest
    flipped = np.sum(direction * contact_normals, axis=1) < 0.0
    contact_normals[flipped] *= -1.0
    contact_normals /= np.maximum(
        np.linalg.norm(contact_normals, axis=1, keepdims=True),
        1.0e-8,
    )
    panel_adjacency = _panel_adjacency(mesh.seam_edges, panel_ids_np)
    seams = np.asarray(mesh.seam_edges, dtype=np.int64).reshape(-1, 2)
    shell = build_kirchhoff_love_shell_geometry(
        torch.as_tensor(material),
        torch.as_tensor(mesh.faces, dtype=torch.long),
        torch.as_tensor(panel_ids_np, dtype=torch.long),
        torch.as_tensor(seams, dtype=torch.long),
        reference_positions=(
            torch.as_tensor(mesh.material_vertices, dtype=torch.float32)
            if reference_surface_metric
            else None
        ),
    )
    return _PreparedGeometry(
        material_coordinates=torch.as_tensor(material),
        panel_ids=torch.as_tensor(panel_ids_np, dtype=torch.long),
        panel_names=panel_names,
        surface_features=torch.as_tensor(surface_features),
        laplace_eigenvalues=torch.as_tensor(eigenvalues),
        laplace_eigenvectors=torch.as_tensor(eigenvectors),
        vertex_mass=torch.as_tensor(mass),
        edge_index=torch.as_tensor(edge_pairs.T, dtype=torch.long),
        edge_pairs=torch.as_tensor(edge_pairs, dtype=torch.long),
        rest_lengths=torch.as_tensor(rest_lengths),
        cotangent_weights=torch.as_tensor(cotangent),
        seam_pairs=torch.as_tensor(seams, dtype=torch.long),
        boundary_weight=torch.as_tensor(boundary),
        cloth_anchor_ids=torch.as_tensor(cloth_anchor_ids, dtype=torch.long),
        body_anchor_positions=torch.as_tensor(body_positions),
        body_anchor_normals=torch.as_tensor(body_normals),
        body_anchor_mask=torch.as_tensor(body_mask, dtype=torch.bool),
        local_query_vectors=torch.as_tensor(local_query_vectors),
        tactile_edge_index=torch.as_tensor(tactile_edges, dtype=torch.long),
        contact_closest=torch.as_tensor(np.asarray(closest, dtype=np.float32)),
        contact_normals=torch.as_tensor(contact_normals),
        panel_adjacency=panel_adjacency,
        shell=shell,
    )


def _cotangent_spectrum(
    vertices: np.ndarray,
    faces: np.ndarray,
    basis_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.linalg import eigh

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    vertex_count = len(points)
    laplacian = np.zeros((vertex_count, vertex_count), dtype=np.float64)
    mass = np.zeros(vertex_count, dtype=np.float64)
    edge_weight: dict[tuple[int, int], float] = {}
    for face in triangles:
        a, b, c = points[face]
        cross_norm = float(np.linalg.norm(np.cross(b - a, c - a)))
        if cross_norm <= 1.0e-12:
            continue
        area = 0.5 * cross_norm
        mass[face] += area / 3.0
        cotangents = (
            float(np.dot(b - a, c - a) / cross_norm),
            float(np.dot(a - b, c - b) / cross_norm),
            float(np.dot(a - c, b - c) / cross_norm),
        )
        opposite_edges = ((int(face[1]), int(face[2])), (int(face[0]), int(face[2])), (int(face[0]), int(face[1])))
        for edge, cotangent in zip(opposite_edges, cotangents):
            key = tuple(sorted(edge))
            edge_weight[key] = edge_weight.get(key, 0.0) + 0.5 * cotangent
    mass = np.maximum(mass, max(float(np.mean(mass[mass > 0.0])) * 1.0e-6, 1.0e-10))
    edges = np.asarray(sorted(edge_weight), dtype=np.int64)
    weights = np.asarray([edge_weight[tuple(edge)] for edge in edges], dtype=np.float64)
    for (a, b), weight in zip(edges, weights):
        laplacian[a, a] += weight
        laplacian[b, b] += weight
        laplacian[a, b] -= weight
        laplacian[b, a] -= weight
    inverse_sqrt_mass = 1.0 / np.sqrt(mass)
    normalized = inverse_sqrt_mass[:, None] * laplacian * inverse_sqrt_mass[None, :]
    normalized = 0.5 * (normalized + normalized.T)
    count = min(max(2, basis_size), max(2, vertex_count - 1))
    values, vectors = eigh(normalized, subset_by_index=(0, count - 1), check_finite=False)
    values = np.maximum(values, 0.0)
    mass_orthonormal = inverse_sqrt_mass[:, None] * vectors
    return (
        values.astype(np.float32),
        mass_orthonormal.astype(np.float32),
        mass.astype(np.float32),
        edges,
        weights.astype(np.float32),
    )


def _surface_scalar_features(
    material: np.ndarray,
    panel_ids: np.ndarray,
    panel_names: list[str],
    boundary: np.ndarray,
    avatar: BodyMeshAvatar,
    canonical: np.ndarray,
    channels: int,
    *,
    semantic_labels: list[str] | None = None,
) -> np.ndarray:
    centered = material - np.mean(material, axis=0, keepdims=True)
    scale = np.maximum(np.std(centered, axis=0, keepdims=True), 1.0e-5)
    normalized = centered / scale
    names = np.asarray(
        semantic_labels
        if semantic_labels is not None
        else [panel_names[index] for index in panel_ids]
    )
    front = np.asarray([name.startswith("front_") for name in names], dtype=np.float32)
    sleeve = np.asarray(["sleeve" in name for name in names], dtype=np.float32)
    signed = avatar._proximity.signed_distance(canonical)
    clearance = np.clip(-(signed + avatar.clearance) / 0.03, -1.0, 1.0).astype(np.float32)
    base = np.column_stack(
        (
            normalized[:, 0],
            normalized[:, 1],
            front,
            sleeve,
            boundary,
            clearance,
        )
    ).astype(np.float32)
    if channels <= base.shape[1]:
        return base[:, :channels]
    padding = np.zeros((len(base), channels - base.shape[1]), dtype=np.float32)
    return np.concatenate((base, padding), axis=1)


def _material_chart_label(label: str) -> str:
    """Keep render-side front/back sleeve labels on one physical chart."""

    if "left_sleeve" in label:
        return "left_sleeve"
    if "right_sleeve" in label:
        return "right_sleeve"
    return label


def _panel_farthest_samples(
    coordinates: np.ndarray,
    panel_ids: np.ndarray,
    panel_count: int,
    samples_per_panel: int,
) -> np.ndarray:
    selected: list[int] = []
    for panel in range(panel_count):
        ids = np.flatnonzero(panel_ids == panel)
        if ids.size == 0:
            raise ValueError(f"panel {panel} has no vertices")
        count = min(samples_per_panel, ids.size)
        local = coordinates[ids]
        span = np.maximum(np.ptp(local, axis=0), 1.0e-6)
        local = (local - np.min(local, axis=0)) / span
        local_selected = _farthest_point_indices(local, count)
        selected.extend(ids[local_selected].tolist())
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def _farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    if count >= len(points):
        return np.arange(len(points), dtype=np.int64)
    center = np.mean(points, axis=0)
    first = int(np.argmax(np.linalg.norm(points - center, axis=1)))
    selected = [first]
    minimum_distance = np.linalg.norm(points - points[first], axis=1)
    for _ in range(1, count):
        candidate = int(np.argmax(minimum_distance))
        selected.append(candidate)
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(points - points[candidate], axis=1),
        )
    return np.asarray(selected, dtype=np.int64)


def _sample_body_anchors(
    avatar: BodyMeshAvatar,
    garment_vertices: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    body = np.asarray(avatar.mesh_vertices, dtype=np.float32)
    normals = np.asarray(avatar._collision_mesh.vertex_normals, dtype=np.float32)
    lower = garment_vertices.min(axis=0) - np.asarray([0.08, 0.08, 0.10], dtype=np.float32)
    upper = garment_vertices.max(axis=0) + np.asarray([0.08, 0.08, 0.10], dtype=np.float32)
    mask = np.all((body >= lower) & (body <= upper), axis=1)
    candidate_ids = np.flatnonzero(mask)
    if candidate_ids.size < count:
        candidate_ids = np.arange(len(body), dtype=np.int64)
    candidates = body[candidate_ids]
    span = np.maximum(np.ptp(candidates, axis=0), 1.0e-6)
    sampled_local = _farthest_point_indices((candidates - candidates.min(axis=0)) / span, min(count, len(candidates)))
    sampled_ids = candidate_ids[sampled_local]
    sampled_normals = normals[sampled_ids]
    sampled_normals /= np.maximum(np.linalg.norm(sampled_normals, axis=1, keepdims=True), 1.0e-8)
    return body[sampled_ids], sampled_normals.astype(np.float32)


def _nearest_anchor_mask(points: np.ndarray, anchors: np.ndarray, count: int) -> np.ndarray:
    distances = np.linalg.norm(points[:, None, :] - anchors[None, :, :], axis=-1)
    count = min(count, len(anchors))
    nearest = np.argpartition(distances, count - 1, axis=1)[:, :count]
    mask = np.zeros(distances.shape, dtype=bool)
    mask[np.arange(len(points))[:, None], nearest] = True
    return mask


def _knn_edges(points: np.ndarray, neighbors: int) -> np.ndarray:
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    neighbors = min(neighbors, max(1, len(points) - 1))
    target = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
    source = np.repeat(np.arange(len(points)), neighbors)
    return np.stack((source, target.reshape(-1)), axis=0).astype(np.int64)


def _panel_adjacency(
    seam_edges: list[tuple[int, int]],
    panel_ids: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    adjacency: set[tuple[int, int]] = set()
    for a, b in seam_edges:
        panel_a = int(panel_ids[int(a)])
        panel_b = int(panel_ids[int(b)])
        if panel_a != panel_b:
            adjacency.add((panel_a, panel_b))
            adjacency.add((panel_b, panel_a))
    return tuple(sorted(adjacency))


def _panel_local_vector_basis(
    material: np.ndarray,
    panel_ids: np.ndarray,
    canonical: np.ndarray,
) -> np.ndarray:
    """Fit polar material tangents and add the scene gravity direction."""

    basis = np.zeros((len(material), 3, 3), dtype=np.float32)
    for panel in np.unique(panel_ids):
        ids = np.flatnonzero(panel_ids == panel)
        design = np.column_stack((material[ids], np.ones(len(ids), dtype=np.float32)))
        coefficients = np.linalg.lstsq(design, canonical[ids], rcond=None)[0]
        tangent_u = coefficients[0]
        tangent_v = coefficients[1]
        tangent_u /= max(float(np.linalg.norm(tangent_u)), 1.0e-8)
        tangent_v -= tangent_u * float(np.dot(tangent_u, tangent_v))
        tangent_v /= max(float(np.linalg.norm(tangent_v)), 1.0e-8)
        basis[ids, 0] = tangent_u
        basis[ids, 1] = tangent_v
    basis[:, 2] = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    return basis


def _build_field_bundle(
    geometry: _PreparedGeometry,
    config: AnnyCloth4DConfig,
) -> _FieldBundle:
    panel_count = len(geometry.panel_names)
    conditioner = Cloth4DGeometricConditioner(
        config.intrinsic_channels,
        config.scene_feature_dim,
        1,
        material_latent_dim=config.material_latent_dim,
        hidden_dim=config.hidden_dim,
        spectral_cap=0.9,
        causal_scene_graph=True,
    )
    field = E3AnchorNeuralClothField(
        panel_count,
        material_latent_dim=config.material_latent_dim,
        query_feature_dim=config.scene_feature_dim,
        cloth_anchor_feature_dim=config.intrinsic_channels,
        body_anchor_feature_dim=1,
        query_vector_feature_dim=5,
        panel_embedding_dim=6,
        hidden_dim=config.hidden_dim,
        hidden_layers=2,
        spectral_cap=0.9,
        material_bandwidth=0.075,
        material_scale=0.25,
        spatial_scale=0.25,
        velocity_scale=0.5,
        time_scale=0.025,
        displacement_limit=0.075,
    )
    latent = SceneMaterialLatent(config.material_latent_dim, initial_scale=0.01)
    shell_model = HyperNetworkKirchhoffLoveShell(
        panel_count,
        latent_dim=config.material_latent_dim,
        hyper_hidden_dim=max(24, config.hidden_dim),
        node_hidden_dim=4,
        node_steps=4,
        spectral_cap=0.9,
    )
    return _FieldBundle(field, conditioner, latent, shell_model)


def _conditioning_for_frame(
    bundle: _FieldBundle,
    geometry: _PreparedGeometry,
    frames: Tensor,
    frame_index: int,
    config: AnnyCloth4DConfig,
) -> Cloth4DGeometricConditioning:
    positions = frames[frame_index]
    if frame_index == 0:
        velocities = (frames[min(1, len(frames) - 1)] - positions) / config.frame_dt
    else:
        velocities = (positions - frames[frame_index - 1]) / config.frame_dt
    scene = _scene_graph_tensors(
        frames[max(0, frame_index - 1)],
        positions,
        geometry,
        frame_index * config.frame_dt,
        config,
    )
    body = geometry.body_anchor_positions
    body_normals = geometry.body_anchor_normals
    distance = torch.cdist(body, positions).min(dim=1).values
    pressure = torch.exp(-distance / 0.025).clamp(0.0, 1.0)
    force = pressure[:, None] * body_normals
    tactile = adapt_genesis_taxel_field(
        body,
        body_normals,
        force,
        geometry.tactile_edge_index,
    )
    return bundle.conditioner(
        surface_features=geometry.surface_features,
        laplace_eigenvalues=geometry.laplace_eigenvalues,
        laplace_eigenvectors=geometry.laplace_eigenvectors,
        vertex_mass=geometry.vertex_mass,
        particle_positions=positions,
        particle_velocities=velocities,
        particle_edge_index=geometry.edge_index,
        rest_lengths=geometry.rest_lengths,
        cotangent_weights=geometry.cotangent_weights,
        material_latent=bundle.material_latent(),
        scene_scalar_features=scene["scalar_features"],
        scene_positions=scene["positions"],
        scene_times=scene["times"],
        scene_hierarchy_levels=scene["levels"],
        scene_edge_index=scene["edge_index"],
        scene_relation_ids=scene["relation_ids"],
        query_scene_node_ids=scene["query_node_ids"],
        tactile_features=tactile.tangent_features,
        tactile_edge_index=geometry.tactile_edge_index,
        tactile_connection_angle=tactile.connection_angle,
        tactile_tangent_u=tactile.tangent_u,
        tactile_tangent_v=tactile.tangent_v,
        tactile_normal_force=tactile.normal_force,
        body_anchor_count=len(body),
    )


def _scene_graph_tensors(
    previous: Tensor,
    current: Tensor,
    geometry: _PreparedGeometry,
    current_time: float,
    config: AnnyCloth4DConfig,
) -> dict[str, Tensor]:
    panel_count = len(geometry.panel_names)
    object_count = 3 + panel_count

    def object_positions(vertices: Tensor) -> Tensor:
        body_center = geometry.body_anchor_positions.mean(dim=0)
        garment_center = vertices.mean(dim=0)
        panels = []
        for panel in range(panel_count):
            panels.append(vertices[geometry.panel_ids == panel].mean(dim=0))
        return torch.stack((body_center, body_center, garment_center, *panels), dim=0)

    previous_positions = object_positions(previous)
    current_positions = object_positions(current)
    positions = torch.cat((previous_positions, current_positions), dim=0)
    previous_time = max(0.0, current_time - config.frame_dt)
    times = torch.cat(
        (
            torch.full((object_count,), previous_time, dtype=current.dtype),
            torch.full((object_count,), current_time, dtype=current.dtype),
        )
    )
    levels_one = torch.as_tensor([0.0, 1.0, 2.0] + [3.0] * panel_count)
    levels = torch.cat((levels_one, levels_one), dim=0)

    def scalar_features_at(time_value: float) -> Tensor:
        scalar = torch.zeros((object_count, config.scene_feature_dim), dtype=current.dtype)
        scalar[0, 0] = 1.0
        scalar[1, 1] = 1.0
        scalar[2, 2] = 1.0
        behavior = config.behavior_profile.sample(time_value, current)
        if config.scene_feature_dim >= 16:
            scalar[0, 8:13] = behavior.personality_ocean
            scalar[1, 13:16] = behavior.emotion_vad
        for panel, name in enumerate(geometry.panel_names):
            row = 3 + panel
            scalar[row, 3] = 1.0
            scalar[row, 4] = panel / max(panel_count - 1, 1)
            scalar[row, 5] = float(name.startswith("front_"))
            scalar[row, 6] = float("sleeve" in name)
        scalar[:, 7] = 1.0
        return scalar

    scalar_features = torch.cat(
        (scalar_features_at(previous_time), scalar_features_at(current_time)),
        dim=0,
    )

    current_offset = object_count
    receivers: list[int] = []
    senders: list[int] = []
    relations: list[int] = []
    for object_id in range(object_count):
        receivers.append(current_offset + object_id)
        senders.append(object_id)
        relations.append(0)
    # Explicit hierarchy: person/root -> body -> garment -> panels. The body
    # to panel relation is the typed contact shortcut.
    receivers.extend((current_offset + 1, current_offset + 2))
    senders.extend((current_offset, current_offset + 1))
    relations.extend((1, 1))
    for panel in range(panel_count):
        panel_node = current_offset + 3 + panel
        receivers.extend((panel_node, panel_node))
        senders.extend((current_offset + 2, current_offset + 1))
        relations.extend((1, 2))
    for receiver_panel, sender_panel in geometry.panel_adjacency:
        receivers.append(current_offset + 3 + receiver_panel)
        senders.append(current_offset + 3 + sender_panel)
        relations.append(3)
    edge_index = torch.as_tensor([receivers, senders], dtype=torch.long)
    relation_ids = torch.as_tensor(relations, dtype=torch.long)
    query_node_ids = current_offset + 3 + geometry.panel_ids
    return {
        "scalar_features": scalar_features,
        "positions": positions,
        "times": times,
        "levels": levels,
        "edge_index": edge_index,
        "relation_ids": relation_ids,
        "query_node_ids": query_node_ids,
    }


def _query_field(
    bundle: _FieldBundle,
    conditioning: Cloth4DGeometricConditioning,
    geometry: _PreparedGeometry,
    canonical: Tensor,
    query_ids: Tensor,
    physical_time: float,
) -> Tensor:
    anchors = geometry.cloth_anchor_ids
    geometric_vectors = torch.cat(
        (
            conditioning.query_vectors[query_ids],
            geometry.local_query_vectors[query_ids],
        ),
        dim=1,
    )
    vector_roles = torch.eye(5, dtype=geometric_vectors.dtype)
    vector_features = vector_roles[None, :, :].expand(len(query_ids), -1, -1)
    output = bundle.field(
        material_coordinates=geometry.material_coordinates[query_ids],
        physical_time=physical_time,
        panel_ids=geometry.panel_ids[query_ids],
        material_latent=bundle.material_latent(),
        cloth_anchor_positions=canonical[anchors],
        cloth_anchor_material_coordinates=geometry.material_coordinates[anchors],
        cloth_anchor_panel_ids=geometry.panel_ids[anchors],
        body_anchor_positions=geometry.body_anchor_positions,
        query_reference_positions=canonical[query_ids],
        cloth_anchor_features=conditioning.cloth_anchor_features[anchors],
        body_anchor_features=conditioning.body_anchor_features,
        body_anchor_normals=geometry.body_anchor_normals,
        body_anchor_velocities=torch.zeros_like(geometry.body_anchor_positions),
        body_tactile_vectors=conditioning.body_tactile_vectors,
        body_anchor_mask=geometry.body_anchor_mask[query_ids],
        query_scalar_features=conditioning.query_scalar_features[query_ids],
        query_vectors=geometric_vectors,
        query_vector_features=vector_features,
        boundary_weight=geometry.boundary_weight[query_ids],
    )
    return output.positions


def _fit_scene_field(
    bundle: _FieldBundle,
    geometry: _PreparedGeometry,
    teacher_frames: np.ndarray,
    material: MaterialSpec,
    config: AnnyCloth4DConfig,
) -> list[dict[str, float]]:
    frames = torch.as_tensor(teacher_frames, dtype=torch.float32)
    canonical = frames[0]
    parameters = (
        list(bundle.field.parameters())
        + list(bundle.conditioner.parameters())
        + list(bundle.material_latent.parameters())
        + list(bundle.shell_model.parameters())
    )
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 1)
    vertex_count = frames.shape[1]
    validation_ids = torch.linspace(
        0,
        vertex_count - 1,
        min(128, vertex_count),
    ).long()
    best_rmse = float("inf")
    best_state: tuple[
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
    ] | None = None
    best_physics_score = float("inf")
    best_physics_state: tuple[
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
    ] | None = None
    stable_validations = 0
    history: list[dict[str, float]] = []

    for step in range(config.training_steps):
        frame_index = 1 + (step % (len(frames) - 1))
        vertex_ids = torch.randint(
            vertex_count,
            (min(config.query_batch_size, vertex_count),),
            generator=generator,
        )
        edge_ids = torch.randint(
            len(geometry.edge_pairs),
            (min(config.sampled_edges, len(geometry.edge_pairs)),),
            generator=generator,
        )
        sampled_edges = geometry.edge_pairs[edge_ids]
        if len(geometry.seam_pairs):
            seam_ids = torch.randint(
                len(geometry.seam_pairs),
                (min(config.sampled_seams, len(geometry.seam_pairs)),),
                generator=generator,
            )
            sampled_seams = geometry.seam_pairs[seam_ids]
        else:
            sampled_seams = geometry.seam_pairs
        query_ids = torch.unique(
            torch.cat((vertex_ids, sampled_edges.reshape(-1), sampled_seams.reshape(-1)))
        )
        inverse = torch.full((vertex_count,), -1, dtype=torch.long)
        inverse[query_ids] = torch.arange(len(query_ids))

        optimizer.zero_grad(set_to_none=True)
        conditioning = _conditioning_for_frame(bundle, geometry, frames, frame_index, config)
        prediction = _query_field(
            bundle,
            conditioning,
            geometry,
            canonical,
            query_ids,
            frame_index * config.frame_dt,
        )
        target = frames[frame_index, query_ids]
        position_loss = (prediction - target).square().mean() / (0.01**2)

        predicted_edge_length = torch.linalg.vector_norm(
            prediction[inverse[sampled_edges[:, 1]]] - prediction[inverse[sampled_edges[:, 0]]],
            dim=-1,
        )
        target_edge_length = torch.linalg.vector_norm(
            frames[frame_index, sampled_edges[:, 1]] - frames[frame_index, sampled_edges[:, 0]],
            dim=-1,
        )
        edge_loss = ((predicted_edge_length - target_edge_length) / 0.005).square().mean()
        if len(sampled_seams):
            predicted_seam_gap = torch.linalg.vector_norm(
                prediction[inverse[sampled_seams[:, 1]]] - prediction[inverse[sampled_seams[:, 0]]],
                dim=-1,
            )
            target_seam_gap = torch.linalg.vector_norm(
                frames[frame_index, sampled_seams[:, 1]] - frames[frame_index, sampled_seams[:, 0]],
                dim=-1,
            )
            seam_loss = ((predicted_seam_gap - target_seam_gap) / 0.002).square().mean()
        else:
            seam_loss = prediction.new_zeros(())
        gap = (
            (prediction - geometry.contact_closest[query_ids])
            * geometry.contact_normals[query_ids]
        ).sum(dim=-1) - 0.003
        contact_loss = (F.relu(-gap) / 0.003).square().mean()
        latent_loss = bundle.material_latent.prior_loss()
        loss = position_loss + 0.06 * edge_loss + 0.04 * seam_loss + 0.15 * contact_loss + 1.0e-3 * latent_loss
        shell_field_loss = prediction.new_zeros(())
        shell_equilibrium_loss = prediction.new_zeros(())
        shell_traction_loss = prediction.new_zeros(())
        action_stationarity_loss = prediction.new_zeros(())
        phase_energy_loss = prediction.new_zeros(())
        contact_kkt_loss = prediction.new_zeros(())
        phase_dissipation_loss = prediction.new_zeros(())
        metric_hamiltonian_loss = prediction.new_zeros(())
        extended_phase_constraint_loss = prediction.new_zeros(())
        bundle_port_hamiltonian_total_loss = prediction.new_zeros(())
        bundle_measure_continuity_loss = prediction.new_zeros(())
        bundle_hamiltonian_balance_loss = prediction.new_zeros(())
        bundle_passivity_loss = prediction.new_zeros(())
        bundle_dissipation_loss = prediction.new_zeros(())
        shell_active = (
            len(frames) >= 3
            and step + 1 > config.shell_physics_warmup_steps
            and (step + 1 - config.shell_physics_warmup_steps)
            % config.shell_physics_interval
            == 0
        )
        if shell_active:
            physics_frame = min(max(frame_index, 1), len(frames) - 2)
            (
                shell_field_loss,
                shell_equilibrium_loss,
                shell_traction_loss,
                action_stationarity_loss,
                phase_energy_loss,
                contact_kkt_loss,
                phase_dissipation_loss,
                metric_hamiltonian_loss,
                extended_phase_constraint_loss,
                bundle_port_hamiltonian_total_loss,
                bundle_measure_continuity_loss,
                bundle_hamiltonian_balance_loss,
                bundle_passivity_loss,
                bundle_dissipation_loss,
            ) = _shell_physics_training_terms(
                bundle,
                geometry,
                frames,
                physics_frame,
                material,
                config,
            )
            ramp = min(
                1.0,
                (step + 1 - config.shell_physics_warmup_steps)
                / max(3 * config.shell_physics_interval, 1),
            )
            loss = loss + ramp * (
                config.shell_energy_weight * shell_field_loss
                + config.shell_equilibrium_weight * shell_equilibrium_loss
                + config.shell_traction_weight * shell_traction_loss
                + config.variational_pinn_weight
                * (
                    action_stationarity_loss
                    + phase_energy_loss
                    + contact_kkt_loss
                    + phase_dissipation_loss
                    + metric_hamiltonian_loss
                    + extended_phase_constraint_loss
                )
                + config.bundle_measure_weight
                * bundle_port_hamiltonian_total_loss
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
        optimizer.step()

        rmse = float(torch.sqrt((prediction.detach() - target).square().mean()).item())
        record = {
            "step": float(step + 1),
            "loss": float(loss.detach().item()),
            "position_rmse_m": rmse,
            "edge": float(edge_loss.detach().item()),
            "seam": float(seam_loss.detach().item()),
            "contact": float(contact_loss.detach().item()),
            "kirchhoff_love_shell": float(shell_field_loss.detach().item()),
            "dynamic_equilibrium": float(shell_equilibrium_loss.detach().item()),
            "free_edge_traction": float(shell_traction_loss.detach().item()),
            "phase_action_stationarity": float(action_stationarity_loss.detach().item()),
            "phase_energy_balance": float(phase_energy_loss.detach().item()),
            "contact_kkt": float(contact_kkt_loss.detach().item()),
            "phase_dissipation": float(phase_dissipation_loss.detach().item()),
            "hamiltonian_metric_constraint": float(
                metric_hamiltonian_loss.detach().item()
            ),
            "biquaternion_extended_phase_constraint": float(
                extended_phase_constraint_loss.detach().item()
            ),
            "bundle_port_hamiltonian_total": float(
                bundle_port_hamiltonian_total_loss.detach().item()
            ),
            "bundle_measure_continuity": float(
                bundle_measure_continuity_loss.detach().item()
            ),
            "bundle_hamiltonian_balance": float(
                bundle_hamiltonian_balance_loss.detach().item()
            ),
            "bundle_port_passivity": float(bundle_passivity_loss.detach().item()),
            "bundle_dissipation_sign": float(
                bundle_dissipation_loss.detach().item()
            ),
        }
        validation_rmse: float | None = None
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == config.training_steps:
            validation_rmse = _temporal_validation_rmse(
                bundle,
                geometry,
                frames,
                validation_ids,
                config,
            )
            record["temporal_validation_rmse_m"] = validation_rmse
            print(
                "cloth4d fit "
                f"{step + 1:04d}/{config.training_steps:04d} "
                f"batch={rmse * 1000.0:.2f}mm "
                f"temporal={validation_rmse * 1000.0:.2f}mm "
                f"loss={record['loss']:.4f}",
                flush=True,
            )
        history.append(record)
        if validation_rmse is not None and validation_rmse < best_rmse:
            best_rmse = validation_rmse
            best_state = (
                copy.deepcopy(bundle.field.state_dict()),
                copy.deepcopy(bundle.conditioner.state_dict()),
                copy.deepcopy(bundle.material_latent.state_dict()),
                copy.deepcopy(bundle.shell_model.state_dict()),
            )
        if validation_rmse is not None and shell_active:
            phase_residual = (
                action_stationarity_loss
                + phase_energy_loss
                + contact_kkt_loss
                + phase_dissipation_loss
                + metric_hamiltonian_loss
                + extended_phase_constraint_loss
                + bundle_port_hamiltonian_total_loss
            )
            physics_selection_score = (
                validation_rmse / config.target_rmse_m
                + 0.05 * torch.log1p(phase_residual.detach())
            )
            record["physics_selection_score"] = float(physics_selection_score)
            if float(physics_selection_score) < best_physics_score:
                best_physics_score = float(physics_selection_score)
                best_physics_state = (
                    copy.deepcopy(bundle.field.state_dict()),
                    copy.deepcopy(bundle.conditioner.state_dict()),
                    copy.deepcopy(bundle.material_latent.state_dict()),
                    copy.deepcopy(bundle.shell_model.state_dict()),
                )
        if validation_rmse is not None:
            if validation_rmse <= config.target_rmse_m:
                stable_validations += 1
            else:
                stable_validations = 0
            if step + 1 >= config.minimum_training_steps and stable_validations >= 3:
                break

    selected_state = best_physics_state if best_physics_state is not None else best_state
    if selected_state is not None:
        bundle.field.load_state_dict(selected_state[0])
        bundle.conditioner.load_state_dict(selected_state[1])
        bundle.material_latent.load_state_dict(selected_state[2])
        bundle.shell_model.load_state_dict(selected_state[3])
    bundle.field.eval()
    bundle.conditioner.eval()
    bundle.shell_model.eval()
    return history


def _shell_physics_training_terms(
    bundle: _FieldBundle,
    geometry: _PreparedGeometry,
    frames: Tensor,
    frame_index: int,
    material: MaterialSpec,
    config: AnnyCloth4DConfig,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    query_ids = torch.arange(frames.shape[1])
    strain_limits = material_strain_limits(material)
    conditioning = _conditioning_for_frame(bundle, geometry, frames, frame_index, config)
    predicted = _query_field(
        bundle,
        conditioning,
        geometry,
        frames[0],
        query_ids,
        frame_index * config.frame_dt,
    )

    shell_parameter_flags = [parameter.requires_grad for parameter in bundle.shell_model.parameters()]
    try:
        for parameter in bundle.shell_model.parameters():
            parameter.requires_grad_(False)
        predicted_shell = bundle.shell_model(
            predicted,
            geometry.shell,
            bundle.material_latent().detach(),
            compute_bending_forces=False,
            create_graph=False,
            maximum_tensile_strain=strain_limits.maximum_tensile_strain,
            maximum_compressive_strain=strain_limits.maximum_compressive_strain,
        )
    finally:
        for parameter, requires_grad in zip(bundle.shell_model.parameters(), shell_parameter_flags):
            parameter.requires_grad_(requires_grad)
    shell_area = geometry.shell.rest_areas.to(predicted).sum().clamp_min(1.0e-8)
    field_energy_density = (
        predicted_shell.membrane_energy.sum()
        + predicted_shell.strain_limit_energy.sum()
        + predicted_shell.bending_energy.sum()
    ) / shell_area
    field_shell_loss = torch.log1p(field_energy_density / 100.0)

    previous_conditioning = _conditioning_for_frame(
        bundle,
        geometry,
        frames,
        frame_index - 1,
        config,
    )
    next_conditioning = _conditioning_for_frame(
        bundle,
        geometry,
        frames,
        frame_index + 1,
        config,
    )
    predicted_previous = _query_field(
        bundle,
        previous_conditioning,
        geometry,
        frames[0],
        query_ids,
        (frame_index - 1) * config.frame_dt,
    )
    predicted_next = _query_field(
        bundle,
        next_conditioning,
        geometry,
        frames[0],
        query_ids,
        (frame_index + 1) * config.frame_dt,
    )
    shell = geometry.shell.to(predicted)
    areal_density_kg_m2 = 0.18
    mass = areal_density_kg_m2 * shell.vertex_areas.clamp_min(1.0e-8)
    phase_state = IntrinsicClothPhaseState.from_velocity(
        geometry.material_coordinates.to(predicted),
        predicted,
        (predicted_next - predicted_previous) / (2.0 * config.frame_dt),
        mass,
        frame_index * config.frame_dt,
    )
    complex_phase = KahlerClothPhase.from_canonical_phase(
        phase_state,
        frames[0],
        reference_angular_frequency=1.0 / config.frame_dt,
    )
    complex_phase_state = complex_phase.to_canonical_phase()
    pinned = torch.nonzero(
        geometry.boundary_weight[:, 0] <= 0.0,
        as_tuple=False,
    ).flatten()
    dynamics = kirchhoff_love_dynamics_loss(
        bundle.shell_model,
        geometry.shell,
        predicted_previous,
        complex_phase_state.positions,
        predicted_next,
        bundle.material_latent(),
        dt=config.frame_dt,
        contact_closest=geometry.contact_closest,
        contact_normals=geometry.contact_normals,
        pinned_vertices=pinned,
        maximum_tensile_strain=strain_limits.maximum_tensile_strain,
        maximum_compressive_strain=strain_limits.maximum_compressive_strain,
    )
    gravity = predicted.new_tensor((0.0, -9.81, 0.0))
    external_force = mass[:, None] * gravity
    contact_normal = F.normalize(
        geometry.contact_normals.to(predicted), dim=-1, eps=1.0e-8
    )
    contact_gap = (
        (predicted - geometry.contact_closest.to(predicted)) * contact_normal
    ).sum(dim=-1)
    metric_constraint_energy = dynamics.output.strain_limit_energy.sum()
    hamiltonian = assemble_cloth_hamiltonian(
        complex_phase_state,
        dynamics.output.membrane_energy,
        dynamics.output.bending_energy,
        metric_constraint_energy=metric_constraint_energy,
    )
    # The physical terms stay real; the complex chart supplies the intrinsic
    # phase calculus and the scene ledger preserves their exact audited sum.
    complex_scene_hamiltonian = assemble_complex_scene_hamiltonian(
        node_energies=torch.stack((hamiltonian.kinetic + hamiltonian.gravity,)),
        relation_energies=torch.stack(
            (
                hamiltonian.membrane,
                hamiltonian.bending,
                hamiltonian.contact_barrier,
            )
        ),
        node_hierarchy_levels=torch.tensor(
            [2], device=predicted.device, dtype=torch.long
        ),
        relation_index=torch.zeros((2, 3), device=predicted.device, dtype=torch.long),
        constraint_energy=hamiltonian.metric_constraint,
    )
    if complex_phase.coordinates.shape != predicted.shape:
        raise RuntimeError("complex cloth phase must preserve the simulation mesh")
    extended_phase = BiquaternionVertexPhase.from_canonical_phase(
        complex_phase_state,
        frames[0],
        hamiltonian=complex_scene_hamiltonian.total,
    )
    extended_phase_constraint = extended_phase.extended_hamiltonian_constraint(
        complex_scene_hamiltonian.total
    )
    extended_phase_constraint_loss = (
        extended_phase_constraint / 1.0e-4
    ).square()
    phase_pinn = variational_phase_pinn_residual(
        predicted_previous,
        predicted,
        predicted_next,
        mass,
        dynamics.output.membrane_internal_force
        + dynamics.output.bending_internal_force,
        complex_scene_hamiltonian.total - hamiltonian.kinetic,
        dt=config.frame_dt,
        external_force=external_force,
        contact_gap=contact_gap,
        contact_normals=contact_normal,
        metric_constraint_energy=metric_constraint_energy,
        config=VariationalPhasePINNConfig(),
    )
    incoming_velocity = (predicted - predicted_previous) / config.frame_dt
    outgoing_velocity = (predicted_next - predicted) / config.frame_dt
    reference_map = ReferenceMapFiberBundle.from_shell_geometry(
        geometry.shell,
        frames[0],
    )
    measure_builder = BundleMeasureBuilder(
        reference_map,
        geometry.shell,
        areal_density_kg_m2=areal_density_kg_m2,
    )
    current_measure = measure_builder.lift(
        predicted,
        incoming_velocity,
        bundle.material_latent(),
        time=frame_index * config.frame_dt,
        shell_output=dynamics.output,
    )
    following_shell = bundle.shell_model(
        predicted_next,
        geometry.shell,
        bundle.material_latent(),
        compute_bending_forces=False,
        create_graph=False,
    )
    following_measure = measure_builder.lift(
        predicted_next,
        outgoing_velocity,
        bundle.material_latent(),
        time=(frame_index + 1) * config.frame_dt,
        shell_output=following_shell,
    )
    current_measure_hamiltonian = assemble_bundle_measure_hamiltonian(
        current_measure,
        dynamics.output.membrane_energy,
        dynamics.output.bending_energy,
    )
    following_measure_hamiltonian = assemble_bundle_measure_hamiltonian(
        following_measure,
        following_shell.membrane_energy,
        following_shell.bending_energy,
    )
    midpoint_velocity = 0.5 * (incoming_velocity + outgoing_velocity)
    supplied_power = (phase_pinn.contact_force * midpoint_velocity).sum()
    dissipative_power = (
        (phase_pinn.friction_force + phase_pinn.damping_force)
        * midpoint_velocity
    ).sum()
    measure_force = (
        dynamics.output.membrane_internal_force
        + dynamics.output.bending_internal_force
        + external_force
        + phase_pinn.contact_force
        + phase_pinn.friction_force
        + phase_pinn.damping_force
    )
    measure_ph = bundle_measure_port_hamiltonian_loss(
        current_measure,
        following_measure,
        measure_force,
        current_measure_hamiltonian,
        following_measure_hamiltonian,
        dt=config.frame_dt,
        supplied_power=supplied_power,
        dissipative_power=dissipative_power,
    )
    return (
        field_shell_loss,
        dynamics.equilibrium,
        dynamics.traction,
        phase_pinn.action_stationarity,
        phase_pinn.energy_balance,
        phase_pinn.contact_kkt,
        phase_pinn.dissipation,
        phase_pinn.metric_constraint,
        extended_phase_constraint_loss,
        measure_ph.total,
        measure_ph.continuity,
        measure_ph.energy_balance,
        measure_ph.passivity,
        measure_ph.dissipation,
    )


def _temporal_validation_rmse(
    bundle: _FieldBundle,
    geometry: _PreparedGeometry,
    frames: Tensor,
    query_ids: Tensor,
    config: AnnyCloth4DConfig,
) -> float:
    squared_error = 0.0
    value_count = 0
    with torch.no_grad():
        for frame_index in range(1, len(frames)):
            conditioning = _conditioning_for_frame(bundle, geometry, frames, frame_index, config)
            prediction = _query_field(
                bundle,
                conditioning,
                geometry,
                frames[0],
                query_ids,
                frame_index * config.frame_dt,
            )
            difference = prediction - frames[frame_index, query_ids]
            squared_error += float(difference.square().sum().item())
            value_count += difference.numel()
    return float(np.sqrt(squared_error / max(value_count, 1)))


def _infer_rollout(
    bundle: _FieldBundle,
    geometry: _PreparedGeometry,
    teacher_frames: np.ndarray,
    config: AnnyCloth4DConfig,
) -> np.ndarray:
    frames = torch.as_tensor(teacher_frames, dtype=torch.float32)
    canonical = frames[0]
    query_all = torch.arange(frames.shape[1])
    output_frames: list[np.ndarray] = []
    with torch.no_grad():
        for frame_index in range(len(frames)):
            conditioning = _conditioning_for_frame(bundle, geometry, frames, frame_index, config)
            chunks = []
            for query_ids in query_all.split(config.inference_chunk_size):
                chunks.append(
                    _query_field(
                        bundle,
                        conditioning,
                        geometry,
                        canonical,
                        query_ids,
                        frame_index * config.frame_dt,
                    )
                )
            output_frames.append(torch.cat(chunks, dim=0).cpu().numpy())
    output = np.asarray(output_frames, dtype=np.float32)
    output[0] = teacher_frames[0]
    return output


def _apply_hard_constraints(
    raw_frames: np.ndarray,
    teacher_frames: np.ndarray,
    mesh: GarmentMesh,
    avatar: BodyMeshAvatar,
    geometry: _PreparedGeometry,
) -> np.ndarray:
    faces = torch.as_tensor(mesh.faces, dtype=torch.long)
    inverse_mass = torch.ones(len(mesh.vertices), dtype=torch.float32)
    inverse_mass[torch.as_tensor(mesh.pinned_vertices, dtype=torch.long)] = 0.0
    movable = inverse_mass > 0.0
    constrained: list[np.ndarray] = []
    for frame_index, raw in enumerate(raw_frames):
        if frame_index == 0:
            constrained.append(teacher_frames[0].copy())
            continue
        positions = torch.as_tensor(raw, dtype=torch.float32).clone()
        target_lengths = geometry.rest_lengths
        canonical = torch.as_tensor(teacher_frames[0], dtype=torch.float32)
        seam_targets = (
            torch.linalg.vector_norm(
                canonical[geometry.seam_pairs[:, 1]] - canonical[geometry.seam_pairs[:, 0]],
                dim=-1,
            ).clamp(5.0e-4, 1.5e-3)
            if len(geometry.seam_pairs)
            else torch.empty(0)
        )
        for _ in range(6):
            positions = project_distance_constraints(
                positions,
                geometry.edge_pairs,
                target_lengths,
                inverse_mass,
                iterations=12,
                compliance=0.0,
                dt=1.0,
                min_ratio=0.985,
                max_ratio=1.015,
            )
            if len(geometry.seam_pairs):
                positions = project_distance_constraints(
                    positions,
                    geometry.seam_pairs,
                    seam_targets,
                    inverse_mass,
                    iterations=12,
                    compliance=0.0,
                    dt=1.0,
                )
            positions, _reports = _project_avatar_constraints(
                avatar,
                positions,
                faces,
                mesh.panel_labels,
                geometry.seam_pairs,
                collision_mask=movable,
            )
        if len(geometry.seam_pairs):
            positions = project_distance_constraints(
                positions,
                geometry.seam_pairs,
                seam_targets,
                inverse_mass,
                iterations=16,
                compliance=0.0,
                dt=1.0,
            )
            positions, _ = avatar.project_seam_groups_torch(
                positions,
                geometry.seam_pairs,
                mesh.panel_labels,
                collision_mask=movable,
            )
        positions[mesh.pinned_vertices] = torch.as_tensor(teacher_frames[0, mesh.pinned_vertices])
        constrained.append(positions.numpy())
    return np.asarray(constrained, dtype=np.float32)


def _render_comparison(
    teacher: np.ndarray,
    field: np.ndarray,
    mesh: GarmentMesh,
    avatar: BodyMeshAvatar,
    texture: Image.Image,
    gif_path: Path,
    final_path: Path,
    caption: str,
) -> None:
    images: list[Image.Image] = []
    for frame_index in range(min(len(teacher), len(field))):
        left = render_frame(
            teacher[frame_index],
            mesh.faces,
            mesh.pinned_vertices,
            frame_index,
            caption,
            avatar=avatar,
            texture_image=texture,
            mesh=mesh,
            size=560,
            method_label="intrinsic shell teacher",
            semantic_sleeve_occlusion=any(
                "sleeve" in label for label in mesh.panel_labels
            ),
        )
        right = render_frame(
            field[frame_index],
            mesh.faces,
            mesh.pinned_vertices,
            frame_index,
            caption,
            avatar=avatar,
            texture_image=texture,
            mesh=mesh,
            size=560,
            method_label="E3 + Hyper-NODE KL shell",
            semantic_sleeve_occlusion=any(
                "sleeve" in label for label in mesh.panel_labels
            ),
        )
        combined = Image.new("RGB", (1120, 590), (238, 242, 246))
        combined.paste(left, (0, 30))
        combined.paste(right, (560, 30))
        draw = ImageDraw.Draw(combined)
        draw.text((18, 8), "PHYSICS TRAJECTORY", fill=(33, 45, 58))
        draw.text((578, 8), "OPTIMIZED GEOMETRIC 4D FIELD", fill=(33, 45, 58))
        images.append(combined)
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=110, loop=0)
    images[-1].save(final_path)


def _rollout_metrics(
    raw: np.ndarray,
    constrained: np.ndarray,
    teacher: np.ndarray,
    mesh: GarmentMesh,
    avatar: BodyMeshAvatar,
    geometry: _PreparedGeometry,
    bundle: _FieldBundle,
    config: AnnyCloth4DConfig,
) -> dict[str, float]:
    raw_error = np.linalg.norm(raw - teacher, axis=-1)
    constrained_error = np.linalg.norm(constrained - teacher, axis=-1)
    edges = geometry.edge_pairs.numpy()
    reference_lengths = np.linalg.norm(
        constrained[0, edges[:, 1]] - constrained[0, edges[:, 0]],
        axis=-1,
    )
    final_lengths = np.linalg.norm(
        constrained[-1, edges[:, 1]] - constrained[-1, edges[:, 0]],
        axis=-1,
    )
    ratio = final_lengths / np.maximum(reference_lengths, 1.0e-8)
    seams = geometry.seam_pairs.numpy()
    seam_gap = (
        np.linalg.norm(constrained[-1, seams[:, 1]] - constrained[-1, seams[:, 0]], axis=-1)
        if len(seams)
        else np.asarray([0.0], dtype=np.float32)
    )
    collision = avatar.collision_diagnostics_np(constrained[-1], mesh.faces)
    equivariance = _field_equivariance_error(bundle, geometry, teacher, config)
    displacement = np.linalg.norm(constrained - constrained[0:1], axis=-1)
    metrics: dict[str, float] = {
        "raw_field_mean_error_m": float(raw_error.mean()),
        "raw_field_p95_error_m": float(np.percentile(raw_error, 95)),
        "raw_field_max_error_m": float(raw_error.max()),
        "constrained_field_mean_error_m": float(constrained_error.mean()),
        "constrained_field_p95_error_m": float(np.percentile(constrained_error, 95)),
        "constrained_field_max_error_m": float(constrained_error.max()),
        "final_mean_canonical_edge_ratio": float(ratio.mean()),
        "final_p95_canonical_edge_ratio": float(np.percentile(ratio, 95)),
        "final_p99_canonical_edge_ratio": float(np.percentile(ratio, 99)),
        "final_max_canonical_edge_ratio": float(ratio.max()),
        "final_mean_seam_gap_m": float(seam_gap.mean()),
        "final_max_seam_gap_m": float(seam_gap.max()),
        "mean_motion_m": float(displacement[-1].mean()),
        "max_motion_m": float(displacement.max()),
        "e3_equivariance_max_abs_m": equivariance,
    }
    metrics.update({key: float(value) for key, value in collision.items()})
    metrics.update(_shell_rollout_metrics(constrained, geometry, bundle))
    return metrics


def _shell_rollout_metrics(
    frames: np.ndarray,
    geometry: _PreparedGeometry,
    bundle: _FieldBundle,
) -> dict[str, float]:
    with torch.no_grad():
        initial = bundle.shell_model(
            torch.as_tensor(frames[0], dtype=torch.float32),
            geometry.shell,
            bundle.material_latent(),
            compute_bending_forces=False,
            create_graph=False,
        )
        final = bundle.shell_model(
            torch.as_tensor(frames[-1], dtype=torch.float32),
            geometry.shell,
            bundle.material_latent(),
            compute_bending_forces=False,
            create_graph=False,
        )
        initial_strain = initial.membrane_strains.abs().flatten()
        final_strain = final.membrane_strains.abs().flatten()
        area = geometry.shell.rest_areas.sum().clamp_min(1.0e-8)
        final_energy_density = (
            final.membrane_energy.sum() + final.bending_energy.sum()
        ) / area
        traction_norm = torch.linalg.vector_norm(final.free_boundary_traction, dim=-1)
        stiffness = final.face_stiffness.mean(dim=0)
        return {
            "kl_valid_material_faces": float(len(geometry.shell.faces)),
            "kl_rejected_degenerate_material_faces": float(geometry.shell.rejected_face_count),
            "kl_initial_median_abs_membrane_strain": float(torch.quantile(initial_strain, 0.5)),
            "kl_initial_p95_abs_membrane_strain": float(torch.quantile(initial_strain, 0.95)),
            "kl_final_median_abs_membrane_strain": float(torch.quantile(final_strain, 0.5)),
            "kl_final_p95_abs_membrane_strain": float(torch.quantile(final_strain, 0.95)),
            "kl_final_energy_density_n_m": float(final_energy_density),
            "kl_final_mean_free_edge_traction_n_m": (
                float(traction_norm.mean()) if len(traction_norm) else 0.0
            ),
            "node_mean_warp_stiffness_n_m": float(stiffness[0]),
            "node_mean_weft_stiffness_n_m": float(stiffness[1]),
            "node_mean_shear_stiffness_n_m": float(stiffness[2]),
            "node_mean_area_stiffness_n_m": float(stiffness[3]),
            "node_mean_bending_stiffness_n_m": float(stiffness[4]),
        }


def _field_equivariance_error(
    bundle: _FieldBundle,
    geometry: _PreparedGeometry,
    teacher: np.ndarray,
    config: AnnyCloth4DConfig,
) -> float:
    frames = torch.as_tensor(teacher, dtype=torch.float32)
    frame_index = len(frames) - 1
    with torch.no_grad():
        conditioning = _conditioning_for_frame(bundle, geometry, frames, frame_index, config)
        query_ids = torch.linspace(0, frames.shape[1] - 1, min(40, frames.shape[1])).long()
        canonical = frames[0]
        anchors = geometry.cloth_anchor_ids
        baseline = _query_field(
            bundle,
            conditioning,
            geometry,
            canonical,
            query_ids,
            frame_index * config.frame_dt,
        )
        orthogonal = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        translation = torch.tensor([0.7, -0.4, 0.2])
        moved = bundle.field(
            material_coordinates=geometry.material_coordinates[query_ids],
            physical_time=frame_index * config.frame_dt,
            panel_ids=geometry.panel_ids[query_ids],
            material_latent=bundle.material_latent(),
            cloth_anchor_positions=canonical[anchors] @ orthogonal.T + translation,
            cloth_anchor_material_coordinates=geometry.material_coordinates[anchors],
            cloth_anchor_panel_ids=geometry.panel_ids[anchors],
            body_anchor_positions=geometry.body_anchor_positions @ orthogonal.T + translation,
            query_reference_positions=canonical[query_ids] @ orthogonal.T + translation,
            cloth_anchor_features=conditioning.cloth_anchor_features[anchors],
            body_anchor_features=conditioning.body_anchor_features,
            body_anchor_normals=geometry.body_anchor_normals @ orthogonal.T,
            body_anchor_velocities=torch.zeros_like(geometry.body_anchor_positions),
            body_tactile_vectors=conditioning.body_tactile_vectors @ orthogonal.T,
            body_anchor_mask=geometry.body_anchor_mask[query_ids],
            query_scalar_features=conditioning.query_scalar_features[query_ids],
            query_vectors=torch.cat(
                (
                    conditioning.query_vectors[query_ids],
                    geometry.local_query_vectors[query_ids],
                ),
                dim=1,
            )
            @ orthogonal.T,
            query_vector_features=torch.eye(5)[None, :, :].expand(len(query_ids), -1, -1),
            boundary_weight=geometry.boundary_weight[query_ids],
        ).positions
        expected = baseline @ orthogonal.T + translation
        return float(torch.max(torch.abs(moved - expected)).item())


def _detect_body_sdf_events(
    frames: np.ndarray,
    mesh: GarmentMesh,
    avatar: BodyMeshAvatar,
    dt: float,
) -> list[dict[str, Any]]:
    detector = ZeroCrossingEventDetector(emit_initial_events=True)
    panel_names = sorted(set(mesh.panel_labels))
    labels = np.asarray(mesh.panel_labels)
    for panel_name in panel_names:
        detector.register(
            EventSurface(
                surface_id=f"body_sdf:{panel_name}",
                kind=EventSurfaceKind.BODY_CONTACT,
                active_side=ActiveSide.BELOW,
                threshold=0.0,
                hysteresis=5.0e-4,
                metadata={"panel": panel_name},
            )
        )
    events: list[dict[str, Any]] = []
    for frame_index, vertices in enumerate(frames):
        signed = avatar._proximity.signed_distance(vertices)
        gap = -(avatar.clearance + signed)
        values = {
            f"body_sdf:{panel_name}": float(np.min(gap[labels == panel_name]))
            for panel_name in panel_names
        }
        batch = detector.observe(frame_index * dt, values)
        for event in batch.events:
            events.append(
                {
                    "kind": event.kind.value,
                    "event_id": event.event_id,
                    "time": event.time,
                    "bracket": list(event.bracket),
                    "localized": event.localized,
                    "active": event.active,
                    "payload": dict(event.payload),
                }
            )
    return events


def _export_genesis_ipc_inputs(
    out_dir: Path,
    mesh: GarmentMesh,
    canonical: np.ndarray,
    avatar: BodyMeshAvatar,
    backend: GenesisIPCBackendStatus,
    config: AnnyCloth4DConfig,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    welded_vertices, welded_faces, original_to_welded = _weld_seam_vertices(
        canonical,
        mesh.faces,
        mesh.seam_edges,
    )
    cloth_path = out_dir / "tshirt_welded.obj"
    body_path = out_dir / "anny_collision.obj"
    mapping_path = out_dir / "original_to_welded.npy"
    config_path = out_dir / "scene_config.json"
    status_path = out_dir / "backend_status.json"
    _write_triangle_obj(cloth_path, welded_vertices, welded_faces)
    np.save(mapping_path, original_to_welded)

    body_vertices = np.asarray(avatar.mesh_vertices, dtype=np.float32)
    body_faces = np.asarray(avatar.mesh_faces, dtype=np.int32)
    try:
        import trimesh

        body_surface = trimesh.Trimesh(body_vertices, body_faces, process=False)
        simplified = body_surface.simplify_quadric_decimation(face_count=2500)
        body_vertices = np.asarray(simplified.vertices, dtype=np.float32)
        body_faces = np.asarray(simplified.faces, dtype=np.int32)
    except Exception:
        pass
    _write_triangle_obj(body_path, body_vertices, body_faces)
    scene_config = {
        "dt": config.dt,
        "steps": config.teacher_steps,
        "gravity": [0.0, -9.81, 0.0],
        "contact_d_hat": 0.004,
        "cloth": {
            "E_pa": 100000.0,
            "nu": 0.35,
            "rho_kg_m3": 180.0,
            "thickness_m": 0.0012,
            "bending_stiffness": 35.0,
            "friction_mu": 0.45,
        },
        "stock_pbd_used": False,
        "note": "Run scripts/run_genesis_ipc_teacher.py on a pyuipc-supported host.",
    }
    config_path.write_text(json.dumps(scene_config, indent=2), encoding="utf-8")
    status_path.write_text(json.dumps(backend.to_dict(), indent=2), encoding="utf-8")
    return {
        "cloth_obj": str(cloth_path),
        "body_obj": str(body_path),
        "original_to_welded": str(mapping_path),
        "scene_config": str(config_path),
        "backend_status": str(status_path),
        "runner": str(Path(__file__).resolve().parents[1] / "scripts" / "run_genesis_ipc_teacher.py"),
    }


def _weld_seam_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    seam_edges: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parent = np.arange(len(vertices), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for a, b in seam_edges:
        union(int(a), int(b))
    roots = np.asarray([find(index) for index in range(len(vertices))], dtype=np.int64)
    unique_roots, mapping = np.unique(roots, return_inverse=True)
    welded = np.zeros((len(unique_roots), 3), dtype=np.float64)
    counts = np.zeros(len(unique_roots), dtype=np.float64)
    np.add.at(welded, mapping, vertices)
    np.add.at(counts, mapping, 1.0)
    welded /= counts[:, None]
    remapped = mapping[np.asarray(faces, dtype=np.int64)]
    nondegenerate = (
        (remapped[:, 0] != remapped[:, 1])
        & (remapped[:, 1] != remapped[:, 2])
        & (remapped[:, 0] != remapped[:, 2])
    )
    remapped = remapped[nondegenerate]
    canonical_faces = np.sort(remapped, axis=1)
    _, unique_ids = np.unique(canonical_faces, axis=0, return_index=True)
    remapped = remapped[np.sort(unique_ids)]
    return welded.astype(np.float32), remapped.astype(np.int32), mapping.astype(np.int64)


def _write_triangle_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = ["# Genesis FEM.Cloth IPC input\n"]
    lines.extend(f"v {x:.8f} {y:.8f} {z:.8f}\n" for x, y, z in vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)
    path.write_text("".join(lines), encoding="utf-8")
