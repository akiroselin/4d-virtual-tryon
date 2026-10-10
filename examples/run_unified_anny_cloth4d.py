from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.anny_cloth4d_simulation import (  # noqa: E402
    AnnyCloth4DConfig,
    _build_field_bundle,
    _load_conditioner_checkpoint_state,
    _load_complete_obj,
    _prepare_geometry,
)
from aigc_fashion.body_mesh_avatar import BodyMeshAvatar  # noqa: E402
from aigc_fashion.adaptive_intrinsic_mesh import (  # noqa: E402
    AdaptiveAMRConfig,
    AdaptiveAMRStateTransfer,
    adaptive_refine_garment,
)
from aigc_fashion.behavioral_phase import BehavioralPhaseTrajectory  # noqa: E402
from aigc_fashion.data import AvatarSpec, GarmentMesh, MaterialSpec  # noqa: E402
from aigc_fashion.implicit_shell import (  # noqa: E402
    _ordered_boundary_loops,
    build_body_sdf_tshirt_shell,
)
from aigc_fashion.learned_panel_mesh import garment_mesh_from_sewformer_spec  # noqa: E402
from aigc_fashion.mesh_refinement import midpoint_subdivide_garment  # noqa: E402
from aigc_fashion.measured_materials import (  # noqa: E402
    measured_material_profile,
    material_motion_tuning,
)
from aigc_fashion.hypernetwork_kirchhoff_love import BENDING_BRANCH  # noqa: E402
from aigc_fashion.neuralclothsim_dynamics import (  # noqa: E402
    NeuralCloth4DDynamics,
    NeuralCloth4DDynamicsConfig,
    NeuralCloth4DState,
)
from aigc_fashion.patterns import export_obj  # noqa: E402
from aigc_fashion.pleated_skirt import (  # noqa: E402
    build_body_conditioned_pleated_skirt_shell,
    infer_pleated_skirt_spec,
)
from aigc_fashion.render import (  # noqa: E402
    GarmentRenderLayer,
    render_layered_frame,
    render_layered_rollout,
)
from aigc_fashion.rigid_lie_message_passing import SE3JointMessagePassing  # noqa: E402
from aigc_fashion.unified_cloth4d_scene import (  # noqa: E402
    Body4DState,
    Garment4DSystem,
    MovingBodyContactQuery,
    CapturedWalkingSE3BodyTrajectory,
    SinusoidalSE3BodyTrajectory,
    WalkingSE3BodyTrajectory,
    UnifiedCloth4DSceneConfig,
    UnifiedNeuralCloth4DScene,
)
from aigc_fashion.lie_group_diffusion import SE3GenerativeDynamicsHead  # noqa: E402


SHIRT_RUN = ROOT / "outputs" / "neuralclothsim4d_dynamic_final"
DEFAULT_SKIRT_RUN = ROOT / "outputs" / "anny_pleated_skirt_patterncode_v12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one continuous-time Anny + T-shirt + skirt NeuralClothSim scene."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "anny_unified_neuralclothsim4d",
    )
    parser.add_argument(
        "--body-mesh",
        type=Path,
        default=ROOT / "assets" / "body_meshes" / "anny_child_neutral_lowpoly_12k.obj",
    )
    parser.add_argument(
        "--shirt-texture",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--skirt-image",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--skirt-run",
        type=Path,
        default=DEFAULT_SKIRT_RUN,
        help="Validated or continued skirt run containing its intrinsic shell and checkpoint.",
    )
    parser.add_argument(
        "--shirt-run",
        type=Path,
        default=SHIRT_RUN,
        help="Trained T-shirt run containing its intrinsic shell and checkpoint.",
    )
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--frame-dt", type=float, default=0.02)
    parser.add_argument("--internal-dt", type=float, default=0.004)
    parser.add_argument(
        "--contact-constraint-tolerance-m",
        type=float,
        default=0.0033,
        help=(
            "Difference between the 3.5 mm force-contact range and the 0.2 mm "
            "hard non-penetration shell."
        ),
    )
    parser.add_argument(
        "--contact-projection-iterations",
        type=int,
        default=12,
        help="Outer body-contact projection passes per physical substep.",
    )
    parser.add_argument(
        "--phase-hamiltonian-iterations",
        type=int,
        default=5,
        help="Incremental Hamiltonian minimization iterations per substep.",
    )
    parser.add_argument(
        "--phase-hamiltonian-line-search-steps",
        type=int,
        default=6,
        help="Armijo line-search evaluations per Hamiltonian iteration.",
    )
    parser.add_argument(
        "--phase-hamiltonian-stride",
        type=int,
        default=1,
        help=(
            "Run the full Hyper-NODE Hamiltonian every N internal steps; "
            "intervening steps use the bounded E(3) preview flow."
        ),
    )
    parser.add_argument(
        "--initial-contact-equilibration-steps",
        type=int,
        default=2,
        help="Frozen-body Hamiltonian/contact warm-up steps before physical time.",
    )
    parser.add_argument(
        "--metric-projection-iterations",
        type=int,
        default=None,
        help="Optional metric recovery iterations per garment; default keeps the trained setting.",
    )
    parser.add_argument(
        "--enable-metric-projection",
        action="store_true",
        help=(
            "Enable the non-variational metric recovery projector. Disabled by "
            "default so Hyper-NODE and contact impulses own the physical rollout."
        ),
    )
    parser.add_argument("--render-frames", type=int, default=60)
    parser.add_argument("--step-duration-s", type=float, default=0.12)
    parser.add_argument("--walking-steps", type=int, default=1)
    parser.add_argument("--step-length-m", type=float, default=0.02)
    parser.add_argument(
        "--motion-bvh",
        type=Path,
        default=None,
        help="Optional captured BVH walk; retargets root/thigh/upper-arm signals onto Anny.",
    )
    parser.add_argument(
        "--motion-limb-scale",
        type=float,
        default=0.50,
        help="Scale captured leg rotations during retargeting; root travel is unchanged.",
    )
    parser.add_argument(
        "--motion-arm-scale",
        type=float,
        default=1.00,
        help="Scale captured upper-arm sagittal swing independently from the legs.",
    )
    parser.add_argument(
        "--motion-yaw-scale",
        type=float,
        default=0.35,
        help="Scale captured root yaw during retargeting; zero keeps the walk facing forward.",
    )
    parser.add_argument(
        "--body-vertical-bob-mm",
        type=float,
        default=3.0,
        help="Small gait-phase pelvis rise/fall restored after BVH retargeting.",
    )
    parser.add_argument(
        "--body-lateral-sway-mm",
        type=float,
        default=2.0,
        help="Small gait-phase pelvis side shift restored after BVH retargeting.",
    )
    parser.add_argument(
        "--leg-swing-deg",
        type=float,
        default=12.0,
        help="Peak alternating hip/leg swing for the walking trajectory.",
    )
    parser.add_argument(
        "--knee-flex-deg",
        type=float,
        default=34.0,
        help="Peak swing-leg knee flexion in the articulated body trajectory.",
    )
    parser.add_argument(
        "--arm-swing-deg",
        type=float,
        default=11.0,
        help="Peak anti-phase arm swing prior for the walking trajectory.",
    )
    parser.add_argument(
        "--personality-ocean",
        type=float,
        nargs=5,
        metavar=("O", "C", "E", "A", "N"),
        default=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    parser.add_argument(
        "--emotion-vad",
        type=float,
        nargs=3,
        metavar=("V", "A", "D"),
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument(
        "--emotion-amplitude",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--emotion-frequency-hz", type=float, default=0.0)
    parser.add_argument(
        "--shirt-subdivision-levels", type=int, choices=(0, 1, 2), default=1
    )
    parser.add_argument(
        "--tshirt-areal-density-kg-m2",
        type=float,
        default=None,
        help="Optional measured override; otherwise use the inferred cotton-family material card.",
    )
    parser.add_argument(
        "--material-softness-scale",
        type=float,
        default=0.72,
        help=(
            "Scale elastic, bending, and retention stiffness in the physical rollout; "
            "0.72 is the softer walking-cloth preset, while strain/contact guards remain active."
        ),
    )
    parser.add_argument(
        "--body-momentum-transfer-scale",
        type=float,
        default=0.75,
        help=(
            "Scale the moving-body inertial port -m*a_body in the Hamiltonian; "
            "this restores garment momentum without adding wind or anchors."
        ),
    )
    parser.add_argument(
        "--amr-passes",
        type=int,
        default=0,
        help="Optional intrinsic shell AMR passes; disabled on the default physical mesh.",
    )
    parser.add_argument("--amr-refine-fraction", type=float, default=0.12)
    parser.add_argument("--amr-indicator-threshold", type=float, default=0.55)
    parser.add_argument("--amr-max-faces", type=int, default=20000)
    parser.add_argument(
        "--skip-cpu-render",
        action="store_true",
        help="Write simulation data and meshes without CPU GIF/PNG rendering.",
    )
    return parser.parse_args()


def _material_from_summary(run_dir: Path) -> MaterialSpec:
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return MaterialSpec(**payload["garment"]["material"])


def _shirt_mesh_from_validated_shell(
    avatar: BodyMeshAvatar, run_dir: Path
) -> GarmentMesh:
    validated_vertices, faces = _load_complete_obj(run_dir / "intrinsic_rest_shell.obj")
    flat = garment_mesh_from_sewformer_spec(
        ROOT / "examples" / "synthetic_learned_tshirt_spec.json",
        resolution=11,
    )
    semantic_shell, _report = build_body_sdf_tshirt_shell(
        avatar,
        flat,
        target_faces=320,
    )
    # Keep the validated shell topology for checkpoint compatibility, but do
    # not reuse its old world-space vertex placement. Project that topology
    # onto the current body's intrinsic SDF-conditioned shell so shoulders and
    # back are graded from this avatar's measurements.
    semantic_surface = trimesh.Trimesh(
        vertices=np.asarray(semantic_shell.vertices, dtype=np.float32),
        faces=np.asarray(semantic_shell.faces, dtype=np.int32),
        process=False,
    )
    conditioned_vertices, _distance, _triangle_ids = (
        trimesh.proximity.ProximityQuery(semantic_surface).on_surface(
            np.asarray(validated_vertices, dtype=np.float32)
        )
    )
    # A full closest-surface projection can collapse a tiny legacy triangle.
    # Retain part of the validated material metric while moving most of the
    # placement toward the current avatar-conditioned shell.
    conditioned_vertices = np.asarray(validated_vertices, dtype=np.float32) + 0.75 * (
        np.asarray(conditioned_vertices, dtype=np.float32)
        - np.asarray(validated_vertices, dtype=np.float32)
    )
    body_closest, body_distance, body_triangle_ids = avatar._proximity.on_surface(
        conditioned_vertices
    )
    body_normals = np.asarray(
        avatar._collision_mesh.face_normals[body_triangle_ids], dtype=np.float32
    )
    body_normals /= np.maximum(
        np.linalg.norm(body_normals, axis=-1, keepdims=True), 1.0e-8
    )
    body_gap = np.sum(
        (conditioned_vertices - body_closest) * body_normals, axis=-1
    )
    minimum_clearance = max(float(getattr(avatar, "clearance", 0.003)), 0.003)
    conditioned_vertices = conditioned_vertices + np.maximum(
        minimum_clearance - body_gap, 0.0
    )[:, None] * body_normals
    correspondence = np.linalg.norm(
        validated_vertices[:, None, :] - semantic_shell.vertices[None, :, :], axis=-1
    ).argmin(axis=1)
    labels = [semantic_shell.panel_labels[int(index)] for index in correspondence]
    # The implicit shell supplies the garment topology, but a legacy static
    # surface can retain too much wearing ease. Contract only the narrow
    # body-facing band along the exact avatar normal before dynamics. The hem
    # keeps more ease and sleeve openings remain wearable rather than glued.
    signed_distance = avatar._proximity.signed_distance(conditioned_vertices)
    label_array = np.asarray(labels, dtype=str)
    hem_fade = np.clip(
        (conditioned_vertices[:, 1] - float(avatar._shirt_landmarks["hem_y"])) / 0.10,
        0.0,
        1.0,
    )
    target_gap = np.where(
        np.char.find(label_array, "sleeve") >= 0,
        0.006,
        0.006 + 0.004 * (1.0 - hem_fade),
    )
    pull = np.clip(body_gap - target_gap, 0.0, 0.010)
    pull *= (signed_distance < 0.0) & (body_distance < 0.035)
    conditioned_vertices = conditioned_vertices - pull[:, None] * body_normals
    # Re-apply the exact hard side only for vertices already inside the body.
    # This initializes contact; it is not a persistent garment anchor.
    conditioned_tensor, _initial_collision = avatar.project_torch(
        torch.as_tensor(conditioned_vertices, dtype=torch.float32),
        panel_labels=labels,
    )
    conditioned_vertices = conditioned_tensor.detach().cpu().numpy().astype(np.float32)
    uvs = (
        semantic_shell.uvs[correspondence]
        if semantic_shell.uvs is not None
        else None
    )
    return GarmentMesh(
        vertices=np.asarray(conditioned_vertices, dtype=np.float32),
        faces=faces,
        pinned_vertices=np.empty(0, dtype=np.int32),
        panel_labels=labels,
        seam_edges=[],
        rest_vertices=np.asarray(conditioned_vertices, dtype=np.float32).copy(),
        uvs=None if uvs is None else np.asarray(uvs, dtype=np.float32),
    )


def _add_tshirt_neckband_prestrain(
    mesh: GarmentMesh,
    current_positions: np.ndarray,
    *,
    negative_ease_fraction: float = 0.06,
) -> tuple[GarmentMesh, np.ndarray, dict[str, float | str]]:
    """Encode collar-rib negative ease in the material metric, not an anchor."""

    if not 0.0 < negative_ease_fraction < 0.15:
        raise ValueError("neckband negative ease must lie in (0, 0.15)")
    current = np.asarray(current_positions, dtype=np.float32)
    if current.shape != mesh.vertices.shape:
        raise ValueError("T-shirt current state must match its rest shell")
    loops = _ordered_boundary_loops(mesh.faces)
    if len(loops) != 4:
        raise ValueError("implicit T-shirt shell must expose neck, hem, and two cuffs")
    neck = max(
        loops,
        key=lambda loop: float(np.mean(mesh.material_vertices[loop, 1])),
    )
    rest = np.asarray(mesh.material_vertices, dtype=np.float32)
    before = np.linalg.norm(
        np.roll(rest[neck], -1, axis=0) - rest[neck], axis=1
    ).sum()
    after = (1.0 - negative_ease_fraction) * before
    retained = GarmentMesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        pinned_vertices=mesh.pinned_vertices.copy(),
        panel_labels=list(mesh.panel_labels),
        seam_edges=list(mesh.seam_edges),
        rest_vertices=mesh.material_vertices.copy(),
        uvs=None if mesh.uvs is None else mesh.uvs.copy(),
        seam_interfaces=list(mesh.seam_interfaces),
    )
    return retained, current.copy(), {
        "negative_ease_fraction": negative_ease_fraction,
        "neckband_reference_circumference_before_m": float(before),
        "neckband_reference_circumference_after_m": float(after),
        "retention": "collar-rib hoop prestrain plus body SDF contact; no world anchor",
    }


def _skirt_mesh_from_validated_shell(
    avatar: BodyMeshAvatar, skirt_image: Path, run_dir: Path
) -> GarmentMesh:
    template, _report = build_body_conditioned_pleated_skirt_shell(
        avatar, infer_pleated_skirt_spec(skirt_image)
    )
    _validated_vertices, validated_faces = _load_complete_obj(
        run_dir / "intrinsic_rest_shell.obj"
    )
    if (
        validated_faces.shape != template.faces.shape
        or not np.array_equal(validated_faces, template.faces)
    ):
        raise RuntimeError("validated skirt shell no longer matches the garment-family topology")
    # The previous run supplies topology and learned weights only. Its world
    # vertices belong to another body/placement and must never become the new
    # body's material reference configuration.
    vertices = template.vertices.copy()
    faces = template.faces.copy()
    return GarmentMesh(
        vertices=vertices.copy(),
        faces=faces,
        pinned_vertices=np.empty(0, dtype=np.int32),
        panel_labels=list(template.panel_labels),
        seam_edges=[],
        rest_vertices=vertices.copy(),
        uvs=None if template.uvs is None else template.uvs.copy(),
    )


def _add_skirt_walking_ease(
    mesh: GarmentMesh,
    current_positions: np.ndarray,
    *,
    body_center_z: float,
    waistband_prestrain_m: float = 0.003,
    prestrain_fade_fraction: float = 0.38,
) -> tuple[GarmentMesh, np.ndarray, dict[str, float | str]]:
    """Add a fitted waistband reference metric while keeping placement outside."""

    if waistband_prestrain_m <= 0.0 or not 0.0 < prestrain_fade_fraction <= 1.0:
        raise ValueError("waistband prestrain settings are invalid")
    current = np.asarray(current_positions, dtype=np.float32)
    if current.shape != mesh.vertices.shape:
        raise ValueError("skirt current state must match its rest shell")
    eased = GarmentMesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        pinned_vertices=mesh.pinned_vertices.copy(),
        panel_labels=list(mesh.panel_labels),
        seam_edges=list(mesh.seam_edges),
        rest_vertices=mesh.material_vertices.copy(),
        uvs=None if mesh.uvs is None else mesh.uvs.copy(),
        seam_interfaces=list(mesh.seam_interfaces),
    )
    return eased, current.copy(), {
        "waistband_radial_prestrain_m": waistband_prestrain_m,
        "prestrain_fade_fraction": prestrain_fade_fraction,
        "placement_offset_m": 0.0,
        "grading": "smooth hoop prestrain from waistband to upper hip; no world anchor",
    }


def _retention_boundary_edges(mesh: GarmentMesh) -> np.ndarray:
    loops = _ordered_boundary_loops(mesh.faces)
    if not loops:
        return np.empty((0, 2), dtype=np.int64)
    upper = max(
        loops,
        key=lambda loop: float(np.mean(mesh.material_vertices[loop, 1])),
    )
    return np.stack((upper, np.roll(upper, -1)), axis=-1).astype(np.int64)


def _condition_inserted_amr_vertices(
    mesh: GarmentMesh,
    state: AdaptiveAMRStateTransfer,
    *,
    original_vertex_count: int,
    contact_query: MovingBodyContactQuery,
    clearance_m: float = 0.002,
) -> tuple[GarmentMesh, AdaptiveAMRStateTransfer, dict[str, float | int | str]]:
    """Place new curvature samples on the body-conditioned reference shell."""

    if not 0 <= original_vertex_count <= len(mesh.vertices):
        raise ValueError("AMR source vertex count is invalid")
    positions = np.asarray(state.positions, dtype=np.float32).copy()
    inserted = np.arange(original_vertex_count, len(positions), dtype=np.int64)
    if not len(inserted):
        return mesh, state, {
            "conditioned_vertices": 0,
            "maximum_correction_m": 0.0,
            "target_gap_m": clearance_m,
            "role": "no inserted AMR vertices",
        }
    points = torch.as_tensor(positions[inserted], dtype=torch.float32)
    sample = contact_query(points, 0.0)
    normals = torch.nn.functional.normalize(sample.normals.to(points), dim=-1)
    gap = ((points - sample.closest_points.to(points)) * normals).sum(dim=-1)
    correction = torch.relu(float(clearance_m) - gap)[:, None] * normals
    correction_np = correction.detach().cpu().numpy()
    positions[inserted] += correction_np

    base = np.asarray(mesh.vertices, dtype=np.float32).copy()
    rest = np.asarray(mesh.material_vertices, dtype=np.float32).copy()
    base[inserted] += correction_np
    rest[inserted] += correction_np
    conditioned = GarmentMesh(
        vertices=base,
        faces=mesh.faces.copy(),
        pinned_vertices=mesh.pinned_vertices.copy(),
        panel_labels=list(mesh.panel_labels),
        seam_edges=list(mesh.seam_edges),
        rest_vertices=rest,
        uvs=None if mesh.uvs is None else mesh.uvs.copy(),
    )

    triangles = positions[conditioned.faces]
    area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    mass = np.zeros(len(positions), dtype=np.float64)
    for column in range(3):
        np.add.at(mass, conditioned.faces[:, column], area / 3.0)
    mass = np.maximum(mass, 1.0e-12)
    mass *= float(np.sum(state.vertex_mass)) / float(np.sum(mass))
    transferred = AdaptiveAMRStateTransfer(
        positions=positions,
        velocities=np.asarray(state.velocities, dtype=np.float32),
        vertex_mass=mass.astype(np.float32),
        material_field=state.material_field,
        linear_momentum_error=state.linear_momentum_error,
        angular_momentum_error=state.angular_momentum_error,
    )
    correction_norm = np.linalg.norm(correction_np, axis=1)
    return conditioned, transferred, {
        "conditioned_vertices": int(np.count_nonzero(correction_norm > 1.0e-8)),
        "inserted_vertices": int(len(inserted)),
        "maximum_correction_m": float(correction_norm.max(initial=0.0)),
        "target_gap_m": clearance_m,
        "role": "curvature-reconstructed AMR samples conditioned on the body SDF before physical time",
    }


def _system_from_checkpoint(
    name: str,
    run_dir: Path,
    mesh: GarmentMesh,
    avatar: BodyMeshAvatar,
    material: MaterialSpec,
    trajectory: SinusoidalSE3BodyTrajectory,
    internal_dt: float,
    *,
    initial_positions: np.ndarray | None = None,
    initial_velocities: np.ndarray | None = None,
    vertex_mass: np.ndarray | None = None,
    areal_density_override: float | None = None,
    phase_hamiltonian_iterations: int = 5,
    phase_hamiltonian_line_search_steps: int = 6,
    metric_projection_iterations: int | None = None,
    enable_metric_projection: bool = False,
    material_softness_scale: float = 1.0,
    body_momentum_transfer_scale: float = 0.75,
) -> Garment4DSystem:
    if not 0.1 <= material_softness_scale <= 1.0:
        raise ValueError("material_softness_scale must be in [0.1, 1.0]")
    if body_momentum_transfer_scale < 0.0:
        raise ValueError("body_momentum_transfer_scale must be non-negative")
    checkpoint = torch.load(
        run_dir / "e3_anchor_4d_checkpoint.pt", map_location="cpu", weights_only=False
    )
    config = AnnyCloth4DConfig(**checkpoint["config"])
    initial_np = (
        np.asarray(
            np.load(run_dir / "e3_anchor_constrained_rollout.npy")[-1],
            dtype=np.float32,
        )
        if initial_positions is None
        else np.asarray(initial_positions, dtype=np.float32)
    )
    if initial_np.shape != mesh.vertices.shape:
        raise RuntimeError(
            f"{name} checkpoint has {len(initial_np)} vertices but its rest shell has "
            f"{len(mesh.vertices)}"
        )
    geometry = _prepare_geometry(
        mesh,
        avatar,
        initial_np,
        config,
        reference_surface_metric=True,
    )
    bundle = _build_field_bundle(geometry, config)
    _load_conditioner_checkpoint_state(bundle.conditioner, checkpoint["conditioner"])
    bundle.material_latent.load_state_dict(checkpoint["material_latent"])
    bundle.shell_model.load_state_dict(checkpoint["hypernetwork_kirchhoff_love"])
    material_profile = measured_material_profile(
        "cotton_single_jersey_reference"
        if name == "tshirt"
        else "wool_blend_suiting_reference"
    )
    motion_tuning = material_motion_tuning(material.name)
    measured_bending_values = [
        value.midpoint
        for value in (
            material_profile.bending_rigidity_warp,
            material_profile.bending_rigidity_weft,
        )
        if value is not None
    ]
    bending_force_scale = 1.0
    if measured_bending_values:
        with torch.no_grad():
            shell_probe = bundle.shell_model(
                torch.as_tensor(initial_np, dtype=torch.float32),
                geometry.shell,
                bundle.material_latent().detach(),
                compute_internal_forces=False,
                compute_bending_forces=False,
            )
            learned_bending = shell_probe.hinge_stiffness[:, BENDING_BRANCH]
            learned_bending = learned_bending[torch.isfinite(learned_bending)]
            if learned_bending.numel():
                learned_mean = float(learned_bending.mean())
                target_mean = float(np.mean(measured_bending_values))
                bending_force_scale = float(
                    np.clip(target_mean / max(learned_mean, 1.0e-12), 0.25, 8.0)
                )
    if name == "pleated_skirt":
        # The photographed garment is structured suiting. Its measured mass
        # card plus the published bending envelope already carry the family
        # difference; an extra multiplier here over-constrained the waistband
        # during gait contact.
        bending_force_scale *= 1.0
    areal_density = (
        float(motion_tuning.areal_density_kg_m2)
        if areal_density_override is None
        else float(areal_density_override)
    )
    if areal_density <= 0.0:
        raise ValueError("areal density must be positive")
    dynamics = NeuralCloth4DDynamics(
        bundle.shell_model,
        geometry.shell,
        rest_positions=torch.as_tensor(mesh.material_vertices, dtype=torch.float32),
        edge_index=geometry.edge_index,
        rest_lengths=geometry.rest_lengths,
        cotangent_weights=geometry.cotangent_weights,
        pinned_vertices=torch.as_tensor(mesh.pinned_vertices, dtype=torch.long),
        laplace_eigenvalues=geometry.laplace_eigenvalues,
        laplace_eigenvectors=geometry.laplace_eigenvectors,
        vertex_mass=(
            geometry.vertex_mass
            if vertex_mass is None
            else torch.as_tensor(vertex_mass, dtype=torch.float32)
        ),
        seam_pairs=geometry.seam_pairs,
        retention_edges=torch.as_tensor(
            _retention_boundary_edges(mesh), dtype=torch.long
        ),
        retention_rest_scale=(0.94 if name == "tshirt" else 0.97),
        material=material,
        residual_model=bundle.conditioner.particle,
        config=NeuralCloth4DDynamicsConfig(
            dt=internal_dt,
            substeps_per_frame=1,
            areal_density_kg_m2=areal_density,
            # This is a material modulus multiplier, not a render-time
            # smoother. Hyper-NODE still owns the strain envelope and the
            # contact solver still owns non-penetration.
            shell_force_scale=material_softness_scale,
            maximum_acceleration_m_s2=min(
                config.dynamic_maximum_acceleration_m_s2, 30.0
            ),
            # This is the material's modal Rayleigh dissipation, not an
            # export-time trajectory filter.
            velocity_damping_per_second=float(
                max(
                    # Convert the legacy per-step loss using the actual
                    # physical clock. The old fixed 8 ms denominator made
                    # the cotton shirt overdamped at the 20 ms walk step and
                    # erased the body's incoming momentum.
                    -np.log(material.damping) / max(2.0 * internal_dt, 1.0e-6),
                    motion_tuning.velocity_damping_per_second,
                )
            ),
            maximum_velocity_m_s=0.55,
            initial_shell_acceleration_target_m_s2=(
                config.dynamic_initial_shell_acceleration_target_m_s2
            ),
            residual_force_scale=min(
                config.dynamic_residual_force_scale,
                motion_tuning.residual_force_scale,
            ),
            residual_acceleration_limit_m_s2=(
                config.dynamic_residual_acceleration_limit_m_s2
            ),
            minimum_constitutive_force_scale=motion_tuning.constitutive_force_floor,
            # Keep the Hyper-NODE envelope active, but scale its finite
            # barrier by fabric family. The previous common value made the
            # wool pleat shell dominate the incremental action and created
            # upward corrective kicks before the hard envelope was reached.
            strain_limit_barrier_scale=(0.01 if name == "tshirt" else 0.004),
            bending_force_scale=bending_force_scale * material_softness_scale,
            bending_damping_per_second=motion_tuning.bending_damping_per_second,
            initial_shell_relaxation_force_scale=0.25,
            # Keep a visible cloth-thickness gap while preventing shoulder,
            # axilla, waist, and thigh vertices from settling into the body.
            contact_clearance_m=0.0035,
            # The smooth area-measure barrier should resolve ordinary body
            # approach inside the Hamiltonian. The non-smooth 1.5 mm shell is
            # only the final non-penetration guard, not the main force law.
            contact_pressure_stiffness_n_m3=2.0e6,
            friction_coefficient=motion_tuning.friction_coefficient,
            contact_damping_n_s_m=motion_tuning.contact_damping_n_s_m,
            seam_damping_n_s_m=motion_tuning.seam_damping_n_s_m,
            retention_band_axial_rigidity_n=(
                (3.0 if name == "tshirt" else 12.0) * material_softness_scale
            ),
            retention_band_damping_n_s_m=0.04,
            metric_projection_iterations=(
                max(96, 2 * config.teacher_edge_iterations)
                if metric_projection_iterations is None
                else metric_projection_iterations
            ),
            metric_projection_tolerance=0.002 if name == "tshirt" else 0.005,
            # The hard material envelope remains active in Hyper-NODE. This
            # The hard material envelope is already part of the Hyper-NODE
            # Hamiltonian. Metric projection is only an emergency recovery;
            # using a lower skirt threshold made the pleat solver inject a
            # corrective impulse on nearly every substep and looked like lift.
            # Layer contact can create a short-lived local metric excursion
            # at the waistband. Keep the emergency projector for genuinely
            # large failures, but do not reject a positive-gap state merely
            # because this interface crosses the old 0.30 cutoff.
            metric_projection_severe_violation=(0.30 if name == "pleated_skirt" else 0.30),
            # The long shared-scene rollout uses a bounded implicit solve. The
            # learned KL energy owns material recovery; metric projection is
            # opt-in because it is outside the Hamiltonian flow.
            enable_metric_projection=enable_metric_projection,
            phase_hamiltonian_iterations=phase_hamiltonian_iterations,
            phase_hamiltonian_line_search_steps=phase_hamiltonian_line_search_steps,
            # This bounds one nonlinear search direction, not the physical
            # frame displacement. At 20 ms the old 0.5 mm radius prevented the
            # variational solve from following a normal walking contact and
            # left the skirt at its rigid transport target with zero local KE.
            phase_hamiltonian_trust_radius_m=(
                min(0.002, max(0.0005, 0.08 * internal_dt))
                if not enable_metric_projection
                else 0.003
            ),
            # The score head may propose a body trajectory, but it must not
            # inject an unmeasured silk-like drive into cloth momentum.
            se3_generative_potential_weight=motion_tuning.se3_generative_potential_weight,
            body_inertial_force_scale=body_momentum_transfer_scale,
        ),
    )
    if not enable_metric_projection:
        # Make only the imported t=0 shell feasible. The projector is never
        # called by the physical rollout; subsequent material recovery must
        # come from Hyper-NODE and the Hamiltonian action solve.
        initial_tensor = torch.as_tensor(initial_np, dtype=torch.float32)
        with torch.no_grad():
            initial_tensor = dynamics.metric_projector(
                initial_tensor,
                pinned_positions=initial_tensor[dynamics.pinned_vertices],
            )
        initial_np = initial_tensor.cpu().numpy()
    return Garment4DSystem(
        name=name,
        dynamics=dynamics,
        material_latent=bundle.material_latent().detach(),
        initial_positions=torch.as_tensor(initial_np),
        initial_velocities=(
            torch.zeros_like(torch.as_tensor(initial_np))
            if initial_velocities is None
            else torch.as_tensor(initial_velocities, dtype=torch.float32)
        ),
        faces=torch.as_tensor(mesh.faces, dtype=torch.long),
        attachment_trajectory=(
            trajectory.transform_points if len(mesh.pinned_vertices) else None
        ),
    )


def _mesh_at(mesh: GarmentMesh, vertices: np.ndarray) -> GarmentMesh:
    return GarmentMesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=mesh.faces.copy(),
        pinned_vertices=mesh.pinned_vertices.copy(),
        panel_labels=list(mesh.panel_labels),
        seam_edges=list(mesh.seam_edges),
        rest_vertices=mesh.rest_vertices,
        uvs=mesh.uvs,
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    avatar_spec = AvatarSpec(
        chest_cm=64.0,
        waist_cm=56.0,
        shoulder_cm=30.0,
        height_cm=128.0,
        pose="a-pose",
    )
    avatar = BodyMeshAvatar.from_obj(
        args.body_mesh,
        avatar_spec,
        source_up_axis="z",
        source_front_axis="-y",
    )
    body_origin = torch.as_tensor(np.mean(avatar.mesh_vertices, axis=0), dtype=torch.float32)
    behavior_profile = BehavioralPhaseTrajectory(
        personality_ocean=tuple(args.personality_ocean),
        emotion_vad_base=tuple(args.emotion_vad),
        emotion_vad_amplitude=tuple(args.emotion_amplitude),
        emotion_frequency_hz=args.emotion_frequency_hz,
    )
    trajectory_kwargs = dict(
        step_duration_s=args.step_duration_s,
        steps=args.walking_steps,
        step_length_m=args.step_length_m,
        leg_swing_rad=np.deg2rad(args.leg_swing_deg),
        knee_flex_rad=np.deg2rad(args.knee_flex_deg),
        knee_y=(
            float(avatar.mesh_vertices[:, 1].min())
            + 0.34
            * float(np.ptp(avatar.mesh_vertices[:, 1]))
        ),
        arm_swing_rad=np.deg2rad(args.arm_swing_deg),
        vertical_bob_m=args.body_vertical_bob_mm * 1.0e-3,
        lateral_sway_m=args.body_lateral_sway_mm * 1.0e-3,
        hip_y=avatar.center_y - avatar.torso_half_height,
        shoulder_y=float(avatar._shirt_landmarks["shoulder_y"]),
        shoulder_half_width_m=float(avatar._shirt_landmarks["shoulder_half"]),
        body_center_z=avatar.base_z,
        behavior_profile=behavior_profile,
    )
    if args.motion_bvh is not None:
        trajectory = CapturedWalkingSE3BodyTrajectory(
            body_origin,
            str(args.motion_bvh),
            motion_limb_scale=args.motion_limb_scale,
            motion_arm_scale=args.motion_arm_scale,
            motion_yaw_scale=args.motion_yaw_scale,
            **trajectory_kwargs,
        )
    else:
        trajectory = WalkingSE3BodyTrajectory(body_origin, **trajectory_kwargs)
    contact_query = MovingBodyContactQuery(avatar, trajectory)
    shirt_material_evidence = measured_material_profile(
        "cotton_single_jersey_reference"
    )
    skirt_material_evidence = measured_material_profile(
        "wool_blend_suiting_reference"
    )
    with torch.random.fork_rng():
        torch.manual_seed(20260718)
        rigid_lie_model = SE3JointMessagePassing(
            node_feature_dim=8,
            joint_feature_dim=4,
            hidden_dim=16,
            spectral_cap=0.9,
            wrench_limit=1.0,
        ).eval()
        se3_generative_model = SE3GenerativeDynamicsHead(
            context_dim=8 + 8 + 4,
            diffusion_steps=8,
            rotation_scale=0.3,
            translation_scale=0.03,
            score_scale=1.0,
            hidden_dim=16,
            spectral_cap=0.9,
        ).eval()
    shirt_mesh = _shirt_mesh_from_validated_shell(avatar, args.shirt_run)
    # The old scene rollout predates the Hyper-NODE material metric and can be
    # far outside its feasible set. Start from the body-conditioned intrinsic
    # shell; load learned weights, never stale deformed coordinates.
    shirt_reference_vertices_before_refinement = len(shirt_mesh.vertices)
    shirt_initial = shirt_mesh.vertices.copy()
    shirt_mesh, shirt_initial, shirt_refinement = midpoint_subdivide_garment(
        shirt_mesh,
        shirt_initial,
        levels=args.shirt_subdivision_levels,
    )
    shirt_mesh, shirt_initial, shirt_neckband_prestrain = (
        _add_tshirt_neckband_prestrain(shirt_mesh, shirt_initial)
    )
    skirt_mesh = _skirt_mesh_from_validated_shell(
        avatar, args.skirt_image, args.skirt_run
    )
    skirt_initial = skirt_mesh.vertices.copy()
    skirt_mesh, skirt_initial, skirt_walking_ease = _add_skirt_walking_ease(
        skirt_mesh,
        skirt_initial,
        body_center_z=avatar.base_z,
    )
    amr_reports: dict[str, dict[str, object]] = {}
    if args.amr_passes:
        if args.amr_passes < 0:
            raise ValueError("--amr-passes must be non-negative")
        amr_config = AdaptiveAMRConfig(
            passes=args.amr_passes,
            refine_fraction=args.amr_refine_fraction,
            indicator_threshold=args.amr_indicator_threshold,
            maximum_faces=args.amr_max_faces,
            # Rest curvature has priority at pressed pleats, side silhouette,
            # axilla and hem. Strain/contact still trigger refinement once a
            # dynamic residual field is supplied by a later AMR event.
            curvature_weight=3.0,
            strain_weight=1.0,
            material_gradient_weight=1.0,
            contact_weight=1.0,
            energy_residual_weight=1.0,
        )

        def contact_gap_for(mesh: GarmentMesh, positions: np.ndarray) -> np.ndarray:
            sample = contact_query(
                torch.as_tensor(positions, dtype=torch.float32), 0.0
            )
            point = torch.as_tensor(positions, dtype=torch.float32)
            return (
                (point - sample.closest_points) * sample.normals
            ).sum(dim=-1).detach().cpu().numpy()

        shirt_mesh, shirt_state, shirt_amr = adaptive_refine_garment(
            shirt_mesh,
            shirt_initial,
            contact_gap=contact_gap_for(shirt_mesh, shirt_initial),
            config=amr_config,
        )
        shirt_mesh, shirt_state, shirt_amr_conditioning = (
            _condition_inserted_amr_vertices(
                shirt_mesh,
                shirt_state,
                original_vertex_count=shirt_reference_vertices_before_refinement,
                contact_query=contact_query,
            )
        )
        shirt_initial = shirt_state.positions
        shirt_amr_velocity = shirt_state.velocities
        amr_reports["tshirt"] = {
            **shirt_amr.to_dict(),
            "reference_surface_conditioning": shirt_amr_conditioning,
        }
        skirt_vertices_before_amr = len(skirt_mesh.vertices)
        skirt_mesh, skirt_state, skirt_amr = adaptive_refine_garment(
            skirt_mesh,
            skirt_initial,
            contact_gap=contact_gap_for(skirt_mesh, skirt_initial),
            config=amr_config,
        )
        skirt_mesh, skirt_state, skirt_amr_conditioning = (
            _condition_inserted_amr_vertices(
                skirt_mesh,
                skirt_state,
                original_vertex_count=skirt_vertices_before_amr,
                contact_query=contact_query,
            )
        )
        skirt_initial = skirt_state.positions
        skirt_amr_velocity = skirt_state.velocities
        amr_reports["pleated_skirt"] = {
            **skirt_amr.to_dict(),
            "reference_surface_conditioning": skirt_amr_conditioning,
        }
    else:
        shirt_amr_velocity = np.zeros_like(shirt_initial)
        skirt_amr_velocity = np.zeros_like(skirt_initial)
        amr_reports = {"enabled": False}
    systems = [
        _system_from_checkpoint(
            "tshirt",
            args.shirt_run,
            shirt_mesh,
            avatar,
            _material_from_summary(args.shirt_run),
            trajectory,
            args.internal_dt,
            initial_positions=shirt_initial,
            initial_velocities=shirt_amr_velocity,
            vertex_mass=(shirt_state.vertex_mass if args.amr_passes else None),
            areal_density_override=args.tshirt_areal_density_kg_m2,
            phase_hamiltonian_iterations=args.phase_hamiltonian_iterations,
            phase_hamiltonian_line_search_steps=args.phase_hamiltonian_line_search_steps,
            metric_projection_iterations=args.metric_projection_iterations,
            enable_metric_projection=args.enable_metric_projection,
            material_softness_scale=args.material_softness_scale,
            body_momentum_transfer_scale=args.body_momentum_transfer_scale,
        ),
        _system_from_checkpoint(
            "pleated_skirt",
            args.skirt_run,
            skirt_mesh,
            avatar,
            _material_from_summary(args.skirt_run),
            trajectory,
            args.internal_dt,
            initial_positions=skirt_initial,
            initial_velocities=skirt_amr_velocity,
            vertex_mass=(skirt_state.vertex_mass if args.amr_passes else None),
            # Use the wool-family card unless the user supplies a swatch
            # measurement.  The previous fixed 0.220 kg/m2 was too light for
            # this structured pleated suiting skirt.
            areal_density_override=None,
            phase_hamiltonian_iterations=args.phase_hamiltonian_iterations,
            phase_hamiltonian_line_search_steps=args.phase_hamiltonian_line_search_steps,
            metric_projection_iterations=args.metric_projection_iterations,
            enable_metric_projection=args.enable_metric_projection,
            material_softness_scale=args.material_softness_scale,
            body_momentum_transfer_scale=args.body_momentum_transfer_scale,
        ),
    ]

    body_reference = torch.as_tensor(
        np.asarray(avatar.mesh_vertices, dtype=np.float32)
    )
    body_vertex_mass = 35.0 / max(len(body_reference), 1)

    def body_phase_state(time: float) -> Body4DState:
        """Expose the same articulated body mesh as the scene root q_b/p_b."""

        positions = trajectory.transform_body_points(body_reference, time)
        # q_b and p_b must use the same articulated map. The root-only
        # point_velocities() omits captured leg/arm joint motion and makes the
        # body Hamiltonian inconsistent with the body mesh that drives SDF
        # contact.
        velocities = trajectory.body_point_velocities(body_reference, time)
        momentum = body_vertex_mass * velocities
        return Body4DState(positions, velocities, momentum, float(time))

    def scene_phase_force_provider(
        name: str,
        state: NeuralCloth4DState,
        all_states: dict[str, NeuralCloth4DState],
        body_state: Body4DState | None,
        time: float,
    ) -> torch.Tensor:
        """Stable graph-to-physics baseline for body -> garment phase flow.

        The learned attention port is enabled only after calibration. Until
        then, the hierarchy edge transfers the body's tangential contact
        velocity, while the SDF owns normal impulses and non-penetration.
        """
        del all_states
        system = next(item for item in systems if item.name == name)
        sample = contact_query(state.positions.detach(), time)
        normals = torch.nn.functional.normalize(sample.normals.to(state.positions), dim=-1)
        body_velocity = (
            torch.zeros_like(state.positions)
            if sample.body_velocities is None
            else sample.body_velocities.to(state.positions)
        )
        relative = body_velocity - state.velocities
        tangential = relative - (relative * normals).sum(dim=-1, keepdim=True) * normals
        gap = ((state.positions - sample.closest_points.to(state.positions)) * normals).sum(dim=-1)
        # The SDF contact solver is intentionally strict and only activates
        # at the cloth clearance. A lifted thigh can still transfer momentum
        # to a nearby skirt hem before literal contact, so this phase port is
        # a distance-weighted tangential body coupling, not a position anchor
        # or a wind term. The skirt gets a slightly stronger textile/body
        # coupling than the shirt because its hem crosses the moving legs.
        local_body_momentum_radius_m = 0.018
        local_body_momentum_coupling = (
            0.55 if name == "pleated_skirt" else 0.42
        )
        contact_weight = torch.exp(
            -torch.relu(gap) / local_body_momentum_radius_m
        ).clamp(0.0, 1.0)
        mass = system.dynamics._vertex_mass(state.positions)
        body_phase_drag = torch.zeros_like(state.positions)
        if body_state is not None:
            # The exact local body velocity comes from the SDF query. The
            # small COM term couples local contact friction to the explicit
            # body root momentum. Global garment transport is handled by the
            # moving-reference Hamiltonian, not by this dissipative port.
            com_velocity = body_state.velocities.mean(dim=0)
            body_phase_drag = com_velocity[None, :] - state.velocities.mean(dim=0, keepdim=True)
        # This is a dissipative momentum port, not a position correction.
        contact_port = contact_weight[:, None] * (
            local_body_momentum_coupling * tangential + 0.02 * body_phase_drag
        )
        return mass[:, None] * contact_port

    scene_config = UnifiedCloth4DSceneConfig(
        frame_count=args.frames,
        frame_dt=args.frame_dt,
        internal_dt=args.internal_dt,
        # Keep the validated body-contact tolerance. Layer separation is
        # reported separately until triangle-level CCD/IPC is available.
        inter_garment_clearance_m=0.006,
        contact_constraint_tolerance_m=args.contact_constraint_tolerance_m,
        # The narrow phase itself is exact; twelve outer passes are enough to
        # settle shared body/garment contact without repeating thousands of
        # identical trimesh queries on a long rollout.
        contact_projection_iterations=args.contact_projection_iterations,
        initial_contact_equilibration_steps=args.initial_contact_equilibration_steps,
        phase_hamiltonian_stride=args.phase_hamiltonian_stride,
    )
    rollout = UnifiedNeuralCloth4DScene(
        systems,
        body_contact_query=contact_query,
        rigid_lie_message_model=rigid_lie_model,
        rigid_link_graph_query=trajectory.lie_message_graph,
        scene_phase_force_provider=scene_phase_force_provider,
        body_phase_state_provider=body_phase_state,
        body_root_transport_provider=trajectory.root_transport,
        se3_generative_dynamics=se3_generative_model,
        config=scene_config,
    ).rollout()

    physical_times = rollout.fields["tshirt"].times.detach().cpu().numpy()
    body_knots = np.stack(
        [
            trajectory.transform_body_points(
                torch.as_tensor(avatar.mesh_vertices), float(time)
            ).numpy()
            for time in physical_times
        ]
    )
    body_velocity_knots = np.stack(
        [
            trajectory.body_point_velocities(body_reference, float(time)).numpy()
            for time in physical_times
        ]
    )
    field_path = args.out_dir / "unified_neuralclothsim4d_field.npz"
    np.savez_compressed(
        field_path,
        physical_times_s=physical_times,
        tshirt_positions=rollout.fields["tshirt"].positions.detach().numpy(),
        tshirt_velocities=rollout.fields["tshirt"].velocities.detach().numpy(),
        tshirt_faces=shirt_mesh.faces,
        tshirt_uvs=np.asarray(shirt_mesh.uvs, dtype=np.float32),
        tshirt_printed_faces=np.all(
            np.asarray(
                [label.startswith("front_front_body") for label in shirt_mesh.panel_labels],
                dtype=bool,
            )[shirt_mesh.faces],
            axis=1,
        ),
        pleated_skirt_positions=rollout.fields["pleated_skirt"].positions.detach().numpy(),
        pleated_skirt_velocities=rollout.fields["pleated_skirt"].velocities.detach().numpy(),
        pleated_skirt_faces=skirt_mesh.faces,
        pleated_skirt_uvs=np.asarray(skirt_mesh.uvs, dtype=np.float32),
        body_positions=body_knots,
        body_velocities=body_velocity_knots,
    )
    physics_report_path = args.out_dir / "physics_report.json"
    physics_report = rollout.report.to_dict()
    physics_report["metric_projection_policy"] = {
        "runtime_enabled": bool(args.enable_metric_projection),
        "initial_shell_only": not bool(args.enable_metric_projection),
        "hamiltonian_path_uses_projection": False,
        "material_recovery": "Hyper-NODE strain-limit energy and incremental Hamiltonian solve",
    }
    physics_report["intrinsic_amr"] = amr_reports
    trajectory_limit_report: dict[str, dict[str, float | int]] = {}
    for system in systems:
        field = rollout.fields[system.name]
        samples = [
            system.dynamics.hypernode_strain_limit_diagnostics(frame)
            for frame in field.positions
        ]
        peak_index = max(
            range(len(samples)),
            key=lambda index: float(samples[index]["maximum_limit_violation"]),
        )
        peak = dict(samples[peak_index])
        peak["frame_index"] = peak_index
        peak["physical_time_s"] = float(field.times[peak_index])
        trajectory_limit_report[system.name] = peak
    physics_report["trajectory_hypernode_strain_limit_peak"] = trajectory_limit_report
    physics_report_path.write_text(json.dumps(physics_report, indent=2), encoding="utf-8")

    # Use the stored physical knot as the endpoint.  Recomputing the duration
    # from frame_count*dt can overshoot the final knot by one float32 ulp and
    # make the continuous field query reject the last render sample.
    dense_times = np.linspace(0.0, float(physical_times[-1]), args.render_frames)
    dense_shirt = [rollout.fields["tshirt"].query(float(time)).detach().numpy() for time in dense_times]
    dense_skirt = [
        rollout.fields["pleated_skirt"].query(float(time)).detach().numpy()
        for time in dense_times
    ]
    dense_body = [
        trajectory.transform_body_points(
            torch.as_tensor(avatar.mesh_vertices), float(time)
        ).numpy()
        for time in dense_times
    ]
    gif_path = args.out_dir / "anny_unified_neuralclothsim4d.gif"
    final_png = args.out_dir / "anny_unified_neuralclothsim4d_final.png"
    if not args.skip_cpu_render:
        shirt_texture = Image.open(args.shirt_texture).convert("RGB")
        skirt_texture_path = args.skirt_run / "image_conditioned_plaid.png"
        skirt_texture = Image.open(skirt_texture_path).convert("RGB")
        render_layered_rollout(
            gif_path,
            [
                (dense_skirt, skirt_mesh, skirt_texture, False),
                (dense_shirt, shirt_mesh, shirt_texture, True),
            ],
            "continuous X_g(u,v,t), shared SE(3) body/contact clock, Hyper-NODE KL materials",
            avatar=avatar,
            avatar_vertex_rollout=dense_body,
            duration_ms=70,
            method_label="unified continuous NeuralClothSim 4D",
            view_yaw_rad=np.deg2rad(40.0),
        )
        render_layered_frame(
            [
                GarmentRenderLayer(dense_skirt[-1], skirt_mesh, skirt_texture),
                GarmentRenderLayer(
                    dense_shirt[-1],
                    shirt_mesh,
                    shirt_texture,
                    closed_sleeve_visibility=True,
                ),
            ],
            args.render_frames - 1,
            "continuous X_g(u,v,t), shared SE(3) body/contact clock, Hyper-NODE KL materials",
            avatar=avatar,
            avatar_vertices=dense_body[-1],
            method_label="unified continuous NeuralClothSim 4D",
            view_yaw_rad=np.deg2rad(40.0),
        ).save(final_png)
    export_obj(args.out_dir / "tshirt_final.obj", _mesh_at(shirt_mesh, dense_shirt[-1]))
    export_obj(args.out_dir / "pleated_skirt_final.obj", _mesh_at(skirt_mesh, dense_skirt[-1]))

    manifest = {
        "status": "completed",
        "definition_of_4d": "continuous material-surface embedding X_g(u,v,t)->R3",
        "scene_config": asdict(scene_config),
        "body_motion": {
            "group": (
                "captured BVH retargeted articulated SE(3) walking trajectory"
                if args.motion_bvh is not None
                else "articulated SE(3) walking trajectory"
            ),
            "period_s": trajectory.period_s,
            "steps": trajectory.steps,
            "translation_amplitude_m": trajectory.translation_amplitude_m,
            "yaw_amplitude_rad": trajectory.yaw_amplitude_rad,
            "step_duration_s": trajectory.step_duration_s,
            "step_length_m": trajectory.step_length_m,
            "leg_swing_rad": trajectory.leg_swing_rad,
            "arm_swing_rad": trajectory.arm_swing_rad,
            "capture": (
                trajectory.capture_summary()
                if hasattr(trajectory, "capture_summary")
                else None
            ),
            "moving_reference": (
                "canonical body-root SE(3) displacement in the incremental "
                "Hamiltonian; garment momentum is relative to that frame"
            ),
            "collision": "inverse-skinned moving-body SDF with transformed normals/velocities",
            "body_collision_mesh": {
                "path": str(args.body_mesh),
                "vertices": int(len(avatar.mesh_vertices)),
                "faces": int(len(avatar.mesh_faces)),
                "role": "保形低面数碰撞与渲染人体；服装材料网格不变",
            },
            "behavior_control_port": {
                "profile": asdict(behavior_profile),
                "bounded_controls": asdict(behavior_profile.body_control_port(0.0)),
                "energy_path": "person/body control -> SE(3) motion -> SDF contact -> cloth momentum",
                "status": "engineering prior, not a validated psychological measurement model",
            },
            "local_body_momentum_port": {
                "radius_m": 0.018,
                "coupling_per_second": {
                    "tshirt": 0.42,
                    "pleated_skirt": 0.55,
                },
                "force_type": "distance-weighted tangential body velocity difference",
                "position_anchor": False,
                "wind_force": False,
            },
        },
        "wearing_constraints": {
            "tshirt_hard_anchors": 0,
            "tshirt_retention": (
                "6% collar-rib negative ease + intrinsic 1D trim material + "
                "signed moving-body SDF and static/Coulomb friction"
            ),
            "skirt_hard_anchors": 0,
            "skirt_retention": (
                "closed prestrained waistband + intrinsic 1D trim material + "
                "signed moving-body SDF and body-shape locking"
            ),
            "boundary_chord_sdf_tolerance_m": 0.00015,
            "boundary_chord_note": (
                "linear boundary-edge samples may lie slightly inside a curved SDF "
                "while both simulation vertices remain outside"
            ),
        },
        "tshirt_simulation_mesh": shirt_refinement.to_dict(),
        "initial_state_source": (
            "body-conditioned intrinsic shells; legacy projected rollout coordinates rejected"
        ),
        "runtime_material": {
            "material_softness_scale": args.material_softness_scale,
            "body_momentum_transfer_scale": args.body_momentum_transfer_scale,
            "tshirt_areal_density_kg_m2": systems[0].dynamics.config.areal_density_kg_m2,
            "pleated_skirt_areal_density_kg_m2": systems[1].dynamics.config.areal_density_kg_m2,
            "evidence_status": "literature priors; exact photographed swatches are unmeasured",
            "tshirt_evidence": shirt_material_evidence.to_dict(),
            "pleated_skirt_evidence": skirt_material_evidence.to_dict(),
            "inter_garment_contact": {
                "clearance_m": scene_config.inter_garment_clearance_m,
                "penalty_stiffness_n_m": scene_config.inter_garment_contact_stiffness_n_m,
                "friction_coefficient": scene_config.inter_garment_friction,
                "energy_audit_term": "inter_garment_contact_j",
                "final_projection": "sampled layer impulse plus velocity friction",
            },
            "metric_projection_role": "numerical/failure guard, not normal elastic response",
            "retention_band_axial_rigidity_n": {
                system.name: system.dynamics.config.retention_band_axial_rigidity_n
                for system in systems
            },
            "velocity_damping_per_second": {
                system.name: system.dynamics.config.velocity_damping_per_second
                for system in systems
            },
            "bending_damping_per_second": {
                system.name: system.dynamics.config.bending_damping_per_second
                for system in systems
            },
            "constitutive_force_scale_floor": {
                system.name: system.dynamics.config.minimum_constitutive_force_scale
                * system.dynamics.config.shell_force_scale
                for system in systems
            },
            "bending_force_scale_from_measured_rigidity": {
                system.name: system.dynamics.config.bending_force_scale
                for system in systems
            },
            "effective_constitutive_force_scale": {
                name: float(scale) for name, scale in rollout.report.constitutive_force_scale.items()
            },
        },
        "skirt_walking_ease": skirt_walking_ease,
        "tshirt_neckband_prestrain": shirt_neckband_prestrain,
        "intrinsic_deformation_operator": {
            "operator": "cotangent Laplace-Beltrami heat kernel",
            "formula": "Phi exp(-tau Lambda) Phi^T M f",
            "reference_surface": "subdivided intrinsic rest shell",
            "recomputed_after_subdivision": True,
            "parameterization_independent": True,
            "eigenbasis_size": int(systems[0].dynamics.laplace_eigenvalues.numel()),
            "maximum_eigenvalue_m-2": float(
                systems[0].dynamics.laplace_eigenvalues.max().detach()
            ),
            "diffusion_time_m2": (
                systems[0].dynamics.config.laplace_beltrami_diffusion_time_m2
            ),
            "role": (
                "intrinsic propagation of the bounded E(3) residual force; "
                "Hyper-NODE Kirchhoff-Love remains the authoritative shell law"
            ),
        },
        "rendering": {
            "camera": "40 degree three-quarter view exposing sagittal walking motion",
            "skirt_normals": "area-weighted smooth vertex normals",
            "skirt_shading": "barycentrically interpolated Lambert shading",
            "geometry_source": "unchanged physical 4D skirt trajectory",
        },
        "report": rollout.report.to_dict(),
        "artifacts": {
            "continuous_field": str(field_path),
            "physics_report": str(physics_report_path),
            "animation": None if args.skip_cpu_render else str(gif_path),
            "final_png": None if args.skip_cpu_render else str(final_png),
            "tshirt_final_obj": str(args.out_dir / "tshirt_final.obj"),
            "pleated_skirt_final_obj": str(args.out_dir / "pleated_skirt_final.obj"),
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["artifacts"], indent=2))


if __name__ == "__main__":
    main()
