"""Single-image body reconstruction: SAM 3D Body -> MHR -> simulation-ready avatar.

SAM 3D Body (Meta) predicts, for each person in a photo, the parameters of the
Momentum Human Rig (MHR): 45 identity coefficients, 68 skeletal scales and a
136-value pose. MHR decouples skeleton and surface shape, which is what this
module relies on: it keeps the person's identity and bone scales, discards the
photographed pose, and re-evaluates MHR in its rest pose. That rest pose is
already an A-pose (upper arms about 48 degrees below horizontal), which is what
the cloth pipeline expects, and its arms are clear of the torso cross-sections
used for girth measurement. The body is exported as an OBJ in metres with +Y up
and +Z front (``BodyMeshAvatar.from_obj(..., source_up_axis="y", source_front_axis="z")``).

Licences: MHR code and model assets are Apache-2.0. SAM 3D Body code and weights
are under the SAM License and its checkpoints are gated on Hugging Face; this
module only calls SAM 3D Body through ``scripts/sam3d_body_infer.py`` in a
separate environment, or consumes a JSON of its predictions.

Single-view limits: metric scale comes from the model's learned priors and the
estimated camera, clothing hides the body surface, and girths are measured on
the parametric surface. Treat the measurements as estimates with uncertainty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MHR_ASSETS_URL = "https://github.com/facebookresearch/MHR/releases/latest/download/assets.zip"

NUM_IDENTITY = 45
NUM_POSE = 136  # 3 root translation + 3 root rotation + 130 body joints
NUM_SCALE = 68
NUM_MODEL_PARAMETERS = NUM_POSE + NUM_SCALE
NUM_EXPRESSION = 72
# Left/right shoulder (glenohumeral) joints of the MHR LOD-1 skeleton; fixed across identities.
MHR_SHOULDER_JOINTS = (75, 39)



def default_mhr_model_path() -> Path:
    return Path(os.environ.get("MHR_MODEL_PATH", ROOT / "work" / "mhr" / "assets" / "mhr_model.pt"))


def ensure_mhr_model(path: Path | None = None) -> Path:
    """Return the MHR TorchScript model, downloading the Apache-2.0 release assets if missing."""

    path = Path(path) if path is not None else default_mhr_model_path()
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(MHR_ASSETS_URL, headers={"User-Agent": "aigc-fashion"})
    with urllib.request.urlopen(request, timeout=600) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for member in ("assets/mhr_model.pt", "assets/LICENSE.txt"):
        (path.parent / Path(member).name).write_bytes(archive.read(member))
    if path.name != "mhr_model.pt":
        (path.parent / "mhr_model.pt").replace(path)
    return path


@dataclass(frozen=True)
class MhrPrediction:
    """The SAM 3D Body outputs this module needs, for one person."""

    shape_params: np.ndarray  # (45,) MHR identity coefficients
    mhr_model_params: np.ndarray  # (204,) pose (136) + skeletal scales (68)
    expr_params: np.ndarray | None = None
    bbox: np.ndarray | None = None

    @classmethod
    def from_record(cls, record: dict) -> "MhrPrediction":
        shape = np.asarray(record["shape_params"], dtype=np.float32).reshape(-1)
        params = np.asarray(record["mhr_model_params"], dtype=np.float32).reshape(-1)
        if shape.shape != (NUM_IDENTITY,) or params.shape != (NUM_MODEL_PARAMETERS,):
            raise ValueError(
                f"expected shape_params ({NUM_IDENTITY},) and mhr_model_params ({NUM_MODEL_PARAMETERS},), "
                f"got {shape.shape} and {params.shape}"
            )
        expr = record.get("expr_params")
        bbox = record.get("bbox")
        return cls(
            shape,
            params,
            None if expr is None else np.asarray(expr, dtype=np.float32).reshape(-1),
            None if bbox is None else np.asarray(bbox, dtype=np.float32).reshape(-1),
        )


@dataclass(frozen=True)
class BodyMeasurements:
    height_cm: float
    chest_cm: float
    waist_cm: float
    hip_cm: float
    shoulder_cm: float
    method: str = (
        "girths: convex-hull perimeter of the central cross-section loop of the rest-pose mesh (chest at 72% of height, "
        "waist = minimum in 58-66%, hip = maximum in 47-55%); shoulder: shoulder-joint span + 15%"
    )

    def avatar_kwargs(self) -> dict[str, float]:
        return {
            "height_cm": round(self.height_cm, 1),
            "chest_cm": round(self.chest_cm, 1),
            "waist_cm": round(self.waist_cm, 1),
            "shoulder_cm": round(self.shoulder_cm, 1),
        }


class MhrBodyModel:
    """Thin wrapper around the MHR TorchScript model (LOD 1, centimetres, +Y up, +Z front)."""

    def __init__(self, model_path: Path | None = None):
        self.path = ensure_mhr_model(model_path)
        self.model = torch.jit.load(str(self.path), map_location="cpu")
        self.faces = self.model.character_torch.mesh.faces.detach().cpu().numpy().astype(np.int64)

    def evaluate(self, identity: np.ndarray, model_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vertices and joint positions in metres."""

        with torch.no_grad():
            vertices, skeleton = self.model(
                torch.as_tensor(identity, dtype=torch.float32)[None],
                torch.as_tensor(model_params, dtype=torch.float32)[None],
                torch.zeros(1, NUM_EXPRESSION),
            )
        return vertices[0].numpy() / 100.0, skeleton[0, :, :3].numpy() / 100.0


def canonical_model_params(prediction: MhrPrediction) -> np.ndarray:
    """Keep the person's skeletal scales; replace the photographed pose by the MHR rest (A-)pose."""

    params = np.zeros(NUM_MODEL_PARAMETERS, dtype=np.float32)
    params[NUM_POSE:] = prediction.mhr_model_params[NUM_POSE:]
    return params


def _hull_perimeter(points_xz: np.ndarray) -> float:
    from scipy.spatial import ConvexHull

    if len(points_xz) < 3:
        return 0.0
    hull = ConvexHull(points_xz)
    ring = points_xz[hull.vertices]
    return float(np.linalg.norm(ring - np.roll(ring, 1, axis=0), axis=1).sum())


def _torso_girth(mesh, y: float) -> float:
    """Tape-measure girth: convex-hull perimeter of the cross-section loop around the body axis."""

    section = mesh.section(plane_origin=[0.0, y, 0.0], plane_normal=[0.0, 1.0, 0.0])
    if section is None:
        return 0.0
    loops = [np.asarray(loop)[:, [0, 2]] for loop in section.discrete]
    central = min(loops, key=lambda loop: abs(float(loop[:, 0].mean())))
    return _hull_perimeter(central)


def measure_rest_body(vertices: np.ndarray, joints: np.ndarray, faces: np.ndarray) -> BodyMeasurements:
    """Body measurements (cm) from an MHR rest-pose surface in metres."""

    import trimesh

    mesh = trimesh.Trimesh(vertices, faces, process=False)

    floor = float(vertices[:, 1].min())
    height = float(vertices[:, 1].max()) - floor
    shoulder_x = 0.5 * abs(float(joints[MHR_SHOULDER_JOINTS[0], 0] - joints[MHR_SHOULDER_JOINTS[1], 0]))

    def girth(fraction: float) -> float:
        return _torso_girth(mesh, floor + fraction * height)

    chest = girth(0.72)
    waist = min(girth(f) for f in np.linspace(0.58, 0.66, 9))
    hip = max(girth(f) for f in np.linspace(0.47, 0.55, 9))
    shoulder = 2.0 * shoulder_x * 1.15
    return BodyMeasurements(
        height_cm=100.0 * height,
        chest_cm=100.0 * chest,
        waist_cm=100.0 * waist,
        hip_cm=100.0 * hip,
        shoulder_cm=100.0 * shoulder,
    )


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# MHR body (Apache-2.0), metres, +Y up, +Z front\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces + 1:
            handle.write(f"f {a} {b} {c}\n")
    return path


def render_body_preview(path: Path, vertices: np.ndarray, faces: np.ndarray, size: int = 512) -> Path:
    """Flat-shaded orthographic front and side views (PIL only), for a quick visual check."""

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (2 * size, size), (240, 242, 242))
    draw = ImageDraw.Draw(image)
    light = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])
    height = float(np.ptp(vertices[:, 1]))
    scale = 0.9 * size / max(height, 1e-6)
    for panel, (axis_u, axis_depth, sign) in enumerate(((0, 2, 1.0), (2, 0, -1.0))):
        tri = vertices[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        view = np.zeros(3)
        view[axis_depth] = sign
        visible = normals @ view > 0
        rotated_light = light.copy()
        rotated_light[[0, 2]] = light[[axis_u, axis_depth]] if panel else light[[0, 2]]
        shade = np.clip(np.abs(normals @ rotated_light), 0.15, 1.0)
        depth = sign * tri[:, :, axis_depth].mean(axis=1)
        for face in np.argsort(depth):
            if not visible[face]:
                continue
            u = (sign if panel else 1.0) * tri[face, :, axis_u]
            points = [
                (panel * size + size / 2 + scale * uu, size * 0.95 - scale * (vv - vertices[:, 1].min()))
                for uu, vv in zip(u, tri[face, :, 1])
            ]
            tone = int(90 + 150 * shade[face])
            draw.polygon(points, fill=(tone, int(tone * 0.86), int(tone * 0.76)))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def run_sam3d_body(image: Path, output_json: Path) -> Path:
    """Run SAM 3D Body in its own environment (see scripts/sam3d_body_infer.py).

    Needs SAM3D_PYTHON (interpreter of the SAM 3D Body env), SAM3D_REPO (checkout)
    and SAM3D_CHECKPOINT (model.ckpt); SAM3D_MHR_PATH defaults to the checkpoint's
    assets/mhr_model.pt.
    """

    missing = [name for name in ("SAM3D_PYTHON", "SAM3D_REPO", "SAM3D_CHECKPOINT") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "SAM 3D Body is not configured: set " + ", ".join(missing) + ". The checkpoints are gated on "
            "Hugging Face (request access to facebook/sam-3d-body-dinov3), then install its environment per "
            "its INSTALL.md. Alternatively pass a SAM 3D Body prediction JSON."
        )
    checkpoint = Path(os.environ["SAM3D_CHECKPOINT"])
    mhr_path = Path(os.environ.get("SAM3D_MHR_PATH", checkpoint.parent / "assets" / "mhr_model.pt"))
    command = [
        os.environ["SAM3D_PYTHON"], str(ROOT / "scripts" / "sam3d_body_infer.py"),
        "--sam3d-repo", os.environ["SAM3D_REPO"], "--checkpoint", str(checkpoint),
        "--mhr-model", str(mhr_path), "--image", str(image), "--output", str(output_json),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError("SAM 3D Body failed:\n" + completed.stderr[-2000:])
    return output_json


def select_person(people: list[dict], person_index: int | None = None) -> tuple[int, MhrPrediction]:
    if not people:
        raise ValueError("SAM 3D Body found no person in the image")
    if person_index is None:
        def area(record: dict) -> float:
            box = record.get("bbox")
            if box is None:
                return 0.0
            x0, y0, x1, y1 = np.asarray(box, dtype=float).reshape(-1)[:4]
            return (x1 - x0) * (y1 - y0)

        person_index = int(np.argmax([area(record) for record in people]))
    return person_index, MhrPrediction.from_record(people[person_index])


@dataclass(frozen=True)
class BodyReconstruction:
    body_mesh: str
    preview_png: str
    image_pose_mesh: str
    measurements: BodyMeasurements
    person_index: int
    people_detected: int
    source: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["avatar"] = self.measurements.avatar_kwargs()
        data["body_mesh_axes"] = {"source_up_axis": "y", "source_front_axis": "z", "units": "metres"}
        return data


def reconstruct_body(
    out_dir: Path,
    *,
    image: Path | None = None,
    prediction_json: Path | None = None,
    person_index: int | None = None,
    mhr_model_path: Path | None = None,
) -> BodyReconstruction:
    """Image (or a saved SAM 3D Body prediction) -> A-pose MHR body + measurements."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if prediction_json is None:
        if image is None:
            raise ValueError("pass an image or a SAM 3D Body prediction JSON")
        prediction_json = run_sam3d_body(Path(image), out_dir / "sam3d_body_prediction.json")
        source = "SAM 3D Body inference on the image"
    else:
        source = f"saved SAM 3D Body prediction {prediction_json}"
    payload = json.loads(Path(prediction_json).read_text(encoding="utf-8"))
    people = payload["people"] if isinstance(payload, dict) and "people" in payload else payload
    index, prediction = select_person(people, person_index)

    body = MhrBodyModel(mhr_model_path)
    rest_vertices, rest_joints = body.evaluate(prediction.shape_params, canonical_model_params(prediction))
    image_pose = prediction.mhr_model_params.copy()
    image_pose[:6] = 0.0  # drop root translation/rotation: body frame, not camera frame
    posed_vertices, _ = body.evaluate(prediction.shape_params, image_pose)
    for vertices in (rest_vertices, posed_vertices):
        vertices[:, 1] -= vertices[:, 1].min()  # stand on y = 0
    measurements = measure_rest_body(rest_vertices, rest_joints, body.faces)
    reconstruction = BodyReconstruction(
        body_mesh=str(write_obj(out_dir / "body_apose.obj", rest_vertices, body.faces)),
        preview_png=str(render_body_preview(out_dir / "body_apose_preview.png", rest_vertices, body.faces)),
        image_pose_mesh=str(write_obj(out_dir / "body_image_pose.obj", posed_vertices, body.faces)),
        measurements=measurements,
        person_index=index,
        people_detected=len(people),
        source=source,
    )
    (out_dir / "body_reconstruction.json").write_text(json.dumps(reconstruction.to_dict(), indent=2), encoding="utf-8")
    return reconstruction


def mean_body_prediction() -> dict:
    """A SAM 3D Body-shaped record for the MHR mean body in its rest pose (for tests and demos)."""

    return {"shape_params": [0.0] * NUM_IDENTITY, "mhr_model_params": [0.0] * NUM_MODEL_PARAMETERS, "bbox": [0, 0, 1, 1]}

