from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the project T-shirt through official Genesis FEM.Cloth + IPC."
    )
    parser.add_argument("--genesis-repo", type=Path, required=True)
    parser.add_argument("--cloth-obj", type=Path, required=True)
    parser.add_argument("--body-obj", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--original-to-welded", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu", "metal"), default="gpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.genesis_repo.resolve()))
    try:
        import uipc  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Genesis FEM.Cloth requires pyuipc. Run this script on a supported "
            "Windows/Linux host with `pip install pyuipc`."
        ) from exc

    import genesis as gs
    from genesis.utils.misc import tensor_to_array

    settings = json.loads(args.scene_config.read_text(encoding="utf-8"))
    backend = getattr(gs, args.backend)
    gs.init(backend=backend, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=float(settings["dt"]),
            gravity=tuple(settings["gravity"]),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_enable=True,
            contact_d_hat=float(settings["contact_d_hat"]),
            contact_friction_enable=True,
            enable_rigid_rigid_contact=False,
            two_way_coupling=False,
        ),
        show_viewer=False,
    )
    material = settings["cloth"]
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(args.cloth_obj.resolve()),
            file_meshes_are_zup=True,
        ),
        material=gs.materials.FEM.Cloth(
            E=float(material["E_pa"]),
            nu=float(material["nu"]),
            rho=float(material["rho_kg_m3"]),
            thickness=float(material["thickness_m"]),
            bending_stiffness=float(material["bending_stiffness"]),
            friction_mu=float(material["friction_mu"]),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(args.body_obj.resolve()),
            fixed=True,
            convexify=False,
            decimate=False,
            file_meshes_are_zup=True,
        ),
        material=gs.materials.Rigid(
            coup_type="ipc_only",
            coup_friction=float(material["friction_mu"]),
        ),
    )
    scene.build(n_envs=1)

    mapping = (
        np.load(args.original_to_welded).astype(np.int64)
        if args.original_to_welded is not None
        else None
    )

    def positions() -> np.ndarray:
        value = np.asarray(tensor_to_array(cloth.get_state().pos), dtype=np.float32)
        if value.ndim == 3:
            value = value[0]
        return value[mapping] if mapping is not None else value

    frames = [positions()]
    for _ in range(int(settings["steps"])):
        scene.step()
        frames.append(positions())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, np.asarray(frames, dtype=np.float32))
    print(f"wrote {len(frames)} Genesis FEM.Cloth + IPC frames to {args.out}")


if __name__ == "__main__":
    main()
