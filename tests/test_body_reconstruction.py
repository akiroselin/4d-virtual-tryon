from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aigc_fashion.body_mesh_avatar import BodyMeshAvatar
from aigc_fashion.body_reconstruction import (
    NUM_IDENTITY,
    NUM_MODEL_PARAMETERS,
    MhrPrediction,
    default_mhr_model_path,
    mean_body_prediction,
    reconstruct_body,
    run_sam3d_body,
    select_person,
)
from aigc_fashion.data import AvatarSpec

needs_mhr = pytest.mark.skipif(
    not default_mhr_model_path().is_file(),
    reason="MHR model not downloaded (aigc_fashion.body_reconstruction.ensure_mhr_model fetches it)",
)


def test_prediction_validation_and_person_selection():
    with pytest.raises(ValueError):
        MhrPrediction.from_record({"shape_params": [0.0] * 10, "mhr_model_params": [0.0] * NUM_MODEL_PARAMETERS})
    small = mean_body_prediction()
    large = mean_body_prediction()
    large["bbox"] = [0, 0, 50, 80]
    index, prediction = select_person([small, large])
    assert index == 1
    assert prediction.shape_params.shape == (NUM_IDENTITY,)
    with pytest.raises(ValueError):
        select_person([])


def test_image_path_requires_configured_sam3d_body(monkeypatch, tmp_path):
    for name in ("SAM3D_PYTHON", "SAM3D_REPO", "SAM3D_CHECKPOINT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="SAM3D_PYTHON"):
        run_sam3d_body(tmp_path / "person.jpg", tmp_path / "out.json")


@needs_mhr
def test_mean_body_reconstruction_is_a_plausible_a_pose_avatar(tmp_path: Path):
    prediction = tmp_path / "prediction.json"
    prediction.write_text(json.dumps({"people": [mean_body_prediction()]}), encoding="utf-8")
    result = reconstruct_body(tmp_path / "body", prediction_json=prediction)
    m = result.measurements
    assert 165 < m.height_cm < 180
    assert 85 < m.chest_cm < 115 and 75 < m.waist_cm < 110 and 95 < m.hip_cm < 120
    assert 35 < m.shoulder_cm < 46
    assert Path(result.preview_png).is_file()

    avatar = BodyMeshAvatar.from_obj(
        result.body_mesh,
        AvatarSpec(**result.measurements.avatar_kwargs()),
        source_up_axis="y",
        source_front_axis="z",
    )
    vertices = np.asarray(avatar.mesh_vertices)
    assert vertices.shape[0] == 18439
    # A-pose: the hands hang below the shoulders, beside the hips.
    landmarks = avatar._shirt_landmarks
    assert landmarks["sleeve_cuff_y"] < landmarks["shoulder_y"]
