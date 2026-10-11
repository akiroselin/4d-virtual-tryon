# Licensed SMPL/SMPL-X Models

Put official model files here after accepting the license:

- SMPL-X: https://smpl-x.is.tue.mpg.de/
- SMPL: https://smpl.is.tue.mpg.de/

Expected layout for the exporter:

```text
assets/smpl_models/
  smplx/
    SMPLX_NEUTRAL.npz
    SMPLX_MALE.npz
    SMPLX_FEMALE.npz
```

Then export an OBJ body mesh:

```bash
python3 outputs/aigc_fashion_prototype/scripts/export_smpl_obj.py \
  --model-folder outputs/aigc_fashion_prototype/assets/smpl_models \
  --model-type smplx \
  --gender neutral \
  --height-cm 178 \
  --output outputs/aigc_fashion_prototype/assets/body_meshes/smplx_neutral.obj
```

Use the exported OBJ in the simulator:

```bash
python3 outputs/aigc_fashion_prototype/examples/run_mvp.py \
  --texture-image "/Users/muwusah/Documents/AI/WorldModel/Genesis/tshirt.png" \
  --fit-from-image \
  --smpl-mesh outputs/aigc_fashion_prototype/assets/body_meshes/smplx_neutral.obj \
  --out-dir outputs/aigc_fashion_prototype/outputs/real_smpl_tshirt_run
```
