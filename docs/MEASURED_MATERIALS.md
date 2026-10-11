# Measured fabric calibration

The garment photograph identifies appearance and a likely construction family. It does not identify a physical material card. Two visually similar fabrics can have different yarn count, knit tightness, finish, humidity response, and elastane content.

## Required swatch measurements

For production calibration, store the raw loading and unloading curves rather than one scalar stiffness:

1. Areal density in `kg/m^2` and thickness in `m`.
2. Warp/wale and weft/course force-extension curves in `N/m` versus strain.
3. Bias-extension or pure-shear loading/unloading curves in `N/m` versus shear angle.
4. Warp, weft, and bias moment-curvature curves in `N*m` versus `1/m`.
5. Tensile, shear, and bending hysteresis for dissipation.
6. Surface friction against skin and the relevant under-garment.

KES-FB provides tensile, shear, bending, compression, surface, weight, and thickness measurements. Browzwear FAB similarly tests on-grain, cross-grain, and bias strips and exports digital fabric data. ASTM D1388 supplies a standardized bending/flexural-rigidity test, but it does not replace tensile and shear tests.

## Solver mapping

The measured curves supervise the Hyper-network NODE at material points:

- warp/weft curves -> membrane branches 0 and 1;
- bias/shear curve -> membrane branch 2;
- biaxial area response -> membrane branch 3;
- moment-curvature curves -> Kirchhoff-Love bending branch;
- loading/unloading hysteresis -> dissipative state, not a larger elastic stiffness;
- areal density -> lumped shell mass.

Metric projection is only a final guard at a measured service/failure boundary. It must not generate the normal drape response.

## Current evidence status

`aigc_fashion.measured_materials` contains literature envelopes for cotton single jersey and wool-blend suiting. They are useful priors for hypothesis ranking, not exact cards for the two photographed garments. Exact simulation requires a fabric SKU, U3M/FAB/KES file, or a measured swatch.

Sources:

- MIT Fabric Properties Dataset: https://people.csail.mit.edu/klbouman/pw/projects/materialproperties/dataset.html
- NC State KES overview: https://textiles.ncsu.edu/tpacc/comfort-performance/kawabata-evaluation-system/
- IEEE critical review for virtual simulation: https://standards.ieee.org/wp-content/uploads/import/governance/iccom/3DBP-Measurement_Fabric_Properties-Virtual_Simulation.pdf
- ASTM D1388: https://store.astm.org/d1388-23.html
- Yarn-to-shell homogenization: https://arxiv.org/abs/2401.15169
- Browzwear FAB: https://browzwear.com/products/fabric-analyzer

