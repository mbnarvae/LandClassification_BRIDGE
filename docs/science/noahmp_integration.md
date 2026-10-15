# Noah-MP integration

## Goal
Test whether replacing the default land cover (MODIS IGBP-20) with the BRIDGE classification improves Noah-MP
simulations of the Andes–Amazon water and energy budgets.

## Experiments
| Run | Land cover | Parameters | Question |
|-----|------------|------------|----------|
| CTRL | Default MODIS IGBP-20 | Default MPTABLE | Baseline |
| B-IGBP | BRIDGE map crosswalked to IGBP-20 (`scheme=igbp`) | Default MPTABLE | Does a better *map* alone help? |
| B-EXT | BRIDGE map with extended codes (`scheme=noahmp_ext`) | MPTABLE + new Andean columns | Do new *ecosystem types* help? |

Keep forcing, period, spin-up, soils and namelist identical. Evaluate against streamflow, ET (flux towers,
remote-sensing ET), soil moisture, snow/glacier extent, LST.

## Workflow
```bash
bridge-lc export-noahmp classification.tif lu_noahmp.tif --scheme noahmp_ext   # remap codes
bridge-lc mptable-stub --out mptable_bridge_ext.txt                            # parameters to fill in
```
Then, in Python: aggregate with `bridge_lc.noahmp.export.dominant_category` to the model grid and
write `LU_INDEX` with `write_lu_index(geo_em, codes, out)`.

## Checklist for extended categories (>20)
- [ ] New columns in `MPTABLE.TBL` (MODIS section) with literature/field values for each parameter in
      `bridge_lc.noahmp.crosswalk.MPTABLE_PARAMS`; document sources here.
- [ ] Update `NUM_LAND_CAT` consistently (geo_em attributes, namelist, tables).
- [ ] Verify Noah-MP code paths that hard-code category indices (water, urban, snow/ice, barren) still work.
- [ ] Monthly LAI/SAI climatologies per new category (e.g. from MODIS LAI over the classified pixels).
- [ ] Sensitivity tests: perturb each new parameter set to see which matter most.

## Parameter sources (to fill)
| Category | Parameter | Value | Source |
|----------|-----------|-------|--------|
| montane_cloud_forest | CH2OP | TODO | TODO |
| paramo_grassland | soil water retention (via soil params) | TODO | TODO |
