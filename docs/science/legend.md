# BRIDGE legend (v0 draft)

Source of truth: [`configs/legend/bridge_andes_v0.yaml`](../../configs/legend/bridge_andes_v0.yaml).
Run `bridge-lc info` to print it.

## Design principles
1. **Hierarchical.** `level1` groups (forest, shrubland, grassland, wetland, barren, snow_ice, water, agriculture,
   urban) are globally applicable; fine classes are region-specific. A global algorithm predicts level1 everywhere
   and fine classes where a regional legend exists.
2. **Ecohydrologically motivated.** A fine class exists only if it is expected to behave differently in the
   water/energy budget (interception, ET, rooting depth, soil water retention, albedo, roughness, snow/ice).
3. **Crosswalkable.** Every class has an IGBP-20 parent (standard Noah-MP) and optionally a new extended code (>20).
4. **Mappable.** Classes must be separable with the available inputs; otherwise merge them.

## Classes proposed as new Noah-MP categories (v0)
| Ext code | Class | IGBP parent | Key hydrological difference |
|----------|-------|-------------|------------------------------|
| 21 | lowland_flooded_forest | 2 EBF | Seasonal inundation, surface water–ET coupling |
| 22 | montane_cloud_forest | 2 EBF | Fog/cloud water interception, epiphyte storage, low ET |
| 23 | polylepis_woodland | 8 Woody savanna | High-altitude woodland canopy, soil water |
| 24 | paramo_grassland | 10 Grassland | Andosols/peat with very high water retention, tussock structure |
| 25 | humid_puna_grassland | 10 Grassland | Strong wet/dry seasonality, frost |
| 26 | bofedal_peatland | 11 Wetland | Near-saturated cushion peat, baseflow buffering |

These are **hypotheses**: whether separate parameter sets improve Noah-MP is the core experiment of BRIDGE.

## Open questions
- [ ] Align fine classes with Becker et al. and regional ecosystem maps (Josse et al. 2003; national maps).
- [ ] Is superpáramo better mapped to IGBP 19 (mixed tundra) or 16 (barren)?
- [ ] Should páramo and humid puna share a parameter set initially (data scarcity)?
- [ ] How to represent degraded páramo / burned grassland / plantations?
- [ ] Minimum mapping unit for bofedales at 30 m?

## Changelog
- v0.1.0 — initial draft (18 classes, 6 proposed extended categories).
