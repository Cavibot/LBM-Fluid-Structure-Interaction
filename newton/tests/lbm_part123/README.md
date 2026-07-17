# LBM Part 0～3 isolated acceptance suite

This directory owns the tests introduced with the Part 0～3 architecture work.
They are kept separate from historical LBM regression and visualization tests
so the new contracts can be reviewed, selected and archived as one unit.

Coverage:

- `test_part0_contracts.py`: naming axes, capability states, fail-fast and
  boundary conflict resolution;
- `test_part1_collision_spaces.py`: SRT/TRT, FullF Raw MRT, FullF NOCM MRT and
  HOME NOCM zero-force paths;
- `test_part2_force_pipeline.py`: force-density units, half-step hydro closure
  and per-space injection;
- `test_part3_boundary_pipeline.py`: completion-before-collection, Zou-He,
  pressure, history-based convective outlet, edge/corner fallback, moving wall
  and cut-link interpolation.

The suite is intentionally not mixed with visual acceptance.  Its tests use
small deterministic CPU domains and independent NumPy assertions.

Suggested explicit invocation:

```text
python -m unittest discover -s newton/tests/lbm_part123 -p "test_*.py" -v
```
