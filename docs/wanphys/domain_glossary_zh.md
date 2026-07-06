# WanPhys Fluid LBM

This context names the fluid-grid LBM concepts used while integrating multiphase flow with solid boundaries and rigid bodies.

## Language

**Shan-Chen wall force**:
A fluid-solid interaction term in Shan-Chen multiphase LBM that models how a solid surface attracts or repels the liquid phase. It is the preferred term here for the wall wettability force, not for rigid-body force feedback.
_Avoid_: fluid-solid interaction force when the meaning could be confused with two-way rigid coupling

**Wettability**:
The tendency of a solid surface to attract or repel the liquid phase in a multiphase simulation. Wettability is usually tuned through the Shan-Chen wall force and observed through the contact angle.
_Avoid_: stickiness, suction

**Contact angle**:
The observable angle where liquid, gas, and solid meet. It is the calibration target for wettability rather than a direct solver state.
_Avoid_: adhesion amount

**Solid indicator**:
A neighborhood value indicating that a Shan-Chen lattice neighbor is solid for wall-force computation. It is distinct from the solid body's velocity used by moving-wall bounce-back.
_Avoid_: fake liquid density

**World rigid velocity**:
The rigid body's velocity in WanPhys/Newton world units. It describes how the rigid pose changes in world space and must not be treated as an LBM lattice velocity.
_Avoid_: lattice speed, wall speed

**LBM wall velocity**:
The lattice-unit velocity sampled by moving-wall bounce-back at fluid-solid links. It is derived from rigid motion but belongs to the LBM solver's stability regime.
_Avoid_: world speed, prescribed sphere speed

**Moving-wall bounce-back**:
An LBM solid-boundary rule where reflected distributions include the wall's lattice velocity. It models no-slip motion at a moving solid surface, not fluid-to-rigid force feedback.
_Avoid_: two-way coupling

**Solid density mask**:
A visualization-only filter that hides density values inside cells marked solid by ``solid_phi``. It does not remove physical density, fix wettability, or solve fluid state left behind by moving solids.
_Avoid_: physical solid-fluid separation, density correction

**Two-way rigid coupling**:
Momentum exchange between fluid and rigid bodies where the fluid exerts force and torque back onto rigid bodies. It is separate from Shan-Chen wall force, which controls wetting behavior.
_Avoid_: wall force

## Example Dialogue

Developer: The sphere pulls the liquid before contact. Is that two-way rigid coupling?

Domain expert: No. That is a Shan-Chen wall-force or wettability problem. Two-way rigid coupling is about force and torque applied back to the sphere.

Developer: Should I tune the contact angle by changing solid velocity?

Domain expert: No. Solid velocity belongs to moving-wall bounce-back. Tune wettability through the Shan-Chen wall force.

Developer: The sphere speed is 0.15. Is that safe for LBM?

Domain expert: Only after you know which velocity you mean. World rigid velocity must be converted before it becomes LBM wall velocity.

Developer: If we mask density inside the sphere, did we fix the water sticking to the sphere?

Domain expert: No. Solid density masking only fixes the display. Wetting and leftover solid-cell fluid state are separate physical/modeling problems.
