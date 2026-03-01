# Simulink DVPP Rewrite

This is the clean restart of the DVPP port from the original Simulink model in:

- [DVPP.slx](/Users/jellelourens/Documents/DVPP_python/Original%20DVPP%20Simulink/DVPP.slx)

The new package lives in:

- [simulink_dvpp](/Users/jellelourens/Documents/DVPP_python/simulink_dvpp)

## Ported in this pass

- `chart_37` -> `simulink_dvpp.waves.wave_properties`
- `chart_90` -> `simulink_dvpp.model.SimulinkDVPPModel.state_derivative`
- `chart_131` -> `simulink_dvpp.mass.mass_and_added_mass`
- `chart_155` -> `simulink_dvpp.gravity.gravity_force`
- `chart_170` -> `simulink_dvpp.hydrostatics.hydrostatic_forces_and_moments`
- `chart_321` -> `simulink_dvpp.non_inertial.rigid_body_coriolis`
- `chart_328` -> `simulink_dvpp.kinematics.body_to_ned_velocity`
- `chart_457` -> `simulink_dvpp.diffraction.diffraction_force`
- `chart_468` -> `simulink_dvpp.radiation.CumminsRadiationModel`
- `chart_54` -> `simulink_dvpp.resistance.resistance_force`
- sail-force subsystem -> `simulink_dvpp.sails`
- active keel/rudder path -> `simulink_dvpp.appendages`
- foil subsystem -> `simulink_dvpp.foils`

## Not ported yet

- STL/mesh loading convenience layer
- simulation driver and IO matching the Simulink signal routing
- some diagnostic side-branches that are present in Simulink but not used in the
  force summation path

## Intent

This package is not a light refactor of the existing Python DVPP. It is a new,
parallel implementation that follows the original Simulink subsystem boundaries
so the equations can be checked block-by-block against the source model.

## Current entry points

- `SimulinkDVPPModel.translated_subsystems(...)`
- `SimulinkDVPPModel.assembled_step(...)`
- `SimulinkDVPPModel.state_derivative(...)`
