# CityLearn Building Energy Control

A building-energy control benchmark with an optional CityLearn execution layer.

The CI-safe core implements a battery/load/tariff simulator and a transparent price-threshold controller. The optional adapter constructs a real `CityLearnEnv` so the same evaluation contract can be moved to standardized building datasets.

The intended experimental ladder is:

`rule-based baseline -> deterministic MPC -> constrained RL challenger -> CityLearn held-out evaluation`.

Report energy cost, peak demand, comfort violations, battery cycling and inference/solve latency. A policy is not considered production-ready merely because it obtains higher simulator reward.
