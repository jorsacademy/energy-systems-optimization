"""Optional CityLearn environment construction."""

from __future__ import annotations


def make_citylearn_env(schema: str):
    try:
        from citylearn.citylearn import CityLearnEnv
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the optional CityLearn dependency") from exc
    return CityLearnEnv(schema)
