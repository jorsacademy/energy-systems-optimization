"""CI-safe building energy control benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Battery:
    capacity: float = 10.0
    max_power: float = 3.0
    efficiency: float = 0.95


def simulate(
    load: ArrayLike,
    price: ArrayLike,
    action: ArrayLike,
    *,
    battery: Battery | None = None,
    initial_soc: float = 5.0,
) -> dict[str, float]:
    battery = Battery() if battery is None else battery
    demand = np.asarray(load, dtype=float)
    tariff = np.asarray(price, dtype=float)
    control = np.asarray(action, dtype=float)
    if not (demand.shape == tariff.shape == control.shape):
        raise ValueError("load, price and action must share shape")

    soc = float(initial_soc)
    grid: list[float] = []
    for load_t, action_t in zip(demand, control, strict=True):
        power = float(np.clip(action_t, -battery.max_power, battery.max_power))
        if power >= 0.0:
            charge = min(power, (battery.capacity - soc) / battery.efficiency)
            soc += battery.efficiency * charge
            grid.append(float(load_t + charge))
        else:
            discharge = min(-power, soc * battery.efficiency)
            soc -= discharge / battery.efficiency
            grid.append(float(max(load_t - discharge, 0.0)))

    grid_array = np.asarray(grid)
    return {
        "cost": float(np.sum(grid_array * tariff)),
        "peak": float(np.max(grid_array)),
        "final_soc": soc,
    }


def price_threshold_policy(
    load: ArrayLike,
    price: ArrayLike,
    battery: Battery | None = None,
) -> np.ndarray:
    del load
    battery = Battery() if battery is None else battery
    tariff = np.asarray(price, dtype=float)
    low = float(np.quantile(tariff, 0.30))
    high = float(np.quantile(tariff, 0.70))
    return np.where(
        tariff <= low,
        battery.max_power,
        np.where(tariff >= high, -battery.max_power, 0.0),
    )
