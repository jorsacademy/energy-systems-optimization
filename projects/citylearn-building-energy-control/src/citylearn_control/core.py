"""CI-safe building energy control benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Battery:
    capacity: float=10.0
    max_power: float=3.0
    efficiency: float=0.95


def simulate(
    load: ArrayLike,
    price: ArrayLike,
    action: ArrayLike,
    *,
    battery: Battery=Battery(),
    initial_soc: float=5.0,
) -> dict[str,float]:
    demand=np.asarray(load,dtype=float)
    tariff=np.asarray(price,dtype=float)
    control=np.asarray(action,dtype=float)
    if not (demand.shape == tariff.shape == control.shape):
        raise ValueError("load, price and action must share shape")
    soc=float(initial_soc)
    grid=[]
    for l,a in zip(demand,control,strict=True):
        p=float(np.clip(a,-battery.max_power,battery.max_power))
        if p >= 0:
            charge=min(p,(battery.capacity-soc)/battery.efficiency)
            soc += battery.efficiency*charge
            grid.append(float(l+charge))
        else:
            discharge=min(-p,soc*battery.efficiency)
            soc -= discharge/battery.efficiency
            grid.append(float(max(l-discharge,0.0)))
    g=np.asarray(grid)
    return {
        "cost":float(np.sum(g*tariff)),
        "peak":float(np.max(g)),
        "final_soc":soc,
    }


def price_threshold_policy(load:ArrayLike,price:ArrayLike,battery:Battery=Battery()) -> np.ndarray:
    tariff=np.asarray(price,dtype=float)
    low=float(np.quantile(tariff,0.30))
    high=float(np.quantile(tariff,0.70))
    return np.where(tariff<=low,battery.max_power,np.where(tariff>=high,-battery.max_power,0.0))
