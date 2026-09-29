import numpy as np

from citylearn_control.core import price_threshold_policy, simulate


def test_threshold_policy_reduces_cost_on_simple_tariff() -> None:
    load=np.full(24,2.0)
    price=np.array([1.0]*8+[3.0]*8+[6.0]*8)
    baseline=simulate(load,price,np.zeros(24))
    action=price_threshold_policy(load,price)
    controlled=simulate(load,price,action)
    assert controlled["cost"] < baseline["cost"]


def test_soc_stays_bounded() -> None:
    load=np.ones(30)
    price=np.linspace(1,5,30)
    result=simulate(load,price,price_threshold_policy(load,price))
    assert 0.0 <= result["final_soc"] <= 10.0
