# SPDX-FileCopyrightText: Christoph Komanns, Florian Maurer, Ralf Schemm
#
# SPDX-License-Identifier: MIT

import inspect
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from battery_utility_calculator.battery_utility_calculator import (
    DEFAULT_GRID_FEE_BETWEEN_LOCATIONS as BUC_DEFAULT_GRID_FEES,
)
from battery_utility_calculator.energy_costs_calculator import (
    DEFAULT_GRID_FEE_BETWEEN_LOCATIONS,
    EnergyCostCalculator,
    Storage,
)

idx_2 = pd.date_range("2025-01-01", freq="h", periods=2)
idx_3 = pd.date_range("2025-01-01", freq="h", periods=3)
idx_4 = pd.date_range("2025-01-01", freq="h", periods=4)
idx_5 = pd.date_range("2025-01-01", freq="h", periods=5)


def community_prices(
    index: pd.DatetimeIndex,
    locations: tuple[str, ...] = ("aachen",),
    values: list[float] | None = None,
) -> dict[str, pd.Series]:
    if values is None:
        values = [0.0] * len(index)
    return {location: pd.Series(values, index=index) for location in locations}


def test_default_grid_fees_are_symmetric():
    for from_location, fees in DEFAULT_GRID_FEE_BETWEEN_LOCATIONS.items():
        assert fees[from_location] == 0.0, f"{from_location} to itself is not free"
        for to_location, fee in fees.items():
            reverse = DEFAULT_GRID_FEE_BETWEEN_LOCATIONS[to_location][from_location]
            assert fee == reverse, (
                f"{from_location}->{to_location} is {fee} but "
                f"{to_location}->{from_location} is {reverse}"
            )


def test_default_grid_fees_are_defined_for_every_location_pair():
    locations = set(DEFAULT_GRID_FEE_BETWEEN_LOCATIONS)
    for from_location, fees in DEFAULT_GRID_FEE_BETWEEN_LOCATIONS.items():
        assert set(fees) == locations, f"row {from_location} is incomplete"


def test_default_grid_fee_tables_match_across_modules():
    # the table is duplicated; BUC passes its copy as the default argument and
    # ECC overlays it onto its own, so a divergence would silently change fees
    assert BUC_DEFAULT_GRID_FEES == DEFAULT_GRID_FEE_BETWEEN_LOCATIONS


def test_ECC_baseline():
    # buying 1 kWh for 1 €/kWh should equal to 3€ total
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=0),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([1, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    assert round(costs, 0) == -3

    # after optimization we should be able to read individual cashflows
    cashflows = calculator.get_cashflows()
    assert cashflows == {
        "community": 0.0,
        "supplier": -3.0,
        "eeg": 0.0,
        "wholesale": 0.0,
        "grid_fees": 0.0,
    }


def test_ECC_opti_storage():
    # buying 2 kWh for 0€/kWh and storing 1 kWh of this should equal 1€ total
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=1),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    assert round(costs, 0) == -1


def test_ECC_opti_storage_2():
    # now we need 2 kWh at each timestep
    # on timestep=0, we can buy for 0€/kWh and should buy 3kWh
    # as we use 2 kWh during timestep=0 and use 1 kWh for timestep=1
    # total cost should be 3*0 + 1*1 + 2*1 = 3
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=1),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([2, 2, 2], index=idx_3),
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    assert round(costs) == -3


def test_ECC_selling_pv():
    # here we should gain 1€ from selling pv
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=0),
        eeg_prices=pd.Series([1, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([1, 1, 1], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 0], index=idx_3),
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    assert round(costs, 0) == 1


def test_ECC_selling_pv_w_storage():
    # same as above, but we can store PV and sell at
    # timestep=1 instead of timestep=0, as we can get 2€/kWh
    # in timestep=1
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=1),
        eeg_prices=pd.Series([1, 2, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([1, 1, 1], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 0], index=idx_3),
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    assert round(costs, 0) == 2

    # charge from solar_generation in ts=0,1 and discharge at ts=2
    calculator = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=2, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([5, 10, 20], index=idx_3),
        solar_generation=pd.Series([1, 1, 0], index=idx_3),
        demand=pd.Series([0, 0, 2], index=idx_3),
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    assert round(costs, 0) == 0


def test_ECC_negative_prices():
    # buy 2 kWh in ts=1 because we get paid for this
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=2),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([5, -20, 5], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 2], index=idx_3),
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    assert round(costs, 0) == 40


def test_ECC_c_rate():
    # check if c_rate is respected
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=0.5, volume=2),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 10, 0], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([2, 2, 0], index=idx_3),
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    assert round(costs, 0) == -10


def test_ECC_wholesale():
    storage = Storage(id=0, c_rate=1, volume=1)

    storage = Storage(id=0, c_rate=1, volume=1)
    # no demand, just buying from wholesale when price is 0 and selling again when price is 5
    # should be able to just do this once, as we only have volume of 1
    # buy for 0, sell for 5 -> gain of 5€, but 50% fee -> 2.5€
    calculator = EnergyCostCalculator(
        storage=storage,
        demand=pd.Series([0, 0, 0, 0], index=idx_4),
        solar_generation=pd.Series([0, 0, 0, 0], index=idx_4),
        supplier_prices=pd.Series([10, 10, 10, 10], index=idx_4),
        eeg_prices=pd.Series([0, 0, 0, 0], index=idx_4),
        community_market_prices={"aachen": pd.Series([0, 0, 0, 0], index=idx_4)},
        wholesale_market_prices=pd.Series([0, 0, 5, 5], index=idx_4),
        wholesale_fee=0.5,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    assert np.isclose(costs, 2.449999, atol=1e-6)

    # same as above, but volume of 2, so should be able to do two times for total gain of 4
    # no fee, so 100% of profit goes to customer
    storage = Storage(id=0, c_rate=1, volume=2)
    calculator = EnergyCostCalculator(
        storage=storage,
        demand=pd.Series([0, 0, 0, 0], index=idx_4),
        solar_generation=pd.Series([0, 0, 0, 0], index=idx_4),
        supplier_prices=pd.Series([10, 10, 10, 10], index=idx_4),
        eeg_prices=pd.Series([0, 0, 0, 0], index=idx_4),
        community_market_prices={"aachen": pd.Series([0, 0, 0, 0], index=idx_4)},
        wholesale_market_prices=pd.Series([3, 3, 5, 5], index=idx_4),
        wholesale_fee=0,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    assert round(costs, 0) == 4


def test_ECC_pv_to_wholesale_toggle_sets_bounds():
    common_kwargs = dict(
        storage=Storage(id=0, c_rate=1, volume=0),
        demand=pd.Series([0, 0, 0], index=idx_3),
        solar_generation=pd.Series([1, 1, 1], index=idx_3),
        supplier_prices=pd.Series([0, 0, 0], index=idx_3),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=community_prices(idx_3),
        wholesale_market_prices=pd.Series([10, 10, 10], index=idx_3),
        wholesale_fee=0.0,
    )

    ecc_disabled = EnergyCostCalculator(
        **common_kwargs,
        allow_pv_to_wholesale=False,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
    )
    ecc_enabled = EnergyCostCalculator(
        **common_kwargs,
        allow_pv_to_wholesale=True,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
    )

    assert ecc_disabled.model.pv_to_wholesale[0].ub == 0
    assert ecc_enabled.model.pv_to_wholesale[0].ub is None


def test_ECC_wholesale_cashflow_includes_pv_to_wholesale():
    # regression guard: direct PV wholesale flow must be part of wholesale cashflow
    source = inspect.getsource(EnergyCostCalculator.calculate_wholesale_cashflow)
    assert "self.model.pv_to_wholesale" in source


def test_ECC_charge_discharge_eff():
    # buying 2 kWh for 0€/kWh and storing 0.5 kWh (1kWh with a c-eff of 0.5) of this should equal 1.5€ total
    calculator = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=0.5, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    assert round(costs, 1) == -1.5

    # buying 2 kWh for 0€/kWh and storing 0.5 kWh (1kWh with a disc-eff of 0.5) of this should equal 1.5€ total
    calculator = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=0.5
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    assert np.isclose(costs, -1.5000005, atol=1e-6)

    # combine those two for total costs of 1.75
    calculator = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=0.5, discharge_efficiency=0.5
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    assert round(costs, 2) == -1.75


def test_ECC_discharge_penalty_is_applied():
    base_calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        discharge_penalty_per_kwh=0.0,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    base_costs = base_calc.optimize(solver="appsi_highs")

    penalized_calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        discharge_penalty_per_kwh=0.1,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    penalized_costs = penalized_calc.optimize(solver="highs")

    flows = penalized_calc.get_energy_flows()
    discharged_kwh = float(flows["storage_to_home"].sum())
    assert penalized_costs < base_costs
    assert (base_costs - penalized_costs) >= 0.1 * discharged_kwh
    assert np.isclose(penalized_calc.calculate_costs(), penalized_costs)


def test_ECC_cycle_cost_per_kwh_is_applied():
    base_calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        cycle_cost_per_kwh=0.0,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    base_costs = base_calc.optimize(solver="appsi_highs")

    cycle_cost_calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        cycle_cost_per_kwh=0.05,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    cycle_costs = cycle_cost_calc.optimize(solver="appsi_highs")

    flows = cycle_cost_calc.get_energy_flows()
    discharged_kwh = float(flows["storage_to_home"].sum())
    expected_delta = 0.05 * discharged_kwh
    assert cycle_costs < base_costs
    assert (base_costs - cycle_costs) >= expected_delta
    assert np.isclose(
        cycle_cost_calc.calculate_cycle_cost_penalty(use_values=True),
        expected_delta,
        atol=1e-6,
    )
    assert np.isclose(cycle_cost_calc.calculate_costs(), cycle_costs)


def test_ECC_hours_per_timestep():
    # using 1kW each timestep so 1kWh in total (0.25hours per timestep)
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=0),
        eeg_prices=pd.Series([0, 0, 0, 0], index=idx_4),
        wholesale_market_prices=pd.Series([0, 0, 0, 0], index=idx_4),
        community_market_prices={"aachen": pd.Series([0, 0, 0, 0], index=idx_4)},
        supplier_prices=pd.Series([1, 1, 1, 1], index=idx_4),
        solar_generation=pd.Series([0, 0, 0, 0], index=idx_4),
        demand=pd.Series([1, 1, 1, 1], index=idx_4),
        hours_per_timestep=0.25,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    assert np.isclose(costs, -1)


def test_ECC_hours_per_timestep_storage_shift():
    # 1 kW PV and 1 kW demand -> 0.25 kWh per 15 min step; shift via storage at zero cost
    calculator = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0], index=idx_2),
        wholesale_market_prices=pd.Series([0, 0], index=idx_2),
        community_market_prices=community_prices(idx_2),
        supplier_prices=pd.Series([100, 10], index=idx_2),
        solar_generation=pd.Series([1, 0], index=idx_2),
        demand=pd.Series([0, 1], index=idx_2),
        hours_per_timestep=0.25,
        allow_pv_to_wholesale=False,
        storage_use_cases=["home"],
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
    )
    costs = calculator.optimize(solver="appsi_highs")
    flows = calculator.get_energy_flows()
    soc = calculator.get_storage_soc_timeseries_df()

    assert np.isclose(costs, 0.0, atol=1e-6)
    assert np.isclose(flows["pv_to_storage_for_home"].iloc[0], 0.25)
    assert np.isclose(flows["storage_to_home"].iloc[1], 0.25)
    assert np.isclose(soc["soc_home"].iloc[0], 0.25)


def test_ECC_hours_per_timestep_c_rate_energy_limit():
    calculator = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=community_prices(idx_3),
        supplier_prices=pd.Series([0, 100, 100], index=idx_3),
        solar_generation=pd.Series([2, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 1], index=idx_3),
        hours_per_timestep=0.25,
        allow_pv_to_wholesale=False,
        storage_use_cases=["home"],
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
    )
    calculator.optimize(solver="appsi_highs")
    flows = calculator.get_energy_flows()
    max_charge_kwh = float(flows["pv_to_storage_for_home"].iloc[0])
    assert max_charge_kwh <= 0.25 + 1e-6
    assert max_charge_kwh > 0.0


def test_ECC_soc_start():
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=0.5, volume=1),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    soc_df = calculator.get_storage_soc_timeseries_df()
    assert round(soc_df.loc["2025-01-01 00:00:00", "soc_home"], 1) == 0.5


def test_ECC_soc_start_scales_with_hours_per_timestep():
    # one 4h step charges up to c_rate * volume * 4 kWh, so the storage can be
    # filled completely despite the 0.9 charge efficiency
    calculator = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=10, charge_efficiency=0.9, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0], index=idx_2),
        wholesale_market_prices=pd.Series([0, 0], index=idx_2),
        community_market_prices={"aachen": pd.Series([0, 0], index=idx_2)},
        supplier_prices=pd.Series([0, 10], index=idx_2),
        solar_generation=pd.Series([0, 0], index=idx_2),
        demand=pd.Series([0, 2.5], index=idx_2),
        hours_per_timestep=4,
        storage_use_cases=["home"],
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=False,
    )
    costs = calculator.optimize(solver="appsi_highs")
    soc_df = calculator.get_storage_soc_timeseries_df()
    flows = calculator.get_energy_flows()

    # without hours_per_timestep the bound would cap the SOC at 9 kWh and force
    # buying the missing kWh at 10 EUR
    assert np.isclose(soc_df["soc_home"].iloc[0], 10)
    assert np.isclose(flows["supplier_to_home"].iloc[1], 0)
    assert np.isclose(costs, 0, atol=1e-3)


def test_ECC_soc_end():
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=0.5, volume=1),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 0], index=idx_3),
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calculator.optimize(solver="highs")
    soc_df = calculator.get_storage_soc_timeseries_df()
    assert soc_df.loc["2025-01-01 02:00:00", "soc_home"] == 0


def test_green_objective_prefers_direct_pv_to_home():
    # PV matches demand exactly -> should be consumed directly
    calc = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=0),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([1, 1, 1], index=idx_3),
        solar_generation=pd.Series([1, 1, 1], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        goal="max_green_energy",
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    calc.optimize(solver="highs")
    flows = calc.get_energy_flows()

    assert (flows["pv_to_home"].values == [1, 1, 1]).all()


def test_green_objective_stores_pv_for_later_home_use():
    # PV available at t=0, demand at t=1 -> with storage, PV should be stored for 'home'
    # although storage could be used for wholesale operation
    calc = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=1),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([-10, 10, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 0, 0], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 1, 0], index=idx_3),
        wholesale_fee=0,
        goal="max_green_energy",
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    calc.optimize(solver="highs")
    flows = calc.get_energy_flows()

    # PV at t=0 should be sent to storage for home use
    assert flows["pv_to_storage_for_home"].iloc[0] == 1
    # storage should discharge to home at t=1 to cover demand
    assert round(flows["storage_to_home"].iloc[1], 0) == 1
    # costs should be 0, as demand can be met by solar generation
    assert round(calc.calculate_costs(), 0) == 0

    calc = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=1),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([-10, 10, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 0, 0], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 1, 0], index=idx_3),
        wholesale_fee=0,
        goal="max_cashflow",
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    calc.optimize(solver="highs")
    # use wholesale operation if goal is set to max cashflow
    assert round(calc.calculate_costs(), 0) == 20


def test_green_objective_respects_no_home_use_case():
    # if 'home' use-case is not present, only direct pv_to_home is considered
    calc = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=1),
        eeg_prices=pd.Series([20, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([1, 1, 1], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 1, 0], index=idx_3),
        storage_use_cases=["eeg"],
        goal="max_green_energy",
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    calc.optimize(solver="highs")
    costs = calc.calculate_costs()
    flows = calc.get_energy_flows()

    # since no 'home' storage use-case exists, pv should not be put into storage
    assert flows["pv_to_storage_for_home"].sum() == 0
    assert (flows["pv_to_eeg"].round(3) == [1, 0, 0]).all()
    assert (flows["pv_to_storage_for_home"].round(3) == [0, 0, 0]).all()
    assert (flows["storage_to_home"].round(3) == [0, 0, 0]).all()
    assert round(costs, 0) == 19


def test_calculate_storage_worth_eeg_eligible():
    storage = Storage(id=0, c_rate=1, volume=0)

    solar_generation_large = pd.Series([1, 1, 1], index=idx_3)
    solar_generation_small = pd.Series([0.5, 0.5, 0.5], index=idx_3)

    ecc_with = EnergyCostCalculator(
        storage=storage,
        eeg_prices=pd.Series([1, 1, 1], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 0, 0], index=idx_3),
        solar_generation=solar_generation_large,
        demand=pd.Series([0, 0, 0], index=idx_3),
        eeg_eligible=True,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs_with_eeg = ecc_with.optimize("appsi_highs")
    assert round(costs_with_eeg) == 3

    ecc_without = EnergyCostCalculator(
        storage=storage,
        eeg_prices=pd.Series([1, 1, 1], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 0, 0], index=idx_3),
        solar_generation=solar_generation_small,
        demand=pd.Series([0, 0, 0], index=idx_3),
        eeg_eligible=False,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs_without_eeg = ecc_without.optimize("appsi_highs")

    assert round(costs_without_eeg) == 0


def test_storage_usage_kpis_and_summary_plot():
    calc = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=1),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([10, 10, 10], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 1, 0], index=idx_3),
        allow_pv_to_wholesale=False,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
    )
    calc.optimize(solver="appsi_highs")

    kpis = calc.get_storage_usage_kpis()

    assert np.isclose(
        sum(kpis["charged_by_source_kwh"].values()), kpis["charged_kwh_total"]
    )
    assert np.isclose(
        sum(kpis["discharged_by_sink_kwh"].values()), kpis["discharged_kwh_total"]
    )
    assert np.isclose(kpis["charged_by_source_kwh"]["pv"], 1.0)
    expected_discharge = (
        calc.storage.charge_efficiency * calc.storage.discharge_efficiency
    )
    assert np.isclose(kpis["discharged_by_sink_kwh"]["home"], expected_discharge)
    assert np.isclose(kpis["full_cycles_equivalent"], expected_discharge)
    assert np.isclose(kpis["roundtrip_indicator"], expected_discharge)

    fig = calc.plot_storage_usage_summary(show=False)
    assert fig is not None


def _optimized_calc_with_activity() -> EnergyCostCalculator:
    """Calculator whose optimum uses PV, storage and every market."""
    calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=2, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 1, 0], index=idx_4),
        wholesale_market_prices=pd.Series([1, 2, 3, 1], index=idx_4),
        community_market_prices={"aachen": pd.Series([1, 2, 1, 2], index=idx_4)},
        supplier_prices=pd.Series([0.3, 0.4, 0.4, 0.3], index=idx_4),
        solar_generation=pd.Series([2, 1, 0, 0], index=idx_4),
        demand=pd.Series([1, 1, 2, 1], index=idx_4),
    )
    calc.optimize(solver="appsi_highs")
    return calc


def test_plot_storage_charge_timeseries_returns_figure():
    calc = _optimized_calc_with_activity()

    fig = calc.plot_storage_charge_timeseries(show=False)

    # the y column comes from melt(value_name="kWh"); a mismatch here used to
    # raise ValueError before the figure was ever built
    assert fig.layout.yaxis.title.text == "kWh"
    assert len(fig.data) > 0

    charge_df = calc.get_storage_charge_timeseries_df()
    plotted = {trace.name: np.asarray(trace.y, dtype=float) for trace in fig.data}
    assert np.isclose(
        sum(values.sum() for values in plotted.values()),
        charge_df.sum().sum(),
    )


@pytest.mark.parametrize(
    "plot_method",
    [
        "plot_energy_flows",
        "plot_demand_coverage",
        "plot_solar_generation",
        "plot_storage_soc_timeseries",
        "plot_storage_charge_timeseries",
        "plot_prices",
        "plot_supplier_costs",
        "plot_storage_usage_summary",
    ],
)
def test_plot_methods_build_a_figure(plot_method):
    calc = _optimized_calc_with_activity()

    fig = getattr(calc, plot_method)(show=False)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_storage_usage_kpis_zero_volume_storage():
    calc = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=0),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([1, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    calc.optimize(solver="appsi_highs")

    kpis = calc.get_storage_usage_kpis()

    assert kpis["charged_kwh_total"] == 0
    assert kpis["discharged_kwh_total"] == 0
    assert kpis["full_cycles_equivalent"] == 0
    assert kpis["utilization_ratio"] == 0
    assert kpis["roundtrip_indicator"] == 0


def test_ECC_grid_fees_reduce_non_wholesale_cashflow():
    base_calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 2, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 0], index=idx_3),
        allow_pv_to_wholesale=False,
        my_location="aachen",
        storage_location="aachen",
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
    )
    base_costs = base_calc.optimize(solver="appsi_highs")

    assert np.isclose(base_costs, 2, atol=1e-3)

    fee_calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 1], index=idx_3),
        allow_pv_to_wholesale=False,
        my_location="aachen",
        storage_location="aachen",
        grid_fee_between_locations={
            "aachen": {"aachen": 0.05},
        },
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        is_rented_storage=True,
    )
    fee_costs = fee_calc.optimize(solver="appsi_highs")
    fee_cashflows = fee_calc.get_cashflows()

    assert fee_costs < base_costs
    assert fee_cashflows["grid_fees"] == -0.05


def test_ECC_grid_fees_do_not_affect_wholesale_only_operations():
    kwargs = dict(
        storage=Storage(id=0, c_rate=1, volume=1),
        demand=pd.Series([0, 0, 0, 0], index=idx_4),
        solar_generation=pd.Series([0, 0, 0, 0], index=idx_4),
        supplier_prices=pd.Series([10, 10, 10, 10], index=idx_4),
        eeg_prices=pd.Series([0, 0, 0, 0], index=idx_4),
        community_market_prices={"aachen": pd.Series([0, 0, 0, 0], index=idx_4)},
        wholesale_market_prices=pd.Series([0, 0, 5, 5], index=idx_4),
        wholesale_fee=0.0,
    )

    no_fee = EnergyCostCalculator(
        **kwargs,
        my_location="aachen",
        storage_location="aachen",
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    with_fee = EnergyCostCalculator(
        **kwargs,
        my_location="aachen",
        storage_location="liege",
        grid_fee_between_locations={
            "aachen": {"liege": 5.0},
        },
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
        is_rented_storage=True,
    )

    no_fee_costs = no_fee.optimize(solver="appsi_highs")
    with_fee_costs = with_fee.optimize(solver="appsi_highs")

    assert np.isclose(no_fee_costs, with_fee_costs)
    assert np.isclose(with_fee.get_cashflows()["grid_fees"], 0.0)


def test_ECC_grid_location_ordering_changes_costs():
    costs_by_storage_location = {}
    for storage_location in ["aachen", "heerlen", "liege", "juelich"]:
        calc = EnergyCostCalculator(
            storage=Storage(
                id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
            ),
            eeg_prices=pd.Series([0, 0, 0], index=idx_3),
            wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
            community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
            supplier_prices=pd.Series([0, 0, 0], index=idx_3),
            solar_generation=pd.Series([1, 0, 0], index=idx_3),
            demand=pd.Series([0, 0, 1], index=idx_3),
            allow_pv_to_wholesale=False,
            my_location="aachen",
            storage_location=storage_location,
            allow_community_to_home=False,
            allow_community_to_storage=False,
            allow_pv_to_community=False,
            allow_storage_to_community=False,
            is_rented_storage=True,
        )
        costs_by_storage_location[storage_location] = calc.optimize(
            solver="appsi_highs"
        )

    # Storage in my_location avoids any inter-location grid fees (tenant perspective).
    assert costs_by_storage_location["aachen"] >= costs_by_storage_location["juelich"]
    assert costs_by_storage_location["aachen"] >= costs_by_storage_location["heerlen"]
    assert costs_by_storage_location["aachen"] >= costs_by_storage_location["liege"]


def test_ECC_own_storage_requires_storage_at_my_location():
    with pytest.raises(ValueError, match="storage_location must equal my_location"):
        EnergyCostCalculator(
            storage=Storage(id=0, c_rate=1, volume=1),
            eeg_prices=pd.Series([0, 0, 0], index=idx_3),
            wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
            community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
            supplier_prices=pd.Series([0, 0, 0], index=idx_3),
            solar_generation=pd.Series([0, 0, 0], index=idx_3),
            demand=pd.Series([0, 0, 0], index=idx_3),
            my_location="aachen",
            storage_location="liege",
            is_rented_storage=False,
        )


def test_ECC_disables_supplier_to_storage_for_non_local_supplier(caplog):
    caplog.set_level(logging.WARNING, logger="battery_utility")
    calc = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=1),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        my_location="aachen",
        storage_location="liege",
        is_rented_storage=True,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
    )
    calc.optimize(solver="appsi_highs")
    flows = calc.get_energy_flows()

    assert np.allclose(flows["supplier_to_storage"], 0.0)
    assert any(
        "Disabled supplier_to_storage use-case because storage_location" in rec.message
        for rec in caplog.records
    )


def test_ECC_community_location_not_in_grid_fee_raises():
    with pytest.raises(ValueError, match="Unknown location"):
        EnergyCostCalculator(
            storage=Storage(id=0, c_rate=1, volume=0),
            eeg_prices=pd.Series([0, 0, 0], index=idx_3),
            wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
            community_market_prices={
                "unknown_location": pd.Series([0, 0, 0], index=idx_3)
            },
            supplier_prices=pd.Series([0, 0, 0], index=idx_3),
            solar_generation=pd.Series([0, 0, 0], index=idx_3),
            demand=pd.Series([0, 0, 0], index=idx_3),
            allow_community_to_home=False,
            allow_community_to_storage=False,
            allow_pv_to_community=False,
            allow_storage_to_community=False,
            allow_pv_to_wholesale=True,
        )


def test_ECC_multi_location_community_routes_by_price():
    calc = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=0),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={
            "aachen": pd.Series([5, 5, 5], index=idx_3),
            "liege": pd.Series([1, 1, 1], index=idx_3),
        },
        supplier_prices=pd.Series([10, 10, 10], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 1], index=idx_3),
        allow_pv_to_community=True,
        my_location="aachen",
        grid_fee_between_locations={
            "aachen": {"aachen": 0.0, "liege": 0.1},
            "liege": {"aachen": 0.1, "liege": 0.0},
        },
        allow_community_to_home=True,
        allow_community_to_storage=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
        is_rented_storage=True,
    )
    costs = calc.optimize(solver="appsi_highs")
    flows = calc.get_energy_flows()

    assert flows["pv_to_community_aachen"].iloc[0] == 1.0
    assert flows["pv_to_community_liege"].iloc[0] == 0.0
    assert flows["pv_to_community"].iloc[0] == 1.0
    assert np.isclose(costs, 3.9)


def test_ECC_remote_storage():
    calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"liege": pd.Series([5, 5, 5], index=idx_3)},
        supplier_prices=pd.Series([10, 10, 10], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 1], index=idx_3),
        my_location="aachen",
        storage_location="liege",
        grid_fee_between_locations={
            "aachen": {"aachen": 0.0, "liege": 1},
            "liege": {"aachen": 1, "liege": 0.0},
        },
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
        is_rented_storage=True,
    )

    costs = calc.optimize(solver="appsi_highs")
    flows = calc.get_energy_flows()
    grid_fees = calc.calculate_grid_fee_cashflow(use_values=True)

    assert flows["pv_to_storage_for_home"].iloc[0] == 1.0
    assert flows["storage_to_home"].iloc[2] == 1.0
    assert np.isclose(costs, -1)
    assert np.isclose(grid_fees, -1)


def test_ECC_rented_storage_same_city_charges_tenant_flows():
    calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([10, 10, 10], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 1], index=idx_3),
        my_location="aachen",
        storage_location="aachen",
        grid_fee_between_locations={"aachen": {"aachen": 0.05}},
        storage_use_cases=["home"],
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=False,
        is_rented_storage=True,
    )
    costs = calc.optimize(solver="appsi_highs")
    flows = calc.get_energy_flows()
    grid_fees = calc.calculate_grid_fee_cashflow(use_values=True)

    assert np.isclose(costs, -0.05, atol=1e-3)
    assert flows["pv_to_storage_for_home"].iloc[0] == 1.0
    assert flows["storage_to_home"].iloc[2] == 1.0
    assert np.isclose(grid_fees, -0.05)


def test_ECC_storage_provider_no_grid_fees_on_btm_pv_shift():
    calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([10, 10, 10], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 1], index=idx_3),
        my_location="aachen",
        storage_location="aachen",
        grid_fee_between_locations={
            "aachen": {"aachen": 0.0, "liege": 5.0},
            "liege": {"aachen": 5.0, "liege": 0.0},
        },
        storage_use_cases=["home"],
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=False,
        is_rented_storage=False,
    )
    calc.optimize(solver="appsi_highs")
    flows = calc.get_energy_flows()

    assert flows["pv_to_storage_for_home"].iloc[0] == 1.0
    assert flows["storage_to_home"].iloc[2] == 1.0
    assert np.isclose(calc.get_cashflows()["grid_fees"], 0.0)


def test_ECC_storage_provider_charges_community_import_only():
    calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={
            "heerlen": pd.Series([0, 10, 10], index=idx_3),
            "liege": pd.Series([10, 10, 10], index=idx_3),
        },
        supplier_prices=pd.Series([50, 50, 50], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 1], index=idx_3),
        my_location="aachen",
        storage_location="aachen",
        grid_fee_between_locations={
            "aachen": {"aachen": 0.0, "liege": 0.2, "heerlen": 0.2},
            "liege": {"aachen": 0.2, "liege": 0.0, "heerlen": 0.2},
            "heerlen": {"aachen": 0.2, "heerlen": 0.0, "liege": 0.2},
        },
        storage_use_cases=["home"],
        allow_community_to_storage=True,
        allow_community_to_home=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=False,
        is_rented_storage=False,
    )
    calc.optimize(solver="appsi_highs")
    flows = calc.get_energy_flows()
    grid_fees = calc.calculate_grid_fee_cashflow(use_values=True)

    assert calc.model.community_to_storage_for_home[0, "heerlen"].value == 1.0
    assert np.isclose(flows["pv_to_storage_for_home"].sum(), 0.0)
    assert np.isclose(grid_fees, -0.2)


def test_ECC_storage_provider_charges_remote_community_to_home():
    # community_to_home never touches the storage, so the import is charged
    # even though the prosumer owns the storage at its own location
    calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=0, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"liege": pd.Series([1, 1, 1], index=idx_3)},
        supplier_prices=pd.Series([50, 50, 50], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 1], index=idx_3),
        my_location="aachen",
        storage_location="aachen",
        grid_fee_between_locations={
            "aachen": {"aachen": 0.0, "liege": 0.2},
            "liege": {"aachen": 0.2, "liege": 0.0},
        },
        allow_community_to_home=True,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=False,
        is_rented_storage=False,
    )
    costs = calc.optimize(solver="appsi_highs")
    flows = calc.get_energy_flows()
    cashflows = calc.get_cashflows()

    assert flows["community_to_home_liege"].iloc[2] == 1.0
    assert np.isclose(cashflows["grid_fees"], -0.2)
    assert np.isclose(costs, -1.2)


def test_ECC_local_community_to_home_stays_fee_free():
    # same setup but the community market sits in my_location -> zero rate
    calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=0, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([1, 1, 1], index=idx_3)},
        supplier_prices=pd.Series([50, 50, 50], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 1], index=idx_3),
        my_location="aachen",
        storage_location="aachen",
        allow_community_to_home=True,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=False,
        is_rented_storage=False,
    )
    costs = calc.optimize(solver="appsi_highs")
    cashflows = calc.get_cashflows()

    assert np.isclose(cashflows["grid_fees"], 0.0)
    assert np.isclose(costs, -1)


def _no_balance_calc(**kwargs) -> EnergyCostCalculator:
    """Storage with no PV, no demand and no supply - it cannot hold any energy."""
    args = dict(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([50, 50, 50], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 0], index=idx_3),
        wholesale_fee=0.0,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
    )
    args.update(kwargs)
    return EnergyCostCalculator(**args)


def test_ECC_storage_to_wholesale_needs_its_use_case():
    # without the wholesale SOC balance the flow would be unbacked and the model
    # could sell energy it never stored
    calc = _no_balance_calc(
        storage_use_cases=["home"],
        wholesale_market_prices=pd.Series([10, 10, 10], index=idx_3),
        allow_storage_to_wholesale=True,
    )
    objective = calc.optimize(solver="appsi_highs")

    sold = sum(calc.model.storage_to_wholesale[t].value for t in calc.timesteps)
    assert np.isclose(sold, 0.0)
    assert np.isclose(objective, 0.0)


def test_ECC_wholesale_to_storage_needs_its_use_case():
    # with negative prices, charging into a bucket that does not exist would pay
    calc = _no_balance_calc(
        storage_use_cases=["home"],
        wholesale_market_prices=pd.Series([-10, -10, -10], index=idx_3),
        allow_wholesale_to_storage=True,
        allow_storage_to_wholesale=False,
    )
    objective = calc.optimize(solver="appsi_highs")

    bought = sum(calc.model.wholesale_to_storage[t].value for t in calc.timesteps)
    assert np.isclose(bought, 0.0)
    assert np.isclose(objective, 0.0)


def test_ECC_storage_to_community_needs_its_use_case():
    calc = _no_balance_calc(
        storage_use_cases=["home"],
        community_market_prices={"aachen": pd.Series([10, 10, 10], index=idx_3)},
        allow_storage_to_community=True,
        allow_storage_to_wholesale=False,
    )
    objective = calc.optimize(solver="appsi_highs")

    sold = sum(
        calc.model.storage_to_community[t, "aachen"].value for t in calc.timesteps
    )
    assert np.isclose(sold, 0.0)
    assert np.isclose(objective, 0.0)


def test_ECC_supplier_to_storage_needs_the_home_use_case():
    calc = _no_balance_calc(
        storage_use_cases=["wholesale"],
        supplier_prices=pd.Series([-10, -10, -10], index=idx_3),
        allow_storage_to_wholesale=False,
        allow_wholesale_to_storage=False,
    )
    objective = calc.optimize(solver="appsi_highs")

    charged = sum(calc.model.supplier_to_storage[t].value for t in calc.timesteps)
    assert np.isclose(charged, 0.0)
    assert np.isclose(objective, 0.0)


def test_ECC_full_use_cases_still_allow_storage_trading():
    # the guard must not close the paths in the default configuration
    calc = _no_balance_calc(
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 10, 10], index=idx_3),
        allow_storage_to_wholesale=True,
    )
    calc.optimize(solver="appsi_highs")

    sold = sum(calc.model.storage_to_wholesale[t].value for t in calc.timesteps)
    assert sold > 0


def test_ECC_energy_flows_report_non_storage_flows_without_use_cases():
    # pv_to_eeg and supplier_to_home never touch the storage, so restricting
    # storage_use_cases must not zero them out in the report
    calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=0, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([5, 5, 5], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx_3)},
        supplier_prices=pd.Series([1, 1, 1], index=idx_3),
        solar_generation=pd.Series([2, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        storage_use_cases=["wholesale"],
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
    )
    calc.optimize(solver="appsi_highs")
    flows = calc.get_energy_flows()

    assert np.isclose(flows["pv_to_eeg"].sum(), 2.0)
    assert np.isclose(flows["supplier_to_home"].sum(), 3.0)
    # the report must add up to the demand it states
    covered = (
        flows["pv_to_home"] + flows["supplier_to_home"] + flows["storage_to_home"]
    ) + flows["community_to_home"]
    assert np.allclose(covered, flows["demand"])


def test_ECC_energy_flows_report_community_imports_without_community_use_case():
    calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([1, 1, 1], index=idx_3)},
        supplier_prices=pd.Series([50, 50, 50], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 1], index=idx_3),
        storage_use_cases=["home"],
        allow_community_to_storage=True,
        allow_community_to_home=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_storage_to_wholesale=False,
    )
    calc.optimize(solver="appsi_highs")
    flows = calc.get_energy_flows()

    charged = sum(
        calc.model.community_to_storage_for_home[timestep, "aachen"].value
        for timestep in calc.timesteps
    )
    assert np.isclose(charged, 1.0)
    assert np.isclose(flows["community_to_storage"].sum(), charged)
    assert np.isclose(flows["community_to_storage_aachen"].sum(), charged)
    assert np.isclose(
        calc.get_storage_usage_kpis()["charged_by_source_kwh"]["community"], charged
    )


def test_ECC_energy_flows_report_pv_to_community_without_community_use_case():
    calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=0, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={"aachen": pd.Series([5, 5, 5], index=idx_3)},
        supplier_prices=pd.Series([50, 50, 50], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 0], index=idx_3),
        storage_use_cases=["home"],
        allow_pv_to_community=True,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_storage_to_community=False,
    )
    calc.optimize(solver="appsi_highs")
    flows = calc.get_energy_flows()

    # pv_to_community bypasses the storage entirely
    assert np.isclose(flows["pv_to_community"].sum(), 1.0)
    assert np.isclose(flows["pv_to_community_aachen"].sum(), 1.0)


def test_ECC_no_community_market_when_none_or_empty():
    common_kwargs = dict(
        storage=Storage(id=0, c_rate=1, volume=1),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 1, 1], index=idx_3),
        allow_pv_to_community=True,
        allow_community_to_home=True,
        allow_community_to_storage=True,
        allow_storage_to_community=True,
        allow_pv_to_wholesale=True,
    )

    for community_market_prices in (None, {}):
        calc = EnergyCostCalculator(
            **common_kwargs,
            community_market_prices=community_market_prices,
        )
        assert calc.has_community_market is False
        calc.optimize(solver="appsi_highs")
        flows = calc.get_energy_flows()

        assert np.allclose(flows["pv_to_community"], 0.0)
        assert np.allclose(flows["community_to_home"], 0.0)
        assert np.allclose(flows["storage_to_community"], 0.0)
        assert np.allclose(flows["community_to_storage"], 0.0)
        assert calc.get_cashflows()["community"] == 0.0


def test_ECC_remote_storage_remote_community_market():
    calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices={
            "heerlen": pd.Series([0, 10, 10], index=idx_3),
            "liege": pd.Series([10, 10, 10], index=idx_3),
        },
        supplier_prices=pd.Series([50, 50, 50], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 1], index=idx_3),
        my_location="aachen",
        storage_location="liege",
        allow_community_to_storage=True,
        allow_community_to_home=True,
        grid_fee_between_locations={
            "aachen": {"aachen": 0.0, "liege": 1, "heerlen": 1},
            "liege": {"aachen": 1, "liege": 0.0, "heerlen": 2},
            "heerlen": {"aachen": 1, "heerlen": 0.0, "liege": 2},
        },
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        allow_pv_to_wholesale=True,
        is_rented_storage=True,
    )
    costs = calc.optimize(solver="appsi_highs")
    flows = calc.get_energy_flows()
    grid_fees = calc.calculate_grid_fee_cashflow(use_values=True)

    assert flows["community_to_storage_for_home_heerlen"].iloc[0] == 1.0
    assert flows["community_to_storage_heerlen"].iloc[0] == 1.0
    assert flows["storage_to_home"].iloc[2] == 1.0
    assert np.isclose(costs, -3)
    assert np.isclose(grid_fees, -3)


def test_ECC_community_market_arbitrage():
    calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0], index=idx_2),
        wholesale_market_prices=pd.Series([0, 0], index=idx_2),
        community_market_prices={"aachen": pd.Series([1, 10], index=idx_2)},
        supplier_prices=pd.Series([100, 100], index=idx_2),
        solar_generation=pd.Series([0, 0], index=idx_2),
        demand=pd.Series([0, 0], index=idx_2),
        allow_community_market_arbitrage=True,
        allow_storage_to_community=True,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_pv_to_wholesale=True,
    )
    costs = calc.optimize(solver="appsi_highs")
    flows = calc.get_energy_flows()

    assert flows["community_to_storage_for_community_aachen"].iloc[0] == 1.0
    assert flows["storage_to_community_aachen"].iloc[1] == 1.0
    assert flows["supplier_to_home"].sum() == 0.0
    assert np.isclose(costs, 9)


def test_ECC_pv_via_storage_to_community():
    calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0], index=idx_2),
        wholesale_market_prices=pd.Series([0, 0], index=idx_2),
        community_market_prices={"aachen": pd.Series([0, 5], index=idx_2)},
        supplier_prices=pd.Series([100, 100], index=idx_2),
        solar_generation=pd.Series([1, 0], index=idx_2),
        demand=pd.Series([0, 0], index=idx_2),
        allow_storage_to_community=True,
        allow_community_market_arbitrage=False,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_pv_to_wholesale=True,
    )
    calc.optimize(solver="appsi_highs")
    flows = calc.get_energy_flows()

    assert flows["pv_to_storage_for_community"].iloc[0] == 1.0
    assert flows["storage_to_community_aachen"].iloc[1] == 1.0
    assert flows["community_to_storage_for_community"].sum() == 0.0


def _rented_pv_calc(**kwargs) -> EnergyCostCalculator:
    """Rented storage that has to split 2 kWh of PV between the home and eeg sinks.

    PV arrives at t0, the eeg price peaks at t1 and the demand falls at t2, so one
    kWh is worth storing for each sink. The supplier is expensive enough that the
    demand is never covered from the grid.
    """
    args = dict(
        storage=Storage(
            id=0, c_rate=1, volume=2, charge_efficiency=1, discharge_efficiency=1
        ),
        demand=pd.Series([0, 0, 1], index=idx_3),
        solar_generation=pd.Series([2, 0, 0], index=idx_3),
        supplier_prices=pd.Series([10, 10, 10], index=idx_3),
        eeg_prices=pd.Series([0, 2, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=None,
        storage_use_cases=["home", "eeg"],
        my_location="aachen",
        storage_location="aachen",
        grid_fee_between_locations={"aachen": {"aachen": 0.05}},
        is_rented_storage=True,
        allow_pv_to_wholesale=False,
        allow_wholesale_to_storage=False,
        allow_storage_to_wholesale=False,
    )
    args.update(kwargs)
    calc = EnergyCostCalculator(**args)
    calc.optimize(solver="appsi_highs")
    return calc


def test_ECC_rented_pv_grid_fee_charges_only_the_home_sink():
    # PV kept for the household comes back as its own withdrawal and is charged;
    # PV routed through the storage to the eeg sink is sold on, so the fee is
    # the buyer's and must not appear here
    calc = _rented_pv_calc(rented_pv_grid_fee=0.07)
    flows = calc.get_energy_flows()

    assert np.isclose(flows["pv_to_storage_for_home"].sum(), 1.0)
    assert np.isclose(flows["pv_to_storage_for_eeg"].sum(), 1.0)
    assert np.isclose(flows["supplier_to_storage"].sum(), 0.0)
    assert np.isclose(calc.get_cashflows()["grid_fees"], -0.07)


def test_ECC_rented_pv_grid_fee_none_charges_every_use_case():
    # the default keeps the previous behaviour: both kWh at the location rate
    calc = _rented_pv_calc(rented_pv_grid_fee=None)
    flows = calc.get_energy_flows()

    assert np.isclose(flows["pv_to_storage_for_home"].sum(), 1.0)
    assert np.isclose(flows["pv_to_storage_for_eeg"].sum(), 1.0)
    assert np.isclose(calc.get_cashflows()["grid_fees"], -0.10)


def test_ECC_rented_pv_grid_fee_spares_the_community_sink():
    calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        demand=pd.Series([0, 0], index=idx_2),
        solar_generation=pd.Series([1, 0], index=idx_2),
        supplier_prices=pd.Series([100, 100], index=idx_2),
        eeg_prices=pd.Series([0, 0], index=idx_2),
        wholesale_market_prices=pd.Series([0, 0], index=idx_2),
        community_market_prices={"aachen": pd.Series([0, 5], index=idx_2)},
        storage_use_cases=["community"],
        my_location="aachen",
        storage_location="aachen",
        is_rented_storage=True,
        rented_pv_grid_fee=0.07,
        allow_storage_to_community=True,
        allow_community_market_arbitrage=False,
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_pv_to_wholesale=True,
    )
    calc.optimize(solver="appsi_highs")
    flows = calc.get_energy_flows()

    assert np.isclose(flows["pv_to_storage_for_community"].sum(), 1.0)
    assert np.isclose(flows["storage_to_community_aachen"].iloc[1], 1.0)
    assert np.isclose(calc.get_cashflows()["grid_fees"], 0.0)


def test_ECC_rented_pv_grid_fee_leaves_the_supplier_charge_untouched():
    # both flows charge the same storage in the same timestep, each at its own
    # rate: PV at rented_pv_grid_fee, the supplier at the location rate
    calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=2, charge_efficiency=1, discharge_efficiency=1
        ),
        demand=pd.Series([0, 0, 2], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        supplier_prices=pd.Series([1, 10, 10], index=idx_3),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=None,
        storage_use_cases=["home"],
        my_location="aachen",
        storage_location="aachen",
        grid_fee_between_locations={"aachen": {"aachen": 0.05}},
        is_rented_storage=True,
        rented_pv_grid_fee=0.07,
        allow_pv_to_wholesale=False,
        allow_wholesale_to_storage=False,
        allow_storage_to_wholesale=False,
    )
    calc.optimize(solver="appsi_highs")
    flows = calc.get_energy_flows()

    assert np.isclose(flows["pv_to_storage_for_home"].sum(), 1.0)
    assert np.isclose(flows["supplier_to_storage"].sum(), 1.0)
    assert np.isclose(calc.get_cashflows()["grid_fees"], -(0.07 + 0.05))


def test_ECC_rented_pv_grid_fee_only_applies_to_rented_storage():
    calc = _rented_pv_calc(rented_pv_grid_fee=0.07, is_rented_storage=False)

    assert np.isclose(calc.get_cashflows()["grid_fees"], 0.0)


def test_ECC_rented_pv_grid_fee_without_home_use_case_is_zero():
    calc = _rented_pv_calc(
        rented_pv_grid_fee=0.07,
        storage_use_cases=["eeg"],
        demand=pd.Series([0, 0, 0], index=idx_3),
    )
    flows = calc.get_energy_flows()

    assert np.isclose(flows["pv_to_storage_for_eeg"].sum(), 2.0)
    assert np.isclose(calc.get_cashflows()["grid_fees"], 0.0)


def community_gap_calculator(**overrides) -> EnergyCostCalculator:
    """Community market cheaper than the supplier, but closed in the middle hour."""
    kwargs = dict(
        storage=None,
        eeg_prices=pd.Series([0.0, 0.0, 0.0], index=idx_3),
        wholesale_market_prices=pd.Series([0.0, 0.0, 0.0], index=idx_3),
        community_market_prices=community_prices(idx_3, values=[0.05, np.nan, 0.05]),
        supplier_prices=pd.Series([0.30, 0.30, 0.30], index=idx_3),
        solar_generation=pd.Series([0.0, 0.0, 0.0], index=idx_3),
        demand=pd.Series([1.0, 1.0, 1.0], index=idx_3),
        allow_pv_to_wholesale=False,
    )
    kwargs.update(overrides)
    return EnergyCostCalculator(**kwargs)


def test_ECC_community_nan_blocks_buying_in_that_timestep():
    calculator = community_gap_calculator()
    costs = calculator.optimize(solver="appsi_highs")
    coverage = calculator.get_demand_coverage_timeseries_df()

    assert np.isfinite(costs)
    assert coverage["from_community"].tolist() == pytest.approx([1.0, 0.0, 1.0])
    assert coverage["from_grid"].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert np.isclose(costs, -(0.05 + 0.30 + 0.05))


def test_ECC_community_nan_blocks_selling_in_that_timestep():
    calculator = community_gap_calculator(
        solar_generation=pd.Series([1.0, 1.0, 1.0], index=idx_3),
        demand=pd.Series([0.0, 0.0, 0.0], index=idx_3),
        community_market_prices=community_prices(idx_3, values=[0.10, np.nan, 0.10]),
    )
    calculator.optimize(solver="appsi_highs")
    pv_usage = calculator.get_solar_generation_timeseries_df()

    assert pv_usage["to_community"].tolist() == pytest.approx([1.0, 0.0, 1.0])


def test_ECC_community_nan_blocks_every_storage_flow():
    # the community bucket would buy cheap and sell dear right across the gap;
    # neither leg may land in the blocked hour
    calculator = community_gap_calculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        demand=pd.Series([0.0, 0.0, 0.0], index=idx_3),
        community_market_prices=community_prices(idx_3, values=[np.nan, 0.01, np.nan]),
        storage_use_cases=["community", "home"],
    )
    calculator.optimize(solver="appsi_highs")
    flows = calculator.get_energy_flows()

    for column in ("storage_to_community", "community_to_storage"):
        assert flows[column].iloc[0] == 0.0
        assert flows[column].iloc[2] == 0.0
    for var_name in (
        "pv_to_community",
        "storage_to_community",
        "community_to_storage_for_home",
        "community_to_storage_for_community",
        "community_to_home",
    ):
        variable = getattr(calculator.model, var_name)
        assert variable[0, "aachen"].fixed
        assert not variable[1, "aachen"].fixed
        assert variable[2, "aachen"].fixed


def test_ECC_community_nan_is_per_location():
    calculator = community_gap_calculator(
        community_market_prices={
            "aachen": pd.Series([0.05, np.nan, 0.05], index=idx_3),
            "juelich": pd.Series([0.05, 0.05, 0.05], index=idx_3),
        },
        grid_fee_between_locations={
            "aachen": {"juelich": 0.0},
            "juelich": {"aachen": 0.0},
        },
    )
    calculator.optimize(solver="appsi_highs")
    coverage = calculator.get_demand_coverage_timeseries_df()

    assert coverage["from_community_aachen"].iloc[1] == 0.0
    assert np.isclose(coverage["from_community_juelich"].iloc[1], 1.0)


def test_ECC_community_nan_is_reported_in_the_price_df():
    calculator = community_gap_calculator()
    prices = calculator.get_price_df()["community_market_prices_aachen"]

    assert np.isnan(prices.iloc[1])
    assert prices.iloc[[0, 2]].tolist() == pytest.approx([0.05, 0.05])


def test_ECC_community_inf_is_still_rejected():
    with pytest.raises(ValueError, match="infinite"):
        community_gap_calculator(
            community_market_prices=community_prices(idx_3, values=[0.05, np.inf, 0.05])
        )


def test_ECC_nan_outside_the_community_market_is_still_rejected():
    with pytest.raises(ValueError, match="supplier_prices contains 1 NaN"):
        community_gap_calculator(
            supplier_prices=pd.Series([0.30, np.nan, 0.30], index=idx_3)
        )
