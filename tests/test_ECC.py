# SPDX-FileCopyrightText: Christoph Komanns, Florian Maurer, Ralf Schemm
#
# SPDX-License-Identifier: MIT

import inspect

import numpy as np
import pandas as pd

from battery_utility_calculator.energy_costs_calculator import (
    EnergyCostCalculator,
    Storage,
)

idx_2 = pd.date_range("2025-01-01", freq="h", periods=2)
idx_3 = pd.date_range("2025-01-01", freq="h", periods=3)
idx_4 = pd.date_range("2025-01-01", freq="h", periods=4)
idx_5 = pd.date_range("2025-01-01", freq="h", periods=5)


def test_ECC_baseline():
    # buying 1 kWh for 1 €/kWh should equal to 3€ total
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=0),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([1, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
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
    }


def test_ECC_opti_storage():
    # buying 2 kWh for 0€/kWh and storing 1 kWh of this should equal 1€ total
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=1),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
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
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([2, 2, 2], index=idx_3),
    )
    costs = calculator.optimize(solver="highs")
    assert round(costs) == -3


def test_ECC_selling_pv():
    # here we should gain 1€ from selling pv
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=0),
        eeg_prices=pd.Series([1, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([1, 1, 1], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 0], index=idx_3),
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
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([1, 1, 1], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 0], index=idx_3),
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
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([5, 10, 20], index=idx_3),
        solar_generation=pd.Series([1, 1, 0], index=idx_3),
        demand=pd.Series([0, 0, 2], index=idx_3),
    )
    costs = calculator.optimize(solver="highs")
    assert round(costs, 0) == 0


def test_ECC_negative_prices():
    # buy 2 kWh in ts=1 because we get paid for this
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=2),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([5, -20, 5], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 2], index=idx_3),
    )
    costs = calculator.optimize(solver="highs")
    assert round(costs, 0) == 40


def test_ECC_c_rate():
    # check if c_rate is respected
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=0.5, volume=2),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 10, 0], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([2, 2, 0], index=idx_3),
    )
    costs = calculator.optimize(solver="highs")
    assert round(costs, 0) == -10


def test_ECC_wholesale():
    storage = Storage(id=0, c_rate=1, volume=1)

    storage = Storage(id=0, c_rate=1, volume=1)
    # no demand, just buying from wholesale when price is 0 and selling again when price is 5
    # should be able to just do this once, as we only have volume of 1
    # buy for 0, sell for 5 -> gain of 5€, but 50% fee -> 2.5€
    ecc = EnergyCostCalculator(
        storage=storage,
        demand=pd.Series([0, 0, 0, 0], index=idx_4),
        solar_generation=pd.Series([0, 0, 0, 0], index=idx_4),
        supplier_prices=pd.Series([10, 10, 10, 10], index=idx_4),
        eeg_prices=pd.Series([0, 0, 0, 0], index=idx_4),
        community_market_prices=pd.Series([0, 0, 0, 0], index=idx_4),
        wholesale_market_prices=pd.Series([0, 0, 5, 5], index=idx_4),
        wholesale_fee=0.5,
    )
    costs = ecc.optimize(solver="highs")
    assert np.isclose(costs, 2.449999, atol=1e-6)

    # same as above, but volume of 2, so should be able to do two times for total gain of 4
    # no fee, so 100% of profit goes to customer
    storage = Storage(id=0, c_rate=1, volume=2)
    ecc = EnergyCostCalculator(
        storage=storage,
        demand=pd.Series([0, 0, 0, 0], index=idx_4),
        solar_generation=pd.Series([0, 0, 0, 0], index=idx_4),
        supplier_prices=pd.Series([10, 10, 10, 10], index=idx_4),
        eeg_prices=pd.Series([0, 0, 0, 0], index=idx_4),
        community_market_prices=pd.Series([0, 0, 0, 0], index=idx_4),
        wholesale_market_prices=pd.Series([3, 3, 5, 5], index=idx_4),
        wholesale_fee=0,
    )
    costs = ecc.optimize(solver="highs")
    assert round(costs, 0) == 4


def test_ECC_pv_to_wholesale_toggle_sets_bounds():
    common_kwargs = dict(
        storage=Storage(id=0, c_rate=1, volume=0),
        demand=pd.Series([0, 0, 0], index=idx_3),
        solar_generation=pd.Series([1, 1, 1], index=idx_3),
        supplier_prices=pd.Series([0, 0, 0], index=idx_3),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([10, 10, 10], index=idx_3),
        wholesale_fee=0.0,
    )

    ecc_disabled = EnergyCostCalculator(
        **common_kwargs,
        allow_pv_to_wholesale=False,
    )
    ecc_enabled = EnergyCostCalculator(
        **common_kwargs,
        allow_pv_to_wholesale=True,
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
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
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
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
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
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
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
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        discharge_penalty_per_kwh=0.0,
    )
    base_costs = base_calc.optimize(solver="appsi_highs")

    penalized_calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        discharge_penalty_per_kwh=0.1,
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
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        cycle_cost_per_kwh=0.0,
    )
    base_costs = base_calc.optimize(solver="appsi_highs")

    cycle_cost_calc = EnergyCostCalculator(
        storage=Storage(
            id=0, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
        ),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        cycle_cost_per_kwh=0.05,
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
        community_market_prices=pd.Series([0, 0, 0, 0], index=idx_4),
        supplier_prices=pd.Series([1, 1, 1, 1], index=idx_4),
        solar_generation=pd.Series([0, 0, 0, 0], index=idx_4),
        demand=pd.Series([1, 1, 1, 1], index=idx_4),
        hours_per_timestep=0.25,
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
        community_market_prices=pd.Series([0, 0], index=idx_2),
        supplier_prices=pd.Series([100, 10], index=idx_2),
        solar_generation=pd.Series([1, 0], index=idx_2),
        demand=pd.Series([0, 1], index=idx_2),
        hours_per_timestep=0.25,
        allow_pv_to_wholesale=False,
        storage_use_cases=["home"],
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
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 100, 100], index=idx_3),
        solar_generation=pd.Series([2, 0, 0], index=idx_3),
        demand=pd.Series([0, 0, 1], index=idx_3),
        hours_per_timestep=0.25,
        allow_pv_to_wholesale=False,
        storage_use_cases=["home"],
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
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
    )
    costs = calculator.optimize(solver="highs")
    soc_df = calculator.get_storage_soc_timeseries_df()
    assert round(soc_df.loc["2025-01-01 00:00:00", "soc_home"], 1) == 0.5


def test_ECC_soc_end():
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=0.5, volume=1),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 0], index=idx_3),
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
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([1, 1, 1], index=idx_3),
        solar_generation=pd.Series([1, 1, 1], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
        goal="max_green_energy",
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
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 0, 0], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 1, 0], index=idx_3),
        wholesale_fee=0,
        goal="max_green_energy",
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
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 0, 0], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 1, 0], index=idx_3),
        wholesale_fee=0,
        goal="max_cashflow",
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
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([1, 1, 1], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 1, 0], index=idx_3),
        storage_use_cases=["eeg"],
        goal="max_green_energy",
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
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 0, 0], index=idx_3),
        solar_generation=solar_generation_large,
        demand=pd.Series([0, 0, 0], index=idx_3),
        eeg_eligible=True,
    )
    costs_with_eeg = ecc_with.optimize("appsi_highs")
    assert round(costs_with_eeg) == 3

    ecc_without = EnergyCostCalculator(
        storage=storage,
        eeg_prices=pd.Series([1, 1, 1], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([0, 0, 0], index=idx_3),
        solar_generation=solar_generation_small,
        demand=pd.Series([0, 0, 0], index=idx_3),
        eeg_eligible=False,
    )
    costs_without_eeg = ecc_without.optimize("appsi_highs")

    assert round(costs_without_eeg) == 0


def test_storage_usage_kpis_and_summary_plot():
    calc = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=1),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([10, 10, 10], index=idx_3),
        solar_generation=pd.Series([1, 0, 0], index=idx_3),
        demand=pd.Series([0, 1, 0], index=idx_3),
        allow_pv_to_wholesale=False,
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


def test_storage_usage_kpis_zero_volume_storage():
    calc = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=0),
        eeg_prices=pd.Series([0, 0, 0], index=idx_3),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx_3),
        community_market_prices=pd.Series([0, 0, 0], index=idx_3),
        supplier_prices=pd.Series([1, 1, 1], index=idx_3),
        solar_generation=pd.Series([0, 0, 0], index=idx_3),
        demand=pd.Series([1, 1, 1], index=idx_3),
    )
    calc.optimize(solver="appsi_highs")

    kpis = calc.get_storage_usage_kpis()

    assert kpis["charged_kwh_total"] == 0
    assert kpis["discharged_kwh_total"] == 0
    assert kpis["full_cycles_equivalent"] == 0
    assert kpis["utilization_ratio"] == 0
    assert kpis["roundtrip_indicator"] == 0
