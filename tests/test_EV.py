# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

from battery_utility_calculator import (
    ElectricVehicle,
    EnergyCostCalculator,
    Storage,
    calculate_multiple_storage_worth,
    calculate_multiple_storage_worth_by_location,
    calculate_storage_worth,
    calculate_storage_worth_distribution,
    plot_multiple_storage_worth_cashflows,
    sample_scenarios,
)

SOLVER = "appsi_highs"

idx_4 = pd.date_range("2025-01-01", freq="h", periods=4)
idx_5 = pd.date_range("2025-01-01", freq="h", periods=5)
idx_6 = pd.date_range("2025-01-01", freq="h", periods=6)

# keeps the household on PV, supplier and EEG only
SELF_CONSUMPTION = dict(
    storage_use_cases=["home", "eeg"],
    allow_storage_to_wholesale=False,
    allow_wholesale_to_storage=False,
    allow_pv_to_wholesale=False,
    allow_community_to_home=False,
    allow_community_to_storage=False,
    allow_pv_to_community=False,
    allow_storage_to_community=False,
)


def series(values, index):
    return pd.Series(values, index=index, dtype=float)


def make_ev(index, availability, driving, **kwargs):
    params = dict(
        id="ev",
        capacity_kwh=10,
        max_charge_kw=11,
        charge_efficiency=1,
        discharge_efficiency=1,
    )
    params.update(kwargs)
    return ElectricVehicle(
        availability=series(availability, index),
        driving_consumption=series(driving, index),
        **params,
    )


def make_ecc(index, ev, *, demand=None, solar=None, supplier=None, eeg=None, **kw):
    n = len(index)
    params = dict(
        storage=None,
        demand=series(demand if demand is not None else [0] * n, index),
        solar_generation=series(solar if solar is not None else [0] * n, index),
        supplier_prices=series(supplier if supplier is not None else [0.3] * n, index),
        eeg_prices=series(eeg if eeg is not None else [0.08] * n, index),
        wholesale_market_prices=series([0] * n, index),
        community_market_prices=None,
        electric_vehicle=ev,
    )
    params.update(SELF_CONSUMPTION)
    params.update(kw)
    return EnergyCostCalculator(**params)


# ---------------------------------------------------------------------------
# feasibility, dynamics, energy balance
# ---------------------------------------------------------------------------


def test_ev_soc_dynamics_and_energy_balance():
    ev = make_ev(
        idx_6,
        availability=[1, 1, 0, 0, 1, 1],
        driving=[0, 0, 3, 3, 0, 0],
        charge_efficiency=0.9,
        initial_soc=0.5,
    )
    ecc = make_ecc(idx_6, ev, demand=[1] * 6)
    ecc.optimize(solver=SOLVER)
    ev_df = ecc.get_ev_timeseries_df()

    previous = 5.0
    for _, row in ev_df.iterrows():
        expected = (
            previous
            + 0.9 * row["charge"]
            - (row["driving"] - row["public_charging"])
            - row["to_home"]
        )
        assert np.isclose(row["soc"], expected, atol=1e-6)
        assert -1e-6 <= row["soc"] <= 10 + 1e-6
        previous = row["soc"]

    # 6 kWh driven at 90 % charge efficiency have to be recharged
    assert ev_df["soc"].iloc[-1] >= 5 - 1e-6
    assert np.isclose(ev_df["charge"].sum(), 6 / 0.9, atol=1e-5)
    # EV charging is paid at the supplier price on top of the household load
    assert np.isclose(ecc.get_cashflows()["supplier"], -0.3 * (6 + 6 / 0.9), atol=1e-4)
    # household demand stays covered without the EV
    coverage = ecc.get_demand_coverage_timeseries_df()
    assert np.allclose(coverage["from_grid"], 1)


def test_no_charging_while_away():
    # charging would be free while the car is away
    ev = make_ev(
        idx_5,
        availability=[1, 0, 0, 0, 1],
        driving=[0, 1, 1, 0, 0],
    )
    ecc = make_ecc(idx_5, ev, supplier=[1, 0, 0, 0, 1])
    ecc.optimize(solver=SOLVER)
    ev_df = ecc.get_ev_timeseries_df()
    away = ev_df["availability"] == 0
    assert np.allclose(ev_df.loc[away, "charge"], 0)
    assert np.isclose(ev_df["charge"].sum(), 2, atol=1e-6)


def test_fractional_availability_limits_charging_power():
    ev = make_ev(
        idx_4,
        availability=[0.5, 0.25, 0, 1],
        driving=[0, 0, 4, 0],
        max_charge_kw=4,
        initial_soc=0.5,
    )
    # cheapest at t=0 and t=1, so it charges there as much as the plug allows
    ecc = make_ecc(idx_4, ev, supplier=[0.1, 0.1, 0.5, 0.5])
    ecc.optimize(solver=SOLVER)
    ev_df = ecc.get_ev_timeseries_df()
    assert np.isclose(ev_df["charge"].iloc[0], 2, atol=1e-6)
    assert np.isclose(ev_df["charge"].iloc[1], 1, atol=1e-6)
    assert np.isclose(ev_df["charge"].iloc[3], 1, atol=1e-6)


def test_pv_surplus_goes_into_ev():
    ev = make_ev(
        idx_4,
        availability=[1, 0, 0, 1],
        driving=[0, 1.5, 1.5, 0],
        initial_soc=0.5,
    )
    ecc = make_ecc(idx_4, ev, solar=[5, 0, 0, 0], supplier=[0.3] * 4, eeg=[0.08] * 4)
    ecc.optimize(solver=SOLVER)
    ev_df = ecc.get_ev_timeseries_df()

    assert np.isclose(ev_df["charge_from_pv"].sum(), 3, atol=1e-6)
    assert np.isclose(ev_df["charge_from_supplier"].sum(), 0, atol=1e-6)
    assert np.isclose(ecc.get_cashflows()["eeg"], 2 * 0.08, atol=1e-6)
    assert np.isclose(ecc.get_solar_generation_timeseries_df()["to_ev"].sum(), 3)
    assert np.isclose(ecc.get_energy_flows()["pv_to_ev"].sum(), 3)


def test_departure_soc_is_respected():
    common = dict(availability=[1, 1, 0, 0, 1], driving=[0, 0, 1, 1, 0])
    supplier = [1, 1, 1, 1, 0.1]

    free = make_ecc(idx_5, make_ev(idx_5, **common), supplier=supplier)
    free.optimize(solver=SOLVER)
    # charging is cheapest after the trip, so nothing is charged before
    assert np.isclose(free.get_ev_timeseries_df()["soc"].iloc[1], 5, atol=1e-6)

    constrained = make_ecc(
        idx_5, make_ev(idx_5, min_departure_soc=0.9, **common), supplier=supplier
    )
    constrained.optimize(solver=SOLVER)
    ev_df = constrained.get_ev_timeseries_df()
    assert ev_df["soc"].iloc[1] >= 9 - 1e-6
    assert ev_df["soc"].iloc[-1] >= 5 - 1e-6


def test_min_soc_respected():
    ev = make_ev(idx_4, [1, 0, 0, 1], [0, 2, 2, 0], initial_soc=0.5, min_soc=0.3)
    ecc = make_ecc(idx_4, ev, supplier=[1, 0.1, 0.1, 0.1])
    ecc.optimize(solver=SOLVER)
    assert (ecc.get_ev_timeseries_df()["soc"] >= 3 - 1e-6).all()


# ---------------------------------------------------------------------------
# V2H
# ---------------------------------------------------------------------------


def test_v2h_is_off_by_default():
    common = dict(availability=[1, 1, 1], driving=[0, 0, 0], initial_soc=0.5)
    index = idx_4[:3]
    prices = dict(demand=[0, 0, 4], supplier=[0.1, 0.1, 1.0])

    default = make_ecc(index, make_ev(index, **common), **prices)
    default.optimize(solver=SOLVER)
    assert np.allclose(default.get_ev_timeseries_df()["to_home"], 0)
    assert np.allclose(default.get_demand_coverage_timeseries_df()["from_ev"], 0)

    v2h = make_ecc(index, make_ev(index, max_discharge_kw=3, **common), **prices)
    v2h_value = v2h.optimize(solver=SOLVER)
    ev_df = v2h.get_ev_timeseries_df()
    assert np.isclose(ev_df["to_home"].iloc[2], 3, atol=1e-6)
    # no free energy: what went to the home is bought back cheaply first
    assert ev_df["soc"].iloc[-1] >= 5 - 1e-6
    assert v2h_value > default.calculate_costs() + 2


def test_v2h_never_feeds_markets():
    index = idx_4[:3]
    ev = make_ev(index, [1, 1, 1], [0, 0, 0], max_discharge_kw=11, initial_soc=1.0)
    ecc = make_ecc(
        index,
        ev,
        demand=[0, 0, 0],
        eeg=[5, 5, 5],
        storage_use_cases=["eeg", "wholesale", "community", "home"],
        allow_storage_to_wholesale=True,
        wholesale_market_prices=series([5, 5, 5], index),
    )
    ecc.optimize(solver=SOLVER)
    assert np.allclose(ecc.get_ev_timeseries_df()["to_home"], 0)
    assert np.isclose(ecc.get_cashflows()["eeg"], 0)


def test_v2h_only_while_available():
    ev = make_ev(idx_4, [1, 0, 0, 1], [0, 0, 0, 0], max_discharge_kw=5, initial_soc=1.0)
    ecc = make_ecc(idx_4, ev, demand=[1, 1, 1, 1], supplier=[0.3, 1, 1, 0.3])
    ecc.optimize(solver=SOLVER)
    ev_df = ecc.get_ev_timeseries_df()
    assert np.allclose(ev_df["to_home"].iloc[1:3], 0)


# ---------------------------------------------------------------------------
# public charging and infeasibility
# ---------------------------------------------------------------------------


def test_infeasible_driving_without_public_charging_raises():
    # 12 kWh of driving do not fit into a 10 kWh battery
    ev = make_ev(idx_4, [1, 0, 0, 1], [0, 6, 6, 0], initial_soc=0.2)
    with pytest.raises(ValueError, match="public_charging_price"):
        make_ecc(idx_4, ev)


def test_unreachable_departure_soc_raises():
    ev = make_ev(
        idx_4,
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        initial_soc=0.2,
        max_charge_kw=1,
        min_departure_soc=0.9,
    )
    with pytest.raises(ValueError, match="min_departure_soc"):
        make_ecc(idx_4, ev)


def test_public_charging_slack():
    ev = make_ev(
        idx_4, [1, 0, 0, 1], [0, 5, 5, 0], initial_soc=0.2, public_charging_price=0.6
    )
    ecc = make_ecc(idx_4, ev, supplier=[0.3] * 4)
    ecc.optimize(solver=SOLVER)
    ev_df = ecc.get_ev_timeseries_df()

    # 2 kWh in the battery plus a full charge of 8 kWh at t=0 cover 10 kWh exactly;
    # more home charging is impossible, so nothing is bought publicly here
    assert np.isclose(ev_df["public_charging"].sum(), 0, atol=1e-6)

    # a weaker wallbox leaves a gap that only public charging can close
    ev_small = make_ev(
        idx_4,
        [1, 0, 0, 1],
        [0, 5, 5, 0],
        initial_soc=0.2,
        max_charge_kw=4,
        public_charging_price=0.6,
    )
    ecc_small = make_ecc(idx_4, ev_small, supplier=[0.3] * 4)
    ecc_small.optimize(solver=SOLVER)
    ev_df = ecc_small.get_ev_timeseries_df()
    public = ev_df["public_charging"].sum()
    assert np.isclose(public, 4, atol=1e-6)
    cashflows = ecc_small.get_cashflows()
    assert np.isclose(cashflows["ev_public_charging"], -0.6 * public, atol=1e-6)
    assert np.isclose(ecc_small.calculate_costs(), sum(cashflows.values()), atol=1e-4)
    assert np.isclose(ecc_small.get_ev_kpis()["public_charging_cost"], 0.6 * public)


def test_public_charging_used_when_cheaper():
    ev = make_ev(
        idx_4, [1, 0, 0, 1], [0, 2, 2, 0], initial_soc=0.5, public_charging_price=0.1
    )
    ecc = make_ecc(idx_4, ev, supplier=[0.3] * 4)
    ecc.optimize(solver=SOLVER)
    assert np.isclose(ecc.get_ev_timeseries_df()["public_charging"].sum(), 4)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(availability=[1, 1.5, 0, 1]), "availability"),
        (dict(availability=[1, -0.1, 0, 1]), "availability"),
        (dict(driving=[0, -1, 0, 0]), "must not be negative"),
        (dict(driving=[1, 0, 0, 0]), "fully available"),
        (dict(capacity_kwh=0), "capacity_kwh"),
        (dict(max_charge_kw=-1), "max_charge_kw"),
        (dict(max_discharge_kw=-1), "max_discharge_kw"),
        (dict(charge_efficiency=0), "charge_efficiency"),
        (dict(discharge_efficiency=1.2), "discharge_efficiency"),
        (dict(initial_soc=1.2), "initial_soc"),
        (dict(min_soc=-0.1), "min_soc"),
        (dict(min_departure_soc=2), "min_departure_soc"),
        (dict(initial_soc=0.1, min_soc=0.2), "below min_soc"),
        (dict(public_charging_price=-0.1), "public_charging_price"),
        (dict(driving=[0, np.nan, 0, 0]), "NaN"),
    ],
)
def test_ev_validation_errors(kwargs, match):
    params = dict(availability=[1, 0, 0, 1], driving=[0, 1, 1, 0])
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        make_ev(idx_4, **params)


def test_ev_index_errors():
    with pytest.raises(TypeError, match="DateTimeIndex"):
        ElectricVehicle(
            id=1,
            capacity_kwh=10,
            max_charge_kw=11,
            availability=pd.Series([1, 1]),
            driving_consumption=pd.Series([0, 0]),
        )
    with pytest.raises(ValueError, match="share one index"):
        ElectricVehicle(
            id=1,
            capacity_kwh=10,
            max_charge_kw=11,
            availability=series([1, 1, 1, 1], idx_4),
            driving_consumption=series([0, 0, 0, 0, 0], idx_5),
        )
    # EV index has to match the demand index
    ev = make_ev(idx_5, [1] * 5, [0] * 5)
    with pytest.raises(ValueError, match="electric_vehicle.availability"):
        make_ecc(idx_4, ev)
    with pytest.raises(TypeError, match="ElectricVehicle"):
        make_ecc(idx_4, ev={"capacity_kwh": 10})


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------


def test_ev_outputs_and_kpis():
    storage = Storage(
        id=1, c_rate=1, volume=4, charge_efficiency=1, discharge_efficiency=1
    )
    ev = make_ev(idx_4, [0, 0, 1, 1], [2, 2, 0, 0])
    ecc = make_ecc(idx_4, ev, solar=[5, 5, 0, 0], storage=storage)
    ecc.optimize(solver=SOLVER)

    results = ecc.output_results(include_cashflows=True)
    assert set(results) == {"demand", "pv", "storage", "electric_vehicle", "cashflows"}
    assert "ev_public_charging" in results["cashflows"]
    # the storage SOC table is unchanged by the EV
    assert list(results["storage"].columns) == ["soc_home", "soc_eeg"]

    kpis = ecc.get_storage_usage_kpis()
    # PV is stored while the car is away and moved into it in the evening
    assert np.isclose(kpis["discharged_by_sink_kwh"]["ev"], 4, atol=1e-6)
    assert np.isclose(kpis["electric_vehicle"]["charged_by_source_kwh"]["storage"], 4)
    assert np.isclose(kpis["electric_vehicle"]["driving_kwh"], 4)

    flows = ecc.get_energy_flows()
    for column in ("pv_to_ev", "supplier_to_ev", "storage_to_ev", "ev_to_home"):
        assert column in flows.columns

    ecc.plot_storage_usage_summary(show=False)
    ecc.plot_energy_flows(show=False)


def test_ev_timeseries_without_ev_raises():
    ecc = make_ecc(idx_4, None)
    ecc.optimize(solver=SOLVER)
    with pytest.raises(ValueError, match="No electric_vehicle"):
        ecc.get_ev_timeseries_df()


def test_ev_community_charging_pays_price_and_grid_fee():
    ev = make_ev(idx_4, [1, 0, 0, 1], [0, 1, 1, 0])
    ecc = make_ecc(
        idx_4,
        ev,
        supplier=[1] * 4,
        community_market_prices={"heerlen": series([0.2] * 4, idx_4)},
        allow_community_to_home=True,
    )
    ecc.optimize(solver=SOLVER)
    ev_df = ecc.get_ev_timeseries_df()
    assert np.isclose(ev_df["charge_from_community"].sum(), 2, atol=1e-6)
    cashflows = ecc.get_cashflows()
    assert np.isclose(cashflows["community"], -0.4, atol=1e-6)
    # aachen <- heerlen is 0.015 EUR/kWh in the default table
    assert np.isclose(cashflows["grid_fees"], -0.03, atol=1e-6)


def test_ev_community_charging_follows_allow_community_to_home():
    ev = make_ev(idx_4, [1, 0, 0, 1], [0, 1, 1, 0])
    ecc = make_ecc(
        idx_4,
        ev,
        supplier=[1] * 4,
        community_market_prices={"aachen": series([0.2] * 4, idx_4)},
        allow_community_to_home=False,
    )
    ecc.optimize(solver=SOLVER)
    assert np.isclose(ecc.get_ev_timeseries_df()["charge_from_community"].sum(), 0)


# ---------------------------------------------------------------------------
# storage worth with an EV
# ---------------------------------------------------------------------------


def _worth_inputs(index):
    # PV at noon while the car is away, the car comes back empty in the evening
    return dict(
        demand=series([0, 0, 0, 0], index),
        solar_generation=series([5, 5, 0, 0], index),
        supplier_prices=series([0.3] * 4, index),
        eeg_prices=series([0.08] * 4, index),
        wholesale_market_prices=series([0] * 4, index),
        community_market_prices=None,
        solver=SOLVER,
        **SELF_CONSUMPTION,
    )


def test_storage_worth_changes_with_ev():
    baseline = Storage(
        id=0, c_rate=1, volume=0, charge_efficiency=1, discharge_efficiency=1
    )
    candidate = Storage(
        id=1, c_rate=1, volume=4, charge_efficiency=1, discharge_efficiency=1
    )
    ev = make_ev(idx_4, [0, 0, 1, 1], [2, 2, 0, 0])

    without_ev = calculate_storage_worth(
        baseline_storage=baseline,
        storage_to_calculate=candidate,
        **_worth_inputs(idx_4),
    )
    result = calculate_storage_worth(
        baseline_storage=baseline,
        storage_to_calculate=candidate,
        electric_vehicle=ev,
        return_cashflows=True,
        return_soc_timeseries=True,
        **_worth_inputs(idx_4),
    )
    # nothing to shift without EV; with EV, 4 kWh PV replace 4 kWh grid power
    assert np.isclose(without_ev, 0, atol=1e-3)
    assert np.isclose(result["worth"], 4 * (0.3 - 0.08), atol=1e-3)
    assert "baseline_ev_soc_ts" in result
    assert "storage_to_calc_ev_soc_ts" in result
    assert "ev_public_charging" in result["baseline_cashflows"]


def test_rented_storage_worth_with_ev_by_location():
    baseline = Storage(
        id=0, c_rate=1, volume=0, charge_efficiency=1, discharge_efficiency=1
    )
    storages = [
        Storage(id=1, c_rate=1, volume=2, charge_efficiency=1, discharge_efficiency=1),
        Storage(id=2, c_rate=1, volume=4, charge_efficiency=1, discharge_efficiency=1),
    ]
    ev = make_ev(idx_4, [0, 0, 1, 1], [2, 2, 0, 0])
    kwargs = _worth_inputs(idx_4)

    without_ev = calculate_multiple_storage_worth_by_location(
        baseline_storage=baseline,
        storages_to_calculate=storages,
        locations_to_calculate=["aachen", "heerlen"],
        **kwargs,
    )
    with_ev = calculate_multiple_storage_worth_by_location(
        baseline_storage=baseline,
        storages_to_calculate=storages,
        locations_to_calculate=["aachen", "heerlen"],
        electric_vehicle=ev,
        **kwargs,
    )
    assert (with_ev["worth"] >= without_ev["worth"] - 1e-6).all()
    heerlen = with_ev[with_ev["location"] == "heerlen"].set_index("volume")["worth"]
    # PV charged into storage in heerlen pays 0.015 EUR/kWh grid fee
    assert np.isclose(heerlen.loc[4.0], 4 * (0.3 - 0.08 - 0.015), atol=1e-3)


def test_multiple_storage_worth_with_ev_returns_ev_timeseries_and_plots():
    baseline = Storage(id=0, c_rate=1, volume=0)
    storages = [Storage(id=1, c_rate=1, volume=2), Storage(id=2, c_rate=1, volume=4)]
    ev = make_ev(idx_4, [0, 0, 1, 1], [2, 2, 0, 0], public_charging_price=0.5)
    out = calculate_multiple_storage_worth(
        baseline_storage=baseline,
        storages_to_calculate=storages,
        electric_vehicle=ev,
        return_soc_timeseries=True,
        return_cashflows=True,
        **_worth_inputs(idx_4),
    )
    assert set(out["storages_to_calc_ev_soc_ts"]) == {1, 2}
    assert "soc" in out["baseline_ev_soc_ts"].columns
    fig = plot_multiple_storage_worth_cashflows(out, show=False)
    assert "ev_public_charging" in {trace.name for trace in fig.data}


def test_worth_distribution_passes_ev_through():
    index = pd.date_range("2025-06-01", freq="h", periods=4)
    scenarios = sample_scenarios(
        demand=series([0.1] * 4, index),
        solar_generation=series([5, 5, 0, 0], index),
        supplier_prices=series([0.3] * 4, index),
        eeg_prices=series([0.08] * 4, index),
        wholesale_market_prices=series([0.05] * 4, index),
        n_scenarios=2,
        seed=1,
    )
    ev = make_ev(index, [0, 0, 1, 1], [2, 2, 0, 0])
    common = dict(
        baseline_storage=Storage(id=0, c_rate=1, volume=0),
        storages_to_calculate=[Storage(id=1, c_rate=1, volume=4)],
        scenarios=scenarios,
        solver=SOLVER,
        **{k: v for k, v in SELF_CONSUMPTION.items()},
    )
    without_ev = calculate_storage_worth_distribution(**common)
    with_ev = calculate_storage_worth_distribution(electric_vehicle=ev, **common)
    assert (
        with_ev.loc[with_ev["id"] == 1, "worth"].to_numpy()
        > without_ev.loc[without_ev["id"] == 1, "worth"].to_numpy() + 0.5
    ).all()


def test_max_green_energy_counts_pv_into_ev():
    ev = make_ev(idx_4, [1, 0, 0, 1], [0, 1.5, 1.5, 0])
    ecc = make_ecc(idx_4, ev, solar=[5, 0, 0, 0], eeg=[1] * 4, goal="max_green_energy")
    ecc.optimize(solver=SOLVER)
    # all PV ends up in the car, even beyond what driving needs
    assert np.isclose(ecc.get_ev_timeseries_df()["charge_from_pv"].sum(), 5, atol=1e-6)
    assert np.isclose(ecc.get_cashflows()["eeg"], 0, atol=1e-6)


# ---------------------------------------------------------------------------
# regression: electric_vehicle=None reproduces the results before EV support
# ---------------------------------------------------------------------------


def _regression_inputs():
    index = pd.date_range("2025-06-01", freq="h", periods=24)
    hours = np.arange(24)
    return dict(
        demand=pd.Series(0.4 + 0.5 * np.exp(-((hours - 19) ** 2) / 8), index=index),
        solar_generation=pd.Series(
            np.clip(4 * np.sin((hours - 6) * np.pi / 12), 0, None), index=index
        ),
        supplier_prices=pd.Series(
            0.30 + 0.05 * np.cos(hours * np.pi / 12), index=index
        ),
        eeg_prices=pd.Series(0.08, index=index),
        wholesale_market_prices=pd.Series(
            0.05 + 0.08 * np.cos((hours - 1) * np.pi / 12), index=index
        ),
        community_market_prices={
            "aachen": pd.Series(0.15 + 0.05 * np.sin(hours * np.pi / 12), index=index),
            "heerlen": pd.Series(0.17, index=index),
        },
    )


def test_none_ev_reproduces_previous_results():
    # reference values computed with the code before EV support (commit 4c7272b)
    ecc = EnergyCostCalculator(
        storage=Storage(id=1, c_rate=0.5, volume=5),
        cycle_cost_per_kwh=0.02,
        electric_vehicle=None,
        **_regression_inputs(),
    )
    objective = ecc.optimize(solver=SOLVER)
    assert np.isclose(objective, 4.79501801192917, rtol=0, atol=1e-9)
    expected_cashflows = {
        "community": 4.9309897394,
        "supplier": 0.0,
        "eeg": 0.0,
        "wholesale": 0.4882374116,
        "grid_fees": -0.0420000301,
    }
    cashflows = ecc.get_cashflows()
    assert set(cashflows) == set(expected_cashflows)
    for key, value in expected_cashflows.items():
        assert np.isclose(cashflows[key], value, atol=1e-8)
    assert not hasattr(ecc.model, "ev_soc")
    assert "ev" not in ecc.get_storage_usage_kpis()["discharged_by_sink_kwh"]
    assert "electric_vehicle" not in ecc.output_results()

    df = calculate_multiple_storage_worth(
        baseline_storage=Storage(id=0, c_rate=1, volume=0),
        storages_to_calculate=[
            Storage(id=1, c_rate=1, volume=2),
            Storage(id=2, c_rate=1, volume=5),
        ],
        storage_location="heerlen",
        is_rented_storage=True,
        solver=SOLVER,
        **_regression_inputs(),
    )
    assert np.allclose(
        df["worth"].to_numpy(),
        [0.0, 0.6660587402833213, 1.6651468507083091],
        rtol=0,
        atol=1e-9,
    )
