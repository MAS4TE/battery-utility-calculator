# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

from battery_utility_calculator.battery_utility_calculator import (
    Storage,
    calculate_bidding_curve,
    calculate_multiple_storage_worth,
    calculate_multiple_storage_worth_by_city,
    calculate_storage_worth,
    plot_multiple_storage_worth_cashflows,
)

idx = pd.date_range("2025-01-01", freq="h", periods=3)


def test_calculate_storage_worth_without_community_market():
    worth = calculate_storage_worth(
        baseline_storage=Storage(0, 1, 0, 1),
        storage_to_calculate=Storage(0, 1, 1, 1),
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices=None,
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        solver="appsi_highs",
    )
    assert isinstance(worth, float)


def test_calculate_storage_worth():
    baseline_storage = Storage(0, 1, 0, 1)
    storage_to_calc = Storage(0, 1, 1, 1)

    worth = calculate_storage_worth(
        baseline_storage=baseline_storage,
        storage_to_calculate=storage_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        solver="appsi_highs",
    )

    # ask for cashflows as well
    result = calculate_storage_worth(
        baseline_storage=baseline_storage,
        storage_to_calculate=storage_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        return_cashflows=True,
        solver="appsi_highs",
    )
    assert "baseline_cashflows" in result and "storage_to_calc_cashflows" in result
    assert "baseline_soc_ts" not in result and "storage_to_calc_soc_ts" not in result
    # baseline and storage supplier costs should differ by roughly the worth (≈1)
    diff = (
        result["storage_to_calc_cashflows"]["supplier"]
        - result["baseline_cashflows"]["supplier"]
    )
    assert round(diff, 0) == 1

    # with baseline storage costs should be 2
    # with storage costs should be 1
    assert round(worth, 0) == 1

    soc_result = calculate_storage_worth(
        baseline_storage=baseline_storage,
        storage_to_calculate=storage_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        return_soc_timeseries=True,
        solver="appsi_highs",
    )
    assert "baseline_soc_ts" in soc_result and "storage_to_calc_soc_ts" in soc_result


def test_calculate_storage_worth_cycle_cost_default_compatibility():
    baseline_storage = Storage(0, 1, 0, 1)
    storage_to_calc = Storage(0, 1, 1, 1)

    worth_default = calculate_storage_worth(
        baseline_storage=baseline_storage,
        storage_to_calculate=storage_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        solver="appsi_highs",
    )
    worth_explicit_zero = calculate_storage_worth(
        baseline_storage=baseline_storage,
        storage_to_calculate=storage_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        cycle_cost_per_kwh=0.0,
        solver="appsi_highs",
    )
    worth_with_cycle_cost = calculate_storage_worth(
        baseline_storage=baseline_storage,
        storage_to_calculate=storage_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        cycle_cost_per_kwh=0.05,
        solver="appsi_highs",
    )

    assert round(worth_default, 6) == round(worth_explicit_zero, 6)
    assert worth_with_cycle_cost < worth_default


def test_calculate_storage_worth_allow_pv_to_wholesale_changes_result():
    baseline_storage = Storage(0, 1, 0, 1)
    storage_to_calc = Storage(1, 1, 1, 1)

    result_disabled = calculate_storage_worth(
        baseline_storage=baseline_storage,
        storage_to_calculate=storage_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([10, 10, 10], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 0, 0], index=idx),
        solar_generation=pd.Series([1, 1, 1], index=idx),
        demand=pd.Series([0, 0, 0], index=idx),
        allow_pv_to_wholesale=False,
        return_cashflows=True,
        solver="appsi_highs",
    )
    result_enabled = calculate_storage_worth(
        baseline_storage=baseline_storage,
        storage_to_calculate=storage_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([10, 10, 10], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 0, 0], index=idx),
        solar_generation=pd.Series([1, 1, 1], index=idx),
        demand=pd.Series([0, 0, 0], index=idx),
        allow_pv_to_wholesale=True,
        return_cashflows=True,
        solver="appsi_highs",
    )

    assert result_disabled["baseline_cashflows"]["wholesale"] == 0.0
    assert result_enabled["baseline_cashflows"]["wholesale"] > 0.0


def test_calculate_multiple_storage_worth():
    baseline_storage = Storage(0, 1, 0, 1)
    storages_to_calc = [Storage(0, 1, 1, 1), Storage(1, 1, 2, 1)]

    worths = calculate_multiple_storage_worth(
        baseline_storage=baseline_storage,
        storages_to_calculate=storages_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        solver="appsi_highs",
    )

    # request cashflow output too
    df_with_cf = calculate_multiple_storage_worth(
        baseline_storage=baseline_storage,
        storages_to_calculate=storages_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        return_cashflows=True,
        solver="appsi_highs",
    )
    assert "baseline_cashflows" in df_with_cf
    assert isinstance(df_with_cf["storages_to_calc_cashflows"], dict)

    fig = plot_multiple_storage_worth_cashflows(df_with_cf, show=False, stacked=False)
    assert len(fig.data) == 5

    with pytest.raises(ValueError, match="missing"):
        plot_multiple_storage_worth_cashflows({"results_df": df_with_cf["results_df"]})

    df_with_soc = calculate_multiple_storage_worth(
        baseline_storage=baseline_storage,
        storages_to_calculate=storages_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        return_soc_timeseries=True,
        solver="appsi_highs",
    )
    assert "baseline_soc_ts" in df_with_soc
    assert isinstance(df_with_soc["storages_to_calc_soc_ts"], dict)

    print(worths["costs"])
    assert (worths["costs"].round(0).values == [-2, -1, 0]).all()
    assert (worths["worth"].round(0).values[1:] == [1, 2]).all()


def test_calculate_multiple_storage_worth_cycle_cost_is_consistent():
    baseline_storage = Storage(0, 1, 0, 1)
    storages_to_calc = [Storage(0, 1, 1, 1), Storage(1, 1, 2, 1)]

    without_cycle_cost = calculate_multiple_storage_worth(
        baseline_storage=baseline_storage,
        storages_to_calculate=storages_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        cycle_cost_per_kwh=0.0,
        solver="appsi_highs",
    )
    with_cycle_cost = calculate_multiple_storage_worth(
        baseline_storage=baseline_storage,
        storages_to_calculate=storages_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        cycle_cost_per_kwh=0.05,
        solver="appsi_highs",
    )

    # baseline row worth should always be zero
    assert float(without_cycle_cost.loc[0, "worth"]) == 0.0
    assert float(with_cycle_cost.loc[0, "worth"]) == 0.0
    # added cycle cost should lower worth of storage configurations
    assert float(with_cycle_cost.loc[1, "worth"]) < float(
        without_cycle_cost.loc[1, "worth"]
    )
    assert float(with_cycle_cost.loc[2, "worth"]) < float(
        without_cycle_cost.loc[2, "worth"]
    )


def test_calculate_multiple_storage_worth_applies_wholesale_fee_to_all_runs():
    baseline_storage = Storage(0, 1, 0, 1)
    storages_to_calc = [Storage(1, 1, 0, 1)]

    low_fee = calculate_multiple_storage_worth(
        baseline_storage=baseline_storage,
        storages_to_calculate=storages_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([10, 10, 10], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 0, 0], index=idx),
        solar_generation=pd.Series([1, 1, 1], index=idx),
        demand=pd.Series([0, 0, 0], index=idx),
        allow_pv_to_wholesale=True,
        wholesale_fee=0.0,
        solver="appsi_highs",
    )
    high_fee = calculate_multiple_storage_worth(
        baseline_storage=baseline_storage,
        storages_to_calculate=storages_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([10, 10, 10], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 0, 0], index=idx),
        solar_generation=pd.Series([1, 1, 1], index=idx),
        demand=pd.Series([0, 0, 0], index=idx),
        allow_pv_to_wholesale=True,
        wholesale_fee=0.9,
        solver="appsi_highs",
    )

    assert float(low_fee.loc[1, "costs"]) > float(high_fee.loc[1, "costs"])


def test_calc_bid_curve_dtypes():
    input_df = pd.DataFrame()
    input_df["volume"] = [1, 0]
    input_df["worth"] = [10, 0]
    input_df["other"] = ["A", "B"]

    curve = calculate_bidding_curve(input_df, "buyer")


def test_calculate_bidding_curve_buyer():
    volume_worths = pd.DataFrame()
    volume_worths["volume"] = [0, 1, 2, 3, 3.5]
    volume_worths["worth"] = [0, 5, 7, 8, 8.2]

    bidding_curve = calculate_bidding_curve(
        volumes_worth=volume_worths, buy_or_sell_side="buyer"
    )

    correct_df = pd.DataFrame()
    correct_df["volume"] = [1, 1, 1, 0.5]
    correct_df["marginal_price"] = [5, 2, 1, 0.2]
    correct_df["cumulative_volume"] = [1, 2, 3, 3.5]
    correct_df["marginal_price_per_kwh"] = [5, 2, 1, 0.4]

    assert (bidding_curve["volume"] == correct_df["volume"]).all()
    assert (
        bidding_curve["marginal_price"].round(2)
        == correct_df["marginal_price"].round(2)
    ).all()
    assert (bidding_curve["cumulative_volume"] == correct_df["cumulative_volume"]).all()
    assert (
        bidding_curve["marginal_price_per_kwh"].round(2)
        == correct_df["marginal_price_per_kwh"].round(2)
    ).all()


def test_calculate_bidding_curve_seller():
    volume_worths = pd.DataFrame()
    volume_worths["volume"] = [0, 2, 10]
    volume_worths["worth"] = [-10, -7, 0]

    bidding_curve = calculate_bidding_curve(
        volumes_worth=volume_worths, buy_or_sell_side="seller"
    )

    correct_df = pd.DataFrame()
    correct_df["volume"] = [8, 2]
    correct_df["marginal_price"] = [7, 3]
    correct_df["cumulative_volume"] = [8, 10]
    correct_df["marginal_price_per_kwh"] = [7 / 8, 3 / 2]

    assert (bidding_curve["volume"] == correct_df["volume"]).all()
    assert (
        bidding_curve["marginal_price"].round(2)
        == correct_df["marginal_price"].round(2)
    ).all()
    assert (bidding_curve["cumulative_volume"] == correct_df["cumulative_volume"]).all()
    assert (
        bidding_curve["marginal_price_per_kwh"].round(2)
        == correct_df["marginal_price_per_kwh"].round(2)
    ).all()


def test_calculate_storage_worth_with_my_city_pass_through():
    baseline_storage = Storage(0, 1, 0, 1)
    storage_to_calc = Storage(1, 1, 1, 1)

    low_fee_worth = calculate_storage_worth(
        baseline_storage=baseline_storage,
        storage_to_calculate=storage_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        my_city="aachen",
        storage_city="aachen",
        solver="appsi_highs",
    )
    high_fee_worth = calculate_storage_worth(
        baseline_storage=baseline_storage,
        storage_to_calculate=storage_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        my_city="aachen",
        storage_city="liege",
        solver="appsi_highs",
    )

    assert high_fee_worth < low_fee_worth


def test_calculate_multiple_storage_worth_by_city():
    baseline_storage = Storage(0, 1, 0, 1)
    storages_to_calc = [Storage(1, 1, 1, 1), Storage(2, 1, 2, 1)]
    cities = ["aachen", "juelich"]

    result_df = calculate_multiple_storage_worth_by_city(
        baseline_storage=baseline_storage,
        storages_to_calculate=storages_to_calc,
        cities=cities,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices={"aachen": pd.Series([0, 0, 0], index=idx)},
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        solver="appsi_highs",
    )

    assert "city" in result_df.columns
    assert set(result_df["city"].unique()) == set(cities)
    # baseline + 2 storages for each city
    assert len(result_df) == len(cities) * (1 + len(storages_to_calc))
