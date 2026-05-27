# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT

from typing import Literal

import pandas as pd
import plotly.express as px
from pandas.api.types import is_numeric_dtype
from plotly.graph_objects import Figure

from battery_utility_calculator import EnergyCostCalculator as ECC
from battery_utility_calculator import Storage

DEFAULT_GRID_FEE_BETWEEN_CITIES = {
    "juelich": {
        "juelich": 0.0,
        "aachen": 0.01,
        "heerlen": 0.015,
        "liege": 0.02,
    },
    "aachen": {
        "juelich": 0.01,
        "aachen": 0.0,
        "heerlen": 0.015,
        "liege": 0.02,
    },
    "heerlen": {
        "juelich": 0.015,
        "aachen": 0.01,
        "heerlen": 0.0,
        "liege": 0.02,
    },
    "liege": {
        "juelich": 0.02,
        "aachen": 0.01,
        "heerlen": 0.015,
        "liege": 0.0,
    },
}


def calculate_storage_worth(
    baseline_storage: Storage,
    storage_to_calculate: Storage,
    demand: pd.Series,
    solar_generation: pd.Series,
    supplier_prices: pd.Series,
    eeg_prices: pd.Series,
    wholesale_market_prices: pd.Series,
    community_market_prices: dict[str, pd.Series] | None = None,
    hours_per_timestep: int | float = 1,
    storage_use_cases: list[str] = ["eeg", "wholesale", "community", "home"],
    allow_community_to_home: bool = True,
    allow_community_to_storage: bool = True,
    allow_community_market_arbitrage: bool = True,
    allow_pv_to_community: bool = True,
    allow_storage_to_community: bool = True,
    allow_wholesale_to_home: bool = False,
    allow_pv_to_wholesale: bool = False,
    allow_wholesale_to_storage: bool = True,
    allow_storage_to_wholesale: bool = True,
    wholesale_fee: float = 0.3,
    my_city: str = "aachen",
    grid_fee_between_cities: dict[str, dict[str, float]]
    | None = DEFAULT_GRID_FEE_BETWEEN_CITIES,
    storage_city: str | None = None,
    eeg_eligible: bool = True,
    goal: str = "max_cashflow",
    discharge_penalty_per_kwh: float = 1e-6,
    cycle_cost_per_kwh: float = 0.0,
    return_charge_timeseries: bool = False,
    return_soc_timeseries: bool = False,
    return_cashflows: bool = False,
    solver: str = "gurobi",
) -> dict | float:
    """Calculates the worth (value) of a single storage (compared to a baseline storage).

    Args:
        baseline_storage (Storage): The baseline storage to compare to.
        storage_to_calculate (Storage): The storage to calculate the worth (value) for.
        demand (pd.Series): Demand timeseries. Values should be in kWh per hour (kW).
        solar_generation (pd.Series): Solar generation timeseries. Values should be in kWh per hour (kW).
        supplier_prices (pd.Series): Grid prices timeseries. Values should be in EUR per kWh.
        eeg_prices (pd.Series): EEG prices timeseries. Values should be in EUR per kWh.
        community_market_prices (dict[str, pd.Series] | None): Community market prices per city in EUR per kWh. ``None`` disables the community market.
        wholesale_market_prices (pd.Series): Wholesale market timeseries. Values should be in EUR per kWh.
        hours_per_timestep (int | float, optional): Hours per timestep, e.g. 0.25 for 15-minute intervals. Defaults to 1.
        storage_use_cases (list[str], optional): Use cases for storage. Defaults to ECC default.
        allow_community_to_home (bool, optional): Wether to allow using energy from community for home use. Defaults to True.
        allow_community_to_storage (bool, optional): Allow community imports into the home storage bucket. Defaults to True.
        allow_community_market_arbitrage (bool, optional): Allow community round-trip arbitrage via the community storage bucket. Defaults to True.
        allow_pv_to_community (bool, optional): Wether to allow selling PV energy to community. Defaults to True.
        allow_storage_to_community (bool, optional): Allow discharging the community storage bucket to the community market. Defaults to True.
        allow_wholesale_to_home (bool, optional): Allow wholesale imports directly to home. Defaults to False.
        allow_pv_to_wholesale (bool, optional): Allow direct PV export to wholesale. Defaults to False.
        allow_wholesale_to_storage (bool, optional): Allow wholesale imports into storage. Defaults to True.
        allow_storage_to_wholesale (bool, optional): Wether to allow selling from storage to wholesale market. Defaults to True.
        goal (str, optional): Optimization goal. Defaults to "max_cashflow".
        return_charge_timeseries (bool, optional): If True, returns dict with charge timeseries data.
            Default is False.
        return_soc_timeseries (bool, optional): If True, returns dict with SOC timeseries data.
            Default is False.
        return_cashflows (bool, optional): If True, returns cashflow breakdown for baseline and
            candidate storage. Default is False.
        cycle_cost_per_kwh (float, optional): Optional degradation cost in EUR per
            discharged kWh. Default is 0.0.
        solver (str, optional): Which solver to use. Defaults to "gurobi".

    Returns:
        float or dict: If no return flags are
            requested, returns worth as float. Otherwise a dict containing at least
            ``worth`` and additional fields depending on the flags.
    """

    # calculate baseline costs
    baseline_ecc = ECC(
        storage=baseline_storage,
        demand=demand,
        solar_generation=solar_generation,
        supplier_prices=supplier_prices,
        eeg_prices=eeg_prices,
        wholesale_market_prices=wholesale_market_prices,
        community_market_prices=community_market_prices,
        hours_per_timestep=hours_per_timestep,
        storage_use_cases=storage_use_cases,
        allow_community_to_home=allow_community_to_home,
        allow_community_to_storage=allow_community_to_storage,
        allow_community_market_arbitrage=allow_community_market_arbitrage,
        allow_pv_to_community=allow_pv_to_community,
        allow_storage_to_community=allow_storage_to_community,
        allow_wholesale_to_home=allow_wholesale_to_home,
        allow_pv_to_wholesale=allow_pv_to_wholesale,
        allow_wholesale_to_storage=allow_wholesale_to_storage,
        allow_storage_to_wholesale=allow_storage_to_wholesale,
        wholesale_fee=wholesale_fee,
        my_city=my_city,
        grid_fee_between_cities=grid_fee_between_cities,
        storage_city=storage_city,
        eeg_eligible=eeg_eligible,
        goal=goal,
        discharge_penalty_per_kwh=discharge_penalty_per_kwh,
        cycle_cost_per_kwh=cycle_cost_per_kwh,
    )
    baseline_costs = baseline_ecc.optimize(solver=solver)

    # calculate costs for storage to calculate
    to_calc_ecc = ECC(
        storage=storage_to_calculate,
        demand=demand,
        solar_generation=solar_generation,
        supplier_prices=supplier_prices,
        eeg_prices=eeg_prices,
        wholesale_market_prices=wholesale_market_prices,
        community_market_prices=community_market_prices,
        hours_per_timestep=hours_per_timestep,
        storage_use_cases=storage_use_cases,
        allow_community_to_home=allow_community_to_home,
        allow_community_to_storage=allow_community_to_storage,
        allow_community_market_arbitrage=allow_community_market_arbitrage,
        allow_pv_to_community=allow_pv_to_community,
        allow_storage_to_community=allow_storage_to_community,
        allow_wholesale_to_home=allow_wholesale_to_home,
        allow_pv_to_wholesale=allow_pv_to_wholesale,
        allow_wholesale_to_storage=allow_wholesale_to_storage,
        allow_storage_to_wholesale=allow_storage_to_wholesale,
        wholesale_fee=wholesale_fee,
        my_city=my_city,
        grid_fee_between_cities=grid_fee_between_cities,
        storage_city=storage_city,
        eeg_eligible=eeg_eligible,
        goal=goal,
        discharge_penalty_per_kwh=discharge_penalty_per_kwh,
        cycle_cost_per_kwh=cycle_cost_per_kwh,
    )
    to_calc_costs = to_calc_ecc.optimize(solver=solver)

    # storage worth is difference between baseline and new storage
    storage_worth = to_calc_costs - baseline_costs

    # prepare optional outputs
    if return_charge_timeseries or return_soc_timeseries or return_cashflows:
        result = {"worth": storage_worth}

        if return_charge_timeseries:
            # Get storage charge timeseries for both
            baseline_charge = baseline_ecc.get_storage_charge_timeseries_df()
            storage_charge = to_calc_ecc.get_storage_charge_timeseries_df()
            result["baseline_charge_ts"] = baseline_charge
            result["storage_to_calc_charge_ts"] = storage_charge
        if return_soc_timeseries:
            baseline_soc = baseline_ecc.get_storage_soc_timeseries_df()
            storage_soc = to_calc_ecc.get_storage_soc_timeseries_df()
            result["baseline_soc_ts"] = baseline_soc
            result["storage_to_calc_soc_ts"] = storage_soc

        if return_cashflows:
            baseline_cf = baseline_ecc.get_cashflows()
            storage_cf = to_calc_ecc.get_cashflows()
            result["baseline_cashflows"] = baseline_cf
            result["storage_to_calc_cashflows"] = storage_cf

        return result
    else:
        return storage_worth


def calculate_multiple_storage_worth(
    baseline_storage: Storage,
    storages_to_calculate: list[Storage],
    demand: pd.Series,
    solar_generation: pd.Series,
    supplier_prices: pd.Series,
    eeg_prices: pd.Series,
    wholesale_market_prices: pd.Series,
    community_market_prices: dict[str, pd.Series] | None = None,
    hours_per_timestep: int | float = 1,
    storage_use_cases: list[str] = ["eeg", "wholesale", "community", "home"],
    allow_community_to_home: bool = True,
    allow_community_to_storage: bool = True,
    allow_community_market_arbitrage: bool = True,
    allow_pv_to_community: bool = True,
    allow_storage_to_community: bool = True,
    allow_wholesale_to_home: bool = False,
    allow_pv_to_wholesale: bool = False,
    allow_wholesale_to_storage: bool = True,
    allow_storage_to_wholesale: bool = True,
    wholesale_fee: float = 0.3,
    my_city: str = "aachen",
    grid_fee_between_cities: dict[str, dict[str, float]]
    | None = DEFAULT_GRID_FEE_BETWEEN_CITIES,
    storage_city: str | None = None,
    eeg_eligible: bool = True,
    goal: str = "max_cashflow",
    discharge_penalty_per_kwh: float = 1e-6,
    cycle_cost_per_kwh: float = 0.0,
    return_charge_timeseries: bool = False,
    return_soc_timeseries: bool = False,
    return_cashflows: bool = False,
    solver: str = "gurobi",
) -> dict | pd.DataFrame:
    """Calculates the worth (value) of multiple storages compared to a baseline storage.

    Args:
        baseline_storage (Storage): The baseline storage to compare to.
        storages_to_calculate (list[Storage]): List of storages to calculate worth (value) for.
        demand (pd.Series): Demand timeseries. Values should be in kWh per hour (kW).
        solar_generation (pd.Series): Solar generation timeseries. Values should be in kWh per hour (kW).
        supplier_prices (pd.Series): Grid prices timeseries. Values should be in EUR per kWh.
        eeg_prices (pd.Series): EEG prices timeseries. Values should be in EUR per kWh.
        community_market_prices (dict[str, pd.Series] | None): Community market prices per city in EUR per kWh. ``None`` disables the community market.
        wholesale_market_prices (pd.Series): Wholesale market timeseries. Values should be in EUR per kWh.
        hours_per_timestep (int | float, optional): Hours per timestep. Defaults to 1.
        storage_use_cases (list[str], optional): Use cases for storage. Defaults to ECC default.
        allow_community_to_home (bool, optional): Wether to allow using energy from community for home use. Defaults to True.
        allow_community_to_storage (bool, optional): Allow community imports into the home storage bucket. Defaults to True.
        allow_community_market_arbitrage (bool, optional): Allow community round-trip arbitrage via the community storage bucket. Defaults to True.
        allow_pv_to_community (bool, optional): Wether to allow selling PV energy to community. Defaults to True.
        allow_storage_to_community (bool, optional): Allow discharging the community storage bucket to the community market. Defaults to True.
        allow_wholesale_to_home (bool, optional): Allow wholesale imports directly to home. Defaults to False.
        allow_pv_to_wholesale (bool, optional): Allow direct PV export to wholesale. Defaults to False.
        allow_wholesale_to_storage (bool, optional): Allow wholesale imports into storage. Defaults to True.
        allow_storage_to_wholesale (bool, optional): Wether to allow selling from storage to wholesale market. Defaults to True.
        goal (str, optional): Optimization goal. Defaults to "max_cashflow".
        return_charge_timeseries (bool, optional): If True, returns dict with charge timeseries data. Defaults to False.
        return_soc_timeseries (bool, optional): If True, returns dict with SOC timeseries data. Defaults to False.
        return_cashflows (bool, optional): If True, returns dict with cashflow results for each storage. Defaults to False.
        cycle_cost_per_kwh (float, optional): Optional degradation cost in EUR per
            discharged kWh. Default is 0.0.
        check_timeseries (bool, optional): Wether to check time series. Defaults to True.
        solver (str, optional): Which solver to use. Defaults to "gurobi".

    Returns:
        pd.DataFrame or dict: If no return flag is set, returns DataFrame with storage parameters and worth. If any return
        flag is True, a dict is returned with ``results_df``
        plus the requested additional information.
    """

    if return_charge_timeseries and len(
        [stor.id for stor in storages_to_calculate]
    ) != len(set([stor.id for stor in storages_to_calculate])):
        msg = "Multiple storages with same ID are not allowed when returning charge timeseries data as "
        msg += "IDs are used to index storages_to_calc_charge_timeseries dictionary"
        raise ValueError(msg)
    if return_soc_timeseries and len(
        [stor.id for stor in storages_to_calculate]
    ) != len(set([stor.id for stor in storages_to_calculate])):
        msg = "Multiple storages with same ID are not allowed when returning SOC timeseries data as "
        msg += "IDs are used to index storages_to_calc_soc_timeseries dictionary"
        raise ValueError(msg)
    if return_cashflows and len([stor.id for stor in storages_to_calculate]) != len(
        set([stor.id for stor in storages_to_calculate])
    ):
        msg = "Multiple storages with same ID are not allowed when returning cashflow data as "
        msg += "IDs are used to index storages_to_calc_cashflows dictionary"
        raise ValueError(msg)

    # calculate baseline costs
    baseline_ecc = ECC(
        storage=baseline_storage,
        demand=demand,
        solar_generation=solar_generation,
        supplier_prices=supplier_prices,
        eeg_prices=eeg_prices,
        wholesale_market_prices=wholesale_market_prices,
        community_market_prices=community_market_prices,
        hours_per_timestep=hours_per_timestep,
        storage_use_cases=storage_use_cases,
        allow_community_to_home=allow_community_to_home,
        allow_community_to_storage=allow_community_to_storage,
        allow_community_market_arbitrage=allow_community_market_arbitrage,
        allow_pv_to_community=allow_pv_to_community,
        allow_storage_to_community=allow_storage_to_community,
        allow_wholesale_to_home=allow_wholesale_to_home,
        allow_pv_to_wholesale=allow_pv_to_wholesale,
        allow_wholesale_to_storage=allow_wholesale_to_storage,
        allow_storage_to_wholesale=allow_storage_to_wholesale,
        wholesale_fee=wholesale_fee,
        my_city=my_city,
        grid_fee_between_cities=grid_fee_between_cities,
        storage_city=storage_city,
        eeg_eligible=eeg_eligible,
        goal=goal,
        discharge_penalty_per_kwh=discharge_penalty_per_kwh,
        cycle_cost_per_kwh=cycle_cost_per_kwh,
    )
    baseline_costs = baseline_ecc.optimize(solver=solver)

    if return_charge_timeseries:
        baseline_charge = baseline_ecc.get_storage_charge_timeseries_df()
    if return_soc_timeseries:
        baseline_soc = baseline_ecc.get_storage_soc_timeseries_df()
    if return_cashflows:
        baseline_cashflows = baseline_ecc.get_cashflows()

    df = pd.DataFrame(
        columns=[
            "id",
            "c_rate",
            "volume",
            "charge_efficiency",
            "discharge_efficiency",
            "costs",
            "worth",
        ]
    )
    df.loc[
        0,
        [
            "id",
            "c_rate",
            "volume",
            "charge_efficiency",
            "discharge_efficiency",
            "costs",
            "worth",
        ],
    ] = [
        baseline_storage.id,
        baseline_storage.c_rate,
        baseline_storage.volume,
        baseline_storage.charge_efficiency,
        baseline_storage.discharge_efficiency,
        baseline_costs,
        0,
    ]

    storages_charge = {}
    storages_soc = {}
    storages_cashflows = {}
    for storage in storages_to_calculate:
        ecc = ECC(
            storage=storage,
            demand=demand,
            solar_generation=solar_generation,
            supplier_prices=supplier_prices,
            eeg_prices=eeg_prices,
            wholesale_market_prices=wholesale_market_prices,
            community_market_prices=community_market_prices,
            hours_per_timestep=hours_per_timestep,
            storage_use_cases=storage_use_cases,
            allow_community_to_home=allow_community_to_home,
            allow_community_to_storage=allow_community_to_storage,
            allow_community_market_arbitrage=allow_community_market_arbitrage,
            allow_pv_to_community=allow_pv_to_community,
            allow_storage_to_community=allow_storage_to_community,
            allow_wholesale_to_home=allow_wholesale_to_home,
            allow_pv_to_wholesale=allow_pv_to_wholesale,
            allow_wholesale_to_storage=allow_wholesale_to_storage,
            allow_storage_to_wholesale=allow_storage_to_wholesale,
            wholesale_fee=wholesale_fee,
            my_city=my_city,
            grid_fee_between_cities=grid_fee_between_cities,
            storage_city=storage_city,
            eeg_eligible=eeg_eligible,
            goal=goal,
            discharge_penalty_per_kwh=discharge_penalty_per_kwh,
            cycle_cost_per_kwh=cycle_cost_per_kwh,
        )
        costs = ecc.optimize(solver=solver)
        storage_worth = costs - baseline_costs

        stor_df = pd.DataFrame()
        stor_df["id"] = [storage.id]
        stor_df["c_rate"] = [storage.c_rate]
        stor_df["volume"] = [storage.volume]
        stor_df["charge_efficiency"] = [storage.charge_efficiency]
        stor_df["discharge_efficiency"] = [storage.discharge_efficiency]
        stor_df["costs"] = [costs]
        stor_df["worth"] = [storage_worth]

        df = pd.concat([df, stor_df], ignore_index=True)
        df["worth"] = df["worth"].astype(float)
        df["costs"] = df["costs"].astype(float)
        df["c_rate"] = df["c_rate"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["charge_efficiency"] = df["charge_efficiency"].astype(float)
        df["discharge_efficiency"] = df["discharge_efficiency"].astype(float)

        if return_charge_timeseries:
            storages_charge[storage.id] = ecc.get_storage_charge_timeseries_df()
        if return_soc_timeseries:
            storages_soc[storage.id] = ecc.get_storage_soc_timeseries_df()
        if return_cashflows:
            storages_cashflows[storage.id] = ecc.get_cashflows()

    # return depending on requested data
    if return_charge_timeseries or return_soc_timeseries or return_cashflows:
        out = {"results_df": df}
        if return_charge_timeseries:
            out["baseline_charge_ts"] = baseline_charge
            out["storages_to_calc_charge_ts"] = storages_charge
        if return_soc_timeseries:
            out["baseline_soc_ts"] = baseline_soc
            out["storages_to_calc_soc_ts"] = storages_soc
        if return_cashflows:
            out["baseline_cashflows"] = baseline_cashflows
            out["storages_to_calc_cashflows"] = storages_cashflows
        return out
    else:
        return df


def calculate_multiple_storage_worth_by_city(
    baseline_storage: Storage,
    storages_to_calculate: list[Storage],
    cities_to_calculate: list[str],
    demand: pd.Series,
    solar_generation: pd.Series,
    supplier_prices: pd.Series,
    eeg_prices: pd.Series,
    wholesale_market_prices: pd.Series,
    community_market_prices: dict[str, pd.Series] | None = None,
    hours_per_timestep: int | float = 1,
    storage_use_cases: list[str] = ["eeg", "wholesale", "community", "home"],
    allow_community_to_home: bool = True,
    allow_community_to_storage: bool = True,
    allow_community_market_arbitrage: bool = True,
    allow_pv_to_community: bool = True,
    allow_storage_to_community: bool = True,
    allow_wholesale_to_home: bool = False,
    allow_pv_to_wholesale: bool = False,
    allow_wholesale_to_storage: bool = True,
    allow_storage_to_wholesale: bool = True,
    wholesale_fee: float = 0.3,
    my_city: str = "aachen",
    grid_fee_between_cities: dict[str, dict[str, float]]
    | None = DEFAULT_GRID_FEE_BETWEEN_CITIES,
    eeg_eligible: bool = True,
    goal: str = "max_cashflow",
    discharge_penalty_per_kwh: float = 1e-6,
    cycle_cost_per_kwh: float = 0.0,
    return_charge_timeseries: bool = False,
    return_soc_timeseries: bool = False,
    return_cashflows: bool = False,
    solver: str = "gurobi",
) -> pd.DataFrame:
    """Calculate storage worth for multiple storages across multiple cities.

    Returns one row per city and storage configuration (including baseline row).
    """
    rows = []
    for city in cities_to_calculate:
        worth_result = calculate_multiple_storage_worth(
            baseline_storage=baseline_storage,
            storages_to_calculate=storages_to_calculate,
            demand=demand,
            solar_generation=solar_generation,
            supplier_prices=supplier_prices,
            eeg_prices=eeg_prices,
            wholesale_market_prices=wholesale_market_prices,
            community_market_prices=community_market_prices,
            hours_per_timestep=hours_per_timestep,
            storage_use_cases=storage_use_cases,
            allow_community_to_home=allow_community_to_home,
            allow_community_to_storage=allow_community_to_storage,
            allow_community_market_arbitrage=allow_community_market_arbitrage,
            allow_pv_to_community=allow_pv_to_community,
            allow_storage_to_community=allow_storage_to_community,
            allow_wholesale_to_home=allow_wholesale_to_home,
            allow_pv_to_wholesale=allow_pv_to_wholesale,
            allow_wholesale_to_storage=allow_wholesale_to_storage,
            allow_storage_to_wholesale=allow_storage_to_wholesale,
            wholesale_fee=wholesale_fee,
            my_city=my_city,
            grid_fee_between_cities=grid_fee_between_cities,
            storage_city=city,
            eeg_eligible=eeg_eligible,
            goal=goal,
            discharge_penalty_per_kwh=discharge_penalty_per_kwh,
            cycle_cost_per_kwh=cycle_cost_per_kwh,
            return_charge_timeseries=return_charge_timeseries,
            return_soc_timeseries=return_soc_timeseries,
            return_cashflows=return_cashflows,
            solver=solver,
        )
        worth_df = (
            worth_result["results_df"].copy()
            if isinstance(worth_result, dict)
            else worth_result.copy()
        )
        worth_df["city"] = city
        rows.append(worth_df)

    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "c_rate",
                "volume",
                "charge_efficiency",
                "discharge_efficiency",
                "costs",
                "worth",
                "city",
            ]
        )
    df = pd.concat(rows, ignore_index=True)
    df = df[
        ~((df["city"] != my_city) & (df["volume"] == baseline_storage.volume))
    ].reset_index(drop=True)
    return df


def calculate_bidding_curve(
    volumes_worth: pd.DataFrame,
    buy_or_sell_side: Literal["buyer", "seller"],
) -> pd.DataFrame:
    """Calculates the bidding curve for a single product.

    Args:
        volumes_worth (pd.DataFrame): The volumes and their worth (values) in a pd.DataFrame. Columns should be "volume" and "worth". If a "city" column is present, marginal steps are derived per city and combined into one DataFrame. Each city is its own exclusive bid group: ``cumulative_volume`` is restarted per city, and the clearing algorithm must enforce that only **one** city's bids can be awarded at a time (since the same physical storage is considered as alternative placements per city).
        buy_or_sell_side (Literal["buyer", "seller"]): Wether to calculate for buyer side or seller side.

    Returns:
        pd.DataFrame: A new DataFrame with the bidding curve.
    """

    df = volumes_worth.copy()

    if "worth" not in df.columns:
        raise KeyError("'worth' column not found!")
    elif "volume" not in df.columns:
        raise KeyError("'volume' column not found!")

    if not is_numeric_dtype(df["worth"]):
        try:
            df["worth"] = df["worth"].astype(float)
        except ValueError:
            raise ValueError("Column 'worth' not numeric and cannot be converted!")

    if not is_numeric_dtype(df["volume"]):
        try:
            df["volume"] = df["volume"].astype(float)
        except ValueError:
            raise ValueError("Column 'volume' not numeric and cannot be converted!")

    calc_per_city = "city" in df.columns
    for col in list(df.columns):
        if col == "city":
            continue
        if not is_numeric_dtype(df[col]):
            df.drop(columns=col, inplace=True)

    if 0 not in df["worth"].values:
        raise ValueError(
            "Baseline volume is missing! There has to be one volume whose worth is 0!"
        )

    if buy_or_sell_side == "buyer":
        ascending = True
    elif buy_or_sell_side == "seller":
        ascending = False
    else:
        raise ValueError("buy_or_sell_side has to be either 'buyer' or 'seller'")

    use_orig_costs = "costs" in df.columns
    numeric_cols = [col for col in df.columns if col != "city"]

    def _diff_to_marginal_steps(city_df: pd.DataFrame) -> pd.DataFrame:
        city_df = city_df.sort_values("volume", ascending=ascending)
        if use_orig_costs:
            original_costs = city_df["costs"].copy()
        stepped = city_df.diff().dropna().reset_index(drop=True).abs()
        if use_orig_costs:
            stepped["costs"] = original_costs
        return stepped

    if calc_per_city:
        baseline_rows = df.loc[df["worth"] == 0, numeric_cols]
        reference_baseline = baseline_rows.loc[baseline_rows["worth"].idxmax()]

        city_curves = []
        for city in volumes_worth["city"].unique():
            city_df = df.loc[df["city"] == city, numeric_cols].copy()
            if 0 not in city_df["worth"].values:
                city_df = pd.concat(
                    [reference_baseline.to_frame().T, city_df],
                    ignore_index=True,
                )
            city_curve = _diff_to_marginal_steps(city_df)
            city_curve["city"] = city
            city_curves.append(city_curve)
        df = pd.concat(city_curves, ignore_index=True)
    else:
        df = _diff_to_marginal_steps(df[numeric_cols])

    df.rename(columns={"worth": "marginal_price"}, inplace=True)
    df = df[df["volume"] != 0].reset_index(drop=True)
    df["marginal_price_per_kwh"] = df["marginal_price"] / df["volume"]

    if calc_per_city:
        df["cumulative_volume"] = df.groupby("city", sort=False)["volume"].cumsum()
    else:
        df["cumulative_volume"] = df["volume"].cumsum()

    output_cols = [
        "volume",
        "cumulative_volume",
        "marginal_price",
        "marginal_price_per_kwh",
    ]
    if calc_per_city:
        output_cols.append("city")
    return df[output_cols]


_CASHFLOW_COMPONENT_ORDER = (
    "community",
    "supplier",
    "eeg",
    "wholesale",
    "grid_fees",
)


def plot_multiple_storage_worth_cashflows(
    results: dict,
    *,
    stacked: bool = True,
    show: bool = True,
    title: str | None = None,
) -> Figure:
    """Plot cashflow components from :func:`calculate_multiple_storage_worth` with ``return_cashflows=True``.

    Expects ``results`` to contain ``baseline_cashflows``, ``storages_to_calc_cashflows``,
    and ``results_df`` (the same keys returned by that function).

    Args:
        results: Output dict of ``calculate_multiple_storage_worth(..., return_cashflows=True)``.
        stacked: If True, stacked bars per scenario; otherwise grouped bars by component.
        show: If True, call ``show()`` on the figure (Plotly).
        title: Optional plot title; a default is used if omitted.

    Returns:
        Plotly figure (stacked or grouped bar chart, EUR by market component).

    Raises:
        ValueError: If required keys are missing or cashflow dicts have unexpected keys.
    """
    required = ("baseline_cashflows", "storages_to_calc_cashflows", "results_df")
    missing = [k for k in required if k not in results]
    if missing:
        msg = "results must contain keys from calculate_multiple_storage_worth(..., return_cashflows=True); "
        msg += f"missing: {', '.join(missing)}"
        raise ValueError(msg)

    baseline_cf: dict = results["baseline_cashflows"]
    stor_cf: dict = results["storages_to_calc_cashflows"]
    results_df: pd.DataFrame = results["results_df"]

    _validate_cashflow_dict(baseline_cf, "baseline_cashflows")
    for sid, cf in stor_cf.items():
        _validate_cashflow_dict(cf, f"storages_to_calc_cashflows[{sid!r}]")

    rows: list[dict] = []
    rows.extend(
        {"scenario": "Baseline", "component": comp, "EUR": float(baseline_cf[comp])}
        for comp in _CASHFLOW_COMPONENT_ORDER
    )

    for _, row in results_df.iloc[1:].iterrows():
        sid = row["id"]
        if sid not in stor_cf:
            msg = (
                f"results_df storage id {sid!r} not found in storages_to_calc_cashflows"
            )
            raise ValueError(msg)
        vol = row.get("volume")
        if vol is not None and pd.notna(vol):
            scen = f"id={sid}, vol={float(vol)} kWh"
        else:
            scen = f"id={sid}"
        cf = stor_cf[sid]
        rows.extend(
            {"scenario": scen, "component": comp, "EUR": float(cf[comp])}
            for comp in _CASHFLOW_COMPONENT_ORDER
        )

    long = pd.DataFrame(rows)
    long["component"] = pd.Categorical(
        long["component"],
        categories=list(_CASHFLOW_COMPONENT_ORDER),
        ordered=True,
    )

    fig_title = title if title is not None else "Cashflows by scenario (EUR)"
    barmode = "stack" if stacked else "group"
    fig = px.bar(
        long,
        x="scenario",
        y="EUR",
        color="component",
        title=fig_title,
        barmode=barmode,
        category_orders={"component": list(_CASHFLOW_COMPONENT_ORDER)},
    )
    fig.update_layout(xaxis_title=None, legend_title_text="Market")
    fig.update_xaxes(tickangle=-25)

    if show:
        fig.show()
    return fig


def _validate_cashflow_dict(cf: dict, label: str) -> None:
    expected = set(_CASHFLOW_COMPONENT_ORDER)
    if set(cf.keys()) != expected:
        msg = f"{label} must have keys {sorted(expected)}, got {sorted(cf.keys())}"
        raise ValueError(msg)
