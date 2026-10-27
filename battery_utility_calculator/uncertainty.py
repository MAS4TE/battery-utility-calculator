# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT

"""Parametric scenario generation for the uncertainty analysis.

The optimizer is deterministic and has perfect foresight. To say something about
how uncertain a storage worth is, the same problem is solved once per sampled
scenario and the resulting spread is evaluated (wait-and-see).

Scenarios are drawn from the deterministic base timeseries by adding correlated
noise. Two properties matter for the result and are therefore explicit:

* Cross correlation. Retail prices follow the exchange, PV weighs on it, and the
  community market reacts mostly to local PV feed-in. Sampling the series
  independently would create combinations that do not occur and would misstate
  the storage worth. See ``DEFAULT_CORRELATIONS`` for the assumed structure.
* Persistence. Forecast errors last - a cloudy day stays cloudy. Independent
  noise per timestep would average out over the horizon and understate the
  spread of the storage worth considerably.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# imported from the module rather than the package to avoid an import cycle
from battery_utility_calculator.electric_vehicle import ElectricVehicle
from battery_utility_calculator.storage import Storage
from battery_utility_calculator.timeseries import check_identical_indices

# the order fixes rows and columns of the correlation matrix
UNCERTAIN_QUANTITIES = (
    "demand",
    "solar_generation",
    "supplier_prices",
    "wholesale_market_prices",
    "community_market_prices",
)

# quantities scale with their level, prices are shifted; see the module docstring
_DISTURBANCE_KIND = {
    "demand": "relative",
    "solar_generation": "relative",
    "supplier_prices": "absolute",
    "wholesale_market_prices": "absolute",
    "community_market_prices": "absolute",
}

# rough starting points, meant to be overridden with numbers from actual forecasts
DEFAULT_RELATIVE_STD = {
    "demand": 0.10,
    "solar_generation": 0.20,
    "supplier_prices": 0.10,
    "wholesale_market_prices": 0.30,
    "community_market_prices": 0.30,
}

# Pairwise correlations; unlisted pairs default to 0.0.
#
# Reasoning behind the values:
#
# * Retail prices track the exchange one to one, so their disturbance is the same
#   disturbance. As a consequence supplier_prices is not free to have its own
#   relationship with anything else - it inherits every correlation of
#   wholesale_market_prices. The two supplier entries below are therefore not
#   independent choices, they are copies of the wholesale row, and changing one
#   without the other makes the matrix unsamplable.
# * PV weighs on the exchange price, but only weakly: wind can depress prices just
#   as well on an overcast day, so a low price is a poor indicator of sunshine.
# * The community market is driven by local PV feed-in rather than by the
#   exchange - hence nearly independent of wholesale, but clearly negative
#   against solar_generation.
#
# Note that a single common market factor cannot produce this structure: tying
# supplier to wholesale at 1.0 would force correlation(solar, community) to be
# the product of their wholesale correlations, about -0.02 instead of -0.50. The
# matrix is therefore given directly and checked in build_correlation_matrix.
DEFAULT_CORRELATIONS = {
    ("wholesale_market_prices", "supplier_prices"): 1.00,
    ("wholesale_market_prices", "solar_generation"): -0.20,
    ("wholesale_market_prices", "community_market_prices"): 0.10,
    # forced by the perfect wholesale/supplier coupling above
    ("supplier_prices", "solar_generation"): -0.20,
    ("supplier_prices", "community_market_prices"): 0.10,
    ("community_market_prices", "solar_generation"): -0.50,
}

DEFAULT_PERSISTENCE_HOURS = 12.0


@dataclass(frozen=True)
class Scenario:
    """One complete, internally consistent set of optimizer inputs."""

    demand: pd.Series
    solar_generation: pd.Series
    supplier_prices: pd.Series
    eeg_prices: pd.Series
    wholesale_market_prices: pd.Series
    community_market_prices: dict[str, pd.Series] | None = None

    def to_ecc_kwargs(self) -> dict:
        """Keyword arguments for ``EnergyCostCalculator`` and the worth helpers."""
        return {
            "demand": self.demand,
            "solar_generation": self.solar_generation,
            "supplier_prices": self.supplier_prices,
            "eeg_prices": self.eeg_prices,
            "wholesale_market_prices": self.wholesale_market_prices,
            "community_market_prices": self.community_market_prices,
        }


def build_correlation_matrix(
    correlations: dict[tuple[str, str], float] | None = None,
) -> np.ndarray:
    """Build the correlation matrix over ``UNCERTAIN_QUANTITIES``.

    Args:
        correlations: Pairwise correlations keyed by a tuple of quantity names in
            either order. Unlisted pairs are uncorrelated. ``None`` uses
            ``DEFAULT_CORRELATIONS``.

    Returns:
        Symmetric matrix with ones on the diagonal.

    Raises:
        ValueError: On unknown quantity names, correlations outside [-1, 1],
            contradicting entries for the same pair, or a matrix that is not
            positive semi-definite and therefore not sampleable.
    """
    if correlations is None:
        correlations = DEFAULT_CORRELATIONS

    index = {name: position for position, name in enumerate(UNCERTAIN_QUANTITIES)}
    matrix = np.eye(len(UNCERTAIN_QUANTITIES))

    for (first, second), value in correlations.items():
        for name in (first, second):
            if name not in index:
                msg = (
                    f"Unknown quantity '{name}' in correlations. "
                    f"Allowed values: {sorted(index)}"
                )
                raise ValueError(msg)
        if not -1.0 <= value <= 1.0:
            msg = f"Correlation for ('{first}', '{second}') must be within [-1, 1], got {value}."
            raise ValueError(msg)
        if first == second:
            if value != 1.0:
                msg = f"Correlation of '{first}' with itself must be 1.0, got {value}."
                raise ValueError(msg)
            continue

        row, column = index[first], index[second]
        existing = matrix[row, column]
        if existing != 0.0 and existing != value:
            msg = (
                f"Contradicting correlations given for '{first}' and '{second}': "
                f"{existing} and {value}."
            )
            raise ValueError(msg)
        matrix[row, column] = value
        matrix[column, row] = value

    smallest_eigenvalue = float(np.linalg.eigvalsh(matrix).min())
    if smallest_eigenvalue < -1e-8:
        msg = (
            "The correlation matrix is not positive semi-definite (smallest "
            f"eigenvalue {smallest_eigenvalue:.4f}), so no joint distribution "
            "matches it. Relax the conflicting pairs, e.g. a strong positive "
            "correlation of A with B and of B with C forces A and C to be "
            "correlated as well."
        )
        raise ValueError(msg)

    return matrix


def _persistence_factor(persistence_hours: float, hours_per_timestep: float) -> float:
    """AR(1) coefficient for an e-folding time of ``persistence_hours``."""
    if persistence_hours <= 0:
        return 0.0
    return float(np.exp(-hours_per_timestep / persistence_hours))


def _correlated_ar1_noise(
    n_scenarios: int,
    n_timesteps: int,
    correlation_matrix: np.ndarray,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Standard normal noise, correlated across quantities and over time.

    Returns an array of shape ``(n_scenarios, n_timesteps, len(UNCERTAIN_QUANTITIES))``
    with unit marginal variance, so the caller only has to scale it.
    """
    n_quantities = correlation_matrix.shape[0]

    # eigendecomposition rather than Cholesky, so that merely semi-definite
    # matrices (a correlation of exactly 1.0 between two quantities) still work
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    factor = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))

    white = rng.standard_normal((n_scenarios, n_timesteps, n_quantities))
    correlated = white @ factor.T

    # AR(1) along the time axis; the scaling keeps the marginal variance at one,
    # which also preserves the cross correlation imposed above
    noise = np.empty_like(correlated)
    noise[:, 0, :] = correlated[:, 0, :]
    innovation_scale = float(np.sqrt(1.0 - rho**2))
    for timestep in range(1, n_timesteps):
        noise[:, timestep, :] = (
            rho * noise[:, timestep - 1, :]
            + innovation_scale * correlated[:, timestep, :]
        )

    return noise


def _disturb(base: pd.Series, noise: np.ndarray, relative_std: float, kind: str):
    """Apply one quantity's noise column to its base timeseries."""
    values = base.to_numpy(dtype=float)

    if kind == "relative":
        # scales with the level, so a night without PV stays a night without PV
        disturbed = values * (1.0 + relative_std * noise)
        disturbed = np.clip(disturbed, 0.0, None)
    else:
        # absolute shift derived from the mean magnitude, which keeps working
        # for prices at or below zero
        scale = relative_std * float(np.abs(values).mean())
        disturbed = values + scale * noise

    return pd.Series(disturbed, index=base.index, name=base.name)


def sample_scenarios(
    demand: pd.Series,
    solar_generation: pd.Series,
    supplier_prices: pd.Series,
    eeg_prices: pd.Series,
    wholesale_market_prices: pd.Series,
    community_market_prices: dict[str, pd.Series] | None = None,
    *,
    n_scenarios: int = 100,
    relative_std: dict[str, float] | None = None,
    correlations: dict[tuple[str, str], float] | None = None,
    persistence_hours: float = DEFAULT_PERSISTENCE_HOURS,
    hours_per_timestep: int | float = 1,
    seed: int | None = None,
) -> list[Scenario]:
    """Draw scenarios around deterministic base timeseries.

    ``demand`` and ``solar_generation`` are disturbed multiplicatively and clipped
    at zero, prices are shifted by an absolute amount derived from their mean
    magnitude so that negative prices keep working. Disturbances are correlated
    across quantities and follow an AR(1) process over time.

    ``eeg_prices`` are passed through unchanged - the feed-in tariff is fixed by
    law and carries no forecast uncertainty.

    All locations in ``community_market_prices`` share one disturbance, since
    neighbouring local markets are assumed to move together. Model separate
    location risk by sampling those series yourself and assembling
    :class:`Scenario` objects directly.

    Args:
        demand: Base demand in kW per timestep, with a ``pd.DatetimeIndex``.
        solar_generation: Base PV generation in kW per timestep.
        supplier_prices: Base grid prices in EUR per kWh.
        eeg_prices: EEG prices in EUR per kWh, used unchanged.
        wholesale_market_prices: Base wholesale prices in EUR per kWh.
        community_market_prices: Base community prices per location, or ``None``.
        n_scenarios: Number of scenarios to draw.
        relative_std: Relative standard deviation per quantity, overriding
            ``DEFAULT_RELATIVE_STD``. For prices it refers to the mean magnitude
            of the respective series.
        correlations: Pairwise correlations, see :func:`build_correlation_matrix`.
        persistence_hours: E-folding time of the disturbances in hours. Larger
            values make forecast errors last longer; ``0`` gives white noise.
        hours_per_timestep: Duration of one timestep in hours, used to translate
            ``persistence_hours`` into the AR(1) coefficient.
        seed: Seed for reproducible draws.

    Returns:
        A list of ``n_scenarios`` :class:`Scenario` objects.

    Raises:
        TypeError: If an index is not a ``pd.DatetimeIndex``.
        ValueError: On mismatched indices, ``n_scenarios < 1``, negative standard
            deviations, unknown quantity names or a non-sampleable correlation matrix.
    """
    if n_scenarios < 1:
        msg = f"n_scenarios has to be at least 1, got {n_scenarios}."
        raise ValueError(msg)

    base_series = {
        "demand": demand,
        "solar_generation": solar_generation,
        "supplier_prices": supplier_prices,
        "eeg_prices": eeg_prices,
        "wholesale_market_prices": wholesale_market_prices,
    }
    if community_market_prices is not None:
        for location, series in community_market_prices.items():
            base_series[f"community_market_prices['{location}']"] = series
    index = check_identical_indices(base_series)

    standard_deviations = dict(DEFAULT_RELATIVE_STD)
    if relative_std is not None:
        for name, value in relative_std.items():
            if name not in DEFAULT_RELATIVE_STD:
                msg = (
                    f"Unknown quantity '{name}' in relative_std. "
                    f"Allowed values: {sorted(DEFAULT_RELATIVE_STD)}"
                )
                raise ValueError(msg)
            if value < 0:
                msg = f"relative_std['{name}'] must not be negative, got {value}."
                raise ValueError(msg)
            standard_deviations[name] = float(value)

    correlation_matrix = build_correlation_matrix(correlations)
    rho = _persistence_factor(float(persistence_hours), float(hours_per_timestep))
    rng = np.random.default_rng(seed)
    noise = _correlated_ar1_noise(
        n_scenarios=n_scenarios,
        n_timesteps=len(index),
        correlation_matrix=correlation_matrix,
        rho=rho,
        rng=rng,
    )

    column = {name: position for position, name in enumerate(UNCERTAIN_QUANTITIES)}
    scenarios = []
    for scenario_number in range(n_scenarios):
        scenario_noise = noise[scenario_number]

        def disturb(name: str, base: pd.Series, scenario_noise=scenario_noise):
            return _disturb(
                base=base,
                noise=scenario_noise[:, column[name]],
                relative_std=standard_deviations[name],
                kind=_DISTURBANCE_KIND[name],
            )

        if community_market_prices is None:
            sampled_community = None
        else:
            sampled_community = {
                location: disturb("community_market_prices", series)
                for location, series in community_market_prices.items()
            }

        scenarios.append(
            Scenario(
                demand=disturb("demand", demand),
                solar_generation=disturb("solar_generation", solar_generation),
                supplier_prices=disturb("supplier_prices", supplier_prices),
                eeg_prices=eeg_prices.copy(),
                wholesale_market_prices=disturb(
                    "wholesale_market_prices", wholesale_market_prices
                ),
                community_market_prices=sampled_community,
            )
        )

    return scenarios


def calculate_storage_worth_distribution(
    baseline_storage: Storage,
    storages_to_calculate: list[Storage],
    scenarios: Sequence[Scenario],
    hours_per_timestep: int | float = 1,
    storage_use_cases: list[str] = ["eeg", "wholesale", "community", "home"],
    allow_community_to_home: bool = True,
    allow_community_to_storage: bool = True,
    allow_community_market_arbitrage: bool = True,
    allow_pv_to_community: bool = True,
    allow_storage_to_community: bool = True,
    allow_pv_to_wholesale: bool = False,
    allow_wholesale_to_storage: bool = True,
    allow_storage_to_wholesale: bool = True,
    wholesale_fee: float = 0.3,
    my_location: str = "aachen",
    grid_fee_between_locations: dict[str, dict[str, float]] | None = None,
    storage_location: str | None = None,
    is_rented_storage: bool = False,
    eeg_eligible: bool = True,
    goal: str = "max_cashflow",
    discharge_penalty_per_kwh: float = 1e-6,
    cycle_cost_per_kwh: float = 0.0,
    electric_vehicle: ElectricVehicle | None = None,
    solver: str = "gurobi",
) -> pd.DataFrame:
    """Calculate the storage worth once per scenario (wait-and-see).

    Each scenario is optimized on its own, which means the optimizer knows that
    scenario's future exactly. The spread of the result therefore describes how
    much the worth varies across possible futures - it is not the worth to an
    operator who has to decide without knowing the future, and the mean is
    optimistically biased.

    Args:
        baseline_storage: The baseline storage to compare against.
        storages_to_calculate: Storages to evaluate in every scenario.
        scenarios: Scenarios to run, e.g. from :func:`sample_scenarios`.
        electric_vehicle: Electric vehicle of the household, used unchanged in every
            scenario (its profiles are not disturbed). Its index has to match the
            scenario timeseries.
        solver: Which solver to use. Defaults to "gurobi".

    All remaining arguments are passed to ``calculate_multiple_storage_worth``
    unchanged; the timeseries come from each scenario.

    Returns:
        One row per scenario and storage, including the baseline row of every
        scenario, with the columns of ``calculate_multiple_storage_worth`` plus a
        ``scenario`` column holding the index within ``scenarios``.

    Raises:
        ValueError: If ``scenarios`` is empty.
        TypeError: If an entry of ``scenarios`` is not a :class:`Scenario`.
    """
    # imported here because battery_utility_calculator imports this module
    from battery_utility_calculator.battery_utility_calculator import (
        calculate_multiple_storage_worth,
    )

    if len(scenarios) == 0:
        raise ValueError("scenarios must not be empty.")
    for position, scenario in enumerate(scenarios):
        if not isinstance(scenario, Scenario):
            msg = f"scenarios[{position}] has to be a Scenario, got {type(scenario).__name__}."
            raise TypeError(msg)

    per_scenario = []
    for position, scenario in enumerate(scenarios):
        worth_df = calculate_multiple_storage_worth(
            baseline_storage=baseline_storage,
            storages_to_calculate=storages_to_calculate,
            **scenario.to_ecc_kwargs(),
            hours_per_timestep=hours_per_timestep,
            storage_use_cases=storage_use_cases,
            allow_community_to_home=allow_community_to_home,
            allow_community_to_storage=allow_community_to_storage,
            allow_community_market_arbitrage=allow_community_market_arbitrage,
            allow_pv_to_community=allow_pv_to_community,
            allow_storage_to_community=allow_storage_to_community,
            allow_pv_to_wholesale=allow_pv_to_wholesale,
            allow_wholesale_to_storage=allow_wholesale_to_storage,
            allow_storage_to_wholesale=allow_storage_to_wholesale,
            wholesale_fee=wholesale_fee,
            my_location=my_location,
            grid_fee_between_locations=grid_fee_between_locations,
            storage_location=storage_location,
            is_rented_storage=is_rented_storage,
            eeg_eligible=eeg_eligible,
            goal=goal,
            discharge_penalty_per_kwh=discharge_penalty_per_kwh,
            cycle_cost_per_kwh=cycle_cost_per_kwh,
            electric_vehicle=electric_vehicle,
            solver=solver,
        )
        worth_df.insert(0, "scenario", position)
        per_scenario.append(worth_df)

    return pd.concat(per_scenario, ignore_index=True)


def _quantile_column(quantile: float, prefix: str = "worth") -> str:
    percent = quantile * 100
    if float(percent).is_integer():
        return f"{prefix}_q{int(percent):02d}"
    return f"{prefix}_q{percent:g}"


def _check_quantiles(quantiles: Sequence[float]) -> None:
    for quantile in quantiles:
        if not 0.0 <= quantile <= 1.0:
            msg = f"Quantiles have to lie within [0, 1], got {quantile}."
            raise ValueError(msg)


def summarize_worth_distribution(
    worth_distribution: pd.DataFrame,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> pd.DataFrame:
    """Aggregate the per-scenario worths into one row per storage.

    Args:
        worth_distribution: Output of :func:`calculate_storage_worth_distribution`.
        quantiles: Quantiles of the worth to report, each within [0, 1].

    Returns:
        One row per storage configuration with ``n_scenarios``, ``worth_mean``,
        ``worth_std``, ``worth_min``, ``worth_max``, ``cashflow_mean`` and one
        column per requested quantile, e.g. ``worth_q05``.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If a quantile lies outside [0, 1].
    """
    required = {"scenario", "worth", "cashflow", "volume"}
    missing = required - set(worth_distribution.columns)
    if missing:
        msg = f"worth_distribution is missing the columns {sorted(missing)}."
        raise KeyError(msg)

    _check_quantiles(quantiles)

    group_columns = [
        column
        for column in (
            "id",
            "c_rate",
            "volume",
            "charge_efficiency",
            "discharge_efficiency",
            "location",
        )
        if column in worth_distribution.columns
    ]
    grouped = worth_distribution.groupby(group_columns, sort=False, dropna=False)

    summary = grouped.agg(
        n_scenarios=("scenario", "nunique"),
        worth_mean=("worth", "mean"),
        worth_std=("worth", "std"),
        worth_min=("worth", "min"),
        worth_max=("worth", "max"),
        cashflow_mean=("cashflow", "mean"),
    )
    for quantile in quantiles:
        summary[_quantile_column(quantile)] = grouped["worth"].quantile(quantile)

    # a single scenario has no spread, which pandas reports as NaN
    summary["worth_std"] = summary["worth_std"].fillna(0.0)

    return summary.reset_index()


def calculate_bidding_curve_distribution(
    worth_distribution: pd.DataFrame,
    buy_or_sell_side: Literal["buyer", "seller"],
) -> pd.DataFrame:
    """Derive one complete bidding curve per scenario.

    The curve is built inside each scenario and only then compared across
    scenarios. Taking quantiles of the cumulative worths first and differencing
    afterwards would mix steps from different scenarios and can produce a
    non-monotonic curve that occurs in none of them.

    Args:
        worth_distribution: Output of :func:`calculate_storage_worth_distribution`.
        buy_or_sell_side: Passed to ``calculate_bidding_curve``.

    Returns:
        The stacked per-scenario curves with the columns of
        ``calculate_bidding_curve`` plus a ``scenario`` column.

    Raises:
        KeyError: If the ``scenario`` column is missing.
    """
    # imported here because battery_utility_calculator imports this module
    from battery_utility_calculator.battery_utility_calculator import (
        calculate_bidding_curve,
    )

    if "scenario" not in worth_distribution.columns:
        raise KeyError("worth_distribution is missing the column 'scenario'.")

    curves = []
    for scenario, rows in worth_distribution.groupby("scenario", sort=True):
        curve = calculate_bidding_curve(
            volumes_worth=rows.drop(columns="scenario"),
            buy_or_sell_side=buy_or_sell_side,
        )
        curve.insert(0, "scenario", scenario)
        curves.append(curve)

    return pd.concat(curves, ignore_index=True)


def _bidding_curve_group_columns(curve_distribution: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ("cumulative_volume", "location", "exclusive_id")
        if column in curve_distribution.columns
    ]


def summarize_bidding_curve(
    curve_distribution: pd.DataFrame,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> pd.DataFrame:
    """Aggregate the per-scenario curves into one row per volume step.

    Args:
        curve_distribution: Output of :func:`calculate_bidding_curve_distribution`.
        quantiles: Quantiles of the marginal price to report, each within [0, 1].

    Returns:
        One row per volume step with ``marginal_price_mean``, ``_std``, ``_min``,
        ``_max`` and one column per quantile, plus the matching
        ``marginal_price_per_kwh_*`` columns.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If a quantile lies outside [0, 1].
    """
    required = {"scenario", "volume", "cumulative_volume", "marginal_price"}
    missing = required - set(curve_distribution.columns)
    if missing:
        msg = f"curve_distribution is missing the columns {sorted(missing)}."
        raise KeyError(msg)

    _check_quantiles(quantiles)

    group_columns = _bidding_curve_group_columns(curve_distribution)
    grouped = curve_distribution.groupby(group_columns, sort=False, dropna=False)

    summary = grouped.agg(
        volume=("volume", "first"),
        n_scenarios=("scenario", "nunique"),
        marginal_price_mean=("marginal_price", "mean"),
        marginal_price_std=("marginal_price", "std"),
        marginal_price_min=("marginal_price", "min"),
        marginal_price_max=("marginal_price", "max"),
    )
    for quantile in quantiles:
        summary[_quantile_column(quantile, "marginal_price")] = grouped[
            "marginal_price"
        ].quantile(quantile)

    # a single scenario has no spread, which pandas reports as NaN
    summary["marginal_price_std"] = summary["marginal_price_std"].fillna(0.0)

    # the step volume is constant within a group, so the per kWh figures follow
    # from a plain division and stay consistent with the columns above
    price_columns = [
        column for column in summary.columns if column.startswith("marginal_price_")
    ]
    for column in price_columns:
        summary[column.replace("marginal_price_", "marginal_price_per_kwh_", 1)] = (
            summary[column] / summary["volume"]
        )

    return summary.reset_index()


def _risk_adjusted_value(values: pd.Series, risk_level: float, measure: str, side: str):
    """Conservative point estimate of a marginal price.

    A buyer must not overpay, so the lower tail is the careful choice; a seller
    must not undercut their own opportunity cost and looks at the upper tail.
    """
    if side == "buyer":
        quantile = values.quantile(risk_level)
        if measure == "quantile":
            return quantile
        tail = values[values <= quantile]
    else:
        quantile = values.quantile(1.0 - risk_level)
        if measure == "quantile":
            return quantile
        tail = values[values >= quantile]

    return tail.mean() if len(tail) else quantile


def calculate_risk_adjusted_bidding_curve(
    worth_distribution: pd.DataFrame,
    buy_or_sell_side: Literal["buyer", "seller"],
    risk_level: float = 0.05,
    risk_measure: Literal["quantile", "cvar"] = "quantile",
) -> pd.DataFrame:
    """Condense the scenario curves into a single curve fit to bid with.

    The direction of caution follows the side: a buyer takes the lower tail of
    the marginal prices so that the storage is worth at least the bid in
    ``1 - risk_level`` of the scenarios, a seller takes the upper tail.

    Args:
        worth_distribution: Output of :func:`calculate_storage_worth_distribution`.
        buy_or_sell_side: Which side to bid on.
        risk_level: Tail probability, e.g. ``0.05`` for a 5 % tail.
        risk_measure: ``"quantile"`` uses the tail boundary, ``"cvar"`` the mean
            of the tail beyond it, which also accounts for how bad the tail gets.

    Returns:
        The columns of ``calculate_bidding_curve``, with ``marginal_price`` and
        ``marginal_price_per_kwh`` replaced by their risk adjusted values.

    Raises:
        ValueError: If ``risk_level`` is outside (0, 1) or the measure is unknown.
    """
    if not 0.0 < risk_level < 1.0:
        msg = f"risk_level has to lie within (0, 1), got {risk_level}."
        raise ValueError(msg)
    if risk_measure not in ("quantile", "cvar"):
        msg = f"risk_measure has to be 'quantile' or 'cvar', got '{risk_measure}'."
        raise ValueError(msg)

    curve_distribution = calculate_bidding_curve_distribution(
        worth_distribution=worth_distribution,
        buy_or_sell_side=buy_or_sell_side,
    )

    group_columns = _bidding_curve_group_columns(curve_distribution)
    grouped = curve_distribution.groupby(group_columns, sort=False, dropna=False)

    curve = grouped.agg(volume=("volume", "first"))
    curve["marginal_price"] = grouped["marginal_price"].apply(
        _risk_adjusted_value,
        risk_level=risk_level,
        measure=risk_measure,
        side=buy_or_sell_side,
    )
    curve = curve.reset_index()
    curve["marginal_price_per_kwh"] = curve["marginal_price"] / curve["volume"]

    output_columns = [
        "volume",
        "cumulative_volume",
        "marginal_price",
        "marginal_price_per_kwh",
    ]
    for column in ("location", "exclusive_id"):
        if column in curve.columns:
            output_columns.append(column)
    return curve[output_columns]


# single hue, light to dark; these charts show one quantity, not several series
_BAND_COLOR = "#9ec5f4"
_LINE_COLOR = "#256abf"
_GRID_COLOR = "#e8e8e5"


def _apply_chart_style(fig: go.Figure) -> None:
    """Hairline, recessive grid and axes so the marks carry the chart."""
    fig.update_xaxes(gridcolor=_GRID_COLOR, zeroline=False, linecolor=_GRID_COLOR)
    fig.update_yaxes(gridcolor=_GRID_COLOR, zeroline=False, linecolor=_GRID_COLOR)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")


def plot_worth_distribution(
    worth_distribution: pd.DataFrame,
    show: bool = True,
    title: str | None = None,
) -> go.Figure:
    """Plot the spread of the storage worth per storage size.

    Args:
        worth_distribution: Output of :func:`calculate_storage_worth_distribution`.
        show: If True, call ``show()`` on the figure.
        title: Optional plot title; a default is used if omitted.

    Returns:
        Box plot of the worth per volume, one box per storage size.

    Raises:
        KeyError: If a required column is missing.
    """
    required = {"volume", "worth"}
    missing = required - set(worth_distribution.columns)
    if missing:
        msg = f"worth_distribution is missing the columns {sorted(missing)}."
        raise KeyError(msg)

    df = worth_distribution.copy()
    df["volume"] = df["volume"].astype(float)

    fig = px.box(
        df.sort_values("volume"),
        x="volume",
        y="worth",
        points="outliers",
        title=title if title is not None else "Storage worth across scenarios",
        color_discrete_sequence=[_LINE_COLOR],
    )
    fig.update_traces(line_width=1.5, marker_size=4, fillcolor=_BAND_COLOR)
    fig.update_layout(
        xaxis_title="Storage volume (kWh)",
        yaxis_title="Worth (EUR)",
        showlegend=False,
    )
    _apply_chart_style(fig)

    if show:
        fig.show()
    return fig


def plot_bidding_curve_with_bands(
    curve_distribution: pd.DataFrame,
    quantiles: tuple[float, float] = (0.05, 0.95),
    show: bool = True,
    title: str | None = None,
) -> go.Figure:
    """Plot the bidding curve as capacity blocks with an uncertainty band.

    Each capacity increment holds one price over its whole volume range, so it
    is drawn as a block sitting on that range - the usual merit order reading,
    and every price stays readable by pointing anywhere at its block instead of
    only at a step edge. The band between the quantiles rides on each block as
    an error bar.

    Args:
        curve_distribution: Output of :func:`calculate_bidding_curve_distribution`.
        quantiles: Lower and upper quantile bounding the band.
        show: If True, call ``show()`` on the figure.
        title: Optional plot title; a default is used if omitted.

    Returns:
        Figure with one block per increment, carrying the median price and the
        band as an asymmetric error bar.

    Raises:
        ValueError: If the quantiles are not a rising pair within [0, 1].
    """
    low, high = quantiles
    _check_quantiles(quantiles)
    if low >= high:
        msg = f"quantiles have to rise, got ({low}, {high})."
        raise ValueError(msg)

    bands = summarize_bidding_curve(curve_distribution, quantiles=[low, 0.5, high])
    bands = bands.sort_values("cumulative_volume").reset_index(drop=True)

    low_column = _quantile_column(low, "marginal_price_per_kwh")
    high_column = _quantile_column(high, "marginal_price_per_kwh")
    median_column = _quantile_column(0.5, "marginal_price_per_kwh")

    right_edges = bands["cumulative_volume"].to_numpy(dtype=float)
    widths = bands["volume"].to_numpy(dtype=float)
    left_edges = right_edges - widths
    median = bands[median_column].to_numpy(dtype=float)
    lower = bands[low_column].to_numpy(dtype=float)
    upper = bands[high_column].to_numpy(dtype=float)

    # one block per capacity increment, sitting on its own volume range - the
    # usual merit order reading, and hoverable across its whole width instead
    # of only at the step edges
    gap = 0.004 * float(right_edges[-1]) if len(right_edges) else 0.0
    fig = go.Figure(
        go.Bar(
            x=left_edges + widths / 2,
            y=median,
            width=np.maximum(widths - gap, widths * 0.6),
            marker_color=_BAND_COLOR,
            error_y=dict(
                type="data",
                symmetric=False,
                array=upper - median,
                arrayminus=median - lower,
                color=_LINE_COLOR,
                thickness=1.5,
                width=4,
            ),
            customdata=np.column_stack([left_edges, right_edges, lower, upper]),
            hovertemplate=(
                "<b>%{customdata[0]:.3g} to %{customdata[1]:.3g} kWh</b><br>"
                "median %{y:.3f} EUR/kWh<br>"
                f"{low:.0%} to {high:.0%} of scenarios: "
                "%{customdata[2]:.3f} to %{customdata[3]:.3f}<extra></extra>"
            ),
            name="Marginal price",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title if title is not None else "Bidding curve with uncertainty band",
        xaxis_title="Cumulative volume (kWh)",
        yaxis_title="Marginal price (EUR/kWh)",
        yaxis_rangemode="tozero",
        bargap=0,
        hovermode="closest",
    )
    # ticks on the step edges, so the blocks can be read against the volumes
    fig.update_xaxes(tickmode="array", tickvals=[0.0, *right_edges])
    _apply_chart_style(fig)

    if show:
        fig.show()
    return fig
