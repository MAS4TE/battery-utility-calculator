<!--
SPDX-FileCopyrightText: Christoph Komanns, Florian Maurer

SPDX-License-Identifier: MIT
-->

# Battery Utility Calculator

This tool provides a calculation of the utility a storage provides to an electricity consumer.

The utility is calculated by optimizing the storage dispatch and comparing the utility of the storage vs without it.
Iterating over different storage volumes can create a price curve of the stepwise utility each additional capacity provides.
Such values can be used in bidding projects or to investigate whether an additional storage is beneficial.

## Install & run tests

```sh
pip install -e .[test]
pytest
```

## Input conventions

All timeseries are `pandas.Series` and **must share one identical `pd.DatetimeIndex`** —
`demand`, `solar_generation`, `supplier_prices`, `eeg_prices`, `wholesale_market_prices`
and every series inside `community_market_prices`. A non-datetime index raises `TypeError`,
a mismatched one raises `ValueError`.

Values have to be finite: NaN or `inf` in any series raises `ValueError`. The one
exception is NaN in `community_market_prices`, which marks a timestep without community
trading (see [Locations and grid fees](#locations-and-grid-fees)); `inf` is rejected there
as well.

- `demand` and `solar_generation` are powers in kW; they are converted internally to
  kWh per timestep using `hours_per_timestep`.
- All prices are in EUR per kWh.
- `hours_per_timestep` describes the timestep length (e.g. `0.25` for 15 minutes).

## Usage examples (public helpers)

This package exposes small helper functions for common workflows plus the underlying optimizer class.

- `Storage(id, c_rate, volume, charge_efficiency=0.98, discharge_efficiency=0.98)` — small value
  object describing a storage unit. `c_rate` is a C-rate in 1/h: the energy limit per timestep is
  `c_rate * volume * hours_per_timestep` kWh, for charging and discharging alike.
- `calculate_storage_worth(baseline_storage, storage_to_calculate, demand, solar_generation, supplier_prices, eeg_prices, wholesale_market_prices, community_market_prices=None, ...)` — returns the value (difference in optimized cashflows) of adding `storage_to_calculate` compared to `baseline_storage`.
- `calculate_multiple_storage_worth(...)` — same as above but returns a DataFrame with cashflows and worth for multiple storage sizes.
- `calculate_multiple_storage_worth_by_location(..., locations_to_calculate=[...])` — runs the above once per storage location from the tenant perspective (`is_rented_storage=True`) and returns one combined DataFrame.
- `calculate_bidding_curve(volumes_worth, buy_or_sell_side)` — converts cumulative worths into a marginal bidding curve.
- `plot_multiple_storage_worth_cashflows(results)` — bar chart of the cashflow components returned by `calculate_multiple_storage_worth(..., return_cashflows=True)`.

Minimal examples (adapted from `tests/`):

```py
import pandas as pd
from battery_utility_calculator import (
    Storage,
    calculate_bidding_curve,
    calculate_multiple_storage_worth,
    calculate_storage_worth,
)

index = pd.date_range("2025-01-01", freq="h", periods=3)
baseline = Storage(
    id=0, c_rate=1, volume=0, charge_efficiency=1, discharge_efficiency=1
)
candidate = Storage(
    id=1, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1
)

common = dict(
    demand=pd.Series([1, 1, 1], index=index),
    solar_generation=pd.Series([0, 0, 0], index=index),
    supplier_prices=pd.Series([0, 1, 1], index=index),
    eeg_prices=pd.Series([0, 0, 0], index=index),
    wholesale_market_prices=pd.Series([0, 0, 0], index=index),
    community_market_prices={"aachen": pd.Series([0.5, 0.5, 0.5], index=index)},
    solver="appsi_highs",
)
# the storage can charge for free at t=0 and cover t=1, which would otherwise
# be served from the community market at 0.5 EUR/kWh -> a worth of about 0.5,
# minus whatever cycle_cost_per_kwh is charged for the discharged kWh

# single worth (basic use)
worth = calculate_storage_worth(
    baseline_storage=baseline,
    storage_to_calculate=candidate,
    cycle_cost_per_kwh=0.05,  # optional degradation cost
    **common,
)

# requesting a cashflow breakdown from the same call
result = calculate_storage_worth(
    baseline_storage=baseline,
    storage_to_calculate=candidate,
    return_cashflows=True,
    **common,
)
# result is a dict containing keys 'worth',
# 'baseline_cashflows' and 'storage_to_calc_cashflows'.

# requesting SOC timeseries output
soc_result = calculate_storage_worth(
    baseline_storage=baseline,
    storage_to_calculate=candidate,
    return_soc_timeseries=True,
    **common,
)
# soc_result contains 'baseline_soc_ts' and 'storage_to_calc_soc_ts'.

# multiple worths (cashflows are available by setting return_cashflows=True)
storages = [
    Storage(id=1, c_rate=1, volume=1, charge_efficiency=1, discharge_efficiency=1),
    Storage(id=2, c_rate=1, volume=2, charge_efficiency=1, discharge_efficiency=1),
]
df = calculate_multiple_storage_worth(
    baseline_storage=baseline,
    storages_to_calculate=storages,
    cycle_cost_per_kwh=0.05,
    **common,
)

# bidding curve; one row must have worth 0 to act as the baseline
vol_worth = pd.DataFrame({"volume": [0, 1, 2, 3], "worth": [0, 5, 7, 8]})
curve = calculate_bidding_curve(volumes_worth=vol_worth, buy_or_sell_side="buyer")
```

## Locations and grid fees

Flows between locations are charged with a grid fee taken from a nested dict,
`grid_fee_between_locations[from_location][to_location]` in EUR/kWh. The default
table `DEFAULT_GRID_FEE_BETWEEN_LOCATIONS` covers `juelich`, `aachen`, `heerlen`
and `liege`; values you pass are overlaid on it, so location names outside that
set are rejected with `ValueError`.

- `my_location` is where the prosumer sits, `storage_location` where the storage sits
  (defaults to `my_location`).
- `is_rented_storage=True` models the tenant / buyer: their flows into and out of
  somebody else's storage are charged.
- `is_rented_storage=False` models the storage provider and requires
  `storage_location == my_location`; only external imports into the storage are charged.
- Community imports that go straight to the home bypass the storage and are charged
  in both cases.
- `storage_location != my_location` disables charging the storage from the supplier.

`community_market_prices` defaults to `None` (no community market). Pass a dict keyed by
location name when community trading should be modelled, for example `{"aachen": pd.Series(...)}`.

A NaN in one of these series means that this community market does not trade in that
timestep, for example because it did not clear. All flows from and to that location are
then fixed to zero there: `pv_to_community`, `storage_to_community`,
`community_to_storage_for_home`, `community_to_storage_for_community` and
`community_to_home`. The block applies per location, so other community markets keep
trading in the same timestep. `get_price_df()` reports these timesteps as NaN again.

```py
community_market_prices = {
    # no community trading in Aachen at t=1
    "aachen": pd.Series([0.15, np.nan, 0.15], index=index),
}
```

Do not use NaN to mark faulty data. A NaN in the community prices is not reported as an
error: it is read as "no trading" and changes the result. A gap in any other series still
raises.

## Notes about the optimizer

The core optimizer is `EnergyCostCalculator` (in `battery_utility_calculator/energy_costs_calculator.py`).
It builds a Pyomo `ConcreteModel` with variables like `pv_to_storage[t, use]` and one
state-of-charge variable per use case (`eeg`, `wholesale`, `community`, `home`), which keeps
the buckets accounted for separately while sharing the volume and c-rate limits. The objective
maximizes summed cashflows (community + supplier + EEG + wholesale − grid fees), minus the
discharge penalty and the optional cycle cost. If you need lower-level control or plotting,
instantiate `EnergyCostCalculator` directly and call `optimize(solver=...)`.

Time-index handling: the optimizer normalizes all timeseries to integer timesteps and keeps
the original index in `self.original_index`, which the exporters use to restore timestamps.

### Optional cycle-cost parameter

`EnergyCostCalculator`, `calculate_storage_worth`, and `calculate_multiple_storage_worth`
support `cycle_cost_per_kwh` as an optional degradation cost in EUR per discharged kWh.
This value is subtracted from the objective proportionally to storage throughput.

Rule-of-thumb range for home storage (LFP, 2026 market snapshots):

- Cost basis: roughly `250-450 EUR/kWh` installed storage capacity
- Lifetime: roughly `5,000-10,000` full cycles
- Derived cycle cost range: about `0.025-0.09 EUR/kWh` discharged
- Typical working value for scenario analysis: `~0.05 EUR/kWh`

Quick derivation:

`cycle_cost_per_kwh = storage_cost_per_kwh / cycle_lifetime`

Example:

`300 EUR/kWh / 6000 cycles = 0.05 EUR/kWh`

### Storage usage KPIs (no timeseries plot)

If you want a compact view of how a storage was used after optimization:

```py
import pandas as pd
from battery_utility_calculator import EnergyCostCalculator, Storage

index = pd.date_range("2025-01-01", freq="h", periods=3)
ecc = EnergyCostCalculator(
    storage=Storage(id=0, c_rate=1, volume=1),
    demand=pd.Series([0, 1, 0], index=index),
    solar_generation=pd.Series([1, 0, 0], index=index),
    supplier_prices=pd.Series([10, 10, 10], index=index),
    eeg_prices=pd.Series([0, 0, 0], index=index),
    wholesale_market_prices=pd.Series([0, 0, 0], index=index),
    community_market_prices=None,  # no community market in this example
    allow_pv_to_wholesale=False,
    cycle_cost_per_kwh=0.05,
)
ecc.optimize(solver="appsi_highs")

kpis = ecc.get_storage_usage_kpis()
print(kpis["charged_kwh_total"])
print(kpis["charged_by_source_kwh"])
print(kpis["discharged_by_sink_kwh"])

# simple aggregate visualization without time axis
ecc.plot_storage_usage_summary()
```

## Uncertainty analysis

The optimizer is deterministic and has perfect foresight, so a single run yields a
single number. To see how much that number depends on the assumed forecast, sample
scenarios around the base timeseries and solve each of them.

- `sample_scenarios(...)` — draws scenarios by adding correlated, persistent noise to
  the base timeseries. Returns `Scenario` objects; `scenario.to_ecc_kwargs()` feeds
  straight into the optimizer. NaN in the community prices stays NaN in every scenario,
  so a timestep without community trading stays one.
- `calculate_storage_worth_distribution(...)` — solves every scenario, one row per
  scenario and storage.
- `summarize_worth_distribution(...)` — mean, spread and quantiles per storage size.
- `calculate_bidding_curve_distribution(...)` — one full bidding curve per scenario.
- `summarize_bidding_curve(...)` — quantile bands per volume step.
- `calculate_risk_adjusted_bidding_curve(...)` — a single curve to bid with.
- `plot_worth_distribution(...)`, `plot_bidding_curve_with_bands(...)` — Plotly figures.

```py
import numpy as np
import pandas as pd
from battery_utility_calculator import (
    Storage,
    calculate_bidding_curve_distribution,
    calculate_risk_adjusted_bidding_curve,
    calculate_storage_worth_distribution,
    sample_scenarios,
    summarize_bidding_curve,
)

index = pd.date_range("2025-06-01", freq="h", periods=24)
hours = np.arange(24)

scenarios = sample_scenarios(
    demand=pd.Series(0.4 + 0.5 * np.exp(-((hours - 19) ** 2) / 8), index=index),
    solar_generation=pd.Series(
        np.clip(4 * np.sin((hours - 6) * np.pi / 12), 0, None), index=index
    ),
    supplier_prices=pd.Series(0.32, index=index),
    eeg_prices=pd.Series(0.08, index=index),
    wholesale_market_prices=pd.Series(0.09, index=index),
    community_market_prices=None,
    n_scenarios=50,
    relative_std={"solar_generation": 0.25},  # overrides DEFAULT_RELATIVE_STD
    persistence_hours=12,  # forecast errors decay to 1/e after 12 hours
    hours_per_timestep=1,
    seed=42,
)

# storage may only serve the household, so its value saturates
self_consumption = dict(
    storage_use_cases=["home", "eeg"],
    allow_storage_to_wholesale=False,
    allow_wholesale_to_storage=False,
    allow_pv_to_wholesale=False,
    allow_community_to_home=False,
    allow_community_to_storage=False,
    allow_pv_to_community=False,
    allow_storage_to_community=False,
)

distribution = calculate_storage_worth_distribution(
    baseline_storage=Storage(id=0, c_rate=1, volume=0),
    storages_to_calculate=[
        Storage(id=i, c_rate=1, volume=v) for i, v in enumerate([1, 2, 4, 8], start=1)
    ],
    scenarios=scenarios,
    solver="appsi_highs",
    **self_consumption,
)

bands = summarize_bidding_curve(
    calculate_bidding_curve_distribution(distribution, "buyer")
)
print(bands[["cumulative_volume", "marginal_price_q05", "marginal_price_q95"]])

# a single curve fit to bid with: the buyer takes the lower tail
bid = calculate_risk_adjusted_bidding_curve(distribution, "buyer", risk_level=0.05)
print(bid)
```

### What the numbers do and do not say

Each scenario is optimized on its own, so the optimizer knows *that scenario's*
future exactly (wait-and-see). The spread therefore answers "how much does the worth
vary across possible futures" - it is not the worth to an operator deciding without
knowing the future, and the mean is optimistically biased.

Two modelling points worth knowing before reading the output:

- **Correlations are not free parameters.** `DEFAULT_CORRELATIONS` ties retail prices
  to the exchange one to one, which forces retail to inherit every other correlation
  of the exchange. `build_correlation_matrix` rejects combinations that describe no
  joint distribution, so adjust the set as a whole rather than pair by pair.
- **Persistence drives the spread.** With `persistence_hours=0` the forecast errors
  average out over the horizon and the worth barely moves between scenarios. The
  default of 12 hours means an error still correlates at 37 % half a day later.

### Where storage value comes from

The example above deliberately blocks trading, because the value of a storage needs
a *spread* between buying and selling. A community market modelled with one price per
timestep for both directions removes that spread entirely: selling PV and buying it
back at the same price is a wash, so the optimizer bypasses self-consumption and only
runs intraday arbitrage. Value is then linear in the volume, the marginal price never
falls, and the bidding curve carries no information. With the classic prosumer spread
(0.32 buying, 0.08 feed-in) the value saturates once the evening load is covered, which
is what makes a bidding curve meaningful.

## License
MIT - see [LICENSE](./LICENSE)
