<!--
SPDX-FileCopyrightText: NOWUM Developers

SPDX-License-Identifier: MIT
-->

<!-- Short guidance for AI coding agents working on this repo -->

# Battery Utility Calculator — Copilot instructions (concise)

Goal: get an AI agent productive quickly. This repo is a small Python package that models a prosumer (PV + optional battery) dispatch optimizer using Pyomo. Focus on the optimizer, its data inputs, and the tests which are the ground truth.

Core modules
- `battery_utility_calculator/energy_costs_calculator.py` — EnergyCostCalculator (builds Pyomo model, variables, constraints, objective, exporters).
- `battery_utility_calculator/battery_utility_calculator.py` — helper functions (calculate_storage_worth, calculate_multiple_storage_worth, calculate_bidding_curve) that call ECC.
- `battery_utility_calculator/storage.py` — simple `Storage(id:int, c_rate, volume, efficiency)` value object.

Key tests to read
- `tests/test_BUC.py` and `tests/test_ECC.py` — concise, deterministic examples used as the canonical spec. Copy their small pandas examples when you add new tests or reproduce bugs.

Concrete data shapes & minimal examples
- Prices: pandas.DataFrame (columns: `eeg`, `wholesale`, `community`, `grid`) indexed to timesteps.
- Demand & solar: pandas.Series aligned with the prices index (integers or DatetimeIndex — see below).
- Storage: `Storage(id, c_rate, volume, efficiency)`.
- Example (from tests):

```py
from battery_utility_calculator.battery_utility_calculator import Storage, calculate_storage_worth
worth = calculate_storage_worth(
    baseline_storage=Storage(0,1,0,1),
    storage_to_calculate=Storage(0,1,1,1),
    eeg_prices=pd.Series([0,0,0]),
    wholesale_market_prices=pd.Series([0,0,0]),
    community_market_prices={"aachen": pd.Series([0,0,0])},  # or None to disable
    supplier_prices=pd.Series([0,1,1]),
    solar_generation=pd.Series([0,0,0]),
    demand=pd.Series([1,1,1]),
    solver="appsi_highs",
)
```

Important code patterns & conventions
- Variables map to energy flows with systematic names: `pv_to_home`, `pv_to_eeg`, `pv_to_storage`, `storage_to_home`, `storage_to_eeg`, `wholesale_to_storage`, `supplier_to_home`, etc.
- `storage_use_cases` (default `['eeg','wholesale','community','home']`) is used as a second index for `pv_to_storage` and `storage_level` and drives per-use-case SOC constraints.
- Objective: the model maximizes summed cashflows (community + supplier + EEG + wholesale) in `set_model_objective()`.
- Solver strings: ECC.optimize default is `gurobi`; tests call open solvers like `appsi_highs` (or `highs`). Use the exact tester solver string when running tests.

Integration & dependencies
- Pyomo is used for modeling. Solvers must be available in the runtime environment (`highs`, `appsi_highs`, `gurobi`, etc.). `highspy` / appsi-related solvers are referenced in tests/environments.

Project-specific quirks (important for edits)
- Time index handling: `EnergyCostCalculator.__init__` calls `__check_timeseries_indices__()` which expects identical indices across series and converts `DatetimeIndex` to integer timestep indices while keeping original timestamps in `self.timestamps`. Tests commonly use integer-index Series — adapting indexes is a common source of breakage.
- Exporters and shape assumptions: exporters assume `storage_use_cases` strings exist and access model variables like `pv_to_storage[t,'home']`. If you change variable indices or use-case names, update all exporters and tests.
- `Storage.id` is a plain int in code/tests. Some code paths (or future features) may expect iterable storage identifiers; prefer adding a lightweight compatibility layer to accept either an int or list.

Developer workflows (quick commands)
- Install dev/test deps and local editable package: `pip install -e .[test]`.
- Run full test suite: `pytest -q` (tests live in `tests/`).
- Run a single test quickly: `pytest -q tests/test_BUC.py::test_calculate_storage_worth`.

Small, safe edits an AI agent can start with
- Add a unit test that reproduces exporter shape mismatches (cover `pv_to_storage` indexing and `storage.id` usage).
- Add normalization helpers to accept integer or datetime indices and clearly document the conversion behavior.
- When refactoring model variables or `Storage` shapes, update exporters and tests together.

Where to look first when debugging
- `battery_utility_calculator/energy_costs_calculator.py`: inspect `set_model_variables()`, `set_model_constraints()`, `set_model_objective()` and `optimize()`.
- Use `tests/` as the authoritative behaviour; copy their minimal pandas structures when reproducing problems.

If anything here is unclear or you want a short follow-up (for example: a compatibility helper for `Storage.id`, or one failing exporter unit test), tell me which area and I'll iterate.
<!-- Short, focused guidance for AI coding agents working on this repo -->
