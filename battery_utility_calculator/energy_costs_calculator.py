# SPDX-FileCopyrightText: Christoph Komanns, Florian Maurer, Ralf Schemm
#
# SPDX-License-Identifier: MIT

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyomo.environ as pyo

from battery_utility_calculator import Storage

log = logging.getLogger("battery_utility")
log.setLevel(logging.WARNING)

epsilon = 1e-4
# Nested dict of grid fees for connections between cities
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


class EnergyCostCalculator:
    def __init__(
        self,
        storage: Storage | None,
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
    ):
        """Optimizer for prosumer energy management, calculating minimum costs to cover energy demand.

        Args:
            storage (Storage) | None: The available storage. None if no storage is available.
            demand (pd.Series): Demand in kW per timestep; converted internally to kWh per timestep using hours_per_timestep.
            solar_generation (pd.Series): PV generation in kW per timestep; converted internally to kWh per timestep.
            All model flow variables and storage_level use kWh per timestep.
            supplier_prices (pd.Series): The grid prices for the optimization. Values should be in EUR per kWh. Index has to be pd.DateTimeIndex.
            eeg_prices (pd.Series): The EEG prices for the optimization. Values should be in EUR per kWh. Index has to be pd.DateTimeIndex.
            community_market_prices (dict[str, pd.Series] | None): Community market prices per city. ``None`` or an empty dict disables the community market (flows indexed by ``my_city`` at zero). Otherwise keys must be present in ``grid_fee_between_cities``. Values in EUR per kWh with pd.DateTimeIndex.
            wholesale_market_prices (pd.Series): The wholesale market prices for the optimization. Values should be in EUR per kWh. Index has to be pd.DateTimeIndex.
            hours_per_timestep (int | float): Duration of each timestep in hours (e.g. 0.25 for 15 minutes). Used to convert kW inputs to kWh and to cap storage charge/discharge energy per step.
            storage_use_cases (list[str]): The use cases for energy storage. Allowed values are "eeg", "wholesale", "community", "home"
            allow_community_to_storage (bool): Import from community market into the home SOC bucket (community → storage → home).
            allow_community_market_arbitrage (bool): Import into the community SOC bucket for round-trip arbitrage (requires ``allow_storage_to_community`` to sell back).
            allow_storage_to_community (bool): Discharge community SOC to the community market (PV via storage or arbitrage).
            allow_community_to_home (bool): Direct community import to home without storage.
            allow_pv_to_community (bool): Direct PV export to the community market.
            wholesale_fee (float): Percentage of earned wholesale money that has to be given away (0.0 to 1.0). Default is 0.3.
            my_city (str): City of the prosumer. Must be a key in ``grid_fee_between_cities``. Default is "aachen".
            grid_fee_between_cities (dict[str, dict[str, float]]): Grid fees in EUR/kWh for energy transferred between cities. Default is DEFAULT_GRID_FEE_BETWEEN_CITIES.
            storage_city (str | None): City where the storage is located. Defaults to ``my_city``.
            eeg_eligible (bool): Whether the storage is eligible for EEG. Default is True.
            goal (str): The goal of the optimization. Allowed values are "max_cashflow" and "max_green_energy". Default is "max_cashflow".
            discharge_penalty_per_kwh (float): The discharge penalty per kWh in EUR. Default is 1e-6.
            cycle_cost_per_kwh (float): Optional degradation cost per discharged kWh in EUR.
        """

        if storage:
            self.storage = storage
        else:
            self.storage = Storage(
                id=0, c_rate=1, volume=0, charge_efficiency=1, discharge_efficiency=1
            )

        self.solar_generation = solar_generation
        self.demand = demand
        self.supplier_prices = supplier_prices.copy()
        self.eeg_prices = eeg_prices.copy()
        self.wholesale_market_prices = wholesale_market_prices.copy()
        self.solar_generation = solar_generation.copy()
        self.demand = demand.copy()
        self.hours_per_timestep = hours_per_timestep
        self.charge_efficiency = self.storage.charge_efficiency
        self.discharge_efficiency = self.storage.discharge_efficiency

        self.allow_wholesale_to_home = allow_wholesale_to_home
        self.allow_wholesale_to_storage = allow_wholesale_to_storage
        self.allow_pv_to_wholesale = allow_pv_to_wholesale
        self.allow_storage_to_wholesale = allow_storage_to_wholesale

        self.allow_pv_to_community = allow_pv_to_community
        self.allow_community_to_home = allow_community_to_home
        self.allow_storage_to_community = allow_storage_to_community
        self.allow_community_to_storage = allow_community_to_storage
        self.allow_community_market_arbitrage = allow_community_market_arbitrage
        self.wholesale_fee = wholesale_fee
        self.grid_fee_between_cities = self.__prepare_grid_fee_between_cities__(
            grid_fee_between_cities
        )
        self.my_city = self.__normalize_city__(
            my_city, allowed_cities=set(self.grid_fee_between_cities.keys())
        )
        self.storage_city = self.__normalize_city__(
            storage_city if storage_city is not None else self.my_city,
            allowed_cities=set(self.grid_fee_between_cities.keys()),
        )
        self.community_market_prices, self.community_cities = (
            self.__prepare_community_market_prices__(community_market_prices)
        )
        self.has_community_market = bool(self.community_market_prices)
        self.allow_supplier_to_storage = True
        if self.storage_city != self.my_city:
            self.allow_supplier_to_storage = False
            log.warning(
                "Disabled supplier_to_storage use-case because storage_city '%s' differs from my_city '%s'.",
                self.storage_city,
                self.my_city,
            )
        self.eeg_eligible = eeg_eligible
        self.goal = goal
        self.discharge_penalty_per_kwh = discharge_penalty_per_kwh
        self.cycle_cost_per_kwh = cycle_cost_per_kwh

        self.storage_use_cases = storage_use_cases

        if not self.eeg_eligible:
            self.eeg_prices = self.eeg_prices * 0

        self.__check_prepare_timeseries_indices__()
        self.timesteps = list(range(len(self.demand)))

        self.is_optimized = False

        self.model = pyo.ConcreteModel()
        self.__set_model_variables__()
        self.set_model_constraints()

        if self.goal == "max_cashflow":
            self.__set_max_cashflow_objective__()
        elif self.goal == "max_green_energy":
            self.set_max_green_energy_objective()

    def __normalize_city__(
        self, city: str, allowed_cities: set[str] | None = None
    ) -> str:
        if not isinstance(city, str):
            raise TypeError("city names have to be strings.")

        normalized = city.strip().lower()

        if allowed_cities is None:
            allowed_cities = set(DEFAULT_GRID_FEE_BETWEEN_CITIES.keys())

        if normalized not in allowed_cities:
            msg = f"Unknown city '{city}'. Allowed values: {sorted(allowed_cities)}"
            raise ValueError(msg)

        return normalized

    def __prepare_community_market_prices__(
        self, community_market_prices: dict[str, pd.Series] | None
    ) -> tuple[dict[str, pd.Series], list[str]]:
        if community_market_prices is None:
            return {}, [self.my_city]
        if not isinstance(community_market_prices, dict):
            raise TypeError(
                "community_market_prices has to be a dict[str, pd.Series] or None."
            )
        if not community_market_prices:
            return {}, [self.my_city]

        allowed_cities = set(self.grid_fee_between_cities.keys())
        prepared: dict[str, pd.Series] = {}

        for city, series in community_market_prices.items():
            normalized_city = self.__normalize_city__(
                city, allowed_cities=allowed_cities
            )
            if normalized_city in prepared:
                raise ValueError(
                    f"Duplicate community market city '{normalized_city}' in "
                    "community_market_prices."
                )
            if not isinstance(series, pd.Series):
                raise TypeError(
                    f"community_market_prices['{city}'] has to be a pd.Series."
                )
            prepared[normalized_city] = series.copy()

        return prepared, sorted(prepared.keys())

    def __prepare_grid_fee_between_cities__(
        self,
        grid_fee_between_cities: dict[str, dict[str, float]] | None,
    ) -> dict[str, dict[str, float]]:
        # this prevents modifying the default grid fees by reference
        normalized_fees = {
            from_city: row.copy()
            for from_city, row in DEFAULT_GRID_FEE_BETWEEN_CITIES.items()
        }

        if grid_fee_between_cities is None:
            return normalized_fees
        if not isinstance(grid_fee_between_cities, dict):
            raise TypeError(
                "grid_fee_between_cities has to be a dict[str, dict[str, float]] or None."
            )

        allowed_cities = set(normalized_fees.keys())
        for from_city, to_fees in grid_fee_between_cities.items():
            normalized_from = self.__normalize_city__(
                from_city, allowed_cities=allowed_cities
            )
            if not isinstance(to_fees, dict):
                raise TypeError(
                    f"grid_fee_between_cities['{from_city}'] has to be a dict[str, float]."
                )
            for to_city, value in to_fees.items():
                normalized_to = self.__normalize_city__(
                    to_city, allowed_cities=allowed_cities
                )
                normalized_fees[normalized_from][normalized_to] = float(value)

        return normalized_fees

    def grid_fee_rate(self, from_city: str, to_city: str) -> float:
        return float(self.grid_fee_between_cities[from_city][to_city])

    def __define_community_flow_var__(self, var_name: str, allow: bool) -> None:
        if self.has_community_market and allow:
            setattr(
                self.model,
                var_name,
                pyo.Var(
                    self.timesteps,
                    self.community_cities,
                    domain=pyo.NonNegativeReals,
                ),
            )
        else:
            setattr(
                self.model,
                var_name,
                pyo.Var(
                    self.timesteps,
                    self.community_cities,
                    domain=pyo.NonNegativeReals,
                    bounds=(0, 0),
                ),
            )

    def __community_flow_value__(
        self,
        var_name: str,
        timestep: int,
        *,
        city: str,
        use_values: bool = False,
    ):
        return self.__get_value__(
            getattr(self.model, var_name)[timestep, city], use_values
        )

    def __community_flow_timestep_total__(
        self, var_name: str, timestep: int, use_values: bool = False
    ):
        return sum(
            self.__community_flow_value__(
                var_name, timestep, city=city, use_values=use_values
            )
            for city in self.community_cities
        )

    def __community_to_storage_value__(
        self,
        timestep: int,
        city: str,
        *,
        use_values: bool = False,
    ):
        return self.__community_flow_value__(
            "community_to_storage_for_home",
            timestep,
            city=city,
            use_values=use_values,
        ) + self.__community_flow_value__(
            "community_to_storage_for_community",
            timestep,
            city=city,
            use_values=use_values,
        )

    def __community_to_storage_timestep_total__(
        self, timestep: int, *, use_values: bool = False
    ):
        return sum(
            self.__community_to_storage_value__(
                timestep, city=city, use_values=use_values
            )
            for city in self.community_cities
        )

    def __check_prepare_timeseries_indices__(self) -> None:
        """Check if all timeseries indices are valid."""
        attrs = [
            "solar_generation",
            "supplier_prices",
            "eeg_prices",
            "wholesale_market_prices",
        ]

        if not isinstance(self.demand.index, pd.DatetimeIndex):
            msg = "Index of demand timeseries has to be pd.DateTimeIndex!"
            raise TypeError(msg)
        else:
            ref_index = self.demand.index.copy()

        self.original_index = ref_index
        new_index = pd.RangeIndex(len(ref_index))

        self.demand.index = new_index
        # ensure all indices match the reference
        for name in attrs:
            series = getattr(self, name).copy()

            if not isinstance(series.index, pd.DatetimeIndex):
                msg = f"Index of {name} has to be pd.DateTimeIndex!"
                raise TypeError(msg)

            if not series.index.equals(ref_index):
                raise ValueError(
                    f"All timeseries indices must be identical. Index of {name} does not equal index of demand."
                )

            series.index = new_index
            setattr(self, name, series)

        if self.has_community_market:
            prepared_community: dict[str, pd.Series] = {}
            for city in self.community_cities:
                series = self.community_market_prices[city].copy()
                label = f"community_market_prices['{city}']"
                if not isinstance(series.index, pd.DatetimeIndex):
                    msg = f"Index of {label} has to be pd.DateTimeIndex!"
                    raise TypeError(msg)
                if not series.index.equals(ref_index):
                    raise ValueError(
                        f"All timeseries indices must be identical. Index of {label} "
                        "does not equal index of demand."
                    )
                series.index = new_index
                prepared_community[city] = series
            self.community_market_prices = prepared_community

        h = self.hours_per_timestep
        self.demand = self.demand * h
        self.solar_generation = self.solar_generation * h

    def __set_model_variables__(self):
        log.info("Setting up model variables...")

        # storage level restrction
        self.model.storage_level = pyo.Var(
            self.timesteps,
            self.storage_use_cases,
            domain=pyo.NonNegativeReals,
        )

        #############################
        ##      PV SYSTEM VARS      #
        #############################
        # Selling PV for EEG
        self.model.pv_to_eeg = pyo.Var(self.timesteps, domain=pyo.NonNegativeReals)

        # Selling PV on wholesale
        if self.allow_pv_to_wholesale:
            self.model.pv_to_wholesale = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals
            )
        else:
            self.model.pv_to_wholesale = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals, bounds=(0, 0)
            )

        # Selling PV on community market
        self.__define_community_flow_var__(
            "pv_to_community", self.allow_pv_to_community
        )

        # PV energy to storage
        self.model.pv_to_storage = pyo.Var(
            self.timesteps,
            self.storage_use_cases,
            domain=pyo.NonNegativeReals,
        )

        # using PV energy at home
        self.model.pv_to_home = pyo.Var(self.timesteps, domain=pyo.NonNegativeReals)

        # selling from storage for EEG
        if "eeg" in self.storage_use_cases:
            self.model.storage_to_eeg = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals
            )
        else:
            self.model.storage_to_eeg = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals, bounds=(0, 0)
            )

        # selling from storage on wholesale
        if self.allow_storage_to_wholesale:
            self.model.storage_to_wholesale = pyo.Var(
                self.timesteps,
                domain=pyo.NonNegativeReals,
            )
        else:
            self.model.storage_to_wholesale = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals, bounds=(0, 0)
            )

        # selling from storage on community
        self.__define_community_flow_var__(
            "storage_to_community", self.allow_storage_to_community
        )

        # using energy from storage at home
        if "home" in self.storage_use_cases:
            self.model.storage_to_home = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals
            )
        else:
            self.model.storage_to_home = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals, bounds=(0, 0)
            )

        # charging storage from wholesale market
        if self.allow_wholesale_to_storage:
            self.model.wholesale_to_storage = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals
            )
        else:
            self.model.wholesale_to_storage = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals, bounds=(0, 0)
            )

        # charging storage from community market (split by SOC bucket)
        self.__define_community_flow_var__(
            "community_to_storage_for_home",
            self.has_community_market
            and self.allow_community_to_storage
            and "home" in self.storage_use_cases,
        )
        self.__define_community_flow_var__(
            "community_to_storage_for_community",
            self.has_community_market
            and self.allow_community_market_arbitrage
            and "community" in self.storage_use_cases,
        )

        # buying energy from community market
        self.__define_community_flow_var__(
            "community_to_home", self.allow_community_to_home
        )

        # charging storage from supplier
        if self.allow_supplier_to_storage:
            self.model.supplier_to_storage = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals
            )
        else:
            self.model.supplier_to_storage = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals, bounds=(0, 0)
            )

        # buying energy from supplier
        self.model.supplier_to_home = pyo.Var(
            self.timesteps, domain=pyo.NonNegativeReals
        )

        log.info("Model variables set up successfully.")

    def set_model_constraints(self):
        log.info("Setting up model constraints...")

        # consumption must equal supply (from PV system, supplier, community market, PV system)
        def restrict_demand(model, timestep):
            community_to_home = sum(
                model.community_to_home[timestep, city]
                for city in self.community_cities
            )

            return (
                model.storage_to_home[timestep]
                + model.pv_to_home[timestep]
                + model.supplier_to_home[timestep]
                + community_to_home
                == self.demand.iloc[timestep]
            )

        self.model.demand_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_demand
        )

        # use of PV system must be smaller or equal than PV system generation
        def restrict_solar_gen(model, timestep):
            pv_to_community = sum(
                model.pv_to_community[timestep, city] for city in self.community_cities
            )

            return (
                sum(
                    model.pv_to_storage[timestep, use] for use in self.storage_use_cases
                )
                + model.pv_to_eeg[timestep]
                + model.pv_to_home[timestep]
                + model.pv_to_wholesale[timestep]
                + pv_to_community
                <= self.solar_generation.iloc[timestep]
            )

        self.model.solar_gen_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_solar_gen
        )

        # energy flow TO storage (kWh per timestep) must be <= max charge energy per step
        def restrict_storage_charge(model, timestep):
            community_to_storage = sum(
                model.community_to_storage_for_home[timestep, city]
                + model.community_to_storage_for_community[timestep, city]
                for city in self.community_cities
            )

            return (
                sum(
                    model.pv_to_storage[timestep, use] for use in self.storage_use_cases
                )
                + model.wholesale_to_storage[timestep]
                + community_to_storage
                + model.supplier_to_storage[timestep]
                <= self.storage.c_rate * self.storage.volume * self.hours_per_timestep
            )

        self.model.storage_charge_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_storage_charge
        )

        # energy flow FROM storage (kWh per timestep) must be <= max discharge energy per step
        def restrict_storage_discharge(model, timestep):
            storage_to_community = sum(
                model.storage_to_community[timestep, city]
                for city in self.community_cities
            )

            return (
                +model.storage_to_eeg[timestep]
                + model.storage_to_wholesale[timestep]
                + storage_to_community
                + model.storage_to_home[timestep]
                <= self.storage.c_rate * self.storage.volume * self.hours_per_timestep
            )

        self.model.storage_discharge_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_storage_discharge
        )

        # storage level must be smaller than volume
        def restrict_soc_max(model, timestep):
            return (
                sum(
                    model.storage_level[timestep, use] for use in self.storage_use_cases
                )
                <= self.storage.volume
            )

        self.model.storage_level_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_soc_max
        )

        # storage level must be larger than 0
        def restrict_soc_min(model, timestep):
            return (
                sum(
                    model.storage_level[timestep, use] for use in self.storage_use_cases
                )
                >= 0
            )

        self.model.storage_level_non_negative = pyo.Constraint(
            self.timesteps, rule=restrict_soc_min
        )

        # storage level must be 0 at beginning
        def restrict_soc_start(model):
            return (
                sum(model.storage_level[0, use] for use in self.storage_use_cases)
                <= self.storage.volume
                * self.storage.c_rate
                * self.storage.charge_efficiency
            )

        self.model.storage_start_level_restriction = pyo.Constraint(
            rule=restrict_soc_start
        )

        # storage level must be 0 at end
        def restrict_soc_end(model):
            return (
                sum(
                    model.storage_level[self.timesteps[-1], use]
                    for use in self.storage_use_cases
                )
                == 0
            )

        self.model.storage_end_level_restriction = pyo.Constraint(rule=restrict_soc_end)

        if "eeg" in self.storage_use_cases:

            def restrict_soc_eeg(model, timestep):
                if timestep == self.timesteps[0]:
                    return model.storage_level[
                        timestep, "eeg"
                    ] == self.charge_efficiency * model.pv_to_storage[
                        timestep, "eeg"
                    ] - (
                        (1 / self.discharge_efficiency) * model.storage_to_eeg[timestep]
                    )
                else:
                    previous_timestep = timestep - 1
                    return model.storage_level[timestep, "eeg"] == model.storage_level[
                        previous_timestep, "eeg"
                    ] + self.charge_efficiency * model.pv_to_storage[
                        timestep, "eeg"
                    ] - (
                        (1 / self.discharge_efficiency) * model.storage_to_eeg[timestep]
                    )

            self.model.soc_eeg_restriction = pyo.Constraint(
                self.timesteps, rule=restrict_soc_eeg
            )

        if "wholesale" in self.storage_use_cases:

            def restrict_soc_wholesale(model, timestep):
                if timestep == self.timesteps[0]:
                    return model.storage_level[
                        timestep, "wholesale"
                    ] == self.charge_efficiency * model.wholesale_to_storage[
                        timestep
                    ] - (
                        (1 / self.discharge_efficiency)
                        * model.storage_to_wholesale[timestep]
                    )
                else:
                    previous_timestep = timestep - 1
                    return model.storage_level[
                        timestep, "wholesale"
                    ] == model.storage_level[
                        previous_timestep, "wholesale"
                    ] + self.charge_efficiency * model.wholesale_to_storage[
                        timestep
                    ] - (
                        (1 / self.discharge_efficiency)
                        * model.storage_to_wholesale[timestep]
                    )

            self.model.soc_wholesale_restriction = pyo.Constraint(
                self.timesteps, rule=restrict_soc_wholesale
            )

        if "community" in self.storage_use_cases:

            def restrict_soc_community(model, timestep):
                community_charge = (
                    sum(
                        model.community_to_storage_for_community[timestep, city]
                        for city in self.community_cities
                    )
                    + model.pv_to_storage[timestep, "community"]
                )
                community_discharge = sum(
                    model.storage_to_community[timestep, city]
                    for city in self.community_cities
                )
                if timestep == self.timesteps[0]:
                    return model.storage_level[
                        timestep, "community"
                    ] == self.charge_efficiency * community_charge - (
                        (1 / self.discharge_efficiency) * community_discharge
                    )
                else:
                    previous_timestep = timestep - 1
                    return model.storage_level[
                        timestep, "community"
                    ] == model.storage_level[
                        previous_timestep, "community"
                    ] + self.charge_efficiency * community_charge - (
                        (1 / self.discharge_efficiency) * community_discharge
                    )

            self.model.soc_community_restriction = pyo.Constraint(
                self.timesteps, rule=restrict_soc_community
            )

        if "home" in self.storage_use_cases:

            def restrict_soc_home(model, timestep):
                community_to_storage_for_home = sum(
                    model.community_to_storage_for_home[timestep, city]
                    for city in self.community_cities
                )
                if timestep == self.timesteps[0]:
                    return model.storage_level[
                        timestep, "home"
                    ] == self.charge_efficiency * model.supplier_to_storage[
                        timestep
                    ] + self.charge_efficiency * model.pv_to_storage[
                        timestep, "home"
                    ] + self.charge_efficiency * community_to_storage_for_home - (
                        (1 / self.discharge_efficiency)
                        * model.storage_to_home[timestep]
                    )
                else:
                    previous_timestep = timestep - 1
                    return model.storage_level[timestep, "home"] == model.storage_level[
                        previous_timestep, "home"
                    ] + self.charge_efficiency * model.supplier_to_storage[
                        timestep
                    ] + self.charge_efficiency * model.pv_to_storage[
                        timestep, "home"
                    ] + self.charge_efficiency * community_to_storage_for_home - (
                        (1 / self.discharge_efficiency)
                        * model.storage_to_home[timestep]
                    )

            self.model.soc_home_restriction = pyo.Constraint(
                self.timesteps, rule=restrict_soc_home
            )

        log.info("Model constraints set up successfully.")

    def __get_value__(self, var, use_values):
        """Helper to conditionally apply .value to a Pyomo variable."""
        return var.value if use_values else var

    def __community_flow_series__(self, var_name: str) -> tuple[dict[str, list], list]:
        """Per-city and aggregated timestep series for a community flow variable."""
        model_var = getattr(self.model, var_name)
        by_city = {
            city: [
                float(model_var[timestep, city].value or 0.0)
                for timestep in self.timesteps
            ]
            for city in self.community_market_prices
        }
        aggregate = [
            sum(
                float(model_var[timestep, city].value or 0.0)
                for city in self.community_cities
            )
            for timestep in self.timesteps
        ]
        return by_city, aggregate

    def __set_max_cashflow_objective__(self):
        log.info("Setting up model objective...")

        cashflows = self.calculate_cashflows(use_values=False)
        community_cf = cashflows["community"]
        supplier_cf = cashflows["supplier"]
        eeg_cf = cashflows["eeg"]
        wholesale_cf = cashflows["wholesale"]
        grid_fee_cf = cashflows["grid_fees"]
        discharge_penalty = self.calculate_discharge_penalty(use_values=False)
        cycle_cost_penalty = self.calculate_cycle_cost_penalty(use_values=False)

        # maximize sum of cashflows
        self.model.objective = pyo.Objective(
            expr=community_cf
            + supplier_cf
            + eeg_cf
            + wholesale_cf
            + grid_fee_cf
            - discharge_penalty
            - cycle_cost_penalty,
            sense=pyo.maximize,
        )

        log.info("Model objective set up successfully.")

    def calculate_cashflows(self, use_values=False):
        # community market cashflow
        community_cf = self.calculate_community_cashflow(use_values=use_values)

        # supplier cashflow
        supplier_cf = self.calculate_supplier_cashflow(use_values=use_values)

        # EEG cashflow
        eeg_cf = self.calculate_eeg_cashflow(use_values=use_values)

        # wholesale cashflow
        wholesale_cf = self.calculate_wholesale_cashflow(use_values=use_values)
        grid_fee_cf = self.calculate_grid_fee_cashflow(use_values=use_values)

        return {
            "community": community_cf,
            "supplier": supplier_cf,
            "eeg": eeg_cf,
            "wholesale": wholesale_cf,
            "grid_fees": grid_fee_cf,
        }

    def calculate_community_cashflow(self, use_values=False):
        if not self.has_community_market:
            return 0.0

        return sum(
            (
                self.__community_flow_value__(
                    var_name="storage_to_community",
                    timestep=timestep,
                    city=city,
                    use_values=use_values,
                )
                - self.__community_to_storage_value__(
                    timestep,
                    city=city,
                    use_values=use_values,
                )
                - self.__community_flow_value__(
                    var_name="community_to_home",
                    timestep=timestep,
                    city=city,
                    use_values=use_values,
                )
            )
            * self.community_market_prices[city].loc[timestep]
            + self.__community_flow_value__(
                var_name="pv_to_community",
                timestep=timestep,
                city=city,
                use_values=use_values,
            )
            * self.community_market_prices[city].loc[timestep]
            for timestep in self.timesteps
            for city in self.community_market_prices
        )

    def calculate_supplier_cashflow(self, use_values=False):
        return -(1 - epsilon) * sum(
            self.__get_value__(self.model.supplier_to_storage[timestep], use_values)
            * self.supplier_prices.loc[timestep]
            for timestep in self.timesteps
        ) - sum(
            self.__get_value__(self.model.supplier_to_home[timestep], use_values)
            * self.supplier_prices.loc[timestep]
            for timestep in self.timesteps
        )

    def calculate_eeg_cashflow(self, use_values=False):
        return (1 - epsilon) * sum(
            self.__get_value__(self.model.storage_to_eeg[timestep], use_values)
            * self.eeg_prices.loc[timestep]
            for timestep in self.timesteps
        ) + sum(
            self.__get_value__(self.model.pv_to_eeg[timestep], use_values)
            * self.eeg_prices.loc[timestep]
            for timestep in self.timesteps
        )

    def calculate_wholesale_cashflow(self, use_values=False):
        wholesale_earnings = sum(
            self.__get_value__(self.model.storage_to_wholesale[timestep], use_values)
            * self.wholesale_market_prices.loc[timestep]
            for timestep in self.timesteps
        )
        wholesale_earnings += sum(
            self.__get_value__(self.model.pv_to_wholesale[timestep], use_values)
            * self.wholesale_market_prices.loc[timestep]
            for timestep in self.timesteps
        )
        wholesale_costs = sum(
            self.__get_value__(self.model.wholesale_to_storage[timestep], use_values)
            * self.wholesale_market_prices.loc[timestep]
            for timestep in self.timesteps
        )
        return (wholesale_earnings - wholesale_costs) * (1 - self.wholesale_fee)

    def calculate_grid_fee_cashflow(self, use_values=False):
        pv_charge = sum(
            self.__get_value__(self.model.pv_to_storage[timestep, use], use_values)
            for timestep in self.timesteps
            for use in self.storage_use_cases
            if use != "wholesale"
        )
        supplier_charge = sum(
            self.__get_value__(self.model.supplier_to_storage[timestep], use_values)
            for timestep in self.timesteps
        )
        storage_to_home = sum(
            self.__get_value__(self.model.storage_to_home[timestep], use_values)
            for timestep in self.timesteps
        )

        fee_energy = (
            self.grid_fee_rate(self.my_city, self.storage_city) * pv_charge
            + self.grid_fee_rate(self.my_city, self.storage_city) * supplier_charge
            + self.grid_fee_rate(self.storage_city, self.my_city) * storage_to_home
        )

        for city in self.community_market_prices:
            community_to_storage = sum(
                self.__community_to_storage_value__(
                    timestep, city=city, use_values=use_values
                )
                for timestep in self.timesteps
            )
            storage_to_community = sum(
                self.__community_flow_value__(
                    "storage_to_community",
                    timestep,
                    city=city,
                    use_values=use_values,
                )
                for timestep in self.timesteps
            )
            community_to_home = sum(
                self.__community_flow_value__(
                    "community_to_home",
                    timestep,
                    city=city,
                    use_values=use_values,
                )
                for timestep in self.timesteps
            )
            pv_to_community = sum(
                self.__community_flow_value__(
                    "pv_to_community",
                    timestep,
                    city=city,
                    use_values=use_values,
                )
                for timestep in self.timesteps
            )
            fee_energy = (
                fee_energy
                + self.grid_fee_rate(city, self.storage_city) * community_to_storage
                + self.grid_fee_rate(self.storage_city, city) * storage_to_community
                + self.grid_fee_rate(city, self.my_city) * community_to_home
                + self.grid_fee_rate(self.my_city, city) * pv_to_community
            )

        if isinstance(fee_energy, (int, float)) and fee_energy <= 0:
            return 0.0

        return -fee_energy

    def calculate_discharge_penalty(self, use_values=False):
        total_discharge = sum(
            self.__get_value__(self.model.storage_to_eeg[timestep], use_values)
            + self.__get_value__(self.model.storage_to_wholesale[timestep], use_values)
            + self.__community_flow_timestep_total__(
                "storage_to_community", timestep, use_values=use_values
            )
            + self.__get_value__(self.model.storage_to_home[timestep], use_values)
            for timestep in self.timesteps
        )
        return self.discharge_penalty_per_kwh * total_discharge

    def calculate_cycle_cost_penalty(self, use_values=False):
        total_discharge = sum(
            self.__get_value__(self.model.storage_to_eeg[timestep], use_values)
            + self.__get_value__(self.model.storage_to_wholesale[timestep], use_values)
            + self.__community_flow_timestep_total__(
                "storage_to_community", timestep, use_values=use_values
            )
            + self.__get_value__(self.model.storage_to_home[timestep], use_values)
            for timestep in self.timesteps
        )
        return self.cycle_cost_per_kwh * total_discharge

    def set_max_green_energy_objective(self):
        """Set objective to maximize PV self-consumption.

        This objective favors using PV generation for the household either
        immediately (`pv_to_home`) or by storing PV specifically intended
        for home use (`pv_to_storage[..., 'home']`). If the 'home'
        use-case is not present the objective will only maximize
        `pv_to_home`.
        """
        log.info("Setting up green-energy objective (maximize PV self-consumption)...")

        # maximize direct PV consumption (kWh per timestep)
        expr = sum(self.model.pv_to_home[timestep] for timestep in self.timesteps)

        # include PV charged to storage for later home use when 'home' use-case exists
        if "home" in self.storage_use_cases:
            expr = expr + sum(
                self.model.pv_to_storage[timestep, "home"]
                for timestep in self.timesteps
            )

        cashflows = self.calculate_cashflows()
        eeg_cf = cashflows["eeg"]
        community_cf = cashflows["community"]
        supplier_cf = cashflows["supplier"]
        wholesale_cf = cashflows["wholesale"]
        grid_fee_cf = cashflows["grid_fees"]
        total_cashflow = (
            eeg_cf + supplier_cf + community_cf + wholesale_cf + grid_fee_cf
        )

        expr = expr + epsilon * total_cashflow

        self.model.objective = pyo.Objective(expr=expr, sense=pyo.maximize)

        log.info("Green-energy objective set up successfully.")

    def calculate_costs(self) -> float:
        """Calculate monetary cashflow from the optimized model and return it.

        Returns the same cashflow expression that `set_max_cashflow_objective`
        maximizes (including discharge penalty for ``goal="max_cashflow"``).
        Requires the model to be optimized first (raises ValueError otherwise).
        """
        if not self.is_optimized:
            raise ValueError("Model not optimized yet - run optimize() first!")

        cashflows = self.calculate_cashflows(use_values=True)

        total = (
            cashflows["community"]
            + cashflows["supplier"]
            + cashflows["eeg"]
            + cashflows["wholesale"]
            + cashflows["grid_fees"]
        )
        if self.goal == "max_cashflow":
            total -= self.calculate_discharge_penalty(use_values=True)
            total -= self.calculate_cycle_cost_penalty(use_values=True)
        return float(total)

    # ------------------------------------------------------------------
    def get_cashflows(self) -> dict:
        """Return the individual cashflow components from the optimized model.

        The dictionary contains five keys: ``"community"``, ``"supplier"``,
        ``"eeg"``, ``"wholesale"`` and ``"grid_fees"``. Values are floats
        representing EUR cashflow components. A negative value means a cost,
        a positive value means a revenue.

        Raises
        ------
        ValueError
            If called before ``optimize()`` has been run.
        """
        if not self.is_optimized:
            raise ValueError("Model not optimized yet - run optimize() first!")

        # calculate_cashflows with ``use_values=True`` gives numeric results
        raw = self.calculate_cashflows(use_values=True)
        # ensure all entries are floats
        return {k: float(v) for k, v in raw.items()}

    def optimize(self, solver: str = "gurobi"):
        optimizer = pyo.SolverFactory(
            solver,
        )

        optimizer.solve(self.model, tee=False)

        self.is_optimized = True

        return self.model.objective()

    def get_price_df(self) -> pd.DataFrame:
        price_columns = {
            "supplier_prices": self.supplier_prices,
            "eeg_prices": self.eeg_prices,
            "wholesale_market_prices": self.wholesale_market_prices,
        }
        if self.has_community_market:
            for city, series in self.community_market_prices.items():
                price_columns[f"community_market_prices_{city}"] = series

        price_df = pd.DataFrame(price_columns)
        price_df.index = self.original_index.copy()

        return price_df

    def get_energy_flows(self) -> pd.DataFrame:
        energy_flows = pd.DataFrame(index=self.timesteps)

        if "wholesale" in self.storage_use_cases:
            energy_flows["storage_to_wholesale"] = [
                self.model.storage_to_wholesale[timestep].value
                for timestep in self.timesteps
            ]
            energy_flows["wholesale_to_storage"] = [
                self.model.wholesale_to_storage[timestep].value
                for timestep in self.timesteps
            ]
            energy_flows["pv_to_wholesale"] = [
                self.model.pv_to_wholesale[timestep].value
                for timestep in self.timesteps
            ]
            energy_flows["pv_to_storage_for_wholesale"] = [
                self.model.pv_to_storage[timestep, "wholesale"].value
                for timestep in self.timesteps
            ]
        else:
            energy_flows["storage_to_wholesale"] = 0
            energy_flows["wholesale_to_storage"] = 0
            energy_flows["pv_to_wholesale"] = 0
            energy_flows["pv_to_storage_for_wholesale"] = 0

        if "eeg" in self.storage_use_cases:
            energy_flows["pv_to_eeg"] = [
                self.model.pv_to_eeg[timestep].value for timestep in self.timesteps
            ]
            energy_flows["storage_to_eeg"] = [
                self.model.storage_to_eeg[timestep].value for timestep in self.timesteps
            ]
            energy_flows["pv_to_storage_for_eeg"] = [
                self.model.pv_to_storage[timestep, "eeg"].value
                for timestep in self.timesteps
            ]
        else:
            energy_flows["pv_to_eeg"] = 0
            energy_flows["storage_to_eeg"] = 0
            energy_flows["pv_to_storage_for_eeg"] = 0

        if "community" in self.storage_use_cases:
            for var_name in (
                "storage_to_community",
                "pv_to_community",
                "community_to_storage_for_home",
                "community_to_storage_for_community",
            ):
                by_city, aggregate = self.__community_flow_series__(var_name)
                for city, values in by_city.items():
                    energy_flows[f"{var_name}_{city}"] = values
                energy_flows[var_name] = aggregate
            _, home_agg = self.__community_flow_series__(
                "community_to_storage_for_home"
            )
            _, community_agg = self.__community_flow_series__(
                "community_to_storage_for_community"
            )
            energy_flows["community_to_storage"] = [
                h + c for h, c in zip(home_agg, community_agg, strict=True)
            ]
            for city in self.community_market_prices:
                energy_flows[f"community_to_storage_{city}"] = [
                    float(
                        self.model.community_to_storage_for_home[timestep, city].value
                        or 0.0
                    )
                    + float(
                        self.model.community_to_storage_for_community[
                            timestep, city
                        ].value
                        or 0.0
                    )
                    for timestep in self.timesteps
                ]
            energy_flows["pv_to_storage_for_community"] = [
                self.model.pv_to_storage[timestep, "community"].value
                for timestep in self.timesteps
            ]
        else:
            energy_flows["storage_to_community"] = 0
            energy_flows["pv_to_community"] = 0
            energy_flows["community_to_storage"] = 0
            energy_flows["community_to_storage_for_home"] = 0
            energy_flows["community_to_storage_for_community"] = 0
            energy_flows["pv_to_storage_for_community"] = 0

        _, community_to_home = self.__community_flow_series__("community_to_home")
        energy_flows["community_to_home"] = community_to_home
        for city in self.community_market_prices:
            energy_flows[f"community_to_home_{city}"] = [
                float(self.model.community_to_home[timestep, city].value or 0.0)
                for timestep in self.timesteps
            ]

        if "home" in self.storage_use_cases:
            energy_flows["storage_to_home"] = [
                self.model.storage_to_home[timestep].value
                for timestep in self.timesteps
            ]
            energy_flows["supplier_to_home"] = [
                self.model.supplier_to_home[timestep].value
                for timestep in self.timesteps
            ]
            energy_flows["supplier_to_storage"] = [
                self.model.supplier_to_storage[timestep].value
                for timestep in self.timesteps
            ]
            energy_flows["pv_to_storage_for_home"] = [
                self.model.pv_to_storage[timestep, "home"].value
                for timestep in self.timesteps
            ]
            energy_flows["pv_to_home"] = [
                self.model.pv_to_home[timestep].value for timestep in self.timesteps
            ]
        else:
            energy_flows["storage_to_home"] = 0
            energy_flows["supplier_to_home"] = 0
            energy_flows["supplier_to_storage"] = 0
            energy_flows["pv_to_storage_for_home"] = 0
            energy_flows["pv_to_home"] = 0

        energy_flows["demand"] = self.demand.values

        energy_flows["index"] = self.original_index.copy()
        energy_flows.set_index("index", inplace=True)
        energy_flows.index.name = None

        return energy_flows

    def get_demand_coverage_timeseries_df(self) -> pd.DataFrame:
        demand_coverage = pd.DataFrame(index=self.timesteps)

        demand_coverage["demand"] = self.demand.copy()
        demand_coverage["from_grid"] = [
            self.model.supplier_to_home[timestep].value for timestep in self.timesteps
        ]
        demand_coverage["from_pv"] = [
            self.model.pv_to_home[timestep].value for timestep in self.timesteps
        ]
        demand_coverage["from_storage"] = [
            self.model.storage_to_home[timestep].value for timestep in self.timesteps
        ]
        _, demand_coverage["from_community"] = self.__community_flow_series__(
            "community_to_home"
        )
        for city in self.community_market_prices:
            demand_coverage[f"from_community_{city}"] = [
                float(self.model.community_to_home[timestep, city].value or 0.0)
                for timestep in self.timesteps
            ]

        demand_coverage["index"] = self.original_index.copy()
        demand_coverage.set_index("index", inplace=True)
        demand_coverage.index.name = None

        return demand_coverage

    def get_solar_generation_timeseries_df(self) -> pd.DataFrame:
        pv_usage = pd.DataFrame(index=self.timesteps)

        pv_usage["generation"] = self.solar_generation.loc[self.timesteps]

        pv_usage["to_home"] = [self.model.pv_to_home[t].value for t in self.timesteps]
        pv_usage["to_eeg"] = [self.model.pv_to_eeg[t].value for t in self.timesteps]
        _, pv_usage["to_community"] = self.__community_flow_series__("pv_to_community")
        for city in self.community_market_prices:
            pv_usage[f"to_community_{city}"] = [
                float(self.model.pv_to_community[t, city].value or 0.0)
                for t in self.timesteps
            ]
        pv_usage["to_wholesale"] = [
            self.model.pv_to_wholesale[t].value for t in self.timesteps
        ]
        pv_usage["to_storage_home"] = [
            self.model.pv_to_storage[t, "home"].value for t in self.timesteps
        ]
        pv_usage["to_storage_eeg"] = [
            self.model.pv_to_storage[t, "eeg"].value for t in self.timesteps
        ]
        pv_usage["to_storage_wholesale"] = [
            self.model.pv_to_storage[t, "wholesale"].value for t in self.timesteps
        ]
        pv_usage["to_storage_community"] = [
            self.model.pv_to_storage[t, "community"].value for t in self.timesteps
        ]

        pv_usage["index"] = self.original_index.copy()
        pv_usage.set_index("index", inplace=True)
        pv_usage.index.name = None

        return pv_usage

    def get_storage_soc_timeseries_df(self) -> pd.DataFrame:
        storage_usage = pd.DataFrame(index=self.timesteps)

        for use_case in self.storage_use_cases:
            storage_usage[f"soc_{use_case}"] = [
                self.model.storage_level[t, use_case].value for t in self.timesteps
            ]

        storage_usage["index"] = self.original_index.copy()
        storage_usage.set_index("index", inplace=True)
        storage_usage.index.name = None

        return storage_usage

    def get_storage_charge_timeseries_df(self) -> pd.DataFrame:
        charge_df = self.get_storage_soc_timeseries_df()
        charge_df.loc[
            charge_df.index[0] - pd.Timedelta(hours=self.hours_per_timestep), :
        ] = 0
        charge_df = charge_df.sort_index().diff().dropna()

        return charge_df

    def get_storage_usage_kpis(self) -> dict:
        """Return compact storage usage KPIs aggregated over the full horizon."""
        if not self.is_optimized:
            raise ValueError("Model not optimized yet - run optimize() first!")

        energy_flows = self.get_energy_flows()
        h = float(self.hours_per_timestep)

        def col_energy_sum(column: str) -> float:
            if column not in energy_flows.columns:
                return 0.0
            return float(energy_flows[column].sum())

        charged_by_source_kwh = {
            "pv": (
                col_energy_sum("pv_to_storage_for_home")
                + col_energy_sum("pv_to_storage_for_eeg")
                + col_energy_sum("pv_to_storage_for_wholesale")
                + col_energy_sum("pv_to_storage_for_community")
            ),
            "supplier": col_energy_sum("supplier_to_storage"),
            "wholesale": col_energy_sum("wholesale_to_storage"),
            "community": col_energy_sum("community_to_storage"),
        }
        discharged_by_sink_kwh = {
            "home": col_energy_sum("storage_to_home"),
            "eeg": col_energy_sum("storage_to_eeg"),
            "wholesale": col_energy_sum("storage_to_wholesale"),
            "community": col_energy_sum("storage_to_community"),
        }

        charged_kwh_total = float(sum(charged_by_source_kwh.values()))
        discharged_kwh_total = float(sum(discharged_by_sink_kwh.values()))

        max_discharge_kwh = (
            float(self.storage.c_rate)
            * float(self.storage.volume)
            * h
            * len(self.timesteps)
        )

        full_cycles_equivalent = (
            discharged_kwh_total / float(self.storage.volume)
            if self.storage.volume > 0
            else 0.0
        )
        utilization_ratio = (
            discharged_kwh_total / max_discharge_kwh if max_discharge_kwh > 0 else 0.0
        )
        roundtrip_indicator = (
            discharged_kwh_total / charged_kwh_total if charged_kwh_total > 0 else 0.0
        )

        return {
            "charged_kwh_total": charged_kwh_total,
            "discharged_kwh_total": discharged_kwh_total,
            "charged_by_source_kwh": charged_by_source_kwh,
            "discharged_by_sink_kwh": discharged_by_sink_kwh,
            "full_cycles_equivalent": float(full_cycles_equivalent),
            "utilization_ratio": float(utilization_ratio),
            "roundtrip_indicator": float(roundtrip_indicator),
        }

    def plot_storage_usage_summary(self, show: bool = True) -> go.Figure:
        """Plot compact storage usage summary without a time axis."""
        kpis = self.get_storage_usage_kpis()

        rows = []
        for source, value in kpis["charged_by_source_kwh"].items():
            if value > 0:
                rows.append(
                    {
                        "Bucket": f"Charge from {source}",
                        "Energy (kWh)": value,
                        "Flow": "Charge",
                    }
                )
        for sink, value in kpis["discharged_by_sink_kwh"].items():
            if value > 0:
                rows.append(
                    {
                        "Bucket": f"Discharge to {sink}",
                        "Energy (kWh)": value,
                        "Flow": "Discharge",
                    }
                )

        if not rows:
            rows = [
                {
                    "Bucket": "No storage activity",
                    "Energy (kWh)": 0.0,
                    "Flow": "Charge",
                }
            ]

        summary_df = pd.DataFrame(rows)
        summary_df["Label"] = summary_df["Energy (kWh)"].map(lambda v: f"{v:.2f}")
        fig = px.bar(
            summary_df,
            x="Bucket",
            y="Energy (kWh)",
            color="Flow",
            text="Label",
            barmode="group",
            title="Storage usage summary",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_title=None, yaxis_title="Energy (kWh)")
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1,
            y=1.15,
            showarrow=False,
            align="right",
            text=(
                f"Cycles: {kpis['full_cycles_equivalent']:.2f}<br>"
                f"Utilization: {kpis['utilization_ratio']:.1%}<br>"
                f"Roundtrip indicator: {kpis['roundtrip_indicator']:.2f}"
            ),
        )

        if show:
            fig.show()
        return fig

    def output_results(self, include_cashflows: bool = False):
        """Return convenience timeseries dict with optional cashflow breakdown.

        Parameters
        ----------
        include_cashflows : bool, optional
            If ``True`` the returned dictionary will contain an additional
            key ``"cashflows"`` whose value is the output of
            :meth:`get_cashflows`. Default is ``False``.
        """
        if not self.is_optimized:
            raise ValueError(
                "Model not optimized yet - run class method optimize() first!"
            )

        demand_timeseries = self.get_demand_coverage_timeseries_df()
        pv_timeseries = self.get_solar_generation_timeseries_df()
        storage_timeseries = self.get_storage_soc_timeseries_df()

        results = {
            "demand": demand_timeseries,
            "pv": pv_timeseries,
            "storage": storage_timeseries,
        }

        if include_cashflows:
            results["cashflows"] = self.get_cashflows()

        return results

    def plot_energy_flows(self, show: bool = True) -> go.Figure:
        """Quick energy flow plot using plotly.express."""
        energy_flows = self.get_energy_flows()

        df = energy_flows.reset_index().rename(
            columns={
                energy_flows.index.name or "index": "t",
                "pv_to_eeg": "PV to EEG",
                "pv_to_wholesale": "PV to wholesale",
                "pv_to_community": "PV to community",
                "pv_to_home": "PV to home",
                "storage_to_eeg": "Storage to EEG",
                "storage_to_wholesale": "Storage to wholesale",
                "storage_to_community": "Storage to community",
                "storage_to_home": "Storage to home",
                "wholesale_to_storage": "Wholesale to storage",
                "supplier_to_storage": "Supplier to storage",
                "supplier_to_home": "Supplier to home",
            }
        )

        df = df.drop(columns=[col for col in df.columns if (df[col] == 0).all()])

        long = df.melt(
            id_vars=[df.columns[0]],
            var_name="Flow",
            value_name="kWh",
        )
        fig = px.line(long, x="t", y="kWh", color="Flow", title="Energy flows")

        if show:
            fig.show()
        return fig

    def plot_demand_coverage(self, show: bool = True) -> go.Figure:
        """Quick stacked bar + demand line using plotly.express."""
        demand_df = self.get_demand_coverage_timeseries_df()

        df = demand_df.reset_index().rename(
            columns={
                demand_df.index.name or "index": "t",
                "from_pv": "From PV",
                "from_storage": "From storage",
                "from_community": "From community",
                "from_grid": "From grid",
            }
        )

        # only plot those with usage
        df = df.drop(columns=[col for col in df.columns if (df[col] == 0).all()])

        # long format for stacking
        supply_cols = [
            c
            for c in ["From PV", "From storage", "From community", "From grid"]
            if c in df.columns
        ]
        long = df.melt(
            id_vars=[df.columns[0], "demand"],
            value_vars=supply_cols,
            var_name="source",
            value_name="kWh",
        )
        fig = px.area(long, x="t", y="kWh", color="source", title="Demand coverage")

        if "demand" in df.columns:
            fig.add_scatter(x=df[df.columns[0]], y=df["demand"], name="Demand")

        if show:
            fig.show()
        return fig

    def plot_solar_generation(self, show: bool = True) -> go.Figure:
        """Quick PV generation and usage plot using plotly.express."""
        pv_df = self.get_solar_generation_timeseries_df()

        df = pv_df.reset_index()

        df = df.drop(columns=[col for col in df.columns if (df[col] == 0).all()])

        df = df.rename(
            columns={
                df.index.name or "index": "t",
                "channel": "Channel",
                "to_home": "To home",
                "to_eeg": "To EEG",
                "to_community": "To community",
                "to_wholesale": "To wholesale",
                "to_storage_home": "To storage for home",
                "to_storage_eeg": "To storage for EEG",
                "to_storage_wholesale": "To storage for wholesale",
                "to_storage_community": "To storage for community",
            }
        )

        usage_cols = [c for c in df.columns if c.startswith("To ")]
        long = df.melt(
            id_vars=[df.columns[0]],
            value_vars=usage_cols,
            var_name="Channel",
            value_name="kWh",
        )
        fig = px.area(
            long,
            x="t",
            y="kWh",
            color="Channel",
            title="PV generation & usage",
        )

        if "generation" in df.columns:
            fig.add_scatter(x=df[df.columns[0]], y=df["generation"], name="Generation")

        if show:
            fig.show()
        return fig

    def plot_storage_soc_timeseries(self, show: bool = True) -> go.Figure:
        """Plot storage SOC per use case using plotly.express."""
        storage_df = self.get_storage_soc_timeseries_df()

        df = storage_df.reset_index().rename(
            columns={
                storage_df.index.name or "index": "t",
                "soc_home": "Home",
                "soc_eeg": "EEG",
                "soc_wholesale": "Wholesale",
                "soc_community": "Community",
                "use_case": "Use case",
            }
        )

        df = df.drop(columns=[col for col in df.columns if (df[col] == 0).all()])
        long = df.melt(id_vars=[df.columns[0]], var_name="Use case", value_name="kWh")
        fig = px.line(long, x="t", y="kWh", color="Use case", title="Storage SOC")

        if show:
            fig.show()
        return fig

    def plot_storage_charge_timeseries(self, show: bool = True) -> go.Figure:
        """Plot storage charge per use case using plotly.express."""
        storage_df = self.get_storage_charge_timeseries_df()

        df = storage_df.reset_index().rename(
            columns={
                storage_df.index.name or "index": "t",
                "soc_home": "Home",
                "soc_eeg": "EEG",
                "soc_wholesale": "Wholesale",
                "soc_community": "Community",
                "use_case": "Use case",
            }
        )

        df = df.drop(columns=[col for col in df.columns if (df[col] == 0).all()])
        long = df.melt(id_vars=[df.columns[0]], var_name="Use case", value_name="kWh")
        fig = px.line(long, x="t", y="kW", color="Use case", title="Charge / Discharge")

        if show:
            fig.show()
        return fig

    def plot_prices(self, show: bool = True):
        """Plot prices using plotly.express."""
        price_df = self.get_price_df()

        rename_columns = {
            price_df.index.name or "index": "t",
            "supplier_prices": "Grid prices",
            "eeg_prices": "EEG prices",
            "wholesale_market_prices": "Wholesale market prices",
        }
        if self.has_community_market:
            for city in self.community_cities:
                rename_columns[f"community_market_prices_{city}"] = (
                    f"Community market prices ({city})"
                )
        df = price_df.reset_index().rename(columns=rename_columns)

        df = df.drop(columns=[col for col in df.columns if (df[col] == 0).all()])
        long = df.melt(id_vars=[df.columns[0]], var_name="Price type", value_name="EUR")
        fig = px.line(long, x="t", y="EUR", color="Price type", title="Energy prices")

        if show:
            fig.show()
        return fig

    def plot_supplier_costs(self, show: bool = True) -> go.Figure:
        """Plot costs for meeting demand solely from supplier."""

        supplier_price = self.get_price_df()["supplier_prices"].reset_index(drop=True)
        load = self.demand

        costs = supplier_price * load

        fig = px.line(x=self.original_index, y=costs.values)
        fig.update_layout(
            title="Costs for meeting demand solely by supplier",
            xaxis_title="t",
            yaxis_title="Costs in €",
        )

        if show:
            fig.show()
        return fig
