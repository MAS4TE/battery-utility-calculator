# SPDX-FileCopyrightText: Christoph Komanns, Florian Maurer, Ralf Schemm
#
# SPDX-License-Identifier: MIT

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyomo.environ as pyo

from battery_utility_calculator import Storage
from battery_utility_calculator.electric_vehicle import ElectricVehicle
from battery_utility_calculator.timeseries import (
    check_finite,
    describe_index_mismatch,
    indices_match,
)

log = logging.getLogger("battery_utility")
log.setLevel(logging.WARNING)

epsilon = 1e-4
# Nested dict of grid fees for connections between locations
DEFAULT_GRID_FEE_BETWEEN_LOCATIONS = {
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
        "aachen": 0.015,
        "heerlen": 0.0,
        "liege": 0.02,
    },
    "liege": {
        "juelich": 0.02,
        "aachen": 0.02,
        "heerlen": 0.02,
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
        is_rented_storage: bool = False,
        allow_storage_to_community: bool = True,
        allow_pv_to_wholesale: bool = False,
        allow_wholesale_to_storage: bool = True,
        allow_storage_to_wholesale: bool = True,
        wholesale_fee: float = 0.3,
        rented_pv_grid_fee: float | None = None,
        my_location: str = "aachen",
        grid_fee_between_locations: dict[str, dict[str, float]]
        | None = DEFAULT_GRID_FEE_BETWEEN_LOCATIONS,
        storage_location: str | None = None,
        eeg_eligible: bool = True,
        goal: str = "max_cashflow",
        discharge_penalty_per_kwh: float = 1e-6,
        cycle_cost_per_kwh: float = 0.0,
        electric_vehicle: ElectricVehicle | None = None,
    ):
        """Optimizer for prosumer energy management, calculating minimum costs to cover energy demand.

        Args:
            storage (Storage) | None: The available storage. None if no storage is available.
            demand (pd.Series): Demand in kW per timestep; converted internally to kWh per timestep using hours_per_timestep.
            solar_generation (pd.Series): PV generation in kW per timestep; converted internally to kWh per timestep.
            All model flow variables and storage_level use kWh per timestep.
            supplier_prices (pd.Series): The grid prices for the optimization. Values should be in EUR per kWh. Index has to be pd.DateTimeIndex.
            eeg_prices (pd.Series): The EEG prices for the optimization. Values should be in EUR per kWh. Index has to be pd.DateTimeIndex.
            community_market_prices (dict[str, pd.Series] | None): Community market prices per location. ``None`` or an empty dict disables the community market (flows indexed by ``my_location`` at zero). Otherwise keys must be present in ``grid_fee_between_locations``. Values in EUR per kWh with pd.DateTimeIndex.
            wholesale_market_prices (pd.Series): The wholesale market prices for the optimization. Values should be in EUR per kWh. Index has to be pd.DateTimeIndex.
            hours_per_timestep (int | float): Duration of each timestep in hours (e.g. 0.25 for 15 minutes). Used to convert kW inputs to kWh and to cap storage charge/discharge energy per step.
            is_rented_storage (bool): If True, grid fees apply to tenant flows when using someone else's storage (buyer). If False, only external imports into storage are charged (storage provider / seller); then ``storage_location`` must equal ``my_location``. Imports from a community market straight to the home bypass the storage and are charged in both cases.
            storage_use_cases (list[str]): The use cases for energy storage. Allowed values are "eeg", "wholesale", "community", "home"
            allow_community_to_storage (bool): Import from community market into the home SOC bucket (community → storage → home).
            allow_community_market_arbitrage (bool): Import into the community SOC bucket for round-trip arbitrage (requires ``allow_storage_to_community`` to sell back).
            allow_storage_to_community (bool): Discharge community SOC to the community market (PV via storage or arbitrage).
            allow_community_to_home (bool): Direct community import to home without storage.
            allow_pv_to_community (bool): Direct PV export to the community market.
            wholesale_fee (float): Percentage of earned wholesale money that has to be given away (0.0 to 1.0). Default is 0.3.
            rented_pv_grid_fee (float | None): Grid fee in EUR/kWh charged on PV charged into a rented storage for later home use (``pv_to_storage[t, "home"]``), applied only when ``is_rented_storage`` is True. ``None`` keeps the previous behaviour, which charges every ``pv_to_storage`` use case at ``grid_fee_rate(my_location, storage_location)``. Setting a value restricts the charge to the home use case, where the energy comes back as the household's own withdrawal, and decouples it from the ``supplier_to_storage`` rate. Default is None.
            my_location (str): Location of the prosumer. Must be a key in ``grid_fee_between_locations``. Default is "aachen".
            grid_fee_between_locations (dict[str, dict[str, float]]): Grid fees in EUR/kWh for energy transferred between locations. Default is DEFAULT_GRID_FEE_BETWEEN_LOCATIONS.
            storage_location (str | None): Location where the storage is located. Defaults to ``my_location``.
            eeg_eligible (bool): Whether the storage is eligible for EEG. Default is True.
            goal (str): The goal of the optimization. Allowed values are "max_cashflow" and "max_green_energy". Default is "max_cashflow".
            discharge_penalty_per_kwh (float): The discharge penalty per kWh in EUR. Default is 1e-6.
            cycle_cost_per_kwh (float): Optional degradation cost per discharged kWh in EUR.
            electric_vehicle (ElectricVehicle | None): Electric vehicle of the household, modelled
                as a flexible load (see ``ElectricVehicle``). ``demand`` is then the household load
                without EV charging. Its timeseries need the same index as ``demand``. None
                (default) leaves the model unchanged.
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

        self.allow_wholesale_to_storage = allow_wholesale_to_storage
        self.allow_pv_to_wholesale = allow_pv_to_wholesale
        self.allow_storage_to_wholesale = allow_storage_to_wholesale

        self.allow_pv_to_community = allow_pv_to_community
        self.allow_community_to_home = allow_community_to_home
        self.allow_storage_to_community = allow_storage_to_community
        self.allow_community_to_storage = allow_community_to_storage
        self.allow_community_market_arbitrage = allow_community_market_arbitrage
        self.wholesale_fee = wholesale_fee
        self.rented_pv_grid_fee = rented_pv_grid_fee
        self.grid_fee_between_locations = self.__prepare_grid_fee_between_locations__(
            grid_fee_between_locations
        )
        self.my_location = self.__normalize_location__(
            my_location, allowed_locations=set(self.grid_fee_between_locations.keys())
        )
        self.storage_location = self.__normalize_location__(
            storage_location if storage_location is not None else self.my_location,
            allowed_locations=set(self.grid_fee_between_locations.keys()),
        )
        self.rented_storage = is_rented_storage
        if not self.rented_storage and self.storage_location != self.my_location:
            raise ValueError(
                "storage_location must equal my_location when rented_storage is False "
                f"(storage_location='{self.storage_location}', my_location='{self.my_location}')."
            )
        self.community_market_prices, self.community_locations = (
            self.__prepare_community_market_prices__(community_market_prices)
        )
        self.has_community_market = bool(self.community_market_prices)
        self.allow_supplier_to_storage = True
        if self.storage_location != self.my_location:
            self.allow_supplier_to_storage = False
            log.warning(
                "Disabled supplier_to_storage use-case because storage_location '%s' differs from my_location '%s'.",
                self.storage_location,
                self.my_location,
            )
        self.eeg_eligible = eeg_eligible
        self.goal = goal
        self.discharge_penalty_per_kwh = discharge_penalty_per_kwh
        self.cycle_cost_per_kwh = cycle_cost_per_kwh

        self.storage_use_cases = storage_use_cases

        if electric_vehicle is not None and not isinstance(
            electric_vehicle, ElectricVehicle
        ):
            raise TypeError(
                "electric_vehicle has to be an ElectricVehicle or None, got "
                f"{type(electric_vehicle).__name__}."
            )
        self.electric_vehicle = electric_vehicle
        self.has_ev = electric_vehicle is not None

        if not self.eeg_eligible:
            self.eeg_prices = self.eeg_prices * 0

        self.__check_prepare_timeseries_indices__()
        self.timesteps = list(range(len(self.demand)))
        if self.has_ev:
            self.__check_ev_feasibility__()

        self.is_optimized = False

        self.model = pyo.ConcreteModel()
        self.__set_model_variables__()
        self.set_model_constraints()

        if self.goal == "max_cashflow":
            self.__set_max_cashflow_objective__()
        elif self.goal == "max_green_energy":
            self.set_max_green_energy_objective()

    def __normalize_location__(
        self, location: str, allowed_locations: set[str] | None = None
    ) -> str:
        if not isinstance(location, str):
            raise TypeError("location names have to be strings.")

        normalized = location.strip().lower()

        if allowed_locations is None:
            allowed_locations = set(DEFAULT_GRID_FEE_BETWEEN_LOCATIONS.keys())

        if normalized not in allowed_locations:
            msg = f"Unknown location '{location}'. Allowed values: {sorted(allowed_locations)}"
            raise ValueError(msg)

        return normalized

    def __prepare_community_market_prices__(
        self, community_market_prices: dict[str, pd.Series] | None
    ) -> tuple[dict[str, pd.Series], list[str]]:
        if community_market_prices is None:
            return {}, [self.my_location]
        if not isinstance(community_market_prices, dict):
            raise TypeError(
                "community_market_prices has to be a dict[str, pd.Series] or None."
            )
        if not community_market_prices:
            return {}, [self.my_location]

        allowed_locations = set(self.grid_fee_between_locations.keys())
        prepared: dict[str, pd.Series] = {}

        for location, series in community_market_prices.items():
            normalized_location = self.__normalize_location__(
                location, allowed_locations=allowed_locations
            )
            if normalized_location in prepared:
                raise ValueError(
                    f"Duplicate community market location '{normalized_location}' in "
                    "community_market_prices."
                )
            if not isinstance(series, pd.Series):
                raise TypeError(
                    f"community_market_prices['{location}'] has to be a pd.Series."
                )
            prepared[normalized_location] = series.copy()

        return prepared, sorted(prepared.keys())

    def __prepare_grid_fee_between_locations__(
        self,
        grid_fee_between_locations: dict[str, dict[str, float]] | None,
    ) -> dict[str, dict[str, float]]:
        # this prevents modifying the default grid fees by reference
        normalized_fees = {
            from_location: row.copy()
            for from_location, row in DEFAULT_GRID_FEE_BETWEEN_LOCATIONS.items()
        }

        if grid_fee_between_locations is None:
            return normalized_fees
        if not isinstance(grid_fee_between_locations, dict):
            raise TypeError(
                "grid_fee_between_locations has to be a dict[str, dict[str, float]] or None."
            )

        allowed_locations = set(normalized_fees.keys())
        for from_location, to_fees in grid_fee_between_locations.items():
            normalized_from = self.__normalize_location__(
                from_location, allowed_locations=allowed_locations
            )
            if not isinstance(to_fees, dict):
                raise TypeError(
                    f"grid_fee_between_locations['{from_location}'] has to be a dict[str, float]."
                )
            for to_location, value in to_fees.items():
                normalized_to = self.__normalize_location__(
                    to_location, allowed_locations=allowed_locations
                )
                normalized_fees[normalized_from][normalized_to] = float(value)

        return normalized_fees

    def grid_fee_rate(self, from_location: str, to_location: str) -> float:
        return float(self.grid_fee_between_locations[from_location][to_location])

    def __define_community_flow_var__(self, var_name: str, allow: bool) -> None:
        if self.has_community_market and allow:
            setattr(
                self.model,
                var_name,
                pyo.Var(
                    self.timesteps,
                    self.community_locations,
                    domain=pyo.NonNegativeReals,
                ),
            )
        else:
            setattr(
                self.model,
                var_name,
                pyo.Var(
                    self.timesteps,
                    self.community_locations,
                    domain=pyo.NonNegativeReals,
                    bounds=(0, 0),
                ),
            )

    def __community_flow_value__(
        self,
        var_name: str,
        timestep: int,
        *,
        location: str,
        use_values: bool = False,
    ):
        return self.__get_value__(
            getattr(self.model, var_name)[timestep, location], use_values
        )

    def __community_flow_timestep_total__(
        self, var_name: str, timestep: int, use_values: bool = False
    ):
        return sum(
            self.__community_flow_value__(
                var_name, timestep, location=location, use_values=use_values
            )
            for location in self.community_locations
        )

    def __community_to_storage_value__(
        self,
        timestep: int,
        location: str,
        *,
        use_values: bool = False,
    ):
        return self.__community_flow_value__(
            "community_to_storage_for_home",
            timestep,
            location=location,
            use_values=use_values,
        ) + self.__community_flow_value__(
            "community_to_storage_for_community",
            timestep,
            location=location,
            use_values=use_values,
        )

    def __community_to_storage_timestep_total__(
        self, timestep: int, *, use_values: bool = False
    ):
        return sum(
            self.__community_to_storage_value__(
                timestep, location=location, use_values=use_values
            )
            for location in self.community_locations
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

        check_finite(self.demand, "demand")

        self.original_index = ref_index
        new_index = pd.RangeIndex(len(ref_index))

        self.demand.index = new_index
        # ensure all indices match the reference
        for name in attrs:
            series = getattr(self, name).copy()

            if not isinstance(series.index, pd.DatetimeIndex):
                msg = f"Index of {name} has to be pd.DateTimeIndex!"
                raise TypeError(msg)

            if not indices_match(series.index, ref_index):
                raise ValueError(
                    "All timeseries indices must be identical.\n"
                    + describe_index_mismatch(ref_index, series.index, "demand", name)
                )

            check_finite(series, name)

            series.index = new_index
            setattr(self, name, series)

        if self.has_community_market:
            prepared_community: dict[str, pd.Series] = {}
            for location in self.community_locations:
                series = self.community_market_prices[location].copy()
                label = f"community_market_prices['{location}']"
                if not isinstance(series.index, pd.DatetimeIndex):
                    msg = f"Index of {label} has to be pd.DateTimeIndex!"
                    raise TypeError(msg)
                if not indices_match(series.index, ref_index):
                    raise ValueError(
                        "All timeseries indices must be identical.\n"
                        + describe_index_mismatch(
                            ref_index, series.index, "demand", label
                        )
                    )
                check_finite(series, label)

                series.index = new_index
                prepared_community[location] = series
            self.community_market_prices = prepared_community

        if self.has_ev:
            ev = self.electric_vehicle
            for name, series in (
                ("availability", ev.availability),
                ("driving_consumption", ev.driving_consumption),
            ):
                label = f"electric_vehicle.{name}"
                if not indices_match(series.index, ref_index):
                    raise ValueError(
                        "All timeseries indices must be identical.\n"
                        + describe_index_mismatch(
                            ref_index, series.index, "demand", label
                        )
                    )
            # kept separately so the ElectricVehicle object stays untouched
            self.ev_availability = pd.Series(
                ev.availability.to_numpy(dtype=float), index=new_index
            )
            self.ev_driving = pd.Series(
                ev.driving_consumption.to_numpy(dtype=float) * self.hours_per_timestep,
                index=new_index,
            )

        h = self.hours_per_timestep
        self.demand = self.demand * h
        self.solar_generation = self.solar_generation * h

    def __ev_departure_timesteps__(self) -> list[int]:
        """Last available timestep before each departure within the horizon.

        A departure is a transition from ``availability > 0`` to ``availability == 0``.
        """
        availability = self.ev_availability
        return [
            timestep
            for timestep in self.timesteps[:-1]
            if availability.iloc[timestep] > 0 and availability.iloc[timestep + 1] == 0
        ]

    def __check_ev_feasibility__(self) -> None:
        """Fail early if the driving profile cannot be served without public charging.

        Charging at full power whenever the car is at home maximises the SOC at every
        timestep, so if that schedule violates a bound, every schedule does. Like the
        LP, it only looks at the SOC at the end of each timestep.
        """
        ev = self.electric_vehicle
        if ev.public_charging_price is not None:
            return

        capacity = ev.capacity_kwh
        tolerance = 1e-6
        max_charge = ev.max_charge_kw * self.hours_per_timestep * ev.charge_efficiency
        departures = set(self.__ev_departure_timesteps__())
        soc = ev.initial_soc * capacity
        for timestep in self.timesteps:
            soc = min(
                capacity,
                soc
                + max_charge * self.ev_availability.iloc[timestep]
                - self.ev_driving.iloc[timestep],
            )
            when = self.original_index[timestep]
            if soc < ev.min_soc * capacity - tolerance:
                raise ValueError(
                    f"Electric vehicle '{ev.id}' cannot cover its driving consumption: even "
                    f"when charging at full power whenever it is at home, the SOC drops to "
                    f"{soc:.3f} kWh at {when}, below min_soc ({ev.min_soc * capacity:.3f} "
                    "kWh). Set public_charging_price to allow charging elsewhere, or check "
                    "capacity_kwh, max_charge_kw and the availability profile."
                )
            if (
                ev.min_departure_soc is not None
                and timestep in departures
                and soc < ev.min_departure_soc * capacity - tolerance
            ):
                raise ValueError(
                    f"Electric vehicle '{ev.id}' cannot reach min_departure_soc "
                    f"({ev.min_departure_soc * capacity:.3f} kWh) before leaving at {when}; "
                    f"at most {soc:.3f} kWh are possible."
                )
        if soc < ev.initial_soc * capacity - tolerance:
            raise ValueError(
                f"Electric vehicle '{ev.id}' cannot be recharged to its initial SOC "
                f"({ev.initial_soc * capacity:.3f} kWh) by the end of the horizon; at most "
                f"{soc:.3f} kWh are possible. Set public_charging_price or extend the horizon."
            )

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

        # selling from storage on wholesale; without the use case there is no SOC
        # balance backing this flow, so it has to stay closed
        if self.allow_storage_to_wholesale and "wholesale" in self.storage_use_cases:
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
            "storage_to_community",
            self.allow_storage_to_community and "community" in self.storage_use_cases,
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

        # charging storage from wholesale market; same reasoning as above, which
        # matters for negative prices where charging into nowhere would pay off
        if self.allow_wholesale_to_storage and "wholesale" in self.storage_use_cases:
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

        # charging storage from supplier; feeds the home SOC bucket
        if self.allow_supplier_to_storage and "home" in self.storage_use_cases:
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

        if self.has_ev:
            self.__set_ev_variables__()

        log.info("Model variables set up successfully.")

    def __set_ev_variables__(self):
        ev = self.electric_vehicle
        capacity = ev.capacity_kwh

        # state of charge at the end of each timestep (kWh)
        self.model.ev_soc = pyo.Var(
            self.timesteps,
            domain=pyo.NonNegativeReals,
            bounds=(ev.min_soc * capacity, capacity),
        )

        # charging sources, all household consumption
        self.model.pv_to_ev = pyo.Var(self.timesteps, domain=pyo.NonNegativeReals)
        self.model.supplier_to_ev = pyo.Var(self.timesteps, domain=pyo.NonNegativeReals)
        if "home" in self.storage_use_cases:
            self.model.storage_to_ev = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals
            )
        else:
            self.model.storage_to_ev = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals, bounds=(0, 0)
            )
        # follows the same switch as community -> home, since it is home consumption
        self.__define_community_flow_var__(
            "community_to_ev", self.allow_community_to_home
        )

        # vehicle-to-home, only towards the household demand
        if ev.v2h_enabled:
            self.model.ev_to_home = pyo.Var(self.timesteps, domain=pyo.NonNegativeReals)
        else:
            self.model.ev_to_home = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals, bounds=(0, 0)
            )

        # driving energy bought elsewhere; can at most replace what is driven
        def public_bounds(model, timestep):
            if ev.public_charging_price is None:
                return (0, 0)
            return (0, float(self.ev_driving.iloc[timestep]))

        self.model.ev_public_charging = pyo.Var(
            self.timesteps, domain=pyo.NonNegativeReals, bounds=public_bounds
        )

    def set_model_constraints(self):
        log.info("Setting up model constraints...")

        # consumption must equal supply (from PV system, supplier, community market, PV system)
        def restrict_demand(model, timestep):
            community_to_home = sum(
                model.community_to_home[timestep, location]
                for location in self.community_locations
            )

            supply = (
                model.storage_to_home[timestep]
                + model.pv_to_home[timestep]
                + model.supplier_to_home[timestep]
                + community_to_home
            )
            if self.has_ev:
                supply = supply + model.ev_to_home[timestep]
            return supply == self.demand.iloc[timestep]

        self.model.demand_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_demand
        )

        # use of PV system must be smaller or equal than PV system generation
        def restrict_solar_gen(model, timestep):
            pv_to_community = sum(
                model.pv_to_community[timestep, location]
                for location in self.community_locations
            )

            pv_use = (
                sum(
                    model.pv_to_storage[timestep, use] for use in self.storage_use_cases
                )
                + model.pv_to_eeg[timestep]
                + model.pv_to_home[timestep]
                + model.pv_to_wholesale[timestep]
                + pv_to_community
            )
            if self.has_ev:
                pv_use = pv_use + model.pv_to_ev[timestep]
            return pv_use <= self.solar_generation.iloc[timestep]

        self.model.solar_gen_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_solar_gen
        )

        # energy flow TO storage (kWh per timestep) must be <= max charge energy per step
        def restrict_storage_charge(model, timestep):
            community_to_storage = sum(
                model.community_to_storage_for_home[timestep, location]
                + model.community_to_storage_for_community[timestep, location]
                for location in self.community_locations
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
                model.storage_to_community[timestep, location]
                for location in self.community_locations
            )

            discharge = (
                +model.storage_to_eeg[timestep]
                + model.storage_to_wholesale[timestep]
                + storage_to_community
                + model.storage_to_home[timestep]
            )
            if self.has_ev:
                discharge = discharge + model.storage_to_ev[timestep]
            return (
                discharge
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

        # the storage starts empty, so after the first timestep it can hold at most
        # what one full charging step adds to the SOC
        def restrict_soc_start(model):
            return (
                sum(model.storage_level[0, use] for use in self.storage_use_cases)
                <= self.storage.volume
                * self.storage.c_rate
                * self.hours_per_timestep
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
                        model.community_to_storage_for_community[timestep, location]
                        for location in self.community_locations
                    )
                    + model.pv_to_storage[timestep, "community"]
                )
                community_discharge = sum(
                    model.storage_to_community[timestep, location]
                    for location in self.community_locations
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
                    model.community_to_storage_for_home[timestep, location]
                    for location in self.community_locations
                )
                # the home bucket also charges the household's EV
                if self.has_ev:
                    to_ev = (1 / self.discharge_efficiency) * model.storage_to_ev[
                        timestep
                    ]
                else:
                    to_ev = 0
                if timestep == self.timesteps[0]:
                    return (
                        model.storage_level[timestep, "home"]
                        == self.charge_efficiency * model.supplier_to_storage[timestep]
                        + self.charge_efficiency * model.pv_to_storage[timestep, "home"]
                        + self.charge_efficiency * community_to_storage_for_home
                        - (
                            (1 / self.discharge_efficiency)
                            * model.storage_to_home[timestep]
                        )
                        - to_ev
                    )
                else:
                    previous_timestep = timestep - 1
                    return (
                        model.storage_level[timestep, "home"]
                        == model.storage_level[previous_timestep, "home"]
                        + self.charge_efficiency * model.supplier_to_storage[timestep]
                        + self.charge_efficiency * model.pv_to_storage[timestep, "home"]
                        + self.charge_efficiency * community_to_storage_for_home
                        - (
                            (1 / self.discharge_efficiency)
                            * model.storage_to_home[timestep]
                        )
                        - to_ev
                    )

            self.model.soc_home_restriction = pyo.Constraint(
                self.timesteps, rule=restrict_soc_home
            )

        if self.has_ev:
            self.__set_ev_constraints__()

        log.info("Model constraints set up successfully.")

    def __ev_charge_expr__(self, timestep: int, use_values: bool = False):
        """Energy drawn by the EV charger in one timestep (before charge losses)."""
        model = self.model
        return (
            self.__get_value__(model.pv_to_ev[timestep], use_values)
            + self.__get_value__(model.supplier_to_ev[timestep], use_values)
            + self.__get_value__(model.storage_to_ev[timestep], use_values)
            + self.__community_flow_timestep_total__(
                "community_to_ev", timestep, use_values=use_values
            )
        )

    def __set_ev_constraints__(self):
        ev = self.electric_vehicle
        capacity = ev.capacity_kwh
        h = self.hours_per_timestep

        # charging only while plugged in at home, limited by the wallbox
        def restrict_ev_charge(model, timestep):
            limit = ev.max_charge_kw * h * float(self.ev_availability.iloc[timestep])
            return self.__ev_charge_expr__(timestep) <= limit

        self.model.ev_charge_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_ev_charge
        )

        # V2H only while plugged in
        def restrict_ev_discharge(model, timestep):
            limit = ev.max_discharge_kw * h * float(self.ev_availability.iloc[timestep])
            return model.ev_to_home[timestep] <= limit

        self.model.ev_discharge_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_ev_discharge
        )

        # soc[t] = soc[t-1] + eta_c * charge[t] - (driving[t] - public[t])
        #          - ev_to_home[t] / eta_d, with soc[-1] = initial_soc * capacity
        def restrict_ev_soc(model, timestep):
            if timestep == self.timesteps[0]:
                previous = ev.initial_soc * capacity
            else:
                previous = model.ev_soc[timestep - 1]
            return model.ev_soc[timestep] == (
                previous
                + ev.charge_efficiency * self.__ev_charge_expr__(timestep)
                - (
                    float(self.ev_driving.iloc[timestep])
                    - model.ev_public_charging[timestep]
                )
                - (1 / ev.discharge_efficiency) * model.ev_to_home[timestep]
            )

        self.model.ev_soc_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_ev_soc
        )

        # no free energy: the car ends at least as full as it started
        self.model.ev_end_soc_restriction = pyo.Constraint(
            expr=self.model.ev_soc[self.timesteps[-1]] >= ev.initial_soc * capacity
        )

        if ev.min_departure_soc is not None:
            departures = self.__ev_departure_timesteps__()
            self.model.ev_departure_soc_restriction = pyo.Constraint(
                departures,
                rule=lambda model, timestep: (
                    model.ev_soc[timestep] >= ev.min_departure_soc * capacity
                ),
            )

    def __get_value__(self, var, use_values):
        """Helper to conditionally apply .value to a Pyomo variable."""
        return var.value if use_values else var

    def __community_flow_series__(self, var_name: str) -> tuple[dict[str, list], list]:
        """Per-location and aggregated timestep series for a community flow variable."""
        model_var = getattr(self.model, var_name)
        by_location = {
            location: [
                float(model_var[timestep, location].value or 0.0)
                for timestep in self.timesteps
            ]
            for location in self.community_market_prices
        }
        aggregate = [
            sum(
                float(model_var[timestep, location].value or 0.0)
                for location in self.community_locations
            )
            for timestep in self.timesteps
        ]
        return by_location, aggregate

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

        expr = (
            community_cf
            + supplier_cf
            + eeg_cf
            + wholesale_cf
            + grid_fee_cf
            - discharge_penalty
            - cycle_cost_penalty
        )
        if self.has_ev:
            expr = expr + cashflows["ev_public_charging"]

        # maximize sum of cashflows
        self.model.objective = pyo.Objective(expr=expr, sense=pyo.maximize)

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

        cashflows = {
            "community": community_cf,
            "supplier": supplier_cf,
            "eeg": eeg_cf,
            "wholesale": wholesale_cf,
            "grid_fees": grid_fee_cf,
        }
        if self.has_ev:
            cashflows["ev_public_charging"] = (
                self.calculate_ev_public_charging_cashflow(use_values=use_values)
            )
        return cashflows

    def calculate_ev_public_charging_cashflow(self, use_values=False):
        """Cost of driving energy bought elsewhere (negative), 0 without EV or price."""
        if not self.has_ev or self.electric_vehicle.public_charging_price is None:
            return 0.0
        return -self.electric_vehicle.public_charging_price * sum(
            self.__get_value__(self.model.ev_public_charging[timestep], use_values)
            for timestep in self.timesteps
        )

    def calculate_community_cashflow(self, use_values=False):
        if not self.has_community_market:
            return 0.0

        cashflow = sum(
            (
                self.__community_flow_value__(
                    var_name="storage_to_community",
                    timestep=timestep,
                    location=location,
                    use_values=use_values,
                )
                - self.__community_to_storage_value__(
                    timestep,
                    location=location,
                    use_values=use_values,
                )
                - self.__community_flow_value__(
                    var_name="community_to_home",
                    timestep=timestep,
                    location=location,
                    use_values=use_values,
                )
            )
            * self.community_market_prices[location].loc[timestep]
            + self.__community_flow_value__(
                var_name="pv_to_community",
                timestep=timestep,
                location=location,
                use_values=use_values,
            )
            * self.community_market_prices[location].loc[timestep]
            for timestep in self.timesteps
            for location in self.community_market_prices
        )
        if self.has_ev:
            cashflow = cashflow - sum(
                self.__community_flow_value__(
                    var_name="community_to_ev",
                    timestep=timestep,
                    location=location,
                    use_values=use_values,
                )
                * self.community_market_prices[location].loc[timestep]
                for timestep in self.timesteps
                for location in self.community_market_prices
            )
        return cashflow

    def calculate_supplier_cashflow(self, use_values=False):
        cashflow = -(1 - epsilon) * sum(
            self.__get_value__(self.model.supplier_to_storage[timestep], use_values)
            * self.supplier_prices.loc[timestep]
            for timestep in self.timesteps
        ) - sum(
            self.__get_value__(self.model.supplier_to_home[timestep], use_values)
            * self.supplier_prices.loc[timestep]
            for timestep in self.timesteps
        )
        if self.has_ev:
            cashflow = cashflow - sum(
                self.__get_value__(self.model.supplier_to_ev[timestep], use_values)
                * self.supplier_prices.loc[timestep]
                for timestep in self.timesteps
            )
        return cashflow

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
        supplier_charge = sum(
            self.__get_value__(self.model.supplier_to_storage[timestep], use_values)
            for timestep in self.timesteps
        )
        grid_fee = (
            self.grid_fee_rate(self.my_location, self.storage_location)
            * supplier_charge
        )

        if self.rented_storage:
            # extra grid fee for PV -> storage
            if self.rented_pv_grid_fee is None:
                # every use case at the location rate, which is what the flows
                # were charged before rented_pv_grid_fee existed
                pv_charge = sum(
                    self.__get_value__(
                        self.model.pv_to_storage[timestep, use], use_values
                    )
                    for timestep in self.timesteps
                    for use in self.storage_use_cases
                )
                pv_rate = self.grid_fee_rate(self.my_location, self.storage_location)
            elif "home" in self.storage_use_cases:
                # only the home use case comes back as the household's own
                # withdrawal; PV routed to eeg or community is sold on, and its
                # grid fee is the buyer's
                pv_charge = sum(
                    self.__get_value__(
                        self.model.pv_to_storage[timestep, "home"], use_values
                    )
                    for timestep in self.timesteps
                )
                pv_rate = float(self.rented_pv_grid_fee)
            else:
                # without the home use case there is no such flow to charge
                pv_charge = 0.0
                pv_rate = 0.0

            grid_fee = grid_fee + pv_rate * pv_charge

        # community market to storage flows
        for location in self.community_market_prices:
            community_to_storage = sum(
                self.__community_to_storage_value__(
                    timestep, location=location, use_values=use_values
                )
                for timestep in self.timesteps
            )
            if self.rented_storage or location != self.storage_location:
                grid_fee = (
                    grid_fee
                    + self.grid_fee_rate(location, self.storage_location)
                    * community_to_storage
                )
                grid_fee = (
                    grid_fee
                    + self.grid_fee_rate(self.storage_location, self.my_location)
                    * community_to_storage
                )

            # grid fee from community to home; this flow never touches the storage,
            # so it is charged regardless of who owns the storage
            community_to_home = sum(
                self.__community_flow_value__(
                    "community_to_home",
                    timestep,
                    location=location,
                    use_values=use_values,
                )
                for timestep in self.timesteps
            )
            grid_fee = (
                grid_fee
                + self.grid_fee_rate(location, self.my_location) * community_to_home
            )

            # EV charging from the community is household consumption as well
            if self.has_ev:
                community_to_ev = sum(
                    self.__community_flow_value__(
                        "community_to_ev",
                        timestep,
                        location=location,
                        use_values=use_values,
                    )
                    for timestep in self.timesteps
                )
                grid_fee = (
                    grid_fee
                    + self.grid_fee_rate(location, self.my_location) * community_to_ev
                )

        if isinstance(grid_fee, (int, float)) and grid_fee <= 0:
            return 0.0

        return -grid_fee

    def __storage_discharge_total__(self, use_values=False):
        total_discharge = sum(
            self.__get_value__(self.model.storage_to_eeg[timestep], use_values)
            + self.__get_value__(self.model.storage_to_wholesale[timestep], use_values)
            + self.__community_flow_timestep_total__(
                "storage_to_community", timestep, use_values=use_values
            )
            + self.__get_value__(self.model.storage_to_home[timestep], use_values)
            for timestep in self.timesteps
        )
        if self.has_ev:
            total_discharge = total_discharge + sum(
                self.__get_value__(self.model.storage_to_ev[timestep], use_values)
                for timestep in self.timesteps
            )
        return total_discharge

    def calculate_discharge_penalty(self, use_values=False):
        total_discharge = self.__storage_discharge_total__(use_values=use_values)
        if self.has_ev:
            # tie-breaker against pointless V2H round trips
            total_discharge = total_discharge + sum(
                self.__get_value__(self.model.ev_to_home[timestep], use_values)
                for timestep in self.timesteps
            )
        return self.discharge_penalty_per_kwh * total_discharge

    def calculate_cycle_cost_penalty(self, use_values=False):
        # degradation of the stationary storage only; the EV battery is not priced
        total_discharge = self.__storage_discharge_total__(use_values=use_values)
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

        # PV charged into the household's EV is self-consumption as well
        if self.has_ev:
            expr = expr + sum(
                self.model.pv_to_ev[timestep] for timestep in self.timesteps
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
        if self.has_ev:
            total_cashflow = total_cashflow + cashflows["ev_public_charging"]

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
        if self.has_ev:
            total += cashflows["ev_public_charging"]
        if self.goal == "max_cashflow":
            total -= self.calculate_discharge_penalty(use_values=True)
            total -= self.calculate_cycle_cost_penalty(use_values=True)
        return float(total)

    # ------------------------------------------------------------------
    def get_cashflows(self) -> dict:
        """Return the individual cashflow components from the optimized model.

        The dictionary contains five keys: ``"community"``, ``"supplier"``,
        ``"eeg"``, ``"wholesale"`` and ``"grid_fees"``; with an electric vehicle a
        sixth key ``"ev_public_charging"`` holds the cost of public charging. EV
        charging at home is part of ``"supplier"``, ``"community"`` and
        ``"grid_fees"`` like any other household consumption. Values are floats
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

    def optimize(self, solver: str = "gurobi", tee: bool = False):
        optimizer = pyo.SolverFactory(
            solver,
        )

        results = optimizer.solve(self.model, tee=tee)

        condition = results.solver.termination_condition
        if condition != pyo.TerminationCondition.optimal:
            raise RuntimeError(
                f"Solver '{solver}' found no optimal solution: {condition}"
            )

        self.is_optimized = True

        return self.model.objective()

    def get_price_df(self) -> pd.DataFrame:
        price_columns = {
            "supplier_prices": self.supplier_prices,
            "eeg_prices": self.eeg_prices,
            "wholesale_market_prices": self.wholesale_market_prices,
        }
        if self.has_community_market:
            for location, series in self.community_market_prices.items():
                price_columns[f"community_market_prices_{location}"] = series

        price_df = pd.DataFrame(price_columns)
        price_df.index = self.original_index.copy()

        return price_df

    def get_energy_flows(self) -> pd.DataFrame:
        energy_flows = pd.DataFrame(index=self.timesteps)

        # every flow variable exists no matter which use cases are active -
        # disabled paths are declared with bounds=(0, 0) - so all of them can be
        # read directly. Gating them on storage_use_cases would zero out flows
        # that never touch the storage, such as pv_to_eeg or supplier_to_home.
        for var_name in (
            "storage_to_wholesale",
            "wholesale_to_storage",
            "pv_to_wholesale",
            "pv_to_eeg",
            "storage_to_eeg",
            "storage_to_home",
            "supplier_to_home",
            "supplier_to_storage",
            "pv_to_home",
        ):
            model_var = getattr(self.model, var_name)
            energy_flows[var_name] = [
                model_var[timestep].value for timestep in self.timesteps
            ]

        community_by_location: dict[str, dict[str, list]] = {}
        community_totals: dict[str, list] = {}
        for var_name in (
            "storage_to_community",
            "pv_to_community",
            "community_to_storage_for_home",
            "community_to_storage_for_community",
            "community_to_home",
        ):
            by_location, aggregate = self.__community_flow_series__(var_name)
            for location, values in by_location.items():
                energy_flows[f"{var_name}_{location}"] = values
            energy_flows[var_name] = aggregate
            community_by_location[var_name] = by_location
            community_totals[var_name] = aggregate

        # combined view over both community SOC buckets
        energy_flows["community_to_storage"] = [
            for_home + for_community
            for for_home, for_community in zip(
                community_totals["community_to_storage_for_home"],
                community_totals["community_to_storage_for_community"],
                strict=True,
            )
        ]
        for location in self.community_market_prices:
            energy_flows[f"community_to_storage_{location}"] = [
                for_home + for_community
                for for_home, for_community in zip(
                    community_by_location["community_to_storage_for_home"][location],
                    community_by_location["community_to_storage_for_community"][
                        location
                    ],
                    strict=True,
                )
            ]

        # pv_to_storage is the only variable actually indexed by the use cases
        for use_case in ("home", "eeg", "wholesale", "community"):
            column = f"pv_to_storage_for_{use_case}"
            if use_case in self.storage_use_cases:
                energy_flows[column] = [
                    self.model.pv_to_storage[timestep, use_case].value
                    for timestep in self.timesteps
                ]
            else:
                energy_flows[column] = 0

        energy_flows["demand"] = self.demand.values

        if self.has_ev:
            for var_name in (
                "pv_to_ev",
                "supplier_to_ev",
                "storage_to_ev",
                "ev_to_home",
                "ev_public_charging",
            ):
                model_var = getattr(self.model, var_name)
                energy_flows[var_name] = [
                    float(model_var[timestep].value or 0.0)
                    for timestep in self.timesteps
                ]
            by_location, aggregate = self.__community_flow_series__("community_to_ev")
            for location, values in by_location.items():
                energy_flows[f"community_to_ev_{location}"] = values
            energy_flows["community_to_ev"] = aggregate
            energy_flows["ev_driving"] = self.ev_driving.values

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
        for location in self.community_market_prices:
            demand_coverage[f"from_community_{location}"] = [
                float(self.model.community_to_home[timestep, location].value or 0.0)
                for timestep in self.timesteps
            ]
        if self.has_ev:
            demand_coverage["from_ev"] = [
                float(self.model.ev_to_home[timestep].value or 0.0)
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
        for location in self.community_market_prices:
            pv_usage[f"to_community_{location}"] = [
                float(self.model.pv_to_community[t, location].value or 0.0)
                for t in self.timesteps
            ]
        pv_usage["to_wholesale"] = [
            self.model.pv_to_wholesale[t].value for t in self.timesteps
        ]
        # pv_to_storage only exists for the active use cases
        for use_case in ("home", "eeg", "wholesale", "community"):
            if use_case in self.storage_use_cases:
                pv_usage[f"to_storage_{use_case}"] = [
                    self.model.pv_to_storage[t, use_case].value for t in self.timesteps
                ]
            else:
                pv_usage[f"to_storage_{use_case}"] = 0.0
        if self.has_ev:
            pv_usage["to_ev"] = [
                float(self.model.pv_to_ev[t].value or 0.0) for t in self.timesteps
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

    def get_ev_timeseries_df(self) -> pd.DataFrame:
        """Timeseries of the electric vehicle in kWh per timestep.

        Columns: ``availability``, ``soc`` (kWh at the end of the step), ``charge_from_pv``,
        ``charge_from_supplier``, ``charge_from_storage``, ``charge_from_community``,
        ``charge`` (sum, drawn at the wallbox before losses), ``to_home`` (V2H),
        ``driving`` and ``public_charging``.

        Raises:
            ValueError: If no electric vehicle is modelled or before ``optimize()``.
        """
        if not self.has_ev:
            raise ValueError("No electric_vehicle given to this EnergyCostCalculator.")
        if not self.is_optimized:
            raise ValueError("Model not optimized yet - run optimize() first!")

        def values(var_name):
            model_var = getattr(self.model, var_name)
            return [float(model_var[t].value or 0.0) for t in self.timesteps]

        ev_df = pd.DataFrame(index=self.timesteps)
        ev_df["availability"] = self.ev_availability.values
        ev_df["soc"] = values("ev_soc")
        ev_df["charge_from_pv"] = values("pv_to_ev")
        ev_df["charge_from_supplier"] = values("supplier_to_ev")
        ev_df["charge_from_storage"] = values("storage_to_ev")
        _, ev_df["charge_from_community"] = self.__community_flow_series__(
            "community_to_ev"
        )
        ev_df["charge"] = ev_df[
            [
                "charge_from_pv",
                "charge_from_supplier",
                "charge_from_storage",
                "charge_from_community",
            ]
        ].sum(axis=1)
        ev_df["to_home"] = values("ev_to_home")
        ev_df["driving"] = self.ev_driving.values
        ev_df["public_charging"] = values("ev_public_charging")

        ev_df.index = self.original_index.copy()
        return ev_df

    def get_ev_kpis(self) -> dict:
        """Aggregated EV energy flows (kWh) and public charging costs (EUR)."""
        ev_df = self.get_ev_timeseries_df()
        return {
            "charged_kwh_total": float(ev_df["charge"].sum()),
            "charged_by_source_kwh": {
                "pv": float(ev_df["charge_from_pv"].sum()),
                "supplier": float(ev_df["charge_from_supplier"].sum()),
                "storage": float(ev_df["charge_from_storage"].sum()),
                "community": float(ev_df["charge_from_community"].sum()),
            },
            "v2h_kwh": float(ev_df["to_home"].sum()),
            "driving_kwh": float(ev_df["driving"].sum()),
            "public_charging_kwh": float(ev_df["public_charging"].sum()),
            "public_charging_cost": float(
                -self.calculate_ev_public_charging_cashflow(use_values=True)
            ),
            "start_soc_kwh": float(
                self.electric_vehicle.initial_soc * self.electric_vehicle.capacity_kwh
            ),
            "end_soc_kwh": float(ev_df["soc"].iloc[-1]),
        }

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
        if self.has_ev:
            discharged_by_sink_kwh["ev"] = col_energy_sum("storage_to_ev")

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

        kpis = {
            "charged_kwh_total": charged_kwh_total,
            "discharged_kwh_total": discharged_kwh_total,
            "charged_by_source_kwh": charged_by_source_kwh,
            "discharged_by_sink_kwh": discharged_by_sink_kwh,
            "full_cycles_equivalent": float(full_cycles_equivalent),
            "utilization_ratio": float(utilization_ratio),
            "roundtrip_indicator": float(roundtrip_indicator),
        }
        if self.has_ev:
            kpis["electric_vehicle"] = self.get_ev_kpis()
        return kpis

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

        if self.has_ev:
            results["electric_vehicle"] = self.get_ev_timeseries_df()

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
                "pv_to_ev": "PV to EV",
                "supplier_to_ev": "Supplier to EV",
                "storage_to_ev": "Storage to EV",
                "community_to_ev": "Community to EV",
                "ev_to_home": "EV to home",
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
                "from_ev": "From EV",
            }
        )

        # only plot those with usage
        df = df.drop(columns=[col for col in df.columns if (df[col] == 0).all()])

        # long format for stacking
        supply_cols = [
            c
            for c in [
                "From PV",
                "From storage",
                "From community",
                "From grid",
                "From EV",
            ]
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
                "to_ev": "To EV",
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
        fig = px.line(
            long, x="t", y="kWh", color="Use case", title="Charge / Discharge"
        )

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
            for location in self.community_locations:
                rename_columns[f"community_market_prices_{location}"] = (
                    f"Community market prices ({location})"
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
