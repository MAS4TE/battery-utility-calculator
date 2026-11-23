# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT

import math
from collections.abc import Hashable

import pandas as pd

from battery_utility_calculator.timeseries import (
    check_finite,
    describe_index_mismatch,
    indices_match,
)


def _check_fraction(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} has to be a fraction between 0 and 1, got {value}.")
    return value


def _check_non_negative(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} has to be a finite value >= 0, got {value}.")
    return value


def _check_efficiency(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name} has to be in (0, 1], got {value}.")
    return value


class ElectricVehicle:
    def __init__(
        self,
        id: Hashable,
        capacity_kwh: float,
        max_charge_kw: float,
        availability: pd.Series,
        driving_consumption: pd.Series,
        max_discharge_kw: float = 0.0,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
        initial_soc: float = 0.5,
        min_soc: float = 0.0,
        min_departure_soc: float | None = None,
        public_charging_price: float | None = None,
    ):
        """Represents an electric vehicle of the household, modelled as a flexible load.

        The car is not offered on any market. While it is at home and plugged in it can
        be charged from PV, the supplier, the home bucket of the storage or the community
        market; with ``max_discharge_kw > 0`` it can also supply the household (V2H).

        Args:
            id (Hashable): Identifier of the vehicle, like ``Storage.id``.
            capacity_kwh (float): Usable battery capacity in kWh (> 0).
            max_charge_kw (float): Maximum charging power at the home wallbox in kW.
            availability (pd.Series): Share of the timestep the car is at home and
                plugged in, in [0, 1]: 1 = plugged in, 0 = away, fractions arise from
                resampling (e.g. 15 min to hourly). Charging and V2H power are limited
                to ``availability * max_kw``. Needs a ``pd.DatetimeIndex`` identical to
                the demand index.
            driving_consumption (pd.Series): Power in kW drawn from the battery while
                driving (timestep average). May only be > 0 where ``availability`` < 1.
                Same index.
            max_discharge_kw (float): Maximum vehicle-to-home power in kW. 0 (default)
                disables V2H. V2H only serves the household demand, never a market.
            charge_efficiency (float): Share of the charged energy that arrives in the
                battery (0-1].
            discharge_efficiency (float): Share of the discharged energy that reaches
                the home (0-1]. Only relevant for V2H.
            initial_soc (float): State of charge before the first timestep as a fraction
                of ``capacity_kwh``. The SOC at the end of the horizon has to be at
                least this value, so the car cannot hand out free energy.
            min_soc (float): Lower SOC bound as a fraction of ``capacity_kwh``, applied
                at all times.
            min_departure_soc (float | None): Fraction of ``capacity_kwh`` required at
                the end of the last available timestep before each departure, i.e. every
                timestep ``t`` with ``availability[t] > 0`` and ``availability[t+1] == 0``.
                None disables it.
            public_charging_price (float | None): Price in EUR/kWh for driving energy
                that is bought elsewhere (public charging) instead of being taken from
                the battery. Acts as a slack that keeps the optimization feasible. None
                disables public charging; an infeasible driving profile then raises a
                ``ValueError`` when the optimizer is built.

        Raises:
            TypeError: If a timeseries is not a ``pd.Series`` with ``pd.DatetimeIndex``.
            ValueError: On invalid values, mismatched indices or inconsistent fractions.
        """
        self.id = id
        self.capacity_kwh = float(capacity_kwh)
        if not math.isfinite(self.capacity_kwh) or self.capacity_kwh <= 0:
            raise ValueError(f"capacity_kwh has to be > 0, got {capacity_kwh}.")
        self.max_charge_kw = _check_non_negative(max_charge_kw, "max_charge_kw")
        self.max_discharge_kw = _check_non_negative(
            max_discharge_kw, "max_discharge_kw"
        )
        self.charge_efficiency = _check_efficiency(
            charge_efficiency, "charge_efficiency"
        )
        self.discharge_efficiency = _check_efficiency(
            discharge_efficiency, "discharge_efficiency"
        )
        self.initial_soc = _check_fraction(initial_soc, "initial_soc")
        self.min_soc = _check_fraction(min_soc, "min_soc")
        if self.initial_soc < self.min_soc:
            raise ValueError(
                f"initial_soc ({self.initial_soc}) must not be below min_soc ({self.min_soc})."
            )
        self.min_departure_soc = (
            None
            if min_departure_soc is None
            else _check_fraction(min_departure_soc, "min_departure_soc")
        )
        self.public_charging_price = (
            None
            if public_charging_price is None
            else _check_non_negative(public_charging_price, "public_charging_price")
        )

        for name, series in (
            ("availability", availability),
            ("driving_consumption", driving_consumption),
        ):
            if not isinstance(series, pd.Series):
                raise TypeError(f"{name} has to be a pd.Series.")
            if not isinstance(series.index, pd.DatetimeIndex):
                raise TypeError(f"Index of {name} has to be pd.DateTimeIndex!")
            check_finite(series, name)

        if not indices_match(availability.index, driving_consumption.index):
            raise ValueError(
                "availability and driving_consumption must share one index.\n"
                + describe_index_mismatch(
                    availability.index,
                    driving_consumption.index,
                    "availability",
                    "driving_consumption",
                )
            )

        values = availability.to_numpy(dtype=float)
        invalid = (values < 0.0) | (values > 1.0)
        if invalid.any():
            raise ValueError(
                "availability has to be a share between 0 and 1 (1 = plugged in at "
                f"home, 0 = away); found {values[invalid][0]} at "
                f"{availability.index[invalid.argmax()]}."
            )

        consumption = driving_consumption.to_numpy(dtype=float)
        negative = consumption < 0
        if negative.any():
            raise ValueError(
                "driving_consumption must not be negative; found "
                f"{consumption[negative][0]} at {driving_consumption.index[negative.argmax()]}."
            )
        while_home = (values == 1.0) & (consumption > 0)
        if while_home.any():
            raise ValueError(
                "driving_consumption has to be 0 while the car is fully available "
                "(availability == 1, plugged in at home for the whole timestep); "
                f"found {consumption[while_home][0]} kW at "
                f"{driving_consumption.index[while_home.argmax()]}."
            )

        self.availability = availability.astype(float).copy()
        self.driving_consumption = driving_consumption.astype(float).copy()

    @property
    def v2h_enabled(self) -> bool:
        return self.max_discharge_kw > 0

    def __repr__(self) -> str:
        return (
            f"ElectricVehicle(id={self.id!r}, capacity_kwh={self.capacity_kwh}, "
            f"max_charge_kw={self.max_charge_kw}, max_discharge_kw={self.max_discharge_kw})"
        )
