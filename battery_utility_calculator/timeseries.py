# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT

"""Shared validation of the timeseries indices.

All inputs of the optimizer have to sit on the same time axis. Checking that
with ``Index.equals`` compares the dtype as well, so the same instants stored at
different datetime resolutions - ``datetime64[ns]`` against ``datetime64[us]``,
which is what you get when one series comes from ``pd.date_range`` and another
from a file or a database - are reported as different on pandas 2.x. Comparing
the values instead keeps the check on what actually matters.
"""

from collections.abc import Collection

import numpy as np
import pandas as pd


def indices_match(first: pd.Index, second: pd.Index) -> bool:
    """Whether both indices describe the same instants in the same order.

    What is compared are the moments themselves, so two representations of the
    same timeline match: a different datetime resolution, and also a different
    timezone as long as it denotes the same instants (UTC against a converted
    Europe/Berlin). What does not match is the same wall clock in two zones -
    those are different moments - or a naive index against an aware one, which
    cannot be placed on a common timeline at all.
    """
    if first.equals(second):
        return True
    if len(first) != len(second):
        return False
    try:
        # NaT never equals itself, so missing entries are matched separately to
        # stay in line with Index.equals
        same = (first == second) | (first.isna() & second.isna())
        return bool(same.all())
    except TypeError:
        # comparing tz-naive against tz-aware, which is genuinely ambiguous
        return False


def describe_index_mismatch(
    first: pd.Index,
    second: pd.Index,
    first_name: str,
    second_name: str,
) -> str:
    """Explain why two indices differ, so the caller can act on it."""
    lines = [
        f"Index of {second_name} does not match index of {first_name}.",
        f"  {first_name}: len={len(first)}, dtype={first.dtype}",
        f"  {second_name}: len={len(second)}, dtype={second.dtype}",
    ]

    # the timezone comes first: it is the more fundamental problem and stays the
    # actionable hint even when the lengths differ as well
    first_tz = getattr(first, "tz", None)
    second_tz = getattr(second, "tz", None)
    if first_tz != second_tz:
        lines.append(f"  timezones differ: {first_tz} vs {second_tz}")
        return "\n".join(lines)

    if len(first) != len(second):
        missing = first.difference(second)
        extra = second.difference(first)
        if len(missing):
            lines.append(f"  only in {first_name}: {list(missing[:3])}")
        if len(extra):
            lines.append(f"  only in {second_name}: {list(extra[:3])}")
        return "\n".join(lines)

    if set(first) == set(second):
        lines.append("  same timestamps in a different order - try sort_index()")
        return "\n".join(lines)

    differing = [
        position
        for position, (left, right) in enumerate(zip(first, second, strict=True))
        if left != right
    ]
    if differing:
        position = differing[0]
        lines.append(
            f"  first difference at position {position} of {len(first)}: "
            f"{first[position]} vs {second[position]}"
        )
    return "\n".join(lines)


def check_finite(series: pd.Series, name: str, allow_nan: bool = False) -> None:
    """Reject NaN and inf in an input timeseries.

    A NaN in a price series becomes a coefficient of the objective, and no solver
    reports that back: HiGHS answers "optimal" with an objective of nan, or hangs
    outright as soon as the storage variables are not fixed to zero by a volume of
    0. So the values are checked where the indices are checked.

    Args:
        series: The timeseries to check.
        name: Name of the series for the error message.
        allow_nan: Accept NaN and reject only inf. Meant for series in which a gap
            has a meaning of its own, such as community market prices, where NaN
            marks a timestep without trading. The caller then has to keep the NaN
            out of the model.
    """
    values = series.to_numpy(dtype=float)
    invalid = np.isinf(values) if allow_nan else ~np.isfinite(values)
    if invalid.any():
        kind = "infinite" if allow_nan else "NaN or infinite"
        raise ValueError(
            f"{name} contains {int(invalid.sum())} {kind} values, the first "
            f"at {series.index[invalid.argmax()]}. Fill or drop them before optimizing."
        )


def check_identical_indices(
    series_by_name: dict[str, pd.Series],
    allow_nan: Collection[str] = (),
) -> pd.Index:
    """Validate that every series shares one datetime index and return it.

    Args:
        series_by_name: Series keyed by the name to use in error messages. The
            first entry acts as the reference.
        allow_nan: Names of the series in which NaN is accepted, see
            ``check_finite``.

    Returns:
        The index of the first series.

    Raises:
        TypeError: If an index is not a ``pd.DatetimeIndex``.
        ValueError: If an index does not match the reference.
    """
    reference_name, reference = next(iter(series_by_name.items()))

    for name, series in series_by_name.items():
        if not isinstance(series.index, pd.DatetimeIndex):
            msg = f"Index of {name} has to be pd.DateTimeIndex!"
            raise TypeError(msg)
        if not indices_match(series.index, reference.index):
            msg = "All timeseries indices must be identical.\n" + (
                describe_index_mismatch(
                    reference.index, series.index, reference_name, name
                )
            )
            raise ValueError(msg)
        check_finite(series, name, allow_nan=name in allow_nan)

    return reference.index
