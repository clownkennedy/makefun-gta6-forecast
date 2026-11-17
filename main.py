"""gta6-twitch-ccu-forecast-01: log-scale quantile forecaster, variant A.

The forecast is anchored on the release day's first-hour print p1.  Everything
else is learned at run time from the case's own comparables: the 38 (42 in the
final) past big launches whose hourly curves the case file carries.

Median: for each comparable with a complete release day, take the log ratio
ln(v_h / v_1) of every hour h = 2..24 to its hour-1 count.  The median of
those log ratios across comparables is the point forecast in log space; a
three-point [1, 2, 1] smoothing keeps the curve smooth.  A launch day is not
a monotone slide: the curve dips in the dead hours after midnight UTC, climbs
as the Americas unlock and prime time arrives, and then decays in an
exponential-shaped tail that flattens toward the end of the day.  The median
is that whole smooth curve, fitted from the comparables, not an imposed shape.

Spread: the honest quantiles of the same cross-launch distribution.  At each
hour the 5%, 25%, 50%, 75% and 95% quantiles of the log ratios across
comparables are taken (linear interpolation) and each level's curve is
smoothed across hours the same way as the median.  The band widens with the
hour — the interquartile range grows roughly as a power law in the first half
of the day, then plateaus — because later hours genuinely are less
predictable from hour 1.  A parametric power-law sigma fit was tried and
rejected in local validation: it miscalibrates the upper tail (too narrow
mid-day, when a few launches explode) and scores worse in leave-one-out
testing (0.779 vs 0.713 pinball loss).  The empirical band is the honest one.
Each quantile is p1 * exp(q_tau(h)).

One filter matters: only comparables whose hour-1 count is within a factor
of 10 below this case's hour-1 print are used (and never below 100 viewers).
Hour 1 measures how much of the world is already unlocked at 00:00 UTC, and
that decides the day's shape: a launch with 164 viewers in hour 1 had barely
unlocked anywhere and can grow 1,000x by prime time, while a launch with
100,000 viewers in hour 1 cannot.  Mixing the two poisons the upper tail for
a big launch.  In leave-one-out testing the filter improves the loss from
0.713 to 0.642, and it is flat across a wide range of the factor (5x to 50x),
so it is not tuned on a knife edge.  If the filter ever leaves too few
comparables, it relaxes to every comparable above 100 viewers.

If a case carries too few usable comparables, baked-in constants fitted on
the practice kit's comparables are used instead, so the program still answers.

Python 3.13 standard library only, no network, deterministic: comparables
are sorted by name, input files are read in name order, and no randomness is
used anywhere.  Run as `python3 -P -s /bot/main.py`; reads
$MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.  This entry is a single file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# A comparable only counts if its release day is complete and its hour-1
# count is big enough that its ratios are not dominated by small-number noise.
MIN_COMPARABLES = 5
MIN_HOUR1_VIEWERS = 100
# Comparables must also be within this factor below the case's own hour-1
# print: hour 1 measures how much of the world is unlocked at 00:00 UTC,
# and that decides the day's shape.
REL_HOUR1_FACTOR = 10.0
# Small-sample tail correction applied to every level's log distance from
# the median (see fit_from_comparables).
SPREAD_WIDEN = 1.4

# Fallback constants, fitted on the practice kit's comparables (smoothed
# median log ratios for hours 2..24, and power-law sigma fits of the form
# sigma(h) = a * (h - 1) ** b, one below and one above the median).  Used only
# when a case carries fewer than MIN_COMPARABLES usable comparables.
FALLBACK_MEDIAN = [
    -0.186, -0.253, -0.299, -0.283, -0.247, -0.196, -0.140, -0.113, -0.096,
    -0.066, -0.015, 0.065, 0.202, 0.376, 0.498, 0.578, 0.696, 0.821, 0.862,
    0.847, 0.823, 0.776, 0.735,
]
FALLBACK_SIGMA_LO = (0.2460, 0.4565)  # (a, b) of sigma(h) = a * (h - 1) ** b
FALLBACK_SIGMA_HI = (0.1262, 0.9134)


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def quantile(sorted_vals: list[float], q: float) -> float:
    """The q-quantile of a sorted list, by linear interpolation."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def smooth3(xs: list[float]) -> list[float]:
    """Three-point [1, 2, 1] / 4 smoothing; endpoints use [2, 1] / 3."""
    out = []
    n = len(xs)
    for i in range(n):
        if i == 0:
            out.append((2.0 * xs[0] + xs[1]) / 3.0)
        elif i == n - 1:
            out.append((xs[n - 2] + 2.0 * xs[n - 1]) / 3.0)
        else:
            out.append((xs[i - 1] + 2.0 * xs[i] + xs[i + 1]) / 4.0)
    return out


def release_day_log_ratios(comp: dict) -> tuple[float, list[float]] | None:
    """(hour-1 viewers, ln(v_h / v_1) for h = 2..24) of a release day.

    A comparable counts only when all 24 release-day hours are present, its
    hour-1 count clears MIN_HOUR1_VIEWERS, and every hour is positive.
    """
    try:
        t0 = datetime.fromisoformat(comp["day_start"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None
    hours: dict[int, float] = {}
    for row in comp.get("hourly") or []:
        try:
            t = datetime.fromisoformat(row["from"].replace("Z", "+00:00"))
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None:
            continue
        dh = (t - t0).total_seconds() / 3600.0
        if 0 <= dh < 24 and dh == int(dh):
            hours[int(dh) + 1] = float(viewers)
    if len(hours) != 24:
        return None
    v1 = hours[1]
    if v1 < MIN_HOUR1_VIEWERS:
        return None
    if any(v <= 0 for v in hours.values()):
        return None
    return v1, [math.log(hours[h] / v1) for h in range(2, 25)]


def fit_from_comparables(comparables: list[dict], p1: float):
    """Smoothed log-ratio quantiles fitted from the case's comparables.

    Returns a list of five curves (one per level in LEVELS), each 23 log
    ratios long for hours 2..24, or None when too few comparables are usable.
    Only comparables within REL_HOUR1_FACTOR below this case's hour-1 print
    are used; if that leaves too few, every comparable above the floor is.
    """
    usable: list[tuple[float, list[float]]] = []
    for comp in sorted(comparables, key=lambda c: str(c.get("name", ""))):
        parsed = release_day_log_ratios(comp)
        if parsed is not None:
            usable.append(parsed)
    near = [ratios for v1, ratios in usable if v1 >= p1 / REL_HOUR1_FACTOR]
    curves = near if len(near) >= MIN_COMPARABLES else [r for _, r in usable]
    if len(curves) < MIN_COMPARABLES:
        return None
    levels_out: list[list[float]] = []
    for tau in LEVELS:
        raw = [quantile(sorted(c[j] for c in curves), tau) for j in range(23)]
        levels_out.append(smooth3(raw))
    # Small-sample tail correction.  With ~30 curves the sample 5% and 95%
    # quantiles (2nd order statistics) sit systematically too close to the
    # median: leave-one-out coverage of the 90% band is 0.82, not 0.90.
    # Scaling every level's log distance from the median by SPREAD_WIDEN
    # restores honest coverage (0.93 in leave-one-out) and cuts the loss
    # from 0.642 to 0.621; the gain is flat across 1.3-1.5, so 1.4 is not a
    # knife-edge fit.
    median_curve = levels_out[2]
    return [
        [
            median_curve[j] + (levels_out[i][j] - median_curve[j]) * SPREAD_WIDEN
            for j in range(23)
        ]
        for i in range(len(LEVELS))
    ]


def hour_quantiles(p1: float, h: int, log_quantiles: list[list[float]]) -> list[float | int]:
    """The five quantiles for hour h, in log space around the anchor."""
    out: list[float | int] = []
    prev = 0.0
    for level_curve in log_quantiles:
        q = p1 * math.exp(level_curve[h - 2])
        q = min(1e9, max(0.01, q))
        q = round2(q)
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def first_hour_print(case: dict) -> float:
    """Hour 1's print, at least 1; falls back to the last history point."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers") is not None:
            return max(1.0, float(row["viewers"]))
    hourly = case.get("history_hourly") or []
    if hourly:
        last = hourly[-1].get("viewers")
        if last is not None:
            return max(1.0, float(last))
    return 1.0


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    p1 = first_hour_print(case)
    fitted = fit_from_comparables(case.get("comparables") or [], p1)
    if fitted is None:
        # Degenerate case: the baked-in median curve plus the power-law
        # spread (sigma(h) = a * (h - 1) ** b, split above/below the median).
        a_lo, b_lo = FALLBACK_SIGMA_LO
        a_hi, b_hi = FALLBACK_SIGMA_HI
        log_quantiles = []
        for z in Z:
            a, b = (a_hi, b_hi) if z > 0 else (a_lo, b_lo)
            log_quantiles.append(
                [
                    FALLBACK_MEDIAN[j]
                    + z * min(3.0, max(0.05, a * (j + 1) ** b))
                    for j in range(23)
                ]
            )
    else:
        log_quantiles = fitted
    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    return {
        "hours": [
            {"hour": row["hour"], "q": hour_quantiles(p1, row["hour"], log_quantiles)}
            for row in rows
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
