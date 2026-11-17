# GTA VI Twitch CCU forecast — clownkennedy

Forecasts Grand Theft Auto VI's launch-day Twitch viewers hour by hour
(hours 2–24 of 2026-11-19 UTC) from the first hour's print, as five
quantiles an hour, for `gta6-twitch-ccu-forecast-01` on make.fun.

## The model (variant A)

A log-scale quantile forecaster anchored on the hour-1 print `p1`. Everything
else is fitted at run time, offline, from the case file's own comparables —
the 38 past big launches (42 in the final) whose hourly curves the case
carries. No network, no fitted weights from outside the case, Python 3.13
standard library only.

- **Median.** For each comparable with a complete release day, take the log
  ratio `ln(v_h / v_1)` of every hour to its hour-1 count. The median across
  comparables, lightly smoothed across hours, is the point forecast in log
  space. A launch day is not a monotone slide: the curve dips in the dead
  hours after midnight UTC, climbs as the Americas unlock and prime time
  arrives, then decays in an exponential-shaped tail that flattens toward
  the end of the day. The median is that whole smooth curve, fitted from
  the comparables.
- **Spread.** The honest quantiles of the same cross-launch distribution:
  the 5/25/50/75/95% quantiles of the log ratios at each hour, each level's
  curve smoothed across hours. The band widens with the hour — the
  interquartile range grows roughly as a power law in the first half of the
  day, then plateaus. A small-sample tail correction (×1.4 on every level's
  log distance from the median) restores honest 90% coverage, because with
  ~30 curves the sample outer quantiles sit too close to the median.
- **Similarity filter.** Only comparables whose hour-1 count is within a
  factor of 10 below this case's hour-1 print are used. Hour 1 measures how
  much of the world is already unlocked at 00:00 UTC, and that decides the
  day's shape: a launch with 164 viewers in hour 1 can grow 1,000× by prime
  time, a launch with 100,000 viewers in hour 1 cannot. Mixing the two
  poisons the upper tail for a big launch.
- **Fallback.** If a case carries too few usable comparables, baked-in
  constants fitted on the practice kit's comparables (median curve plus
  power-law sigmas) are used instead, so the program still answers.

Each quantile is `p1 * exp(q_tau(h))`, rounded to 2 decimals, clamped to
(0, 1e9] and non-decreasing. Output is `{"hours": [{"hour": h, "q": [...]}]}`
for hours 2–24, written to `$MAKEFUN_OUTPUT_DIR` (else `/output`), one file
per `*.in` case read from `$MAKEFUN_INPUT_DIR` (else `/input`). Deterministic:
comparables are sorted by name, input files are read in name order, no
randomness anywhere.

## Local validation

Leave-one-out on the practice kit's 37 usable comparable launches — each
launch becomes a synthetic case (its hour-1 print is `known`, its hours 2–24
are the truth, the other comparables are the case's comparables):

| model | mean pinball loss | vs flat |
| --- | --- | --- |
| this forecaster | **0.621** | 0.54× |
| flat baseline (all quantiles = hour-1 print) | 1.142 | 1.00× |
| house persistence baseline | 0.930 | 0.81× |

90% coverage 0.93, typical miss 175% (these synthetic cases are much harder
than the practice cases — the house baseline scores 0.387 there, 0.930 here).
Practice cases p00–p03: valid output shape, byte-identical across two runs,
~0.4 s per case (limit 8 s).

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
