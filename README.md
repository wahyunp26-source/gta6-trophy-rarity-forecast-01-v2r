[![Bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?ref=amzcd6wq)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wahyunp26-source/gta6-trophy-rarity-forecast-01-v2r/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?ref=amzcd6wq)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wahyunp26-source/gta6-trophy-rarity-forecast-01-v2r/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?ref=amzcd6wq)

# GTA VI Trophy Rarity Forecast — counts-first, refresh-aware forecaster

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the
percent of Exophase-tracked players who will have earned it at
2026-11-26T00:00:00Z, as five quantiles per trophy, for the
[make.fun competition](https://make.fun/c/gta6-trophy-rarity-forecast-01).

## The model

1. **Counts-first.** Every earn rate is inverted to integer earners
   (`count = round(rate x players / 100)`); earners and players are
   modelled as two separate series and the forecast rate is
   earners / players. On the practice cases the inverted counts sum
   to the page's `trophies_earned` (943) exactly.
2. **Refresh = cohort event, not trend.** Reads that repeat the page
   verbatim are one snapshot (a stale page, not zero growth). The
   next refresh is projected from the launch-night refresh in the
   competition rules (~4.5 h cadence; players ~6.0x, non-100% earners
   ~1.31x), refined by the most recent jump the case's own history
   shows, when it shows one.
3. **Trophy 2 ("Just Like Paradise", 100%) never dilutes** — every
   new player earns it, so its median stays 99.99 with the lower
   quantiles hedged to 99.9.
4. **Median first, then spread** (~0.15 logit on the practice:
   base 0.10 + small-count term + horizon term), wider over the
   final's 4.5-day horizon.
5. **Safety rail**: a rate may rise at most ~4.5x or fall at most
   ~20x in one projection.
6. **No fixed dilution factor** — every ratio is re-fit from the
   case's own history; zero-count trophies are seeded, never left
   at zero. Continued growth between refreshes is fitted per trophy
   (recency-weighted log-growth) and damped with tau = 72 h.

## Practice scores (exact pinball, logit scale, vs reconstructed answers)

| hypothesis | loss | pay line |
|---|---|---|
| H1 — players ~3,000, trophy 2 = 99.9% | 0.0483 | 0.0632 |
| H4 — earners x1.338, players 3,868, trophy 2 = 100% | 0.0372 | 0.0632 |
| H5 — earners x1.29, players 3,868, trophy 2 = 99.9% | 0.0483 | 0.0632 |
| H7 — players ~3,000, trophy 2 = 100% | 0.0400 | 0.0632 |
| H3 — p00 /2,700, p01 /3,868 (known weak spot) | 0.1216 | 0.0632 |

## Run it

Python 3.13, standard library only — no network, no installs:

```sh
python3 -m venv --without-pip /tmp/v2r-venv
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out \
  /tmp/v2r-venv/bin/python -P -s main.py
```

Reads every `*.in` in the input directory and writes one `<id>.out`
per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.
Deterministic: two runs write byte-identical outputs, ~0.03 s/case.
