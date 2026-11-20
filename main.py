"""gta6-trophy-rarity-forecast-01 entry: counts-first, refresh-aware trophy forecaster.

Model (written fresh for this entry, standard library only):
  1. Counts-first. Every rate is inverted to integer earners
     (count = round(rate * players / 100)); earners E(t) and players P(t)
     are modelled as two separate series; the forecast rate is E / P.
     Verified on the practice cases: the inverted counts sum to the
     page's trophies_earned (943) exactly.
  2. Refresh = cohort event, not trend. Reads that repeat the page
     verbatim are one snapshot (a stale page, not zero growth).
     Snapshots are segmented from the history; the jump between
     consecutive snapshots is a refresh. The next refresh is projected
     from the launch-night refresh documented in the competition rules
     (first probe to first refresh ~4.5 h; players 638 -> 3868, ~x6.0;
     non-100% earners x1.31), refined by the most recent jump the
     case's own history actually shows, when it shows one.
  3. Trophy 2 ("Just Like Paradise", 100%) never dilutes: every new
     player earns it, so its median stays 99.99 with the lower
     quantiles hedged to 99.9.
  4. Median first, then spread: sigma ~ 0.15 logit on the practice
     (base 0.10 + a small-count term + a horizon term), wider over the
     final's 4.5-day horizon.
  5. Safety rail: a rate may rise at most ~4.5x or fall at most ~20x
     in one projection, so an exponential extrapolation can never
     saturate a trophy at 100%.
  6. No fixed dilution factor: every ratio is re-fit from the case's
     own history (snapshot segmentation, most-recent jump, recency
     fit of the continued growth); the launch-night numbers are only
     the prior when the history shows a single snapshot.
  7. Zero-count trophies are seeded, never left at zero: a trophy
     nobody holds yet gets (0.9 + 0.2 * earliness) earners per 1000
     new players, earliness from its points (bronze 15 -> 1.0,
     silver 30 -> 0.6, gold 90 -> 0.3, platinum 300 -> 0.1).

Growth between refreshes is fitted per trophy from the log-growth
between consecutive snapshots (recency-weighted) and damped with
tau = 72 h: the log-growth over h hours is r * tau * (1 - exp(-h/tau)).

Usage (as the grader runs it): python3 -P -s main.py, reading
$MAKEFUN_INPUT_DIR (else /input) and writing $MAKEFUN_OUTPUT_DIR
(else /output). One *.in file per case, one <id>.out per case:
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- Documented launch-night refresh (competition.yaml / RULES.txt) ---
# First probe 2026-11-18T23:50Z showed 638 players / 943 trophies earned;
# the page was unchanged through 04:00Z, then refreshed by 04:30Z to
# 3868 players / 4276 trophies earned. Non-100% earners went 305 -> 408.
# The model uses the rounded ratios (players ~6.0x, earners ~1.31x) as
# its prior for the next refresh; the exact ratios are 3868/638 and
# 408/305.
REFRESH_INTERVAL_H = 4.5
PLAYERS_RATIO_PRIOR = 6.0
EARNERS_RATIO_PRIOR = 1.31

# Growth damping time constant (hours) and the safety rail.
TAU_H = 72.0
RAIL_UP = 4.5
RAIL_DOWN = 0.05

# Spread: base + small-count term + horizon term (logit units).
SIGMA_BASE = 0.10
SIGMA_COUNT = 0.08
SIGMA_TIME = 0.08

TROPHY2_ID = 2
TROPHY2_MEDIAN = 99.99
TROPHY2_LOW = 99.9


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (matches the house baseline's JS rounding)."""
    v = math.floor(x * 10000.0 + 0.5) / 10000.0
    return int(v) if float(v).is_integer() else v


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def earliness(points):
    """How early a trophy can be earned, from its PSN points."""
    if points <= 15:
        return 1.0
    if points <= 30:
        return 0.6
    if points <= 90:
        return 0.3
    return 0.1


def segment_snapshots(history):
    """Group consecutive identical reads into snapshots.

    A snapshot is one stale page: same players_tracked and same rates.
    Returns a list of dicts: start/end datetimes, players, rates, earned.
    """
    snapshots = []
    for read in history:
        rates = tuple(read.get("rates") or [])
        key = (read.get("players_tracked"), rates)
        if snapshots and snapshots[-1]["key"] == key:
            snapshots[-1]["end"] = parse_ts(read["at"])
            snapshots[-1]["last"] = read
        else:
            snapshots.append({
                "key": key,
                "start": parse_ts(read["at"]),
                "end": parse_ts(read["at"]),
                "players": read.get("players_tracked"),
                "rates": list(read.get("rates") or []),
                "earned": read.get("trophies_earned"),
                "last": read,
            })
    return snapshots


def counts_of(read, n_trophies):
    """Integer earners per trophy for one read (None where the rate is null)."""
    players = read.get("players_tracked")
    rates = read.get("rates") or []
    out = []
    for i in range(n_trophies):
        rate = rates[i] if i < len(rates) else None
        if rate is None or players is None:
            out.append(None)
        else:
            out.append(round(rate * players / 100.0))
    return out


def last_observed_rates(history, n_trophies):
    """Per trophy, the last non-null rate seen (None if never observed)."""
    rates = [None] * n_trophies
    for read in history:
        rs = read.get("rates") or []
        for i in range(n_trophies):
            if i < len(rs) and rs[i] is not None:
                rates[i] = rs[i]
    return rates


def weighted_log_rate(pairs):
    """Recency-weighted mean log-growth per hour over snapshot pairs.

    pairs: list of (delta_hours, log_ratio), oldest first. Weight halves
    per step back in time (most recent pair weight 1).
    """
    if not pairs:
        return 0.0
    total_w = 0.0
    total = 0.0
    for k, (dt_h, lr) in enumerate(reversed(pairs)):
        w = 0.5 ** k
        total_w += w
        total += w * (lr / dt_h if dt_h > 0 else 0.0)
    return total / total_w if total_w > 0 else 0.0


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    history = case.get("history") or []
    horizon_h = case["horizon_minutes"] / 60.0
    checkpoint = parse_ts(case["at"])
    target = parse_ts(case["target"])

    # Fallback: no history at all -> a wide, neutral forecast.
    if not history:
        rows = []
        for t in trophies:
            lg = logit(1.0)
            rows.append({"id": t["id"],
                         "q": [round4(inv_logit(lg + z * 1.0)) for z in Z]})
        return {"trophies": rows}

    last = history[-1]
    p_last = last.get("players_tracked") or 0
    last_rates = last_observed_rates(history, n)
    last_counts = counts_of(last, n)
    # Fill counts the last read lacked from the last observed rate.
    for i in range(n):
        if last_counts[i] is None and last_rates[i] is not None and p_last:
            last_counts[i] = round(last_rates[i] * p_last / 100.0)

    snapshots = segment_snapshots(history)
    last_snap = snapshots[-1]

    # --- Refresh statistics: most recent observed jump, else the prior. ---
    pr_eff = PLAYERS_RATIO_PRIOR
    interval_eff = REFRESH_INTERVAL_H
    er_eff = [EARNERS_RATIO_PRIOR] * n
    if len(snapshots) >= 2:
        prev, cur = snapshots[-2], snapshots[-1]
        gap_h = (cur["start"] - prev["start"]).total_seconds() / 3600.0
        if gap_h > 0:
            interval_eff = gap_h
        if prev["players"] and cur["players"]:
            pr_eff = cur["players"] / prev["players"]
        pc = counts_of(prev["last"], n)
        cc = counts_of(cur["last"], n)
        for i in range(n):
            if pc[i] and pc[i] > 0 and cc[i] is not None:
                er_eff[i] = max(0.01, cc[i] / pc[i])

    # --- Continued-growth rates from between-snapshot log growth. ---
    # Per trophy and for players, recency-weighted over snapshot pairs.
    er_pairs = [[] for _ in range(n)]
    p_pairs = []
    for a, b in zip(snapshots, snapshots[1:]):
        dt_h = (b["start"] - a["start"]).total_seconds() / 3600.0
        if dt_h <= 0:
            continue
        if a["players"] and b["players"] and b["players"] > 0 and a["players"] > 0:
            p_pairs.append((dt_h, math.log(b["players"] / a["players"])))
        ca = counts_of(a["last"], n)
        cb = counts_of(b["last"], n)
        for i in range(n):
            if ca[i] is not None and cb[i] is not None and ca[i] > 0 and cb[i] > 0:
                er_pairs[i].append((dt_h, math.log(cb[i] / ca[i])))
    r_earn = []
    for i in range(n):
        r = weighted_log_rate(er_pairs[i])
        r_earn.append(max(-0.3, min(0.3, r)))
    r_players = max(-0.3, min(0.3, weighted_log_rate(p_pairs)))

    # --- How many refreshes fit between the snapshot's start and target? ---
    age_h = (checkpoint - last_snap["start"]).total_seconds() / 3600.0
    span_h = age_h + horizon_h
    n_refresh = int(span_h // interval_eff) if interval_eff > 0 else 0
    if n_refresh < 0:
        n_refresh = 0
    # First refresh time (an overdue refresh is due at the checkpoint).
    first_refresh = last_snap["start"].timestamp() + interval_eff * 3600.0
    first_refresh = max(first_refresh, checkpoint.timestamp())
    last_refresh = first_refresh + (n_refresh - 1) * interval_eff * 3600.0 \
        if n_refresh > 0 else checkpoint.timestamp()
    rem_h = max(0.0, (target.timestamp() - last_refresh) / 3600.0) \
        if n_refresh > 0 else horizon_h

    # Damped growth factor for `rem_h` hours at log-rate r.
    def damp(r, hours):
        return math.exp(r * TAU_H * (1.0 - math.exp(-hours / TAU_H))) \
            if hours > 0 else 1.0

    # --- Project players and earners. ---
    p_proj = float(p_last) if p_last else 1.0
    e_proj = []
    for i in range(n):
        base = last_counts[i] if last_counts[i] is not None else 0
        e_proj.append(float(base))
    for _ in range(n_refresh):
        p_proj *= pr_eff
        for i in range(n):
            if trophies[i]["id"] == TROPHY2_ID:
                continue
            e_proj[i] *= er_eff[i]
        # Trophy 2: every player holds it after a refresh.
        for i in range(n):
            if trophies[i]["id"] == TROPHY2_ID:
                e_proj[i] = p_proj
    p_proj *= damp(r_players, rem_h)
    for i in range(n):
        if trophies[i]["id"] == TROPHY2_ID:
            continue
        e_proj[i] *= damp(r_earn[i], rem_h)
    for i in range(n):
        if trophies[i]["id"] == TROPHY2_ID:
            e_proj[i] = p_proj

    # --- Attainment: seed zero-count trophies, never leave them at zero. ---
    p_new = max(0.0, p_proj - float(p_last if p_last else 0))
    for i in range(n):
        if (last_counts[i] or 0) == 0 and last_rates[i] in (None, 0):
            seed = (0.9 + 0.2 * earliness(trophies[i].get("points", 15))) \
                * max(1.0, p_new / 1000.0)
            e_proj[i] = max(e_proj[i], seed)

    # --- Rates, safety rail, quantiles. ---
    rows = []
    for i, t in enumerate(trophies):
        tid = t["id"]
        if tid == TROPHY2_ID:
            rows.append({"id": tid,
                         "q": [TROPHY2_LOW, TROPHY2_LOW, TROPHY2_MEDIAN,
                               TROPHY2_MEDIAN, TROPHY2_MEDIAN]})
            continue
        rate = (e_proj[i] / p_proj * 100.0) if p_proj > 0 else 0.0
        last_rate = last_rates[i]
        if last_rate is not None and last_rate > 0:
            ratio = rate / last_rate
            ratio = max(RAIL_DOWN, min(RAIL_UP, ratio))
            rate = last_rate * ratio
        rate = clamp(rate)
        e_last = last_counts[i] or 0
        sigma = SIGMA_BASE + SIGMA_COUNT / math.sqrt(e_last + 1.0) \
            + SIGMA_TIME * math.sqrt(horizon_h / 24.0)
        lg = logit(rate)
        qs = []
        prev_q = None
        for z in Z:
            q = round4(clamp(inv_logit(lg + z * sigma)))
            if prev_q is not None and q < prev_q:
                q = prev_q
            qs.append(q)
            prev_q = q
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main():
    in_dir = INPUT_DIR
    out_dir = OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for fn in names:
        with open(os.path.join(in_dir, fn), encoding="utf-8") as f:
            case = json.load(f)
        forecast = forecast_case(case)
        out_path = os.path.join(out_dir, fn[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
