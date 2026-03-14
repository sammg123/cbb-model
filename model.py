"""NCAA baseball prediction engine — Negative Binomial Monte Carlo simulation.

Scoring system:
- Each pick gets a Score = EV% (expected value per dollar wagered)
- EV naturally accounts for vig, probability, and payout:
  EV = model_prob × payout_if_win − (1 − model_prob)
- Heavy favorite MLs with bad vig score LOWER than spread bets with
  the same probability edge, because the vig eats the edge
- But if the model sees a massive probability gap on any bet type,
  the score goes up proportionally — that's an exploitable line
- Unit sizing scales continuously with the score
- Daily threshold: only output the top quarter of picks (5-18 range)
"""

import numpy as np
from scipy.stats import nbinom
import pandas as pd
from datetime import datetime

# --- Constants ---

HOME_FIELD_ADV = 0.50           # Runs — college HFA is ~2x MLB's 0.25
SIM_ITERATIONS = 100_000        # Monte Carlo iterations
D1_AVG_RPG = 6.5                # D1 average runs per game (baseline)
MIN_LAMBDA = 1.0                # Floor for projected runs
MAX_LAMBDA = 15.0               # Ceiling for projected runs

# Negative binomial overdispersion.
# In baseball, variance ≈ 2.1 × mean. Poisson assumes variance = mean.
VARIANCE_RATIO = 2.1

# Starting pitcher adjustment
STARTER_IP_PCT_WEEKEND = 0.60
STARTER_IP_PCT_MIDWEEK = 0.45
STARTER_REGRESSION_IP = 40.0

# Market regression: blend model probability toward FAIR (de-vigged) market probability.
# Confidence scales with odds extremity:
# - Close lines (-110 spreads): model can disagree more (market is softer)
# - Extreme lines (+500 dogs, -500 favs): model should agree more (harder to be wrong)
# This naturally nerfs heavy ML bets without an artificial vig penalty.
MODEL_CONFIDENCE_BASE = 0.70    # Trust model 70% on even-money lines
MODEL_CONFIDENCE_MIN = 0.45     # Trust model 45% on extreme lines (+500/-500)

# Minimum EV% to show a pick — this is the "is it worth gambling on?" threshold.
# In college baseball (inefficient markets), 3-5% EV is where real value starts.
# Below that, the edge is likely noise. Above it, you're exploiting a lazy line.
MIN_EV_THRESHOLD = 0.03  # 3% EV minimum to show

# Unit sizing tiers based on EV% — straightforward, not too conservative.
# Higher EV = more confident the line is wrong = bet more.
UNIT_TIERS = [
    (0.15, 3.0),   # 15%+ EV: max confidence, line is very wrong
    (0.10, 2.5),   # 10-15% EV: strong value
    (0.07, 2.0),   # 7-10% EV: solid value
    (0.05, 1.5),   # 5-7% EV: moderate value
    (0.03, 1.0),   # 3-5% EV: minimum playable value
]

# Target pick count (not a hard cap — just for the top-quarter filter)
TARGET_MIN_PICKS = 5
TARGET_MAX_PICKS = 18


# --- Run Projection ---

def _regress_to_mean(raw_value, league_avg, games_played=17, regression_k=15):
    """Regress a raw stat toward the league average based on sample size.

    With only 15-20 games early in a college season, raw stats are noisy.
    This blends observed performance with a prior (league average):
      regressed = (games * raw + k * avg) / (games + k)

    regression_k=15 gives ~53% weight on raw stats at 17 games,
    increasing to ~77% at 50 games. This preserves the model's ability
    to differentiate between high-scoring and low-scoring games
    while still dampening extreme outliers.
    """
    weight = games_played / (games_played + regression_k)
    return weight * raw_value + (1 - weight) * league_avg


def project_runs(home_stats, away_stats, league_avg_rpg=None,
                 league_avg_sos=None, neutral=False):
    """Project game runs using log5-style matchup formula.

    Pipeline:
    1. Regress raw stats toward league average (handles small sample noise)
    2. Apply dampened SOS adjustment (30% weight — log5 already handles opponent quality)
    3. Blend in ERA/WHIP for pitching quality signal
    4. Log5 matchup formula
    5. Home field advantage
    """
    if league_avg_rpg is None or league_avg_rpg <= 0:
        league_avg_rpg = D1_AVG_RPG

    # Step 1: Regress stats toward league average
    # Early season (17 games): ~40% weight on raw stats
    # Late season (50 games): ~67% weight on raw stats
    league_avg_rapg = league_avg_rpg  # RA/G ≈ RPG league-wide
    home_rpg = _regress_to_mean(home_stats["rpg"], league_avg_rpg)
    home_rapg = _regress_to_mean(home_stats["ra_pg"], league_avg_rapg)
    away_rpg = _regress_to_mean(away_stats["rpg"], league_avg_rpg)
    away_rapg = _regress_to_mean(away_stats["ra_pg"], league_avg_rapg)

    # Step 2: SOS adjustment (50% weight)
    # SOS is critical — a team with 10 RPG against a .28 SOS (weak schedule)
    # is NOT the same as 10 RPG against a .55 SOS (SEC schedule).
    # We apply SOS only to raw stats BEFORE log5, not inside it.
    # 50% dampening balances: strong enough to catch schedule inflation,
    # not so strong that it double-counts with log5's opponent adjustment.
    SOS_WEIGHT = 0.50
    if league_avg_sos and league_avg_sos > 0:
        home_sos = home_stats.get("sos") or league_avg_sos
        away_sos = away_stats.get("sos") or league_avg_sos

        home_sos_raw = home_sos / league_avg_sos
        away_sos_raw = away_sos / league_avg_sos

        home_sos_factor = 1.0 + SOS_WEIGHT * (home_sos_raw - 1.0)
        away_sos_factor = 1.0 + SOS_WEIGHT * (away_sos_raw - 1.0)

        home_rpg *= home_sos_factor
        away_rpg *= away_sos_factor
        home_rapg /= home_sos_factor
        away_rapg /= away_sos_factor

    # Step 3: WHIP pitching quality refinement
    # NOTE: ERA blend was removed — ERA < RA/G always (excludes unearned runs),
    # so blending it in created a systematic Under bias on every game.
    # ERA is kept in the data for individual pitcher adjustments only.
    # WHIP is a better signal for team pitching quality relative to opponents.
    home_whip = home_stats.get("whip")
    away_whip = away_stats.get("whip")

    if home_whip is not None and away_whip is not None:
        avg_whip = (home_whip + away_whip) / 2.0
        if avg_whip > 0:
            # 10% weight: lower WHIP = fewer baserunners = fewer runs allowed
            home_rapg *= (1.0 + 0.10 * (home_whip / avg_whip - 1.0))
            away_rapg *= (1.0 + 0.10 * (away_whip / avg_whip - 1.0))

    # Step 4: Momentum adjustment
    # Teams on hot streaks perform ~2-3% better, cold streaks ~2-3% worse.
    # A 5-game win streak adds ~0.15 runs to offense, subtracts ~0.15 from runs allowed.
    # A 5-game losing streak does the opposite.
    # Scoring margin reinforces this: a team winning by 5+/game is dominant,
    # not just lucky. A team losing by 5+/game is collapsing.
    MOMENTUM_WEIGHT = 0.03  # 3% per streak game (capped)
    MAX_MOMENTUM_GAMES = 6  # Cap at 6 games of streak

    home_momentum = home_stats.get("momentum", 0)
    away_momentum = away_stats.get("momentum", 0)
    home_margin = home_stats.get("margin", 0)
    away_margin = away_stats.get("margin", 0)

    # Streak factor: +6 streak → +18% boost, -6 streak → -18% penalty
    home_streak_factor = 1.0 + MOMENTUM_WEIGHT * max(-MAX_MOMENTUM_GAMES, min(MAX_MOMENTUM_GAMES, home_momentum))
    away_streak_factor = 1.0 + MOMENTUM_WEIGHT * max(-MAX_MOMENTUM_GAMES, min(MAX_MOMENTUM_GAMES, away_momentum))

    # Margin factor: scoring margin is a strong signal of true team quality.
    # A team with +5 margin/game is genuinely dominant. +0 is average. -5 is bad.
    # Scale: each +1 margin adds ~1.5% to offense quality
    MARGIN_WEIGHT = 0.015
    home_margin_factor = 1.0 + MARGIN_WEIGHT * max(-6, min(6, home_margin))
    away_margin_factor = 1.0 + MARGIN_WEIGHT * max(-6, min(6, away_margin))

    # Apply momentum + margin to offense (hot team scores more, cold team scores less)
    home_rpg *= home_streak_factor * home_margin_factor
    away_rpg *= away_streak_factor * away_margin_factor

    # Step 5: Log5 matchup projection
    proj_home = home_rpg * (away_rapg / league_avg_rpg)
    proj_away = away_rpg * (home_rapg / league_avg_rpg)

    # Step 6: Common opponent adjustment
    # If both teams played the same opponents, compare how they did.
    # This is one of the strongest signals in college sports — direct comparison
    # through shared opponents cuts through schedule noise.
    # co_edge > 0 means home team did better vs common opponents.
    # Weight increases with number of common opponents (more data = more reliable).
    co_edge = home_stats.get("_co_edge", 0)
    co_count = home_stats.get("_co_count", 0)
    if co_count >= 2 and co_edge != 0:
        # Scale: each +1 run of common opponent edge adjusts projection by ~0.15 runs
        # More common opponents = more weight (capped at 5)
        reliability = min(co_count, 5) / 5.0  # 0.4 at 2 common, 1.0 at 5+
        CO_WEIGHT = 0.15
        co_adj = co_edge * CO_WEIGHT * reliability
        proj_home += co_adj / 2.0
        proj_away -= co_adj / 2.0

    # Step 7: Home field advantage
    if not neutral:
        proj_home += HOME_FIELD_ADV / 2.0
        proj_away -= HOME_FIELD_ADV / 2.0

    proj_home = float(np.clip(proj_home, MIN_LAMBDA, MAX_LAMBDA))
    proj_away = float(np.clip(proj_away, MIN_LAMBDA, MAX_LAMBDA))

    return proj_home, proj_away


def project_runs_fallback(spread, total):
    """Project runs from the betting line when no team stats are available."""
    proj_home = (total / 2.0) - (spread / 2.0)
    proj_away = (total / 2.0) + (spread / 2.0)
    return max(proj_home, MIN_LAMBDA), max(proj_away, MIN_LAMBDA)


def regress_pitcher_era(pitcher_era, pitcher_ip, team_era):
    """Regress pitcher ERA toward team average for small samples."""
    if pitcher_ip is None or pitcher_ip <= 0:
        return team_era
    weight = pitcher_ip / (pitcher_ip + STARTER_REGRESSION_IP)
    return weight * pitcher_era + (1 - weight) * team_era


def adjust_for_starter(team_rapg, starter_era, team_era, starter_ip=None,
                       is_weekend=True):
    """Adjust team runs-allowed projection based on starting pitcher."""
    if starter_era is None or team_era is None or team_era <= 0:
        return team_rapg

    regressed_era = regress_pitcher_era(starter_era, starter_ip, team_era)
    starter_pct = STARTER_IP_PCT_WEEKEND if is_weekend else STARTER_IP_PCT_MIDWEEK
    bullpen_era = max(0.5, (team_era - regressed_era * starter_pct) / (1 - starter_pct))
    game_era = regressed_era * starter_pct + bullpen_era * (1 - starter_pct)

    return team_rapg * (game_era / team_era)


# --- Monte Carlo Simulation ---

def _nb_params(mean_runs):
    """Calculate Negative Binomial parameters from projected runs."""
    r = mean_runs / (VARIANCE_RATIO - 1)
    p = r / (r + mean_runs)
    return r, p


def monte_carlo_nb(lambda_home, lambda_away, n=SIM_ITERATIONS):
    """Run Negative Binomial Monte Carlo simulation."""
    lambda_home = float(np.clip(lambda_home, MIN_LAMBDA, MAX_LAMBDA))
    lambda_away = float(np.clip(lambda_away, MIN_LAMBDA, MAX_LAMBDA))

    r_h, p_h = _nb_params(lambda_home)
    r_a, p_a = _nb_params(lambda_away)

    rng = np.random.default_rng()
    home_scores = rng.negative_binomial(r_h, p_h, n)
    away_scores = rng.negative_binomial(r_a, p_a, n)

    return home_scores, away_scores


# --- Odds & EV Calculation ---

def ml_to_implied_prob(ml):
    """Convert American moneyline to implied probability (with vig)."""
    if ml is None or ml == 0:
        return None
    if ml < 0:
        return abs(ml) / (abs(ml) + 100.0)
    else:
        return 100.0 / (ml + 100.0)


def fair_probabilities(home_ml, away_ml):
    """Get vig-removed fair probabilities from moneylines (multiplicative devig)."""
    home_imp = ml_to_implied_prob(home_ml)
    away_imp = ml_to_implied_prob(away_ml)
    if home_imp is None or away_imp is None:
        return None, None
    vig_total = home_imp + away_imp
    return home_imp / vig_total, away_imp / vig_total


def _model_confidence_for_odds(american_odds):
    """Scale model confidence based on odds extremity.

    Close to even (±100-150): trust model more (70%)
    Moderate (±200-300): trust model moderately (60%)
    Extreme (±500+): trust model less (45%)

    This is because extreme lines are less likely to be significantly
    wrong — the model needs to be more certain to override them.
    """
    if american_odds is None:
        return MODEL_CONFIDENCE_BASE

    # Convert to absolute implied probability distance from 50%
    imp = ml_to_implied_prob(american_odds)
    if imp is None:
        return MODEL_CONFIDENCE_BASE

    # How far from 50/50 — ranges from 0 (even money) to ~0.4 (extreme)
    extremity = abs(imp - 0.5)

    # Linear interpolation: 0 extremity → BASE, 0.35 extremity → MIN
    t = min(1.0, extremity / 0.35)
    return MODEL_CONFIDENCE_BASE - t * (MODEL_CONFIDENCE_BASE - MODEL_CONFIDENCE_MIN)


def calculate_ev(true_prob, american_odds):
    """Calculate Expected Value per $1 wagered.

    This is THE scoring metric. It naturally handles:
    - Vig differences: ML at -300 has worse payout than spread at -110
    - Probability: higher model confidence → higher EV
    - Edge: bigger gap between model and implied prob → higher EV

    A -300 ML where model says 80% (implied 75%): EV = 6.7%
    A -110 spread where model says 62% (implied 52%): EV = 18.3%
    → Spread scores higher because edge is bigger AND vig is better

    But -300 ML where model says 95% (implied 75%): EV = 26.7%
    → Scores huge because the line is massively wrong
    """
    if american_odds is None or true_prob is None or true_prob <= 0:
        return 0.0
    if american_odds > 0:
        profit_if_win = american_odds / 100.0
    else:
        profit_if_win = 100.0 / abs(american_odds)
    return true_prob * profit_if_win - (1 - true_prob)


def ev_to_units(ev, american_odds=-110):
    """Convert EV score to unit size based on EV tiers.

    Higher EV = line is more wrong = bet more units.
    Simple and transparent.
    """
    if ev is None or ev <= 0:
        return 0
    for min_ev, units in UNIT_TIERS:
        if ev >= min_ev:
            return units
    return 0


# --- Edge Calculation ---

def calculate_edges(home_scores, away_scores, run_line, total_line,
                    home_ml, away_ml, has_stats, **kwargs):
    """Calculate EV-based scores for all bet types using actual posted odds."""
    h = np.asarray(home_scores, dtype=np.float64)
    a = np.asarray(away_scores, dtype=np.float64)

    margins = h - a
    totals = h + a

    home_win_pct = float(np.mean(margins > 0) * 100)
    tie_pct = float(np.mean(margins == 0) * 100)
    if tie_pct > 0 and home_win_pct > 0:
        raw_away = float(np.mean(margins < 0) * 100)
        total_decided = home_win_pct + raw_away
        if total_decided > 0:
            home_win_pct = (home_win_pct / total_decided) * 100
    away_win_pct = 100.0 - home_win_pct

    # --- Run Line ---
    home_rl_odds = kwargs.get("home_rl_odds", -110)
    away_rl_odds = kwargs.get("away_rl_odds", -110)
    rl = {"ev": None, "sim_pct": None, "side": None, "odds": None}
    if run_line is not None:
        home_cover_raw = float(np.mean(margins > -run_line) * 100)
        # Devig the run line odds to get fair probability
        home_fair, away_fair = fair_probabilities(home_rl_odds, away_rl_odds)
        fair_home = (home_fair or 0.5) * 100
        conf = _model_confidence_for_odds(home_rl_odds)
        home_cover_pct = conf * home_cover_raw + (1 - conf) * fair_home
        away_cover_pct = 100.0 - home_cover_pct
        home_ev = calculate_ev(home_cover_pct / 100.0, home_rl_odds)
        away_ev = calculate_ev(away_cover_pct / 100.0, away_rl_odds)
        if home_ev >= away_ev:
            rl = {"ev": home_ev, "sim_pct": home_cover_pct, "side": "home", "odds": home_rl_odds}
        else:
            rl = {"ev": away_ev, "sim_pct": away_cover_pct, "side": "away", "odds": away_rl_odds}

    # --- Over/Under ---
    over_odds = kwargs.get("over_odds", -110)
    under_odds = kwargs.get("under_odds", -110)
    ou = {"ev": None, "sim_pct": None, "side": None, "odds": None}
    if total_line is not None:
        over_raw = float(np.mean(totals > total_line) * 100)
        over_fair, under_fair = fair_probabilities(over_odds, under_odds)
        fair_over = (over_fair or 0.5) * 100
        conf = _model_confidence_for_odds(over_odds)
        over_pct = conf * over_raw + (1 - conf) * fair_over
        under_pct = 100.0 - over_pct
        over_ev = calculate_ev(over_pct / 100.0, over_odds)
        under_ev = calculate_ev(under_pct / 100.0, under_odds)
        if over_ev >= under_ev:
            ou = {"ev": over_ev, "sim_pct": over_pct, "side": "Over", "odds": over_odds}
        else:
            ou = {"ev": under_ev, "sim_pct": under_pct, "side": "Under", "odds": under_odds}

    # --- Moneyline ---
    ml = {"ev": None, "sim_pct": None, "side": None, "odds": None}
    if has_stats and home_ml and away_ml:
        # Devig to get fair probabilities (remove vig/hold)
        home_fair, away_fair = fair_probabilities(home_ml, away_ml)
        if home_fair is not None:
            home_fair_pct = home_fair * 100
            away_fair_pct = away_fair * 100

            # Scale confidence by odds extremity — extreme lines get more market trust
            home_conf = _model_confidence_for_odds(home_ml)
            away_conf = _model_confidence_for_odds(away_ml)

            home_blended = home_conf * home_win_pct + (1 - home_conf) * home_fair_pct
            away_blended = away_conf * away_win_pct + (1 - away_conf) * away_fair_pct
            # Normalize to sum to 100
            total_blend = home_blended + away_blended
            home_blended = home_blended / total_blend * 100
            away_blended = away_blended / total_blend * 100

            home_ev = calculate_ev(home_blended / 100.0, home_ml)
            away_ev = calculate_ev(away_blended / 100.0, away_ml)
            if home_ev >= away_ev:
                ml = {"ev": home_ev, "sim_pct": home_blended, "side": "home", "odds": home_ml}
            else:
                ml = {"ev": away_ev, "sim_pct": away_blended, "side": "away", "odds": away_ml}

    return {
        "rl": rl,
        "ou": ou,
        "ml": ml,
        "proj_home": float(np.mean(h)),
        "proj_away": float(np.mean(a)),
        "proj_total": float(np.mean(totals)),
        "proj_spread": float(-np.mean(margins)),
    }


# --- Pick Generation ---

def generate_picks(games, league_avg_rpg=None):
    """Generate picks for all games.

    Scoring: each pick gets Score = EV% (expected value per dollar wagered).
    Unit sizing: continuous, scaled from Score.
    Daily threshold: dynamic — only output the top quarter of qualifying picks,
    targeting 5-18 picks per day.
    """
    all_picks = []

    for game in games:
        home = game["home"]
        away = game["away"]
        neutral = game.get("neutral", False)
        run_line = game.get("spread")
        total = game.get("total")
        home_ml = game.get("home_ml")
        away_ml = game.get("away_ml")
        sportsbook = game.get("sportsbook", "")
        home_stats = game.get("home_stats")
        away_stats = game.get("away_stats")

        # Parse game time → Chicago time
        game_time_str = ""
        raw_time = game.get("game_time", "")
        if raw_time:
            try:
                from zoneinfo import ZoneInfo
                utc_dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                chi_dt = utc_dt.astimezone(ZoneInfo("America/Chicago"))
                game_time_str = chi_dt.strftime("%-I:%M %p")
            except Exception:
                game_time_str = ""

        has_stats = home_stats is not None and away_stats is not None
        has_lines = (run_line is not None or total is not None or
                     (home_ml is not None and away_ml is not None))

        if not has_stats and not has_lines:
            continue

        # Project runs — inject common opponent data into home_stats for the projection
        league_avg_sos = game.get("league_avg_sos")
        if has_stats:
            # Temporarily add common opponent edge to home_stats so project_runs can use it
            home_stats_ext = dict(home_stats)
            home_stats_ext["_co_edge"] = game.get("co_edge", 0)
            home_stats_ext["_co_count"] = game.get("co_count", 0)
            proj_home, proj_away = project_runs(
                home_stats_ext, away_stats, league_avg_rpg, league_avg_sos, neutral
            )
        elif run_line is not None and total is not None:
            proj_home, proj_away = project_runs_fallback(run_line, total)
        else:
            continue

        # Negative Binomial Monte Carlo
        home_scores, away_scores = monte_carlo_nb(proj_home, proj_away)

        edges = calculate_edges(
            home_scores, away_scores, run_line, total, home_ml, away_ml, has_stats,
            over_odds=game.get("over_odds", -110),
            under_odds=game.get("under_odds", -110),
            home_rl_odds=game.get("home_rl_odds", -110),
            away_rl_odds=game.get("away_rl_odds", -110),
        )

        game_label = f"{away} vs {home}"
        proj_total = edges["proj_total"]
        proj_spread = edges["proj_spread"]

        # --- Run Line Pick ---
        if run_line is not None:
            rl = edges["rl"]
            if rl["ev"] is not None and rl["ev"] >= MIN_EV_THRESHOLD:
                pick_team = home if rl["side"] == "home" else away
                pick_line = run_line if rl["side"] == "home" else -run_line
                all_picks.append(_make_pick(
                    game_label, "Run Line", f"{pick_team} {pick_line:+.1f}",
                    rl, sportsbook, edges, proj_spread, proj_total,
                    game_time=game_time_str,
                ))

        # --- Over/Under Pick ---
        if total is not None:
            ou = edges["ou"]
            if ou["ev"] is not None and ou["ev"] >= MIN_EV_THRESHOLD:
                all_picks.append(_make_pick(
                    game_label, "Over/Under", f"{ou['side']} {total}",
                    ou, sportsbook, edges, proj_spread, proj_total,
                    game_time=game_time_str,
                ))

        # --- Moneyline Pick ---
        if home_ml is not None and away_ml is not None:
            ml = edges["ml"]
            if ml["ev"] is not None and ml["ev"] >= MIN_EV_THRESHOLD:
                pick_team = home if ml["side"] == "home" else away
                pick_ml_odds = home_ml if ml["side"] == "home" else away_ml
                all_picks.append(_make_pick(
                    game_label, "Moneyline", f"{pick_team} ML",
                    ml, sportsbook, edges, proj_spread, proj_total,
                    odds_override=f"{pick_ml_odds:+d}", game_time=game_time_str,
                ))

    # --- Dynamic daily threshold ---
    all_picks = _apply_daily_threshold(all_picks)

    return all_picks


def _make_pick(game_label, bet_type, pick_str, edge_data, sportsbook,
               edges, proj_spread, proj_total, odds_override=None, game_time=""):
    """Build a pick dict with EV score and continuous unit sizing."""
    ev = edge_data["ev"]
    bet_odds = edge_data.get("odds", -110) or -110
    units = ev_to_units(ev, bet_odds)
    odds_display = odds_override or str(edge_data.get("odds", "-110"))
    implied = ml_to_implied_prob(edge_data.get("odds", -110)) or 0.524
    model_prob = (edge_data["sim_pct"] or 0) / 100.0

    return {
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Time": game_time,
        "Game": game_label,
        "Bet Type": bet_type,
        "Pick": pick_str,
        "Odds": odds_display,
        "Sportsbook": sportsbook,
        "Unit Size": f"{units}u",
        "Score": f"{ev * 100:.1f}%",
        "Sim %": f"{edge_data['sim_pct']:.1f}%",
        "Edge": f"{(model_prob - implied) * 100:.1f}%",
        "EV+": f"${ev * 100:+.2f}",
        "Proj Score": f"{edges['proj_away']:.1f}-{edges['proj_home']:.1f}",
        "Proj Spread": f"{proj_spread:+.1f}",
        "Proj Total": f"{proj_total:.1f}",
        "Result": "Pending",
        "_ev": ev,
    }


def _apply_daily_threshold(picks):
    """Dynamic daily threshold: take the top quarter of picks.

    1. Sort all picks by EV score (descending)
    2. Take the top 25% — these are the most exploitable lines
    3. Clamp to 5-18 range
    4. If fewer than 5 qualify at 1.5% EV floor, output whatever qualifies

    This adapts to the day's slate: on days with lots of value, you
    get more picks. On tight-line days, fewer picks.
    """
    if not picks:
        return []

    # Sort by EV descending
    picks.sort(key=lambda x: x.get("_ev", 0), reverse=True)

    # Take top quarter
    quarter = max(TARGET_MIN_PICKS, len(picks) // 4)
    threshold_count = min(TARGET_MAX_PICKS, quarter)

    # But also enforce minimum EV
    qualifying = [p for p in picks if p.get("_ev", 0) >= MIN_EV_THRESHOLD]

    if len(qualifying) <= TARGET_MIN_PICKS:
        return qualifying

    # Take top quarter of qualifying picks
    target = max(TARGET_MIN_PICKS, len(qualifying) // 4)
    target = min(TARGET_MAX_PICKS, target)

    return qualifying[:target]


def picks_to_dataframe(picks):
    """Convert picks list to a clean DataFrame for the spreadsheet."""
    if not picks:
        return pd.DataFrame(columns=[
            "Date", "Game", "Bet Type", "Pick", "Odds",
            "Sportsbook", "Unit Size", "Score", "Result"
        ])

    df = pd.DataFrame(picks)
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    return df
