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

HOME_FIELD_ADV = 0.33           # Runs — Yale NCAA model confirmed 0.33
SIM_ITERATIONS = 100_000        # Monte Carlo iterations
D1_AVG_RPG = 6.5                # D1 average runs per game (baseline)
MIN_LAMBDA = 1.0                # Floor for projected runs
MAX_LAMBDA = 15.0               # Ceiling for projected runs

# Negative binomial overdispersion.
# In baseball, variance ≈ 2.1 × mean. Poisson assumes variance = mean.
VARIANCE_RATIO = 2.1
VARIANCE_RATIO_MIDWEEK = 2.3  # Midweek games: weaker pitching = wider distributions

COLD_REGION_KEYWORDS = {
    "michigan", "minnesota", "wisconsin", "iowa", "illinois", "indiana",
    "ohio", "nebraska", "penn", "pennsylvania", "new york", "rutgers",
    "connecticut", "uconn", "massachusetts", "umass", "maine", "vermont",
    "new hampshire", "rhode island", "montana", "north dakota", "south dakota",
    "wyoming", "colorado", "oregon", "washington", "purdue", "northwestern",
    "notre dame", "boston", "cornell", "columbia", "princeton", "yale",
    "harvard", "dartmouth", "brown", "maryland", "virginia tech", "west virginia",
    "pittsburgh", "cincinnati", "kansas", "missouri",
}


def _weather_total_adjustment(home_team_name, game_month):
    """Regional temperature proxy -- cold-weather parks depress scoring in early season."""
    if game_month is None or game_month >= 5:  # May+ = warm everywhere
        return 0.0

    home_lower = home_team_name.lower() if home_team_name else ""
    is_cold = any(kw in home_lower for kw in COLD_REGION_KEYWORDS)

    if not is_cold:
        return 0.0

    if game_month <= 3:   # March
        return -0.40
    elif game_month == 4:  # April
        return -0.20
    return 0.0

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

# Unit sizing based on Edge (model prob - fair implied prob).
# Edge is more honest than EV — it measures how "wrong" the line is
# without the payout multiplier inflating underdog bets.
# Edge tiers are calibrated to college baseball market inefficiency:
#   - 3-6% edge = slight lean, worth a small bet
#   - 6-10% = solid disagreement with market
#   - 10-15% = strong conviction
#   - 15-20% = very strong, line is clearly off
#   - 20%+ = max conviction (could also be model noise — cap at 3u)
UNIT_TIERS_EDGE = [
    (0.28, 3.0),   # 28%+ edge — extreme conviction, almost never happens
    (0.22, 2.5),   # 22-28% edge — very strong
    (0.17, 2.0),   # 17-22% edge — strong
    (0.12, 1.5),   # 12-17% edge — solid
    (0.07, 1.0),   # 7-12% edge — moderate
    (0.03, 0.5),   # 3-7% edge — small lean
]

# Target pick count (not a hard cap — just for the top-quarter filter)
TARGET_MIN_PICKS = 5
TARGET_MAX_PICKS = 18


# --- Run Projection ---

def _regress_to_mean(raw_value, league_avg, games_played=17, regression_k=None, prior=None):
    """Regress a raw stat toward the league average (or prior year value) based on sample size.

    With only 15-20 games early in a college season, raw stats are noisy.
    This blends observed performance with a prior (league average or prior year):
      regressed = (games * raw + k * anchor) / (games + k)

    If prior year data is available, use it as the anchor instead of league avg.
    This gives team-specific Bayesian priors: a team that scored 8 RPG last year
    is regressed toward 8 RPG, not the league-wide 6.5.

    Dynamic k: more aggressive regression early season, less late season.
    At 17 games: k = 18.2 → weight = 48%
    At 40 games: k = 9.0  → weight = 82%
    At 60 games: k = 8.0  → weight = 88%
    """
    if regression_k is None:
        # Dynamic: more aggressive early season, less late season
        regression_k = max(8, 25 - 0.4 * games_played)
    weight = games_played / (games_played + regression_k)
    anchor = prior if prior is not None else league_avg
    return weight * raw_value + (1 - weight) * anchor


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

    # Step 1: Regress stats toward league average (or prior year if available)
    # Early season (17 games): ~40% weight on raw stats
    # Late season (50 games): ~67% weight on raw stats
    # Prior year data provides team-specific anchors instead of league avg
    league_avg_rapg = league_avg_rpg  # RA/G ≈ RPG league-wide
    home_rpg = _regress_to_mean(home_stats["rpg"], league_avg_rpg, prior=home_stats.get("_prior_rpg"))
    home_rapg = _regress_to_mean(home_stats["ra_pg"], league_avg_rapg, prior=home_stats.get("_prior_rapg"))
    away_rpg = _regress_to_mean(away_stats["rpg"], league_avg_rpg, prior=away_stats.get("_prior_rpg"))
    away_rapg = _regress_to_mean(away_stats["ra_pg"], league_avg_rapg, prior=away_stats.get("_prior_rapg"))

    # Step 2: SOS adjustment (65% weight)
    # SOS is critical — a team with 10 RPG against a .28 SOS (weak schedule)
    # is NOT the same as 10 RPG against a .55 SOS (SEC schedule).
    # 65% weight is aggressive but necessary — without strong SOS correction,
    # weak-conference pretenders with inflated stats pollute the picks.
    # The quality opponent adjustment in Step 5 further reinforces this.
    SOS_WEIGHT = 0.65
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
    # ERA is kept as a RELATIVE differential signal (see Step 3b below).
    # WHIP is a better signal for team pitching quality relative to opponents.
    home_whip = home_stats.get("whip")
    away_whip = away_stats.get("whip")

    if home_whip is not None and away_whip is not None:
        avg_whip = (home_whip + away_whip) / 2.0
        if avg_whip > 0:
            # 10% weight: lower WHIP = fewer baserunners = fewer runs allowed
            home_rapg *= (1.0 + 0.10 * (home_whip / avg_whip - 1.0))
            away_rapg *= (1.0 + 0.10 * (away_whip / avg_whip - 1.0))

    # Step 3b: RA9 differential adjustment (or ERA fallback)
    # RA9 (runs allowed per 9 IP) captures unearned runs that ERA misses.
    # College baseball has 2x MLB's error rate, so RA9 is more accurate.
    # Fall back to ERA if RA9 is not available.
    RA9_DIFF_WEIGHT = 0.015  # 1.5% per run of gap
    home_ra9 = home_stats.get("ra9") or home_stats.get("era")
    away_ra9 = away_stats.get("ra9") or away_stats.get("era")

    if home_ra9 is not None and away_ra9 is not None:
        ra9_diff = away_ra9 - home_ra9  # positive = home has better (lower) RA9
        # Home pitches better → away scores less; away pitches better → home scores less
        away_rpg *= (1.0 - RA9_DIFF_WEIGHT * max(-4, min(4, ra9_diff)))
        home_rpg *= (1.0 + RA9_DIFF_WEIGHT * max(-4, min(4, ra9_diff)))

    # Step 3c: JOPS differential — explains 90.9% of runs scored variance
    # JOPS (3*OBP + SLG) is far more predictive than BA for college baseball.
    home_jops = home_stats.get("jops")
    away_jops = away_stats.get("jops")
    if home_jops is not None and away_jops is not None:
        jops_diff = home_jops - away_jops
        # Scale: each 0.1 JOPS difference adjusts RPG by ~2%
        JOPS_WEIGHT = 0.20  # 2% per 0.1 JOPS gap
        jops_diff = max(-0.3, min(0.3, jops_diff))
        home_rpg *= (1.0 + JOPS_WEIGHT * jops_diff)
        away_rpg *= (1.0 - JOPS_WEIGHT * jops_diff)

    # Step 3d: Fielding adjustment — college has 2x MLB errors, this matters
    home_errors = home_stats.get("errors_pg")
    away_errors = away_stats.get("errors_pg")
    if home_errors is not None and away_errors is not None:
        # More errors = more unearned runs allowed
        # Each extra error per game adds ~0.3 runs to opponent's scoring
        ERROR_WEIGHT = 0.30
        avg_errors = (home_errors + away_errors) / 2.0
        if avg_errors > 0:
            home_rapg *= (1.0 + ERROR_WEIGHT * (home_errors / avg_errors - 1.0) * 0.15)
            away_rapg *= (1.0 + ERROR_WEIGHT * (away_errors / avg_errors - 1.0) * 0.15)

    # Step 3e: K/BB ratio — high strikeout-to-walk ratio = better run prevention
    home_kbb = home_stats.get("kbb_ratio")
    away_kbb = away_stats.get("kbb_ratio")
    if home_kbb is not None and away_kbb is not None:
        avg_kbb = (home_kbb + away_kbb) / 2.0
        if avg_kbb > 0:
            KBB_WEIGHT = 0.05
            home_rapg *= (1.0 - KBB_WEIGHT * (home_kbb / avg_kbb - 1.0))
            away_rapg *= (1.0 - KBB_WEIGHT * (away_kbb / avg_kbb - 1.0))

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

    # Step 5: Quality opponent adjustment
    # This is THE key differentiator between real teams and pretenders.
    # How a team performs vs quality opponents (>.500 teams) tells you
    # if their stats are earned against real competition or inflated
    # by beating up on weak teams.
    #
    # A team with +5 quality margin = dominates good teams = elite
    # A team with -10 quality margin = gets destroyed by real competition = pretender
    #
    # This adjustment is intentionally HEAVY — it's the primary mechanism
    # for catching schedule-inflated stats. A 3% per run adjustment means:
    #   +5 quality margin → +15% RPG boost (earned against real teams)
    #   -8 quality margin → -24% RPG penalty (stats are fool's gold)
    QUALITY_WEIGHT = 0.03  # 3% per run of quality margin
    MIN_QUAL_GAMES = 2     # Need at least 2 quality games

    home_qm = home_stats.get("_qual_margin", 0)
    home_qg = home_stats.get("_qual_games", 0)
    away_qm = away_stats.get("_qual_margin", 0) if isinstance(away_stats.get("_qual_margin"), (int, float)) else 0
    away_qg = away_stats.get("_qual_games", 0) if isinstance(away_stats.get("_qual_games"), (int, float)) else 0

    if home_qg >= MIN_QUAL_GAMES:
        reliability = min(home_qg, 8) / 8.0
        # Boost RPG for teams that dominate quality opponents
        home_rpg *= (1.0 + QUALITY_WEIGHT * home_qm * reliability)
        # Also adjust defense — teams that dominate good teams also pitch well against them
        home_rapg *= (1.0 - QUALITY_WEIGHT * home_qm * reliability * 0.5)

    if away_qg >= MIN_QUAL_GAMES:
        reliability = min(away_qg, 8) / 8.0
        away_rpg *= (1.0 + QUALITY_WEIGHT * away_qm * reliability)
        away_rapg *= (1.0 - QUALITY_WEIGHT * away_qm * reliability * 0.5)

    # Step 5b: Recency-weighted margin adjustment
    # Season-long stats treat game 1 and game 20 equally. But recent form
    # matters — a team that scored +8 margin over the last 5 games is
    # performing differently than their season line suggests.
    # The recency margin (exponential decay, lambda=0.95) captures this.
    # Apply as a small RPG modifier: each +1 run of recent margin adjusts by ~1.5%.
    RECENCY_WEIGHT = 0.015  # 1.5% per run of recent margin
    MAX_RECENCY_ADJ = 0.08  # Cap at +/- 8% adjustment

    home_recent_margin = home_stats.get("_recent_margin")
    away_recent_margin = away_stats.get("_recent_margin")

    if home_recent_margin is not None:
        adj = RECENCY_WEIGHT * max(-5, min(5, home_recent_margin))
        adj = max(-MAX_RECENCY_ADJ, min(MAX_RECENCY_ADJ, adj))
        home_rpg *= (1.0 + adj)

    if away_recent_margin is not None:
        adj = RECENCY_WEIGHT * max(-5, min(5, away_recent_margin))
        adj = max(-MAX_RECENCY_ADJ, min(MAX_RECENCY_ADJ, adj))
        away_rpg *= (1.0 + adj)

    # Step 5c: Overall winning percentage adjustment
    # A team with .800 wpct is genuinely better than one with .400.
    # This is a blunt signal (doesn't account for SOS) but complements
    # the quality margin adjustment above. Applied as a small modifier
    # centered on .500 — each .100 of wpct above/below .500 adjusts by ~2%.
    WPCT_WEIGHT = 0.20  # 2% per .100 wpct deviation from .500
    home_wpct = home_stats.get("_overall_wpct")
    away_wpct = away_stats.get("_overall_wpct")

    if home_wpct is not None and away_wpct is not None:
        home_wpct_adj = 1.0 + WPCT_WEIGHT * (home_wpct - 0.500)
        away_wpct_adj = 1.0 + WPCT_WEIGHT * (away_wpct - 0.500)
        # Cap the adjustment at +/- 6% (wpct .800 or .200)
        home_wpct_adj = max(0.94, min(1.06, home_wpct_adj))
        away_wpct_adj = max(0.94, min(1.06, away_wpct_adj))
        home_rpg *= home_wpct_adj
        away_rpg *= away_wpct_adj

    # Step 6: Log5 matchup projection
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
        # Scale: each +1 run of common opponent edge adjusts projection by ~0.25 runs
        # More common opponents = more weight (capped at 6)
        reliability = min(co_count, 6) / 6.0
        CO_WEIGHT = 0.25
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

def _nb_params(mean_runs, variance_ratio=None):
    """Calculate Negative Binomial parameters from projected runs."""
    if variance_ratio is None:
        variance_ratio = VARIANCE_RATIO
    r = mean_runs / (variance_ratio - 1)
    p = r / (r + mean_runs)
    return r, p


def monte_carlo_nb(lambda_home, lambda_away, n=SIM_ITERATIONS, variance_ratio=None):
    """Run Negative Binomial Monte Carlo simulation.

    variance_ratio: override the default VARIANCE_RATIO (e.g. for midweek games).
    """
    lambda_home = float(np.clip(lambda_home, MIN_LAMBDA, MAX_LAMBDA))
    lambda_away = float(np.clip(lambda_away, MIN_LAMBDA, MAX_LAMBDA))

    r_h, p_h = _nb_params(lambda_home, variance_ratio)
    r_a, p_a = _nb_params(lambda_away, variance_ratio)

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


def edge_to_units(edge_pct):
    """Convert Edge (model prob - fair implied prob) to unit size.

    Edge is more stable than EV for sizing because it's not inflated
    by underdog payout multipliers. A 10% edge means the same thing
    whether it's on a -110 spread or a +300 ML.
    """
    if edge_pct is None or edge_pct <= 0:
        return 0
    for min_edge, units in UNIT_TIERS_EDGE:
        if edge_pct >= min_edge:
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
        # O/U gets LOWER model confidence than ML/RL because books set totals
        # carefully and the model's total projections have known systematic bias.
        # Trust model 45% on totals vs 65-70% on ML/RL.
        OU_MODEL_CONFIDENCE = 0.45
        conf = OU_MODEL_CONFIDENCE
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

        # Midweek vs weekend detection — midweek games have weaker pitching
        # on both sides (teams save aces for Fri-Sun conference series).
        # This means more variance in outcomes, modeled via higher NB variance ratio.
        is_midweek = False
        try:
            game_date = datetime.now()  # default to today
            # Try parsing from game_time (ISO format from ESPN)
            if raw_time:
                game_date = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
            day_of_week = game_date.weekday()  # 0=Monday, 6=Sunday
            is_midweek = day_of_week in (0, 1, 2, 3)  # Mon-Thu
        except Exception:
            is_midweek = False

        has_stats = home_stats is not None and away_stats is not None
        has_lines = (run_line is not None or total is not None or
                     (home_ml is not None and away_ml is not None))

        if not has_stats and not has_lines:
            continue

        # Project runs — inject common opponent data into home_stats for the projection
        league_avg_sos = game.get("league_avg_sos")
        if has_stats:
            # Inject matchup-specific data into stats dicts for projection
            home_stats_ext = dict(home_stats)
            home_stats_ext["_co_edge"] = game.get("co_edge", 0)
            home_stats_ext["_co_count"] = game.get("co_count", 0)
            home_stats_ext["_qual_margin"] = game.get("home_qual_margin", 0)
            home_stats_ext["_qual_games"] = game.get("home_qual_games", 0)
            home_stats_ext["_overall_wpct"] = game.get("home_overall_wpct")

            away_stats_ext = dict(away_stats)
            away_stats_ext["_qual_margin"] = game.get("away_qual_margin", 0)
            away_stats_ext["_qual_games"] = game.get("away_qual_games", 0)
            away_stats_ext["_overall_wpct"] = game.get("away_overall_wpct")

            # Inject recency-weighted margin for each team
            home_stats_ext["_recent_margin"] = game.get("home_recent_margin")
            away_stats_ext["_recent_margin"] = game.get("away_recent_margin")

            # Inject prior year stats for Bayesian regression anchors
            home_stats_ext["_prior_rpg"] = game.get("home_prior_rpg")
            home_stats_ext["_prior_rapg"] = game.get("home_prior_rapg")
            away_stats_ext["_prior_rpg"] = game.get("away_prior_rpg")
            away_stats_ext["_prior_rapg"] = game.get("away_prior_rapg")

            # Starting pitcher: adjust RA/G before projection
            if game.get("home_starter_era") is not None and game.get("home_team_era"):
                adjusted_rapg = adjust_for_starter(
                    home_stats_ext["ra_pg"], game["home_starter_era"], game["home_team_era"],
                    starter_ip=game.get("home_starter_ip"), is_weekend=not is_midweek
                )
                home_stats_ext["ra_pg"] = adjusted_rapg

            if game.get("away_starter_era") is not None and game.get("away_team_era"):
                adjusted_rapg = adjust_for_starter(
                    away_stats_ext["ra_pg"], game["away_starter_era"], game["away_team_era"],
                    starter_ip=game.get("away_starter_ip"), is_weekend=not is_midweek
                )
                away_stats_ext["ra_pg"] = adjusted_rapg

            proj_home, proj_away = project_runs(
                home_stats_ext, away_stats_ext, league_avg_rpg, league_avg_sos, neutral
            )

        elif run_line is not None and total is not None:
            proj_home, proj_away = project_runs_fallback(run_line, total)
        else:
            continue

        # Weather adjustment: cold regions depress totals early season
        game_month = None
        try:
            if raw_time:
                game_month = datetime.fromisoformat(raw_time.replace("Z", "+00:00")).month
            else:
                game_month = datetime.now().month
        except Exception:
            game_month = datetime.now().month

        weather_adj = _weather_total_adjustment(home, game_month)
        if weather_adj != 0 and (proj_home + proj_away) > 0:
            ratio = (proj_home + proj_away + weather_adj) / (proj_home + proj_away)
            proj_home *= max(0.5, ratio)
            proj_away *= max(0.5, ratio)

        # Park factor — hitter/pitcher friendly home environment
        park_factor = game.get("park_factor", 1.0)
        if park_factor != 1.0:
            effective_pf = 1.0 + 0.50 * (park_factor - 1.0)  # 50% dampening
            proj_home *= effective_pf
            proj_away *= effective_pf

        # Negative Binomial Monte Carlo
        # Midweek games use higher variance ratio (weaker pitching = more unpredictable)
        vr = VARIANCE_RATIO_MIDWEEK if is_midweek else None
        home_scores, away_scores = monte_carlo_nb(proj_home, proj_away, variance_ratio=vr)

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
                    game_time=game_time_str, pick_team=pick_team,
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
                    pick_team=pick_team,
                ))

    # --- Deduplicate correlated picks ---
    # Remove same-side ML + RL combos (keep higher EV), enforce max 2 per game
    all_picks = _deduplicate_correlated_picks(all_picks)

    # --- Dynamic daily threshold ---
    all_picks = _apply_daily_threshold(all_picks)

    # Sort by date then time (earliest games first)
    all_picks.sort(key=lambda p: (p.get("Date", ""), p.get("Time", "99:99")))

    return all_picks


def _make_pick(game_label, bet_type, pick_str, edge_data, sportsbook,
               edges, proj_spread, proj_total, odds_override=None,
               game_time="", pick_team=None):
    """Build a pick dict with EV score and continuous unit sizing."""
    ev = edge_data["ev"]
    odds_display = odds_override or str(edge_data.get("odds", "-110"))
    implied = ml_to_implied_prob(edge_data.get("odds", -110)) or 0.524
    model_prob = (edge_data["sim_pct"] or 0) / 100.0
    edge_pct = model_prob - implied
    units = edge_to_units(edge_pct)

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
        "Proj Score": f"{max(0, edges['proj_away']):.1f} - {max(0, edges['proj_home']):.1f}",
        "Proj Spread": f"{proj_spread:+.1f}",
        "Proj Total": f"{proj_total:.1f}",
        "Result": "Pending",
        "_ev": ev,
        "_pick_team": pick_team,  # team name for ML/RL, None for O/U
    }


def _deduplicate_correlated_picks(picks):
    """Remove correlated same-side bets within a game.

    Rules:
    1. ML and Run Line on the SAME SIDE (same team) are correlated —
       if the team wins, both hit; if they lose, both miss. This doubles
       exposure without diversification. Keep only the higher EV one.
    2. Over/Under is INDEPENDENT from ML/RL — it's fine to have
       Under + Team ML in the same game.
    3. Max 2 picks per game (one side bet + one total bet).
    """
    from collections import defaultdict

    # Group picks by game
    by_game = defaultdict(list)
    for pick in picks:
        by_game[pick["Game"]].append(pick)

    deduped = []
    for game, game_picks in by_game.items():
        # Separate into side bets (ML, RL) and total bets (O/U)
        side_bets = [p for p in game_picks if p["Bet Type"] in ("Moneyline", "Run Line")]
        total_bets = [p for p in game_picks if p["Bet Type"] == "Over/Under"]

        # Deduplicate correlated side bets:
        # If ML and RL are on the SAME team, keep only the higher EV one.
        # If they're on different teams (rare but possible), both are fine.
        if len(side_bets) == 2:
            ml_pick = next((p for p in side_bets if p["Bet Type"] == "Moneyline"), None)
            rl_pick = next((p for p in side_bets if p["Bet Type"] == "Run Line"), None)
            if ml_pick and rl_pick and ml_pick["_pick_team"] == rl_pick["_pick_team"]:
                # Same team on both — correlated. Keep higher EV.
                if ml_pick.get("_ev", 0) >= rl_pick.get("_ev", 0):
                    side_bets = [ml_pick]
                else:
                    side_bets = [rl_pick]

        # Enforce max 2 picks per game (best side bet + best total bet)
        # Sort each group by EV descending and take the best
        side_bets.sort(key=lambda p: p.get("_ev", 0), reverse=True)
        total_bets.sort(key=lambda p: p.get("_ev", 0), reverse=True)

        kept = []
        if side_bets:
            kept.append(side_bets[0])
        if total_bets:
            tb = total_bets[0]
            # If there's already a side bet in this game, reduce the O/U units
            # by 30% to cap total game exposure (research-backed)
            if side_bets:
                current_units = float(tb["Unit Size"].replace("u", ""))
                reduced = max(0.5, round(current_units * 0.7, 1))
                tb["Unit Size"] = f"{reduced}u"
            kept.append(tb)

        deduped.extend(kept)

    return deduped


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
