"""Daily NCAA baseball picks pipeline: scrape → predict → write to Google Sheets."""

import os
import re
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from scraper import get_schedule, get_team_stats, get_odds, get_game_results, match_team, normalize_name, get_season_results, common_opponent_edge, get_pitcher_stats, get_starters_from_espn, get_team_eras, load_prior_year_stats, save_season_stats
from model import generate_picks, picks_to_dataframe
from sheets import write_picks, update_results_in_all_picks, COLUMNS


def _match_starter(starter_name, team_name, pitcher_db):
    """Match ESPN starter name to NCAA pitcher database."""
    if not starter_name:
        return None
    team_norm = normalize_name(team_name)
    # Exact match
    key = (team_norm, starter_name)
    if key in pitcher_db:
        return pitcher_db[key]
    # Last name fuzzy match
    last = starter_name.split()[-1].lower() if starter_name else ""
    for (t, name), stats in pitcher_db.items():
        if t.lower() == team_norm.lower() and name.split()[-1].lower() == last:
            return stats
    return None


def merge_game_data(games, team_stats, odds, season_results=None, quality_records=None,
                    recency_stats=None, starters=None, pitcher_db=None,
                    team_eras_db=None, prior_stats=None, park_factors=None):
    """Combine schedule, team stats, odds, common opponent, recency, and quality records into model-ready game dicts."""
    merged = []
    matched_count = 0
    no_stats_count = 0
    no_odds_count = 0

    # Calculate league averages from all teams
    if team_stats:
        all_rpg = [s["rpg"] for s in team_stats.values()]
        league_avg_rpg = sum(all_rpg) / len(all_rpg) if all_rpg else 6.5
        all_sos = [s["sos"] for s in team_stats.values() if s.get("sos")]
        league_avg_sos = sum(all_sos) / len(all_sos) if all_sos else None
    else:
        league_avg_rpg = 6.5
        league_avg_sos = None

    for game in games:
        if game.get("status") == "STATUS_FINAL":
            continue

        home = game["home"]
        away = game["away"]

        # Look up team stats
        home_stats = match_team(home, team_stats)
        away_stats = match_team(away, team_stats)

        if not home_stats or not away_stats:
            no_stats_count += 1

        # Look up odds
        home_norm = normalize_name(home)
        away_norm = normalize_name(away)

        # Find all book odds for this game
        all_book_odds = None
        swapped = False
        for (oh, oa), book_odds in odds.items():
            if _fuzzy(oh, home_norm) and _fuzzy(oa, away_norm):
                all_book_odds = book_odds
                matched_count += 1
                break
            elif _fuzzy(oa, home_norm) and _fuzzy(oh, away_norm):
                # Odds key has home/away reversed vs our game — need to swap MLs
                all_book_odds = book_odds
                swapped = True
                matched_count += 1
                break

        # If home/away were swapped in odds vs game, flip the ML values
        if swapped and all_book_odds:
            fixed_odds = {}
            for book_key, lines in all_book_odds.items():
                fixed = dict(lines)
                if "home_ml" in fixed or "away_ml" in fixed:
                    old_home = fixed.get("home_ml")
                    old_away = fixed.get("away_ml")
                    fixed["home_ml"] = old_away
                    fixed["away_ml"] = old_home
                if "spread" in fixed and fixed["spread"] is not None:
                    fixed["spread"] = -fixed["spread"]
                fixed_odds[book_key] = fixed
            all_book_odds = fixed_odds

        # Pick the best book: prefer full lines (spread + total), then ML-only.
        # Priority order: FanDuel -> DraftKings -> any book with full lines -> any book
        primary_odds = None
        primary_book = ""

        if all_book_odds:
            # Book priority order
            preferred = ["fanduel", "draftkings"]

            # First pass: find a preferred book with full lines
            for book_key in preferred:
                lines = all_book_odds.get(book_key, {})
                if lines.get("spread") is not None and lines.get("total") is not None:
                    primary_odds = lines
                    primary_book = book_key.replace("_", " ").title()
                    break

            # Second pass: any book with full lines
            if not primary_odds:
                for book_key, lines in all_book_odds.items():
                    if lines.get("spread") is not None and lines.get("total") is not None:
                        primary_odds = lines
                        primary_book = book_key.replace("_", " ").title()
                        break

            # Third pass: preferred book with ML
            if not primary_odds:
                for book_key in preferred:
                    lines = all_book_odds.get(book_key, {})
                    if lines.get("home_ml") is not None:
                        primary_odds = lines
                        primary_book = book_key.replace("_", " ").title()
                        break

            # Fourth pass: any book with anything
            if not primary_odds:
                for book_key, lines in all_book_odds.items():
                    if lines:
                        primary_odds = lines
                        primary_book = book_key.replace("_", " ").title()
                        break

        if not primary_odds:
            no_odds_count += 1

        spread = primary_odds.get("spread") if primary_odds else None
        total = primary_odds.get("total") if primary_odds else None
        home_ml = primary_odds.get("home_ml") if primary_odds else None
        away_ml = primary_odds.get("away_ml") if primary_odds else None
        # Actual juice per side (defaults to -110 if not available)
        over_odds = primary_odds.get("over_odds", -110) if primary_odds else -110
        under_odds = primary_odds.get("under_odds", -110) if primary_odds else -110
        home_rl_odds = primary_odds.get("home_rl_odds", -110) if primary_odds else -110
        away_rl_odds = primary_odds.get("away_rl_odds", -110) if primary_odds else -110

        # Sanity check: discard clearly broken ML data
        # Valid MLB/college lines: one side negative (fav), other positive (dog)
        # OR both negative for close games (e.g. -120/-110)
        # OR both positive for pick'em (e.g. +105/+105) — this is fine
        # Broken: both sides at same large value (e.g. +450/+450)
        if home_ml and away_ml:
            if home_ml > 0 and away_ml > 0 and abs(home_ml - away_ml) < 20 and home_ml > 200:
                home_ml = None
                away_ml = None

        # Common opponent analysis
        co_edge = 0
        co_count = 0
        if season_results:
            co_edge, co_count = common_opponent_edge(home_norm, away_norm, season_results)

        # Quality records — performance vs winning teams
        home_qual = quality_records.get(home_norm, {}) if quality_records else {}
        away_qual = quality_records.get(away_norm, {}) if quality_records else {}

        # Recency-weighted stats
        home_recency = recency_stats.get(home_norm, {}) if recency_stats else {}
        away_recency = recency_stats.get(away_norm, {}) if recency_stats else {}

        # Starting pitcher data (if available)
        home_starter_era = None
        home_starter_ip = None
        away_starter_era = None
        away_starter_ip = None
        home_team_era = None
        away_team_era = None

        espn_id = game.get("espn_id", "")
        if starters and pitcher_db and espn_id in starters:
            starter_info = starters[espn_id]
            hs = _match_starter(starter_info.get("home_starter"), home, pitcher_db)
            if hs:
                home_starter_era = hs.get("era")
                home_starter_ip = hs.get("ip")
                home_team_era = team_eras_db.get(normalize_name(home)) if team_eras_db else None

            aws = _match_starter(starter_info.get("away_starter"), away, pitcher_db)
            if aws:
                away_starter_era = aws.get("era")
                away_starter_ip = aws.get("ip")
                away_team_era = team_eras_db.get(normalize_name(away)) if team_eras_db else None

        # Prior year Bayesian priors
        home_prior = prior_stats.get(home_norm, {}) if prior_stats else {}
        away_prior = prior_stats.get(away_norm, {}) if prior_stats else {}

        # Park factor for home team
        home_pf = park_factors.get(home_norm, 1.0) if park_factors else 1.0

        merged.append({
            "home": home,
            "away": away,
            "neutral": game.get("neutral", False),
            "spread": spread,
            "total": total,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_stats": home_stats,
            "away_stats": away_stats,
            "sportsbook": primary_book,
            "league_avg_rpg": league_avg_rpg,
            "league_avg_sos": league_avg_sos,
            "game_time": game.get("game_time", ""),
            "co_edge": co_edge,
            "co_count": co_count,
            "home_qual_margin": home_qual.get("qual_margin", 0),
            "home_qual_games": home_qual.get("qual_games", 0),
            "away_qual_margin": away_qual.get("qual_margin", 0),
            "away_qual_games": away_qual.get("qual_games", 0),
            "home_overall_wpct": home_qual.get("overall_wpct"),
            "away_overall_wpct": away_qual.get("overall_wpct"),
            "over_odds": over_odds,
            "under_odds": under_odds,
            "home_rl_odds": home_rl_odds,
            "away_rl_odds": away_rl_odds,
            "home_recent_margin": home_recency.get("recent_margin"),
            "away_recent_margin": away_recency.get("recent_margin"),
            "home_starter_era": home_starter_era,
            "home_starter_ip": home_starter_ip,
            "away_starter_era": away_starter_era,
            "away_starter_ip": away_starter_ip,
            "home_team_era": home_team_era,
            "away_team_era": away_team_era,
            "home_prior_rpg": home_prior.get("rpg"),
            "home_prior_rapg": home_prior.get("ra_pg"),
            "away_prior_rpg": away_prior.get("rpg"),
            "away_prior_rapg": away_prior.get("ra_pg"),
            "park_factor": home_pf,
        })

        # One pick per game per bet type — no duplicates across books.
        # FanDuel → DraftKings → others priority is handled above.

    print(f"  Odds matched: {matched_count}/{len(games)} games")
    print(f"  Missing team stats: {no_stats_count} games")
    print(f"  Missing odds: {no_odds_count} games")

    return merged


def _build_quality_records(season_results):
    """Calculate each team's record and margin vs quality opponents (>.500 teams).

    Returns dict: {team_norm: {"qual_margin": float, "qual_games": int, "overall_wpct": float}}
    """
    import numpy as np

    # First pass: overall records
    team_records = {}
    for team, games in season_results.items():
        wins = sum(1 for _, margin, *_ in games if margin > 0)
        losses = sum(1 for _, margin, *_ in games if margin < 0)
        total = wins + losses
        team_records[team] = {"wpct": wins / total if total > 0 else 0.5, "games": total}

    # Second pass: record vs winning teams
    quality = {}
    for team, games in season_results.items():
        margins_vs_good = []
        for opp, margin, *_ in games:
            opp_rec = team_records.get(opp, {})
            if opp_rec.get("wpct", 0) > 0.500:
                margins_vs_good.append(margin)

        qual_margin = float(np.mean(margins_vs_good)) if margins_vs_good else 0
        quality[team] = {
            "qual_margin": qual_margin,
            "qual_games": len(margins_vs_good),
            "overall_wpct": team_records.get(team, {}).get("wpct", 0.5),
        }

    return quality


def _calculate_recency_stats(season_results):
    """Calculate recency-weighted offensive and defensive stats.

    Uses exponential decay: recent games matter more.
    Lambda=0.95 means each game back is worth 95% of the previous.

    Returns dict: {team_norm: {"recent_margin": float, "recent_games": int}}
    """
    import numpy as np

    recency = {}
    for team, games in season_results.items():
        if not games:
            continue
        # Sort by date descending (most recent first)
        sorted_games = sorted(games, key=lambda x: x[2], reverse=True)

        LAMBDA = 0.95
        weighted_margins = []
        weights = []
        for i, (opp, margin, date_str, *_) in enumerate(sorted_games):
            w = LAMBDA ** i
            weighted_margins.append(margin * w)
            weights.append(w)

        total_w = sum(weights)
        if total_w > 0:
            recency[team] = {
                "recent_margin": sum(weighted_margins) / total_w,
                "recent_games": len(sorted_games),
            }

    return recency


def _calculate_park_factors(season_results):
    """Calculate park factors from home vs away scoring.

    Park factor = avg total runs in home games / avg total runs in away games.
    >1.0 = hitter-friendly park, <1.0 = pitcher-friendly.
    """
    park_factors = {}
    for team, games in season_results.items():
        home_totals = []
        away_totals = []
        for entry in games:
            if len(entry) >= 4:
                opp, margin, date, is_home = entry[0], entry[1], entry[2], entry[3]
            else:
                continue
            # Total runs in game ~ |margin| + 2 * min_score, but we don't have individual scores
            # Instead, use margin as a proxy -- home margins vs away margins
            # Actually, just track average margin differential
            if is_home:
                home_totals.append(margin)
            else:
                away_totals.append(margin)

        if len(home_totals) >= 3 and len(away_totals) >= 3:
            home_avg = sum(home_totals) / len(home_totals)
            away_avg = sum(away_totals) / len(away_totals)
            # Park factor based on how much better team performs at home
            # Positive diff = team scores more at home relative to away
            margin_diff = home_avg - away_avg
            # Convert to a multiplicative factor centered at 1.0
            # Each +1 run margin at home ~ +3% park factor
            pf = 1.0 + 0.03 * margin_diff
            park_factors[team] = max(0.90, min(1.10, pf))

    return park_factors


def _log_picks_for_clv(picks):
    """Save today's picks with current odds for later CLV calculation.

    Writes to clv_log.json in the project directory. Each entry records
    the bet details and timestamp so closing odds can be compared later.
    """
    clv_file = os.path.join(os.path.dirname(__file__), "clv_log.json")
    try:
        with open(clv_file, "r") as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log = []

    for pick in picks:
        log.append({
            "date": pick.get("Date"),
            "game": pick.get("Game"),
            "bet_type": pick.get("Bet Type"),
            "pick": pick.get("Pick"),
            "bet_odds": pick.get("Odds"),
            "sportsbook": pick.get("Sportsbook"),
            "unit_size": pick.get("Unit Size"),
            "score": pick.get("Score"),
            "timestamp": datetime.now().isoformat(),
            "closing_odds": "",  # To be filled later
            "clv": "",           # To be calculated: closing_implied - bet_implied
        })

    with open(clv_file, "w") as f:
        json.dump(log, f, indent=2)

    print(f"  CLV log: saved {len(picks)} picks to {clv_file}")


def _fuzzy(a, b):
    """Quick fuzzy name match."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return True
    if a in b or b in a:
        return True
    return False


def _get_ncaa_results(date_str):
    """Get ALL D1 results from NCAA API (164+ games vs ESPN's 6).

    date_str: YYYY-MM-DD format
    Returns same format as get_game_results: {"Away vs Home": {"home": str, "away": str, "home_score": int, "away_score": int}}
    """
    import requests
    from scraper import normalize_name, NCAA_API_BASE

    url = f"{NCAA_API_BASE}/scoreboard/baseball/d1/{date_str.replace('-', '/')}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return {}
        data = resp.json()
    except Exception:
        return {}

    results = {}
    for g in data.get("games", []):
        game = g.get("game", {})
        if game.get("gameState") != "final":
            continue

        home_info = game.get("home", {})
        away_info = game.get("away", {})
        home_name = home_info.get("names", {}).get("short", "")
        away_name = away_info.get("names", {}).get("short", "")

        try:
            home_score = int(home_info.get("score", 0))
            away_score = int(away_info.get("score", 0))
        except (ValueError, TypeError):
            continue

        if not home_name or not away_name:
            continue

        game_key = f"{away_name} vs {home_name}"
        results[game_key] = {
            "home": home_name,
            "away": away_name,
            "home_score": home_score,
            "away_score": away_score,
        }

    return results


def settle_yesterdays_bets():
    """Check yesterday's results and update the All Picks tab."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_espn = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    print(f"\n=== Settling bets for {yesterday} ===")

    # Use NCAA API (164+ games) as primary, ESPN as fallback
    results = _get_ncaa_results(yesterday)
    espn_results = get_game_results(yesterday_espn)
    # Merge ESPN results in (NCAA API might miss some name formats)
    for key, val in espn_results.items():
        if key not in results:
            results[key] = val

    print(f"Found {len(results)} final results (NCAA API + ESPN)")

    if not results:
        print("No final results found for yesterday.")
        return

    from sheets import get_client, get_or_create_spreadsheet, SPREADSHEET_NAME, TAB_ALL_PICKS
    import gspread

    client = get_client()
    try:
        sheet = client.open(SPREADSHEET_NAME)
        ws = sheet.worksheet(TAB_ALL_PICKS)
    except (gspread.SpreadsheetNotFound, gspread.WorksheetNotFound):
        print(f"No All Picks tab found")
        return

    all_rows = ws.get_all_records()
    if not all_rows:
        return

    result_col = COLUMNS.index("Result") + 1
    updated = 0

    for row_idx, row in enumerate(all_rows, start=2):
        if row.get("Result") != "Pending":
            continue
        # Only settle yesterday's picks (or older pending)
        row_date = row.get("Date", "")
        if row_date > yesterday:
            continue

        game = row.get("Game", "")
        bet_type = row.get("Bet Type", "")
        pick = row.get("Pick", "")

        matched_result = None
        for game_key, res in results.items():
            if _fuzzy(game_key, game) or \
               (_fuzzy(res["home"], game) and _fuzzy(res["away"], game)):
                matched_result = res
                break

        if not matched_result:
            for game_key, res in results.items():
                if _fuzzy(res["home"], game) or _fuzzy(res["away"], game):
                    matched_result = res
                    break

        if not matched_result:
            continue

        home_score = matched_result["home_score"]
        away_score = matched_result["away_score"]
        home_team = matched_result["home"]
        away_team = matched_result["away"]
        actual_margin = home_score - away_score
        actual_total = home_score + away_score

        result = None

        if bet_type == "Run Line":
            spread_match = re.search(r'([+-]?\d+\.?\d*)', pick.split()[-1])
            if spread_match:
                spread_val = float(spread_match.group(1))
                pick_team = " ".join(pick.split()[:-1])
                if _fuzzy(pick_team, home_team):
                    covered = actual_margin + spread_val > 0
                    push = actual_margin + spread_val == 0
                elif _fuzzy(pick_team, away_team):
                    covered = -actual_margin + spread_val > 0
                    push = -actual_margin + spread_val == 0
                else:
                    continue
                result = "Push" if push else ("Hit" if covered else "Miss")

        elif bet_type == "Over/Under":
            total_line = float(pick.split()[-1])
            if "Over" in pick:
                result = "Hit" if actual_total > total_line else ("Push" if actual_total == total_line else "Miss")
            elif "Under" in pick:
                result = "Hit" if actual_total < total_line else ("Push" if actual_total == total_line else "Miss")

        elif bet_type == "Moneyline":
            pick_team = pick.replace(" ML", "").strip()
            if _fuzzy(pick_team, home_team):
                result = "Hit" if home_score > away_score else "Miss"
            elif _fuzzy(pick_team, away_team):
                result = "Hit" if away_score > home_score else "Miss"

        if result:
            ws.update_cell(row_idx, result_col, result)
            updated += 1
            print(f"  {game} | {bet_type} | {pick} -> {result}")

    print(f"Settlement complete ({updated} picks settled).")


def run_daily():
    """Full daily pipeline: settle yesterday + generate today's picks."""

    # Step 1: Settle yesterday's bets
    settle_yesterdays_bets()

    # Step 2: Scrape today's data
    print("\n=== Generating today's picks ===")

    today = datetime.now().strftime("%Y%m%d")

    # Fetch all data sources
    team_stats = get_team_stats()
    pitcher_db = get_pitcher_stats()
    team_eras_db = get_team_eras(pitcher_db)
    odds, odds_api_raw = get_odds(today)
    games = get_schedule(today, odds_data=odds_api_raw)
    starters = get_starters_from_espn(games)
    print(f"  Starters: {len(starters)} games, Pitcher DB: {len(pitcher_db)} pitchers")
    season_results = get_season_results(days_back=25)

    # Load prior year stats for Bayesian priors
    prior_stats = load_prior_year_stats()

    # Build quality records: how each team performs vs winning opponents
    quality_records = _build_quality_records(season_results)

    # Build park factors from home vs away scoring differentials
    park_factors = _calculate_park_factors(season_results)

    # Build recency-weighted stats: recent games matter more
    recency_stats = _calculate_recency_stats(season_results)
    print(f"  Recency stats: {len(recency_stats)} teams")

    if not games:
        print("No games found for today.")
        return

    # Step 3: Merge + cross-check
    print("\n--- Data merge & cross-check ---")
    merged = merge_game_data(games, team_stats, odds, season_results, quality_records,
                             recency_stats, starters=starters, pitcher_db=pitcher_db,
                             team_eras_db=team_eras_db, prior_stats=prior_stats,
                             park_factors=park_factors)
    print(f"  Total game entries: {len(merged)}")

    # Step 4: Run model
    league_avg = merged[0].get("league_avg_rpg", 6.5) if merged else 6.5
    picks = generate_picks(merged, league_avg_rpg=league_avg)
    print(f"\nGenerated {len(picks)} picks")

    if not picks:
        print("No picks met tier thresholds today.")
        return

    # Step 4b: Log picks for CLV tracking
    _log_picks_for_clv(picks)

    # Step 5: Write to Google Sheets
    df = picks_to_dataframe(picks)
    url = write_picks(df)
    print(f"\nSheet URL: {url}")

    # Auto-archive stats at end of season (July+)
    if datetime.now().month >= 7:
        save_season_stats(team_stats)

    # Print summary
    print(f"\n=== Today's Picks Summary ===")
    for pick in picks:
        print(f"  {pick['Score']:>7s} | {pick['Unit Size']:5s} | {pick['Game']} | {pick['Bet Type']} | {pick['Pick']} ({pick['Sportsbook']})")


if __name__ == "__main__":
    run_daily()
