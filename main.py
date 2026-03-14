"""Daily NCAA baseball picks pipeline: scrape → predict → write to Google Sheets."""

import os
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from scraper import get_schedule, get_team_stats, get_odds, get_game_results, match_team, normalize_name, get_season_results, common_opponent_edge
from model import generate_picks, picks_to_dataframe
from sheets import write_picks, update_results_in_all_picks, COLUMNS


def merge_game_data(games, team_stats, odds, season_results=None):
    """Combine schedule, team stats, odds, and common opponent data into model-ready game dicts."""
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

        # Sanity check: if both MLs are the same sign or suspiciously similar, discard them
        # (FanDuel sometimes returns broken ML data like both teams at +450)
        if home_ml and away_ml:
            if (home_ml > 0 and away_ml > 0) or (home_ml < -900 and away_ml < -900):
                home_ml = None
                away_ml = None

        # Common opponent analysis
        co_edge = 0
        co_count = 0
        if season_results:
            co_edge, co_count = common_opponent_edge(home_norm, away_norm, season_results)

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
            "co_edge": co_edge,        # + = home better vs common opponents
            "co_count": co_count,       # how many common opponents (reliability)
        })

        # One pick per game per bet type — no duplicates across books.
        # FanDuel → DraftKings → others priority is handled above.

    print(f"  Odds matched: {matched_count}/{len(games)} games")
    print(f"  Missing team stats: {no_stats_count} games")
    print(f"  Missing odds: {no_odds_count} games")

    return merged


def _fuzzy(a, b):
    """Quick fuzzy name match."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return True
    if a in b or b in a:
        return True
    return False


def settle_yesterdays_bets():
    """Check yesterday's results and update the All Picks tab."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_espn = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    print(f"\n=== Settling bets for {yesterday} ===")

    results = get_game_results(yesterday_espn)
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
    odds, odds_api_raw = get_odds(today)
    games = get_schedule(today, odds_data=odds_api_raw)
    season_results = get_season_results(days_back=25)
    if not games:
        print("No games found for today.")
        return

    # Step 3: Merge + cross-check
    print("\n--- Data merge & cross-check ---")
    merged = merge_game_data(games, team_stats, odds, season_results)
    print(f"  Total game entries: {len(merged)}")

    # Step 4: Run model
    league_avg = merged[0].get("league_avg_rpg", 6.5) if merged else 6.5
    picks = generate_picks(merged, league_avg_rpg=league_avg)
    print(f"\nGenerated {len(picks)} picks")

    if not picks:
        print("No picks met tier thresholds today.")
        return

    # Step 5: Write to Google Sheets
    df = picks_to_dataframe(picks)
    url = write_picks(df)
    print(f"\nSheet URL: {url}")

    # Print summary
    print(f"\n=== Today's Picks Summary ===")
    for pick in picks:
        print(f"  {pick['Score']:>7s} | {pick['Unit Size']:5s} | {pick['Game']} | {pick['Bet Type']} | {pick['Pick']} ({pick['Sportsbook']})")


if __name__ == "__main__":
    run_daily()
