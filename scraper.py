"""Web scraping for NCAA baseball data: schedule, team stats, odds, and results."""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import json
import os

# --- Team name normalization ---

NAME_ALIASES = {
    "UConn": "Connecticut",
    "UCONN": "Connecticut",
    "UNC": "North Carolina",
    "USC": "Southern California",
    "UCF": "Central Florida",
    "UNLV": "Nevada Las Vegas",
    "SMU": "Southern Methodist",
    "VCU": "Virginia Commonwealth",
    "BYU": "Brigham Young",
    "LSU": "Louisiana State",
    "Ole Miss": "Mississippi",
    "Pitt": "Pittsburgh",
    "Miami (FL)": "Miami FL",
    "Miami (OH)": "Miami OH",
    "St. John's": "St John's",
    "Saint Mary's": "Saint Mary's",
    "ETSU": "East Tennessee State",
    "MTSU": "Middle Tennessee",
    "SIU Edwardsville": "SIU-Edwardsville",
    "UMass": "Massachusetts",
    "UIC": "Illinois-Chicago",
    "UTEP": "UT El Paso",
    "UTSA": "UT San Antonio",
    "UT Arlington": "UT-Arlington",
    "UT Martin": "Tennessee-Martin",
    "LIU": "Long Island University",
    "FIU": "Florida International",
    "FAU": "Florida Atlantic",
    "UAB": "Alabama-Birmingham",
    "USC Upstate": "South Carolina Upstate",
    "UNCW": "UNC Wilmington",
    "UNCG": "UNC Greensboro",
    "UALR": "Little Rock",
    "UTA": "UT Arlington",
    "SFA": "Stephen F. Austin",
    "SEMO": "Southeast Missouri State",
    "A&M-Corpus Christi": "Texas A&M-Corpus Christi",
    # FanDuel-specific aliases (FD uses slightly different team names)
    "Ucf": "Central Florida",
    "Umbc": "UMBC",
    "Lsu": "Louisiana State",
    "Tcu": "TCU",
    "Usc": "Southern California",
    "Ucla": "UCLA",
    "Uc Santa Barbara": "UC Santa Barbara",
    "Uc Davis": "UC Davis",
    "Uc Irvine": "UC Irvine",
    "Cal State Fullerton": "Cal State Fullerton",
    "Long Beach State": "Long Beach State",
    "Hawaii": "Hawaii",
    "LA-Lafayette": "Louisiana-Lafayette",
    "Incarnate Word": "Incarnate Word",
    "New Orleans": "New Orleans",
    "Arkansas Little Rock": "Little Rock",
    "Miami Ohio": "Miami OH",
}

MASCOTS = [
    " Bulldogs", " Tigers", " Bears", " Eagles", " Cardinals",
    " Huskies", " Cougars", " Cavaliers", " Seminoles", " Gators",
    " Volunteers", " Razorbacks", " Longhorns", " Aggies", " Wildcats",
    " Blue Devils", " Tar Heels", " Jayhawks", " Crimson Tide",
    " Fighting Irish", " Bruins", " Trojans", " Beavers", " Ducks",
    " Sun Devils", " Golden Bears", " Buffaloes", " Sooners",
    " Cowboys", " Red Raiders", " Mountaineers", " Cyclones",
    " Hawkeyes", " Cornhuskers", " Boilermakers", " Hoosiers",
    " Wolverines", " Spartans", " Buckeyes", " Nittany Lions",
    " Golden Gophers", " Badgers", " Fighting Illini",
    " Commodores", " Rebels", " Gamecocks", " Hokies",
    " Yellow Jackets", " Panthers", " Demon Deacons",
    " Wolfpack", " Orange", " Hurricanes", " Owls",
    " Red Storm", " Pirates", " Musketeers", " Friars",
    " Bluejays", " Shockers", " Thundering Herd",
    " Ragin' Cajuns", " Ragin Cajuns", " Chanticleers", " Paladins",
    " Knights", " Horned Frogs", " Retrievers", " Quakers",
    " Billikens", " Flyers", " Privateers", " Toreros",
    " Anteaters", " Titans", " Hornets", " Rainbow Warriors",
    " Dirtbags", " Gaels", " Gauchos", " Bobcats",
    " Hilltoppers", " Terrapins", " Bearcats", " Green Wave",
    " Bison", " Stags", " Redhawks", " Rams", " Crusaders",
    " Fighting Camels", " Sycamores", " Mustangs", " Hatters",
    " Golden Eagles", " Racers", " Buccaneers", " Antelopes",
    " Patriots", " Roadrunners", " Cardinal", " Wolfpack",
    " Musketeers", " Wave", " Red Foxes",
]


def normalize_name(name):
    """Standardize team name for matching across sources."""
    if not name:
        return ""
    name = name.strip()
    for mascot in MASCOTS:
        name = name.replace(mascot, "")
    name = NAME_ALIASES.get(name, name)
    return name.strip()


def _fuzzy_match(name1, name2):
    """Check if two team names likely refer to the same team."""
    n1 = normalize_name(name1).lower()
    n2 = normalize_name(name2).lower()
    if n1 == n2:
        return True
    if n1 in n2 or n2 in n1:
        return True
    return False


def match_team(team_name, team_dict):
    """Find the best match for a team name in a dictionary keyed by team names."""
    norm = normalize_name(team_name)
    if norm in team_dict:
        return team_dict[norm]
    for key in team_dict:
        if key.lower() == norm.lower():
            return team_dict[key]
    for key in team_dict:
        if _fuzzy_match(norm, key):
            return team_dict[key]
    return None


# --- Schedule & Results (ESPN College Baseball API) ---

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard"


def get_schedule(date_str=None, odds_data=None):
    """Get today's college baseball games from ESPN + Odds API.

    Uses ESPN as primary source, then adds any games from the Odds API
    response that ESPN missed (ESPN sometimes lacks smaller games that
    sportsbooks still have lines for).

    Args:
        date_str: YYYYMMDD or YYYY-MM-DD
        odds_data: raw Odds API response (list of game dicts) to cross-reference

    Returns list of dicts:
        [{"home": str, "away": str, "neutral": bool, "game_time": str,
          "espn_id": str, "status": str, "source": str}, ...]
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    else:
        date_str = date_str.replace("-", "")

    # --- ESPN games ---
    params = {"dates": date_str, "limit": 300, "groups": 50}
    resp = requests.get(ESPN_SCOREBOARD_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    games = []
    seen_matchups = set()

    for event in data.get("events", []):
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        if len(competitors) != 2:
            continue

        home_team = away_team = None
        for c in competitors:
            team_name = c.get("team", {}).get("displayName", "")
            if c.get("homeAway") == "home":
                home_team = team_name
            else:
                away_team = team_name

        neutral = competition.get("neutralSite", False)
        game_time = event.get("date", "")
        espn_id = event.get("id", "")
        status = competition.get("status", {}).get("type", {}).get("name", "")

        games.append({
            "home": home_team,
            "away": away_team,
            "neutral": neutral,
            "game_time": game_time,
            "espn_id": espn_id,
            "status": status,
            "source": "ESPN",
        })
        seen_matchups.add((normalize_name(home_team), normalize_name(away_team)))

    espn_count = len(games)

    # --- Add games from Odds API that ESPN missed ---
    if odds_data:
        for game in odds_data:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            if not home or not away:
                continue

            home_norm = normalize_name(home)
            away_norm = normalize_name(away)

            # Check if already in ESPN list (fuzzy match)
            already_seen = False
            for (sh, sa) in seen_matchups:
                if (_fuzzy_match(home_norm, sh) and _fuzzy_match(away_norm, sa)) or \
                   (_fuzzy_match(away_norm, sh) and _fuzzy_match(home_norm, sa)):
                    already_seen = True
                    break

            if not already_seen:
                game_time = game.get("commence_time", "")
                games.append({
                    "home": home,
                    "away": away,
                    "neutral": False,
                    "game_time": game_time,
                    "espn_id": "",
                    "status": "STATUS_SCHEDULED",
                    "source": "Odds API",
                })
                seen_matchups.add((home_norm, away_norm))

    odds_added = len(games) - espn_count
    source_str = f"ESPN: {espn_count}"
    if odds_added > 0:
        source_str += f", Odds API: +{odds_added}"
    print(f"Found {len(games)} games for {date_str} ({source_str})")
    return games


def get_game_results(date_str=None):
    """Get final scores for games on a given date.

    Returns dict: {"Away vs Home": {"home_score": int, "away_score": int, "home": str, "away": str}}
    """
    if date_str is None:
        date_str = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    else:
        date_str = date_str.replace("-", "")

    params = {"dates": date_str, "limit": 300, "groups": 50}
    resp = requests.get(ESPN_SCOREBOARD_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = {}
    for event in data.get("events", []):
        competition = event.get("competitions", [{}])[0]
        status = competition.get("status", {}).get("type", {}).get("name", "")

        if status != "STATUS_FINAL":
            continue

        competitors = competition.get("competitors", [])
        home_team = away_team = ""
        home_score = away_score = 0

        for c in competitors:
            name = c.get("team", {}).get("displayName", "")
            score = int(c.get("score", 0))
            if c.get("homeAway") == "home":
                home_team = name
                home_score = score
            else:
                away_team = name
                away_score = score

        game_key = f"{away_team} vs {home_team}"
        results[game_key] = {
            "home": home_team,
            "away": away_team,
            "home_score": home_score,
            "away_score": away_score,
        }

    print(f"Found {len(results)} final results for {date_str}")
    return results


# --- Team Stats (WarrenNolan.com) ---

WARREN_NOLAN_BASE = "https://www.warrennolan.com/baseball"


def get_team_stats(year=None):
    """Scrape team stats from WarrenNolan + NCAA API.

    Sources:
      - WarrenNolan: RPG, RA/G, SOS (308 teams)
      - NCAA API: team ERA, WHIP, BA (300 teams)

    Returns dict: {team_name: {"rpg", "ra_pg", "sos", "era", "whip", "ba"}}
    """
    if year is None:
        year = datetime.now().year

    # WarrenNolan stats
    off_stats = _scrape_warren_nolan_stat(year, "stats-off-runs-per-game")
    def_stats = _scrape_warren_nolan_stat(year, "stats-def-runs-per-game")
    margin_stats = _scrape_warren_nolan_stat(year, "stats-off-scoring-margin")
    win_streaks = _scrape_warren_nolan_stat(year, "stats-streaks-current-wins")
    loss_streaks = _scrape_warren_nolan_stat(year, "stats-streaks-current-losses")
    sos_data = _scrape_warren_nolan_sos(year)

    # NCAA API team pitching + batting stats
    ncaa_era = _fetch_ncaa_team_stat(211)    # ERA
    ncaa_whip = _fetch_ncaa_team_stat(597)   # WHIP
    ncaa_ba = _fetch_ncaa_team_stat(210)     # Batting Average

    if not off_stats and not def_stats:
        print("Warning: Could not load team stats from WarrenNolan")
        return {}

    # Merge all sources
    all_teams = set(list(off_stats.keys()) + list(def_stats.keys()))
    stats = {}
    for team in all_teams:
        rpg = off_stats.get(team)
        ra_pg = def_stats.get(team)
        if rpg is not None and ra_pg is not None:
            # Momentum: positive streak = winning, negative = losing
            w_streak = win_streaks.get(team, 0)
            l_streak = loss_streaks.get(team, 0)
            momentum = w_streak if w_streak > 0 else -l_streak

            stats[team] = {
                "rpg": rpg,
                "ra_pg": ra_pg,
                "sos": sos_data.get(team),
                "era": ncaa_era.get(team),
                "whip": ncaa_whip.get(team),
                "ba": ncaa_ba.get(team),
                "margin": margin_stats.get(team, 0),  # avg scoring margin per game
                "momentum": momentum,                   # +N = win streak, -N = loss streak
            }

    ncaa_count = sum(1 for s in stats.values() if s.get("era") is not None)
    margin_count = sum(1 for s in stats.values() if s.get("margin", 0) != 0)
    print(f"Loaded stats for {len(stats)} teams (WarrenNolan + {ncaa_count} ERA/WHIP + {margin_count} margin/momentum)")
    return stats


def _fetch_ncaa_team_stat(stat_seq, max_pages=8):
    """Fetch a team stat from the NCAA API (all pages).

    Returns dict: {normalized_team_name: float_value}
    """
    import time as _time
    all_rows = []
    for page in range(1, max_pages + 1):
        url = f"{NCAA_API_BASE}/stats/baseball/d1/current/team/{stat_seq}"
        if page > 1:
            url += f"/p{page}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                break
            data = resp.json()
            rows = data.get("data", [])
            if not rows:
                break
            all_rows.extend(rows)
            if page >= data.get("pages", 1):
                break
        except Exception:
            break
        _time.sleep(0.25)

    # Parse: find the numeric stat column (last column that's a float)
    result = {}
    for row in all_rows:
        team = row.get("Team", "").strip()
        if not team:
            continue

        # The stat value is the last column (ERA, WHIP, or BA)
        # Try common column names
        value = None
        for col in ["ERA", "WHIP", "BA"]:
            if col in row:
                try:
                    value = float(row[col].replace(",", "").strip())
                    break
                except (ValueError, TypeError):
                    pass

        if value is not None:
            result[normalize_name(team)] = value

    return result


def _scrape_warren_nolan_stat(year, stat_page):
    """Scrape a single stat table from WarrenNolan.

    Tables have 3 columns: Rank | Team (link) | Stat value
    Returns dict: {normalized_team_name: float_value}
    """
    url = f"{WARREN_NOLAN_BASE}/{year}/{stat_page}"
    ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

    try:
        resp = requests.get(url, headers={"User-Agent": ua}, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"Warning: Could not fetch {stat_page}: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")

    stats = {}
    table = soup.find("table")
    if not table:
        rows = soup.find_all("tr")
    else:
        rows = table.find_all("tr")

    for row in rows:
        cells = row.find_all(["td", "th"])
        if len(cells) < 3:
            continue

        # Team name in cell[1], stat value in cell[2]
        team_cell = cells[1]
        value_cell = cells[2]

        # Use full cell text — WarrenNolan wraps team name in nested divs/links
        team = team_cell.get_text(strip=True)

        if not team or team.lower() in ("team", "school", ""):
            continue

        try:
            value = float(value_cell.get_text(strip=True))
            if 0 < value < 30:  # Sanity check for runs per game
                stats[normalize_name(team)] = value
        except (ValueError, IndexError):
            continue

    return stats


def _scrape_warren_nolan_sos(year):
    """Scrape Strength of Schedule from WarrenNolan sos-rpi page.

    SOS is a decimal 0-1 (higher = tougher schedule).
    Returns dict: {normalized_team_name: float}
    """
    url = f"{WARREN_NOLAN_BASE}/{year}/sos-rpi"
    ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

    try:
        resp = requests.get(url, headers={"User-Agent": ua}, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"Warning: Could not fetch SOS data: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    sos = {}

    table = soup.find("table")
    if not table:
        return sos

    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue

        # Find the team name cell (has a link) and SOS value
        team = None
        sos_val = None

        for cell in cells:
            text = cell.get_text(strip=True)
            # Team name: cell with a link that isn't a number
            if cell.find("a") and not text.replace(".", "").replace("-", "").isdigit():
                team = text
            # SOS value: decimal between 0.2 and 0.9
            elif not team:
                continue
            else:
                try:
                    val = float(text)
                    if 0.2 < val < 0.9:
                        sos_val = val
                        break
                except ValueError:
                    continue

        if team and sos_val:
            sos[normalize_name(team)] = sos_val

    return sos


# --- Individual Pitcher Stats (NCAA API via henrygd) ---
# Free JSON wrapper around stats.ncaa.org — no auth required, 5 req/sec limit.
# Returns paginated pitcher leaderboards (ERA, IP, WHIP, K/9).

NCAA_API_BASE = "https://ncaa-api.henrygd.me"
# stat_seq IDs for pitcher stats
NCAA_STAT_ERA = 205
NCAA_STAT_WHIP = 596
NCAA_STAT_K9 = 207


def get_pitcher_stats():
    """Fetch individual pitcher stats from NCAA API.

    Returns dict keyed by (normalized_team, pitcher_name):
    {
        ("Vanderbilt", "John Smith"): {
            "era": 2.45, "ip": 52.0, "whip": 1.05, "k9": 10.2
        }
    }
    """
    era_data = _fetch_ncaa_stat(NCAA_STAT_ERA, max_pages=5)
    whip_data = _fetch_ncaa_stat(NCAA_STAT_WHIP, max_pages=5)
    k9_data = _fetch_ncaa_stat(NCAA_STAT_K9, max_pages=3)

    # Merge all stats by (team, name) key
    pitchers = {}

    for row in era_data:
        name = row.get("Name", "").strip()
        team = normalize_name(row.get("Team", ""))
        if not name or not team:
            continue
        key = (team, name)
        try:
            era = float(row.get("ERA", "0").replace(",", ""))
            ip = float(row.get("IP", "0").replace(",", ""))
        except (ValueError, TypeError):
            continue
        if ip < 5:  # Skip pitchers with trivial innings
            continue
        pitchers[key] = {"era": era, "ip": ip, "whip": None, "k9": None}

    for row in whip_data:
        name = row.get("Name", "").strip()
        team = normalize_name(row.get("Team", ""))
        key = (team, name)
        if key in pitchers:
            try:
                pitchers[key]["whip"] = float(row.get("WHIP", "0"))
            except (ValueError, TypeError):
                pass

    for row in k9_data:
        name = row.get("Name", "").strip()
        team = normalize_name(row.get("Team", ""))
        key = (team, name)
        if key in pitchers:
            try:
                pitchers[key]["k9"] = float(row.get("K/9", row.get("K_9", "0")))
            except (ValueError, TypeError):
                pass

    print(f"Loaded stats for {len(pitchers)} pitchers from NCAA API")
    return pitchers


def _fetch_ncaa_stat(stat_seq, max_pages=5):
    """Fetch paginated NCAA stat leaderboard. Returns list of row dicts."""
    all_rows = []
    for page in range(1, max_pages + 1):
        url = f"{NCAA_API_BASE}/stats/baseball/d1/current/individual/{stat_seq}"
        if page > 1:
            url += f"/p{page}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                break
            data = resp.json()
            rows = data.get("data", [])
            if not rows:
                break
            all_rows.extend(rows)
            if page >= data.get("pages", 1):
                break
        except Exception:
            break
        import time
        time.sleep(0.25)  # Respect 5 req/sec rate limit
    return all_rows


def get_team_eras(pitcher_stats):
    """Calculate average team ERA from individual pitcher stats.

    Returns dict: {normalized_team: avg_era}
    Weighted by innings pitched.
    """
    team_ip = {}
    team_er_ip = {}  # ERA * IP for weighted average

    for (team, _name), stats in pitcher_stats.items():
        if stats["era"] is not None and stats["ip"] is not None and stats["ip"] > 0:
            team_ip[team] = team_ip.get(team, 0) + stats["ip"]
            team_er_ip[team] = team_er_ip.get(team, 0) + stats["era"] * stats["ip"]

    team_eras = {}
    for team in team_ip:
        if team_ip[team] > 0:
            team_eras[team] = team_er_ip[team] / team_ip[team]

    return team_eras


def get_starters_from_espn(games):
    """Try to get starting pitcher names from ESPN game summary API.

    Only works for in-progress or completed games (not pregame).
    games: list of game dicts from get_schedule() (need espn_id, home, away).

    Returns dict: {espn_id: {"home_starter": str, "away_starter": str, "home": str, "away": str}}
    """
    starters = {}
    for game in games:
        eid = game.get("espn_id")
        if not eid:
            continue
        home_team = game.get("home", "")
        away_team = game.get("away", "")

        url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/summary?event={eid}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
        except Exception:
            continue

        boxscore = data.get("boxscore", {})
        players = boxscore.get("players", [])

        game_starters = {"home": home_team, "away": away_team}
        for team_data in players:
            team_name = team_data.get("team", {}).get("displayName", "")
            for stat_group in team_data.get("statistics", []):
                if stat_group.get("type") != "pitching":
                    continue
                for athlete in stat_group.get("athletes", []):
                    if athlete.get("starter"):
                        name = athlete.get("athlete", {}).get("displayName", "")
                        if not name:
                            continue
                        # Match team name to home/away from scoreboard
                        if _fuzzy_match(team_name, home_team):
                            game_starters["home_starter"] = name
                        elif _fuzzy_match(team_name, away_team):
                            game_starters["away_starter"] = name

        if "home_starter" in game_starters or "away_starter" in game_starters:
            starters[eid] = game_starters

    if starters:
        print(f"Found starting pitchers for {len(starters)} games from ESPN")
    return starters


# --- Common Opponent Analysis ---

def get_season_results(days_back=25):
    """Fetch game results from the last N days using NCAA API (all D1 games).

    NCAA API has ~100-160 games/day vs ESPN's ~6. This gives us
    comprehensive common opponent data for the entire D1 landscape.

    Returns dict: {normalized_team: [(opponent_norm, margin, date_str), ...]}
    """
    import time as _time

    results_by_team = {}
    total_games = 0
    dates_fetched = 0

    for days_ago in range(1, days_back + 1):
        date = (datetime.now() - timedelta(days=days_ago))
        date_str = date.strftime("%Y/%m/%d")
        date_label = date.strftime("%Y-%m-%d")

        url = f"{NCAA_API_BASE}/scoreboard/baseball/d1/{date_str}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            data = resp.json()
        except Exception:
            continue

        dates_fetched += 1
        games = data.get("games", [])

        for g in games:
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

            home_norm = normalize_name(home_name)
            away_norm = normalize_name(away_name)
            margin = home_score - away_score

            results_by_team.setdefault(home_norm, []).append(
                (away_norm, margin, date_label)
            )
            results_by_team.setdefault(away_norm, []).append(
                (home_norm, -margin, date_label)
            )
            total_games += 1

        _time.sleep(0.25)  # Respect 5 req/sec rate limit

    print(f"Built common opponent DB: {len(results_by_team)} teams, {total_games} games ({dates_fetched} days)")
    return results_by_team


def common_opponent_edge(team_a_norm, team_b_norm, results_by_team):
    """Calculate common opponent differential between two teams.

    Finds all opponents both teams have played, compares margins.
    Returns (edge_for_a, num_common_opponents):
      - edge > 0 means team_a performed better against common opponents
      - edge < 0 means team_b performed better
      - num = how many common opponents (more = more reliable)

    Example: Team A beat Opp X by +5, Team B lost to Opp X by -3
             → Team A has +8 edge in that comparison
    """
    a_games = results_by_team.get(team_a_norm, [])
    b_games = results_by_team.get(team_b_norm, [])

    if not a_games or not b_games:
        return 0, 0

    # Build opponent -> [margins] for each team
    a_by_opp = {}
    for opp, margin, _ in a_games:
        a_by_opp.setdefault(opp, []).append(margin)

    b_by_opp = {}
    for opp, margin, _ in b_games:
        b_by_opp.setdefault(opp, []).append(margin)

    # Find common opponents (exclude head-to-head)
    common = set(a_by_opp.keys()) & set(b_by_opp.keys())
    common.discard(team_a_norm)
    common.discard(team_b_norm)

    if not common:
        return 0, 0

    # Average margin differential across common opponents
    total_diff = 0
    for opp in common:
        a_avg = sum(a_by_opp[opp]) / len(a_by_opp[opp])
        b_avg = sum(b_by_opp[opp]) / len(b_by_opp[opp])
        total_diff += (a_avg - b_avg)

    edge = total_diff / len(common)
    return edge, len(common)


# --- Odds Scraping ---
# Sources tried in order:
#   1. The Odds API (best, but requires API key — returns DK + FD odds)
#   2. FanDuel API (free, no key, public sportsbook endpoint)
#   3. DraftKings API (free, no key — currently blocked by Cloudflare, kept for
#      future use in case they unblock)
#   4. ESPN fallback (rarely has college baseball odds)

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
ODDS_SPORT_KEY = "baseball_ncaa"

# FanDuel public sportsbook API — eventTypeId 7511 = College Baseball
# The _ak token is a public client key embedded in the FanDuel web app.
# Multiple state subdomains work (nj, co, az, il, tn, mi, pa).
FANDUEL_API_URL = (
    "https://sbapi.{state}.sportsbook.fanduel.com/api/content-managed-page"
    "?page=SPORT&eventTypeId=7511&_ak=FhMFpcPWXMeyZxOx"
    "&timezone=America%2FNew_York"
)
FANDUEL_STATES = ["nj", "co", "az"]  # Fallback states if first fails
FANDUEL_NCAA_COMPETITION_ID = 11363642

# DraftKings sportsbook API — event group 41151 = College Baseball
# Currently blocked by Akamai WAF (403). Kept here so it auto-recovers if
# DraftKings loosens restrictions.
DRAFTKINGS_API_URL = (
    "https://sportsbook-us-{state}.draftkings.com"
    "/sites/US-{state_upper}-SB/api/v5/eventgroups/41151?format=json"
)
DRAFTKINGS_STATES = ["nj", "co", "az"]


def get_odds(date_str=None):
    """Fetch college baseball odds from multiple sources.

    Tries sources in order: The Odds API -> FanDuel -> DraftKings -> ESPN.
    Returns dict keyed by (home_norm, away_norm):
    {
        (home, away): {
            "draftkings": {"spread": float, "total": float, "home_ml": int, "away_ml": int},
            "fanduel": {"spread": float, "total": float, "home_ml": int, "away_ml": int},
        }
    }
    """
    odds = {}
    _odds_api_raw = []  # Store raw response for get_schedule cross-reference

    # Source 1: The Odds API (returns ALL bookmakers)
    api_key = os.environ.get("ODDS_API_KEY", "")
    if api_key:
        odds, _odds_api_raw = _fetch_odds_api(api_key)

    # Source 2: FanDuel free API — always try, supplements with full lines
    fd_odds = _fetch_fanduel_odds()
    if fd_odds:
        for key, book_odds in fd_odds.items():
            if key not in odds:
                odds[key] = {}
            existing = odds[key].get("fanduel", {})
            for k, v in book_odds.items():
                if v is not None and existing.get(k) is None:
                    existing[k] = v
            odds[key]["fanduel"] = existing

    # Source 3: DraftKings free API (may be blocked)
    dk_odds = _fetch_draftkings_odds()
    if dk_odds:
        for key, book_odds in dk_odds.items():
            if key not in odds:
                odds[key] = {}
            existing = odds[key].get("draftkings", {})
            for k, v in book_odds.items():
                if v is not None and existing.get(k) is None:
                    existing[k] = v
            odds[key]["draftkings"] = existing

    # Source 4: ESPN fallback
    if not odds:
        espn_odds = _scrape_espn_odds(date_str)
        if espn_odds:
            for key, book_odds in espn_odds.items():
                odds[key] = {"espn": book_odds}

    _log_odds_summary(odds, "combined sources")
    return odds, _odds_api_raw


def _log_odds_summary(odds, source_label):
    """Print a summary of loaded odds."""
    book_counts = {}
    full_line_count = 0  # Games with spread + total + ML
    for game_odds in odds.values():
        for book, lines in game_odds.items():
            if lines:
                book_counts[book] = book_counts.get(book, 0) + 1
                if lines.get("spread") is not None and lines.get("total") is not None:
                    full_line_count += 1
    books_str = ", ".join(f"{b}: {c}" for b, c in sorted(book_counts.items(), key=lambda x: -x[1])[:5])
    print(f"Loaded odds ({source_label}): {len(odds)} games, {full_line_count} with full lines [{books_str}]")


# --- The Odds API ---

def _fetch_odds_api(api_key):
    """Fetch odds from The Odds API for college baseball — ALL bookmakers.

    Returns (parsed_odds_dict, raw_api_response_list).
    The raw response is used by get_schedule() to find games ESPN missed.
    """
    url = f"{ODDS_API_BASE}/{ODDS_SPORT_KEY}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 422:
            print("Warning: College baseball odds not available via The Odds API (offseason or no lines posted)")
            return {}, []
        resp.raise_for_status()
    except Exception as e:
        print(f"Warning: Odds API request failed: {e}")
        return {}, []

    data = resp.json()
    if not data:
        return {}, []

    odds = {}
    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        if not home or not away:
            continue

        key = (normalize_name(home), normalize_name(away))
        game_odds = {}

        for bm in game.get("bookmakers", []):
            bm_key = bm.get("key", "").lower()
            book_name = bm.get("title", bm_key)

            book_odds = {}
            for market in bm.get("markets", []):
                mkey = market.get("key", "")
                outcomes = market.get("outcomes", [])

                if mkey == "h2h":
                    for o in outcomes:
                        if _fuzzy_match(o.get("name", ""), home):
                            book_odds["home_ml"] = int(o["price"])
                        elif _fuzzy_match(o.get("name", ""), away):
                            book_odds["away_ml"] = int(o["price"])

                elif mkey == "spreads":
                    for o in outcomes:
                        if _fuzzy_match(o.get("name", ""), home):
                            book_odds["spread"] = float(o.get("point", 0))

                elif mkey == "totals":
                    for o in outcomes:
                        if o.get("name") == "Over":
                            book_odds["total"] = float(o.get("point", 0))

            if book_odds:
                game_odds[bm_key] = book_odds

        if game_odds:
            odds[key] = game_odds

    # Log which books we found
    all_books = set()
    for game_odds in odds.values():
        all_books.update(game_odds.keys())
    if all_books:
        print(f"Odds API: {len(odds)} games from {', '.join(sorted(all_books))}")

    return odds, data


# --- FanDuel API ---

def _fetch_fanduel_odds():
    """Fetch college baseball odds from FanDuel's public sportsbook API.

    FanDuel's API structure:
      - layout.coupons.{id}.display[].rows[] maps eventId -> marketIds
      - attachments.events.{id} has event name ("Away Team @ Home Team")
      - attachments.markets.{id} has runners with odds and handicaps
      - competitionId 11363642 = NCAA Baseball

    Returns dict: {(home_norm, away_norm): {"spread": .., "total": .., "home_ml": .., "away_ml": ..}}
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    data = None
    for state in FANDUEL_STATES:
        url = FANDUEL_API_URL.format(state=state)
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                break
        except Exception:
            continue

    if data is None:
        print("Warning: FanDuel API unavailable (all state endpoints failed)")
        return {}

    attachments = data.get("attachments", {})
    events = attachments.get("events", {})
    markets = attachments.get("markets", {})
    coupons = data.get("layout", {}).get("coupons", {})

    if not events or not markets:
        print("Warning: FanDuel returned empty data")
        return {}

    # Find the coupon with the most NCAA Baseball rows (the "all games" coupon)
    best_rows = []
    for coupon in coupons.values():
        for display in coupon.get("display", []):
            if display.get("competitionId") == FANDUEL_NCAA_COMPETITION_ID:
                rows = display.get("rows", [])
                if len(rows) > len(best_rows):
                    best_rows = rows

    if not best_rows:
        # Fallback: try all events that belong to NCAA Baseball competition
        # and find their markets from any coupon
        event_market_map = {}
        for coupon in coupons.values():
            for display in coupon.get("display", []):
                if display.get("competitionId") == FANDUEL_NCAA_COMPETITION_ID:
                    for row in display.get("rows", []):
                        eid = row.get("eventId")
                        if eid and eid not in event_market_map:
                            event_market_map[eid] = row.get("marketIds", [])
        best_rows = [{"eventId": eid, "marketIds": mids} for eid, mids in event_market_map.items()]

    odds = {}
    for row in best_rows:
        event_id = str(row.get("eventId", ""))
        market_ids = row.get("marketIds", [])
        ev = events.get(event_id, {})
        name = ev.get("name", "")

        if " @ " not in name:
            continue

        away_full, home_full = name.split(" @ ", 1)
        home_norm = normalize_name(home_full)
        away_norm = normalize_name(away_full)

        if not home_norm or not away_norm:
            continue

        game_odds = {}

        for mid in market_ids:
            mkt = markets.get(mid, {})
            mtype = mkt.get("marketType", "")
            runners = mkt.get("runners", [])

            if mtype == "MONEY_LINE":
                for r in runners:
                    am_odds = r.get("winRunnerOdds", {}).get(
                        "americanDisplayOdds", {}
                    ).get("americanOdds")
                    if am_odds is None:
                        continue
                    # FanDuel uses -1000000 as a placeholder for "no real
                    # line" — filter these out so they don't confuse the model
                    if abs(int(am_odds)) >= 99999:
                        continue
                    rtype = r.get("result", {}).get("type", "")
                    if rtype == "HOME":
                        game_odds["home_ml"] = int(am_odds)
                    elif rtype == "AWAY":
                        game_odds["away_ml"] = int(am_odds)

            elif "HANDICAP" in mtype:
                for r in runners:
                    rtype = r.get("result", {}).get("type", "")
                    handicap = r.get("handicap")
                    am_odds = r.get("winRunnerOdds", {}).get(
                        "americanDisplayOdds", {}
                    ).get("americanOdds")
                    if rtype == "HOME" and handicap is not None:
                        game_odds["spread"] = float(handicap)
                        if am_odds is not None:
                            game_odds["home_rl_odds"] = int(am_odds)
                    elif rtype == "AWAY" and am_odds is not None:
                        game_odds["away_rl_odds"] = int(am_odds)

            elif "TOTAL" in mtype:
                for r in runners:
                    handicap = r.get("handicap")
                    am_odds = r.get("winRunnerOdds", {}).get(
                        "americanDisplayOdds", {}
                    ).get("americanOdds")
                    if r.get("runnerName") == "Over" and handicap is not None:
                        game_odds["total"] = float(handicap)
                        if am_odds is not None:
                            game_odds["over_odds"] = int(am_odds)
                    elif r.get("runnerName") == "Under" and am_odds is not None:
                        game_odds["under_odds"] = int(am_odds)

        if game_odds:
            key = (home_norm, away_norm)
            odds[key] = game_odds

    if odds:
        print(f"FanDuel: scraped {len(odds)} college baseball games")
    else:
        print("Warning: FanDuel returned no NCAA baseball odds")

    return odds


# --- DraftKings API ---

def _fetch_draftkings_odds():
    """Fetch college baseball odds from DraftKings sportsbook API.

    DraftKings event group 41151 = College Baseball.
    NOTE: As of 2025, these endpoints are blocked by Akamai WAF (HTTP 403)
    when accessed from server-side code. This function is kept so it will
    auto-recover if DraftKings loosens restrictions in the future.

    JSON structure (when it works):
      eventGroup.offerCategories[].offerSubcategoryDescriptors[]
        .offerSubcategory.offers[][] -> outcomes with American odds

    Returns dict: {(home_norm, away_norm): {"spread": .., "total": .., "home_ml": .., "away_ml": ..}}
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    data = None
    for state in DRAFTKINGS_STATES:
        url = DRAFTKINGS_API_URL.format(state=state, state_upper=state.upper())
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                break
        except Exception:
            continue

    if data is None:
        # Silently return — DK being blocked is expected
        return {}

    # Parse the DraftKings JSON structure
    event_group = data.get("eventGroup", {})
    offer_categories = event_group.get("offerCategories", [])

    if not offer_categories:
        return {}

    # Build event ID -> event name mapping
    events_list = event_group.get("events", [])
    event_names = {}
    for ev in events_list:
        eid = ev.get("eventId")
        home = ev.get("teamName1", "")
        away = ev.get("teamName2", "")
        if eid and home and away:
            event_names[eid] = (home, away)

    odds = {}

    for category in offer_categories:
        cat_name = category.get("name", "").lower()
        subcats = category.get("offerSubcategoryDescriptors", [])

        for subcat_desc in subcats:
            subcat_name = subcat_desc.get("name", "").lower()
            subcat = subcat_desc.get("offerSubcategory", {})
            offers = subcat.get("offers", [])

            for offer_group in offers:
                for offer in offer_group:
                    event_id = offer.get("eventId")
                    outcomes = offer.get("outcomes", [])

                    if not event_id or not outcomes:
                        continue

                    # Determine home/away from event mapping
                    if event_id in event_names:
                        home_name, away_name = event_names[event_id]
                    else:
                        continue

                    home_norm = normalize_name(home_name)
                    away_norm = normalize_name(away_name)
                    key = (home_norm, away_norm)

                    if key not in odds:
                        odds[key] = {}

                    # Parse based on category/subcategory
                    if "moneyline" in subcat_name or "moneyline" in cat_name:
                        for o in outcomes:
                            label = o.get("label", "")
                            price = o.get("oddsAmerican")
                            if price is None:
                                continue
                            try:
                                price = int(price)
                            except (ValueError, TypeError):
                                continue
                            if _fuzzy_match(label, home_name):
                                odds[key]["home_ml"] = price
                            elif _fuzzy_match(label, away_name):
                                odds[key]["away_ml"] = price

                    elif "spread" in subcat_name or "run line" in subcat_name:
                        for o in outcomes:
                            label = o.get("label", "")
                            line = o.get("line")
                            if line is not None and _fuzzy_match(label, home_name):
                                odds[key]["spread"] = float(line)

                    elif "total" in subcat_name:
                        for o in outcomes:
                            label = o.get("label", "")
                            line = o.get("line")
                            if line is not None and label.lower() == "over":
                                odds[key]["total"] = float(line)

    # Remove entries with no data
    odds = {k: v for k, v in odds.items() if v}

    if odds:
        print(f"DraftKings: scraped {len(odds)} college baseball games")

    return odds


# --- ESPN Fallback ---

def _scrape_espn_odds(date_str=None):
    """Try to extract odds from ESPN scoreboard API (unlikely for college baseball)."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    else:
        date_str = date_str.replace("-", "")

    params = {"dates": date_str, "limit": 300, "groups": 50}
    try:
        resp = requests.get(ESPN_SCOREBOARD_URL, params=params, timeout=30)
        resp.raise_for_status()
    except Exception:
        return {}

    data = resp.json()
    odds = {}

    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        odds_list = comp.get("odds", [])
        if not odds_list:
            continue

        o = odds_list[0]
        competitors = comp.get("competitors", [])
        home_name = away_name = ""
        for c in competitors:
            name = c.get("team", {}).get("displayName", "")
            if c.get("homeAway") == "home":
                home_name = name
            else:
                away_name = name

        spread = o.get("spread")
        over_under = o.get("overUnder")

        home_ml = away_ml = None
        ml_data = o.get("moneyline", {})
        try:
            home_ml_str = ml_data.get("home", {}).get("close", {}).get("odds", "")
            away_ml_str = ml_data.get("away", {}).get("close", {}).get("odds", "")
            if home_ml_str:
                home_ml = int(home_ml_str)
            if away_ml_str:
                away_ml = int(away_ml_str)
        except (ValueError, TypeError):
            pass

        key = (normalize_name(home_name), normalize_name(away_name))
        book_odds = {}
        if spread is not None:
            book_odds["spread"] = float(spread)
        if over_under is not None:
            book_odds["total"] = float(over_under)
        if home_ml is not None:
            book_odds["home_ml"] = home_ml
        if away_ml is not None:
            book_odds["away_ml"] = away_ml

        if book_odds:
            odds[key] = book_odds

    return odds


if __name__ == "__main__":
    print("\n--- Testing Schedule ---")
    games = get_schedule()
    for g in games[:5]:
        print(f"  {g['away']} @ {g['home']} (neutral={g['neutral']})")

    print("\n--- Testing Team Stats ---")
    stats = get_team_stats()
    if stats:
        sorted_teams = sorted(stats.items(), key=lambda x: x[1]["rpg"], reverse=True)
        for team, s in sorted_teams[:10]:
            print(f"  {team}: RPG={s['rpg']}, RA/G={s['ra_pg']}")
    else:
        print("  No stats loaded!")

    print("\n--- Testing Individual Odds Sources ---")

    print("\n  [FanDuel direct]")
    fd = _fetch_fanduel_odds()
    if fd:
        for (h, a), o in list(fd.items())[:3]:
            print(f"    {a} @ {h}: spread={o.get('spread')}, total={o.get('total')}, ML={o.get('away_ml')}/{o.get('home_ml')}")
    else:
        print("    No FanDuel odds")

    print("\n  [DraftKings direct]")
    dk = _fetch_draftkings_odds()
    if dk:
        for (h, a), o in list(dk.items())[:3]:
            print(f"    {a} @ {h}: spread={o.get('spread')}, total={o.get('total')}, ML={o.get('away_ml')}/{o.get('home_ml')}")
    else:
        print("    No DraftKings odds (expected — blocked by Cloudflare)")

    print("\n--- Testing Combined Odds (get_odds) ---")
    odds = get_odds()
    if odds:
        for (h, a), book_odds in list(odds.items())[:5]:
            print(f"  {a} @ {h}")
            for book, o in book_odds.items():
                if o:
                    print(f"    {book}: spread={o.get('spread')}, total={o.get('total')}, ML={o.get('home_ml')}/{o.get('away_ml')}")
    else:
        print("  No odds loaded")

    print("\n--- Testing Results (yesterday) ---")
    results = get_game_results()
    for key, r in list(results.items())[:5]:
        print(f"  {key}: {r['away_score']}-{r['home_score']}")
