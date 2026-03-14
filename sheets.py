"""Google Sheets output — 3-tab system for NCAA baseball picks.

Tab 1: DASHBOARD — P&L summary, units up/down, bet count, pending, possible profit
Tab 2: ALL PICKS — every pick ever made, growing vertically, with results
Tab 3: TODAY'S PICKS — clears daily, shows only today's recommended picks
"""

import gspread
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pandas as pd
from datetime import datetime
import os

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

TOKEN_PATH = os.path.join(os.path.dirname(__file__), "token.json")
CLIENT_SECRET_PATH = os.path.join(os.path.dirname(__file__), "client_secret.json")

SPREADSHEET_NAME = "NCAA Baseball Picks"

TAB_DASHBOARD = "Dashboard"
TAB_ALL_PICKS = "All Picks"
TAB_TODAY = "Today's Picks"

COLUMNS = [
    "Date", "Game", "Bet Type", "Pick", "Odds", "Sportsbook",
    "Unit Size", "Score", "Sim %", "Edge", "EV+",
    "Proj Score", "Proj Spread", "Proj Total", "Result",
]

# --- Color palette (dark/sleek aesthetic) ---
BLACK = {"red": 0.08, "green": 0.08, "blue": 0.08}
DARK_GRAY = {"red": 0.12, "green": 0.12, "blue": 0.14}
MED_GRAY = {"red": 0.18, "green": 0.18, "blue": 0.20}
LIGHT_GRAY = {"red": 0.90, "green": 0.90, "blue": 0.92}
WHITE = {"red": 1, "green": 1, "blue": 1}
ACCENT_BLUE = {"red": 0.24, "green": 0.52, "blue": 0.88}
ACCENT_GREEN = {"red": 0.18, "green": 0.75, "blue": 0.45}
ACCENT_RED = {"red": 0.90, "green": 0.25, "blue": 0.25}
ACCENT_YELLOW = {"red": 0.95, "green": 0.78, "blue": 0.20}
ACCENT_CYAN = {"red": 0.20, "green": 0.78, "blue": 0.85}

HIT_BG = {"red": 0.12, "green": 0.25, "blue": 0.15}
MISS_BG = {"red": 0.28, "green": 0.10, "blue": 0.10}
PUSH_BG = {"red": 0.28, "green": 0.25, "blue": 0.10}
PENDING_BG = DARK_GRAY


def get_client():
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
    return gspread.authorize(creds)


def get_or_create_spreadsheet(client):
    try:
        return client.open(SPREADSHEET_NAME)
    except gspread.SpreadsheetNotFound:
        sheet = client.create(SPREADSHEET_NAME)
        sheet.share(None, perm_type="anyone", role="reader")
        return sheet


def _get_or_create_tab(sheet, tab_name, rows=100, cols=20):
    try:
        return sheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        return sheet.add_worksheet(title=tab_name, rows=rows, cols=cols)


def _clear_conditional_rules(sheet, ws):
    reqs = [{"deleteConditionalFormatRule": {"sheetId": ws.id, "index": 0}}]
    for _ in range(15):
        try:
            sheet.batch_update({"requests": reqs})
        except Exception:
            break


def _set_col_widths(sheet, ws, widths):
    reqs = []
    for col_idx, px in widths.items():
        reqs.append({
            "updateDimensionProperties": {
                "range": {
                    "sheetId": ws.id, "dimension": "COLUMNS",
                    "startIndex": col_idx, "endIndex": col_idx + 1,
                },
                "properties": {"pixelSize": px},
                "fields": "pixelSize",
            }
        })
    if reqs:
        sheet.batch_update({"requests": reqs})


def _freeze(sheet, ws, rows=1, cols=0):
    sheet.batch_update({"requests": [{
        "updateSheetProperties": {
            "properties": {
                "sheetId": ws.id,
                "gridProperties": {"frozenRowCount": rows, "frozenColumnCount": cols},
            },
            "fields": "gridProperties.frozenRowCount,gridProperties.frozenColumnCount",
        }
    }]})


def _style_picks_tab(sheet, ws, num_data_rows):
    """Apply dark theme + conditional formatting to a picks tab."""
    last_col = chr(64 + len(COLUMNS))
    num_rows = 1 + num_data_rows

    # Batch format: header + data rows
    formats = [
        {"range": f"A1:{last_col}1", "format": {
            "backgroundColor": BLACK,
            "textFormat": {"foregroundColor": ACCENT_CYAN, "bold": True, "fontSize": 10,
                           "fontFamily": "Roboto Mono"},
            "horizontalAlignment": "CENTER",
        }},
    ]
    if num_data_rows > 0:
        formats.append({"range": f"A2:{last_col}{num_rows}", "format": {
            "backgroundColor": DARK_GRAY,
            "textFormat": {"foregroundColor": LIGHT_GRAY, "fontSize": 10,
                           "fontFamily": "Roboto Mono"},
            "horizontalAlignment": "CENTER",
        }})
        # Alternating row shading (limit to avoid rate limits)
        for i in range(2, min(num_rows + 1, 100), 2):
            formats.append({"range": f"A{i}:{last_col}{i}", "format": {"backgroundColor": MED_GRAY}})
    ws.batch_format(formats)

    # Column widths
    _set_col_widths(sheet, ws, {
        0: 95, 1: 260, 2: 90, 3: 210, 4: 65, 5: 105,
        6: 70, 7: 70, 8: 65, 9: 60, 10: 75, 11: 90, 12: 85, 13: 80, 14: 80,
    })

    _freeze(sheet, ws, rows=1)

    # Conditional formatting for results
    _clear_conditional_rules(sheet, ws)
    result_col = chr(65 + COLUMNS.index("Result"))
    rules = [
        (f'=${result_col}2="Hit"', HIT_BG),
        (f'=${result_col}2="Miss"', MISS_BG),
        (f'=${result_col}2="Push"', PUSH_BG),
    ]
    batch_rules = []
    for idx, (formula, bg) in enumerate(rules):
        batch_rules.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{"sheetId": ws.id, "startRowIndex": 1, "endRowIndex": max(num_rows, 500),
                                "startColumnIndex": 0, "endColumnIndex": len(COLUMNS)}],
                    "booleanRule": {
                        "condition": {"type": "CUSTOM_FORMULA",
                                      "values": [{"userEnteredValue": formula}]},
                        "format": {"backgroundColor": bg,
                                   "textFormat": {"foregroundColor": WHITE}},
                    },
                },
                "index": idx,
            }
        })
    sheet.batch_update({"requests": batch_rules})

    # Result dropdown
    result_col_idx = COLUMNS.index("Result")
    sheet.batch_update({"requests": [
        {"setDataValidation": {
            "range": {"sheetId": ws.id, "startRowIndex": 1, "endRowIndex": max(num_rows, 500),
                      "startColumnIndex": result_col_idx, "endColumnIndex": result_col_idx + 1},
            "rule": {"condition": {"type": "ONE_OF_LIST", "values": [
                {"userEnteredValue": "Pending"}, {"userEnteredValue": "Hit"},
                {"userEnteredValue": "Miss"}, {"userEnteredValue": "Push"},
            ]}, "showCustomUi": True, "strict": True},
        }}
    ]})


# =====================================================================
# TAB 1: DASHBOARD
# =====================================================================

def _build_dashboard(sheet):
    """Build the summary dashboard with formulas that auto-calculate from All Picks."""
    ws = _get_or_create_tab(sheet, TAB_DASHBOARD, rows=30, cols=8)
    ws.clear()

    # Dashboard layout
    rows = [
        ["", "", "", "", "", "", "", ""],
        ["", "NCAA BASEBALL MODEL", "", "", "", "", "", ""],
        ["", "Live Dashboard", "", "", "", datetime.now().strftime("%B %d, %Y"), "", ""],
        ["", "", "", "", "", "", "", ""],
        # --- Record Section ---
        # AP = All Picks tab, TP = Today's Picks tab
        # Column G = Unit Size (e.g. "2.0u"), Column O = Result, Column E = Odds
        ["", "RECORD", "", "", "UNITS", "", "", ""],
        ["", "Total Picks",
         "=COUNTA('All Picks'!A2:A)-COUNTIF('All Picks'!O2:O,\"Pending\")", "",
         "Units Wagered",
         "=SUMPRODUCT((('All Picks'!O2:O=\"Hit\")+('All Picks'!O2:O=\"Miss\")+('All Picks'!O2:O=\"Push\"))*VALUE(SUBSTITUTE('All Picks'!G2:G,\"u\",\"\")))",
         "", ""],
        ["", "Hits",
         "=COUNTIF('All Picks'!O2:O,\"Hit\")", "",
         "Units Won",
         "=SUMPRODUCT(('All Picks'!O2:O=\"Hit\")*VALUE(SUBSTITUTE('All Picks'!G2:G,\"u\",\"\")))",
         "", ""],
        ["", "Misses",
         "=COUNTIF('All Picks'!O2:O,\"Miss\")", "",
         "Units Lost",
         "=-SUMPRODUCT(('All Picks'!O2:O=\"Miss\")*VALUE(SUBSTITUTE('All Picks'!G2:G,\"u\",\"\")))",
         "", ""],
        ["", "Pushes",
         "=COUNTIF('All Picks'!O2:O,\"Push\")", "",
         "Net Units", "=F7+F8", "", ""],
        ["", "Win Rate",
         "=IFERROR(C7/(C7+C8),0)", "",
         "ROI", "=IFERROR(F9/F6,0)", "", ""],
        ["", "", "", "", "", "", "", ""],
        # --- Pending Section ---
        ["", "PENDING", "", "", "POSSIBLE", "", "", ""],
        ["", "Pending Picks",
         "=COUNTIF('All Picks'!O2:O,\"Pending\")", "",
         "Possible Profit",
         "=SUMPRODUCT(('All Picks'!O2:O=\"Pending\")*VALUE(SUBSTITUTE('All Picks'!G2:G,\"u\",\"\")))",
         "", ""],
        ["", "Pending Units",
         "=SUMPRODUCT(('All Picks'!O2:O=\"Pending\")*VALUE(SUBSTITUTE('All Picks'!G2:G,\"u\",\"\")))", "",
         "Possible Net", "=F13", "", ""],
        ["", "", "", "", "", "", "", ""],
        # --- Today Section ---
        ["", "TODAY", "", "", "", "", "", ""],
        ["", "Today's Picks",
         "=COUNTA('Today''s Picks'!A2:A)", "", "", "", "", ""],
        ["", "Today's Units",
         "=IFERROR(SUMPRODUCT(VALUE(SUBSTITUTE('Today''s Picks'!G2:G,\"u\",\"\"))),0)",
         "", "", "", "", ""],
    ]

    ws.update(range_name="A1", values=rows, value_input_option="USER_ENTERED")

    # --- Styling (batched to avoid rate limits) ---
    def _f(rng, fmt):
        return {"range": rng, "format": fmt}

    formats = [
        _f("A1:H30", {"backgroundColor": BLACK,
                       "textFormat": {"foregroundColor": LIGHT_GRAY, "fontSize": 11,
                                      "fontFamily": "Roboto Mono"}}),
        _f("B2", {"textFormat": {"foregroundColor": WHITE, "bold": True,
                                  "fontSize": 22, "fontFamily": "Roboto Mono"}}),
        _f("B3", {"textFormat": {"foregroundColor": ACCENT_CYAN, "fontSize": 12}}),
        _f("F3", {"textFormat": {"foregroundColor": ACCENT_CYAN, "fontSize": 10}}),
        _f("B5", {"textFormat": {"foregroundColor": ACCENT_BLUE, "bold": True, "fontSize": 13}}),
        _f("E5", {"textFormat": {"foregroundColor": ACCENT_BLUE, "bold": True, "fontSize": 13}}),
        _f("B12", {"textFormat": {"foregroundColor": ACCENT_BLUE, "bold": True, "fontSize": 13}}),
        _f("E12", {"textFormat": {"foregroundColor": ACCENT_BLUE, "bold": True, "fontSize": 13}}),
        _f("B16", {"textFormat": {"foregroundColor": ACCENT_BLUE, "bold": True, "fontSize": 13}}),
        _f("C6", {"textFormat": {"foregroundColor": WHITE, "bold": True, "fontSize": 14}}),
        _f("C7", {"textFormat": {"foregroundColor": ACCENT_GREEN, "bold": True, "fontSize": 14}}),
        _f("C8", {"textFormat": {"foregroundColor": ACCENT_RED, "bold": True, "fontSize": 14}}),
        _f("C9:C10", {"textFormat": {"foregroundColor": WHITE, "bold": True, "fontSize": 14}}),
        _f("F6:F10", {"textFormat": {"foregroundColor": WHITE, "bold": True, "fontSize": 14}}),
        _f("C10", {"numberFormat": {"type": "PERCENT", "pattern": "0.0%"}}),
        _f("F10", {"numberFormat": {"type": "PERCENT", "pattern": "+0.0%;-0.0%"}}),
        _f("F9", {"textFormat": {"foregroundColor": ACCENT_GREEN, "bold": True, "fontSize": 16}}),
        _f("C13:C14", {"textFormat": {"foregroundColor": ACCENT_YELLOW, "bold": True, "fontSize": 13}}),
        _f("F13:F14", {"textFormat": {"foregroundColor": ACCENT_YELLOW, "bold": True, "fontSize": 13}}),
        _f("C17:C18", {"textFormat": {"foregroundColor": ACCENT_CYAN, "bold": True, "fontSize": 13}}),
    ]
    ws.batch_format(formats)

    _set_col_widths(sheet, ws, {0: 30, 1: 160, 2: 100, 3: 30, 4: 160, 5: 140, 6: 30, 7: 30})

    print(f"Dashboard built")
    return ws


# =====================================================================
# TAB 2: ALL PICKS (append-only, grows over time)
# =====================================================================

def _append_to_all_picks(sheet, picks_df):
    """Append today's picks to the All Picks tab (never clears)."""
    ws = _get_or_create_tab(sheet, TAB_ALL_PICKS, rows=2000, cols=len(COLUMNS))

    existing = ws.get_all_values()
    if not existing or existing[0] != COLUMNS:
        # First time — write headers
        ws.clear()
        ws.update(range_name="A1", values=[COLUMNS])
        existing = [COLUMNS]

    # Append new picks below existing data
    next_row = len(existing) + 1
    new_data = picks_df[COLUMNS].values.tolist()
    if new_data:
        ws.update(range_name=f"A{next_row}", values=new_data)
        _style_picks_tab(sheet, ws, next_row - 1)
        print(f"Appended {len(new_data)} picks to '{TAB_ALL_PICKS}' (total: {next_row - 1} rows)")

    return ws


# =====================================================================
# TAB 3: TODAY'S PICKS (clears and replaces daily)
# =====================================================================

def _write_today_picks(sheet, picks_df):
    """Write today's picks to the Today tab (clears previous day)."""
    ws = _get_or_create_tab(sheet, TAB_TODAY, rows=50, cols=len(COLUMNS))
    ws.clear()

    num_data_rows = len(picks_df)
    data = [COLUMNS] + picks_df[COLUMNS].values.tolist()
    ws.update(range_name="A1", values=data)

    _style_picks_tab(sheet, ws, num_data_rows)
    print(f"Today's picks written ({num_data_rows} picks)")
    return ws


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def write_picks(picks_df):
    """Write picks to all 3 tabs and return the sheet URL."""
    client = get_client()
    sheet = get_or_create_spreadsheet(client)

    if "Result" not in picks_df.columns:
        picks_df = picks_df.copy()
        picks_df["Result"] = "Pending"
    picks_df = picks_df.fillna({"Result": "Pending"})

    # Tab 3: Today's Picks (clear + write)
    _write_today_picks(sheet, picks_df)

    # Tab 2: All Picks (append)
    _append_to_all_picks(sheet, picks_df)

    # Tab 1: Dashboard (rebuild formulas)
    _build_dashboard(sheet)

    # Reorder tabs: Dashboard first, then All Picks, then Today
    try:
        all_ws = sheet.worksheets()
        tab_order = {TAB_DASHBOARD: 0, TAB_ALL_PICKS: 1, TAB_TODAY: 2}
        reqs = []
        for ws in all_ws:
            if ws.title in tab_order:
                reqs.append({"updateSheetProperties": {
                    "properties": {"sheetId": ws.id, "index": tab_order[ws.title]},
                    "fields": "index",
                }})
        if reqs:
            sheet.batch_update({"requests": reqs})
    except Exception:
        pass

    # Delete any old date-named tabs (cleanup from previous version)
    for ws in sheet.worksheets():
        if ws.title not in (TAB_DASHBOARD, TAB_ALL_PICKS, TAB_TODAY):
            try:
                sheet.del_worksheet(ws)
            except Exception:
                pass

    print(f"Sheet URL: {sheet.url}")
    return sheet.url


def update_results_in_all_picks(results, date_str=None):
    """Update Result column in All Picks for settled bets."""
    client = get_client()
    sheet = get_or_create_spreadsheet(client)

    try:
        ws = sheet.worksheet(TAB_ALL_PICKS)
    except gspread.WorksheetNotFound:
        return

    all_data = ws.get_all_records()
    if not all_data:
        return

    result_col = COLUMNS.index("Result") + 1

    def _fuzzy(a, b):
        a, b = a.lower().strip(), b.lower().strip()
        return a == b or a in b or b in a

    updated = 0
    for row_idx, row in enumerate(all_data, start=2):
        if row.get("Result") != "Pending":
            continue
        if date_str and row.get("Date") != date_str:
            continue

        game = row.get("Game", "")
        for game_key, res in results.items():
            if _fuzzy(game_key, game) or \
               (_fuzzy(res.get("home", ""), game) and _fuzzy(res.get("away", ""), game)):
                ws.update_cell(row_idx, result_col, res.get("result", ""))
                updated += 1
                break

    if updated:
        print(f"Updated {updated} results in '{TAB_ALL_PICKS}'")


if __name__ == "__main__":
    sample = pd.DataFrame({
        "Date": ["2026-03-14", "2026-03-14"],
        "Game": ["Florida vs Vanderbilt", "Oregon State vs LSU"],
        "Bet Type": ["Run Line", "Moneyline"],
        "Pick": ["Vanderbilt -1.5", "LSU ML"],
        "Odds": ["-110", "-145"],
        "Sportsbook": ["DraftKings", "FanDuel"],
        "Unit Size": ["2.0u", "1.0u"],
        "Score": ["8.2%", "4.1%"],
        "Sim %": ["58.2%", "54.1%"],
        "Edge": ["4.1%", "1.3%"],
        "EV+": ["+$8.20", "+$4.10"],
        "Proj Score": ["4.2-6.8", "3.9-5.1"],
        "Proj Spread": ["-2.6", "-1.2"],
        "Proj Total": ["11.0", "9.0"],
        "Result": ["Pending", "Pending"],
    })
    url = write_picks(sample)
    print(f"Done: {url}")
