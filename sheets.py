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

# Column layout
COLUMNS = [
    "Date",
    "Game",
    "Bet Type",
    "Pick",
    "Odds",
    "Sportsbook",
    "Unit Size",
    "Score",
    "Sim %",
    "Edge",
    "EV+",
    "Proj Score",
    "Proj Spread",
    "Proj Total",
    "Result",
]

# Formatting
HEADER_BG = {"red": 0.15, "green": 0.15, "blue": 0.15}
HEADER_FG = {"red": 1, "green": 1, "blue": 1}
HIT_BG = {"red": 0.71, "green": 0.88, "blue": 0.71}       # light green
MISS_BG = {"red": 0.92, "green": 0.70, "blue": 0.70}       # light red
PUSH_BG = {"red": 1, "green": 0.95, "blue": 0.70}          # light yellow


def get_client():
    """Authenticate via OAuth and return a gspread client."""
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
    """Open the spreadsheet or create it if it doesn't exist."""
    try:
        return client.open(SPREADSHEET_NAME)
    except gspread.SpreadsheetNotFound:
        sheet = client.create(SPREADSHEET_NAME)
        sheet.share(None, perm_type="anyone", role="reader")
        return sheet


def setup_worksheet(sheet, tab_name, num_rows):
    """Create or clear a worksheet tab and return it."""
    try:
        ws = sheet.worksheet(tab_name)
        ws.clear()
        ws.format("A1:Z1000", {"backgroundColor": {"red": 1, "green": 1, "blue": 1}})
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet(title=tab_name, rows=max(num_rows + 5, 50), cols=len(COLUMNS))
    return ws


def apply_header_format(ws):
    """Style the header row."""
    header_fmt = {
        "backgroundColor": HEADER_BG,
        "textFormat": {
            "foregroundColor": HEADER_FG,
            "bold": True,
            "fontSize": 11,
        },
        "horizontalAlignment": "CENTER",
    }
    ws.format(f"A1:{chr(64 + len(COLUMNS))}1", header_fmt)


def apply_column_widths(sheet, ws):
    """Set column widths for readability."""
    widths = {
        0: 100,   # Date
        1: 240,   # Game
        2: 100,   # Bet Type
        3: 200,   # Pick
        4: 70,    # Odds
        5: 110,   # Sportsbook
        6: 80,    # Unit Size
        7: 90,    # Confidence
        8: 70,    # Sim %
        9: 70,    # Edge
        10: 80,   # EV+
        11: 100,  # Proj Score
        12: 90,   # Proj Spread
        13: 90,   # Proj Total
        14: 90,   # Result
    }
    requests = []
    for col_idx, px in widths.items():
        requests.append({
            "updateDimensionProperties": {
                "range": {
                    "sheetId": ws.id,
                    "dimension": "COLUMNS",
                    "startIndex": col_idx,
                    "endIndex": col_idx + 1,
                },
                "properties": {"pixelSize": px},
                "fields": "pixelSize",
            }
        })
    sheet.batch_update({"requests": requests})


def apply_result_dropdown(sheet, ws, num_data_rows):
    """Add a dropdown validation on the Result column only."""
    result_col_idx = COLUMNS.index("Result")
    requests = [
        {
            "setDataValidation": {
                "range": {
                    "sheetId": ws.id,
                    "startRowIndex": 0,
                    "endRowIndex": 1000,
                    "startColumnIndex": 0,
                    "endColumnIndex": len(COLUMNS) + 5,
                },
                "rule": None,
            }
        },
        {
            "setDataValidation": {
                "range": {
                    "sheetId": ws.id,
                    "startRowIndex": 1,
                    "endRowIndex": 1 + num_data_rows,
                    "startColumnIndex": result_col_idx,
                    "endColumnIndex": result_col_idx + 1,
                },
                "rule": {
                    "condition": {
                        "type": "ONE_OF_LIST",
                        "values": [
                            {"userEnteredValue": "Pending"},
                            {"userEnteredValue": "Hit"},
                            {"userEnteredValue": "Miss"},
                            {"userEnteredValue": "Push"},
                        ],
                    },
                    "showCustomUi": True,
                    "strict": True,
                },
            }
        },
    ]
    sheet.batch_update({"requests": requests})


def apply_conditional_formatting(sheet, ws, num_data_rows):
    """Color rows green/red/yellow based on Result column value."""
    result_col_idx = COLUMNS.index("Result")
    num_rows = 1 + num_data_rows

    requests = [{"deleteConditionalFormatRule": {"sheetId": ws.id, "index": 0}}]
    for _ in range(10):
        try:
            sheet.batch_update({"requests": requests})
        except Exception:
            break

    result_col_letter = chr(65 + result_col_idx)

    rules = [
        {
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId": ws.id,
                        "startRowIndex": 1,
                        "endRowIndex": num_rows,
                        "startColumnIndex": 0,
                        "endColumnIndex": len(COLUMNS),
                    }],
                    "booleanRule": {
                        "condition": {
                            "type": "CUSTOM_FORMULA",
                            "values": [{"userEnteredValue": f'=${result_col_letter}2="Hit"'}],
                        },
                        "format": {"backgroundColor": HIT_BG},
                    },
                },
                "index": 0,
            }
        },
        {
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId": ws.id,
                        "startRowIndex": 1,
                        "endRowIndex": num_rows,
                        "startColumnIndex": 0,
                        "endColumnIndex": len(COLUMNS),
                    }],
                    "booleanRule": {
                        "condition": {
                            "type": "CUSTOM_FORMULA",
                            "values": [{"userEnteredValue": f'=${result_col_letter}2="Miss"'}],
                        },
                        "format": {"backgroundColor": MISS_BG},
                    },
                },
                "index": 1,
            }
        },
        {
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId": ws.id,
                        "startRowIndex": 1,
                        "endRowIndex": num_rows,
                        "startColumnIndex": 0,
                        "endColumnIndex": len(COLUMNS),
                    }],
                    "booleanRule": {
                        "condition": {
                            "type": "CUSTOM_FORMULA",
                            "values": [{"userEnteredValue": f'=${result_col_letter}2="Push"'}],
                        },
                        "format": {"backgroundColor": PUSH_BG},
                    },
                },
                "index": 2,
            }
        },
    ]
    sheet.batch_update({"requests": rules})


def freeze_header(sheet, ws):
    """Freeze the header row."""
    sheet.batch_update({"requests": [{
        "updateSheetProperties": {
            "properties": {
                "sheetId": ws.id,
                "gridProperties": {"frozenRowCount": 1},
            },
            "fields": "gridProperties.frozenRowCount",
        }
    }]})


def write_picks(picks_df):
    """Write picks to the Google Sheet with full formatting."""
    client = get_client()
    sheet = get_or_create_spreadsheet(client)

    tab_name = datetime.now().strftime("%Y-%m-%d")
    num_data_rows = len(picks_df)

    ws = setup_worksheet(sheet, tab_name, num_data_rows)

    if "Result" not in picks_df.columns:
        picks_df = picks_df.copy()
        picks_df["Result"] = "Pending"
    picks_df = picks_df.fillna({"Result": "Pending"})

    data = [COLUMNS] + picks_df[COLUMNS].values.tolist()
    ws.update(range_name="A1", values=data)

    apply_header_format(ws)
    apply_column_widths(sheet, ws)
    apply_result_dropdown(sheet, ws, num_data_rows)
    apply_conditional_formatting(sheet, ws, num_data_rows)
    freeze_header(sheet, ws)

    last_col = chr(64 + len(COLUMNS))
    ws.format(f"A2:{last_col}{num_data_rows + 1}", {"horizontalAlignment": "CENTER"})

    print(f"Picks written to '{SPREADSHEET_NAME}' -> tab '{tab_name}'")
    return sheet.url


def update_results(results_dict, date_str=None):
    """Update the Result column for settled bets.

    results_dict: {game_name: "Hit" | "Miss" | "Push"}
    date_str: tab name (defaults to today)
    """
    client = get_client()
    sheet = get_or_create_spreadsheet(client)

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    ws = sheet.worksheet(date_str)
    all_data = ws.get_all_records()

    result_col_idx = COLUMNS.index("Result") + 1
    game_col_idx = COLUMNS.index("Game")

    for row_idx, row in enumerate(all_data, start=2):
        game = row.get("Game", "")
        if game in results_dict:
            ws.update_cell(row_idx, result_col_idx, results_dict[game])

    print(f"Results updated for tab '{date_str}'")


if __name__ == "__main__":
    sample = pd.DataFrame({
        "Date": ["2026-03-13", "2026-03-13", "2026-03-13"],
        "Game": ["Florida vs Vanderbilt", "Oregon State vs LSU", "Stanford vs Texas"],
        "Bet Type": ["Run Line", "Moneyline", "Over/Under"],
        "Pick": ["Vanderbilt -1.5", "LSU ML", "Over 11.5"],
        "Odds": ["-110", "-145", "-110"],
        "Sportsbook": ["DraftKings", "FanDuel", "DraftKings"],
        "Unit Size": ["2u", "1u", "1.5u"],
        "Confidence": ["ELITE", "MOD", "STRONG"],
        "Sim %": ["58.2%", "54.1%", "56.8%"],
        "Edge": ["4.1%", "1.3%", "3.2%"],
        "EV+": ["+$5.82", "+$1.45", "+$4.36"],
        "Proj Score": ["4.2-6.8", "3.9-5.1", "6.1-6.3"],
        "Proj Spread": ["-2.6", "-1.2", "-0.2"],
        "Proj Total": ["11.0", "9.0", "12.4"],
        "Result": ["Pending", "Pending", "Pending"],
    })
    url = write_picks(sample)
    print(f"Sheet URL: {url}")
