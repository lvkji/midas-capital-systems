# ============================================================
# MIDAS CAPITAL SYSTEMS v3.0
# AI-Powered Multi-User Paper Trading Platform
# Author: Andrew Ignatius | Senior Capstone Project 2026
# ============================================================

import os, time, sqlite3, random, hashlib, secrets
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

try:
    import pytz
    PYTZ_AVAILABLE = True
except Exception:
    PYTZ_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================

DB_PATH = "midas_capital_v3.db"

RH_GREEN  = "#00C805"
RH_RED    = "#FF5000"
RH_GOLD   = "#D4A017"
ZIMA_BLUE = "#6FC3DF"

UNIVERSE = pd.DataFrame([
    ("AAPL",  "Apple",                "Technology"),
    ("MSFT",  "Microsoft",            "Technology"),
    ("NVDA",  "NVIDIA",               "Technology"),
    ("GOOGL", "Alphabet",             "Communication Services"),
    ("META",  "Meta",                 "Communication Services"),
    ("AMZN",  "Amazon",               "Consumer Discretionary"),
    ("TSLA",  "Tesla",                "Consumer Discretionary"),
    ("HD",    "Home Depot",           "Consumer Discretionary"),
    ("JPM",   "JPMorgan Chase",       "Financials"),
    ("V",     "Visa",                 "Financials"),
    ("BAC",   "Bank of America",      "Financials"),
    ("XOM",   "Exxon Mobil",          "Energy"),
    ("CVX",   "Chevron",              "Energy"),
    ("LLY",   "Eli Lilly",            "Health Care"),
    ("UNH",   "UnitedHealth",         "Health Care"),
    ("JNJ",   "Johnson & Johnson",    "Health Care"),
    ("KO",    "Coca-Cola",            "Consumer Staples"),
    ("PEP",   "PepsiCo",              "Consumer Staples"),
    ("WMT",   "Walmart",              "Consumer Staples"),
    ("DIS",   "Disney",               "Communication Services"),
    ("CAT",   "Caterpillar",          "Industrials"),
    ("BA",    "Boeing",               "Industrials"),
    ("SPY",   "SPDR S&P 500 ETF",    "ETF"),
], columns=["Ticker", "Name", "Sector"])

SECTOR_COLORS = {
    "Technology":             "#6366f1",
    "Communication Services": "#06b6d4",
    "Consumer Discretionary": "#f59e0b",
    "Financials":             "#10b981",
    "Energy":                 "#f97316",
    "Health Care":            "#ec4899",
    "Consumer Staples":       "#84cc16",
    "Industrials":            "#8b5cf6",
    "ETF":                    "#94a3b8",
    "Unknown":                "#475569",
}

# GBM parameters per sector (annualised drift μ and daily σ)
SECTOR_SIM_PARAMS = {
    "Technology":             {"mu": 0.00080, "sigma": 0.0220, "skew": -0.15},
    "Communication Services": {"mu": 0.00050, "sigma": 0.0175, "skew": -0.10},
    "Consumer Discretionary": {"mu": 0.00040, "sigma": 0.0195, "skew": -0.12},
    "Financials":             {"mu": 0.00040, "sigma": 0.0160, "skew": -0.20},
    "Energy":                 {"mu": 0.00030, "sigma": 0.0250, "skew": -0.08},
    "Health Care":            {"mu": 0.00055, "sigma": 0.0145, "skew": -0.10},
    "Consumer Staples":       {"mu": 0.00025, "sigma": 0.0095, "skew": -0.05},
    "Industrials":            {"mu": 0.00040, "sigma": 0.0160, "skew": -0.12},
    "ETF":                    {"mu": 0.00042, "sigma": 0.0115, "skew": -0.10},
    "Unknown":                {"mu": 0.00035, "sigma": 0.0180, "skew": -0.10},
}

# Anchor prices (approximate real-world levels for visual realism)
ANCHOR_PRICES = {
    "AAPL": 195, "MSFT": 420, "NVDA": 880, "GOOGL": 172, "META": 525,
    "AMZN": 198, "TSLA": 172, "HD":   370, "JPM":  208, "V":    288,
    "BAC":   40, "XOM":  112, "CVX":  158, "LLY":  830, "UNH":  495,
    "JNJ":  150, "KO":    63, "PEP":  172, "WMT":   70, "DIS":   97,
    "CAT":  355, "BA":   178, "SPY":  535,
}

ALL_TICKERS = sorted(UNIVERSE["Ticker"].tolist())

STOIC_QUOTES = [
    '"The impediment to action advances action. What stands in the way becomes the way." — Marcus Aurelius',
    '"Wealth consists not in having great possessions, but in having few wants." — Epictetus',
    '"He who is brave is free." — Seneca',
    '"Never let the future disturb you." — Marcus Aurelius',
    '"It is not the man who has too little, but the man who craves more, that is poor." — Seneca',
]

FEATURE_COLS = [
    "SMA_5", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "MACD", "MACD_Signal", "RSI", "BB_Middle", "BB_Upper",
    "BB_Lower", "Momentum_5", "Momentum_20", "Volatility"
]

SIM_PERIODS = {
    "1 Month":  21,
    "3 Months": 63,
    "6 Months": 126,
    "1 Year":   252,
    "2 Years":  504,
}

# ============================================================
# PASSWORD UTILITIES
# ============================================================

def _hash_pw(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 260_000
    ).hex()

def new_credentials(password: str):
    salt = secrets.token_hex(16)
    return _hash_pw(password, salt), salt

def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    return _hash_pw(password, salt) == stored_hash

# ============================================================
# DATABASE LAYER  (v3 — user-scoped)
# ============================================================

def get_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = get_db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS db_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt         TEXT NOT NULL,
            is_admin     INTEGER NOT NULL DEFAULT 0,
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS account (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER NOT NULL UNIQUE,
            cash             REAL NOT NULL DEFAULT 10000.0,
            initial_cash     REAL NOT NULL DEFAULT 10000.0,
            starting_capital REAL NOT NULL DEFAULT 10000.0
        );

        CREATE TABLE IF NOT EXISTS portfolio (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id  INTEGER NOT NULL,
            ticker   TEXT NOT NULL,
            shares   REAL NOT NULL,
            avg_cost REAL NOT NULL,
            UNIQUE(user_id, ticker)
        );

        CREATE TABLE IF NOT EXISTS trades (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id  INTEGER NOT NULL,
            time     TEXT NOT NULL,
            side     TEXT NOT NULL,
            ticker   TEXT NOT NULL,
            shares   REAL NOT NULL,
            price    REAL NOT NULL,
            notional REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS equity_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            timestamp   TEXT NOT NULL,
            equity      REAL NOT NULL,
            trade_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS watchlist (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id  INTEGER NOT NULL,
            ticker   TEXT NOT NULL,
            added_at TEXT NOT NULL,
            UNIQUE(user_id, ticker)
        );
    """)
    con.commit()
    con.close()

# ── User ops ──────────────────────────────────────────────

def db_user_count() -> int:
    con = get_db()
    n = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    con.close()
    return n

def db_create_user(username: str, password: str, is_admin: bool = False) -> tuple[bool, str]:
    pw_hash, salt = new_credentials(password)
    try:
        con = get_db()
        cur = con.execute(
            "INSERT INTO users (username,password_hash,salt,is_admin,created_at) VALUES (?,?,?,?,?)",
            (username, pw_hash, salt, int(is_admin), datetime.now().isoformat())
        )
        uid = cur.lastrowid
        con.execute(
            "INSERT INTO account (user_id,cash,initial_cash,starting_capital) VALUES (?,10000,10000,10000)",
            (uid,)
        )
        for t in ["AAPL", "MSFT", "NVDA", "SPY"]:
            con.execute(
                "INSERT OR IGNORE INTO watchlist (user_id,ticker,added_at) VALUES (?,?,?)",
                (uid, t, datetime.now().isoformat())
            )
        con.commit()
        con.close()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already taken."

def db_authenticate(username: str, password: str):
    """Returns user row dict or None."""
    con = get_db()
    row = con.execute(
        "SELECT id,username,password_hash,salt,is_admin FROM users WHERE username=?",
        (username,)
    ).fetchone()
    con.close()
    if not row:
        return None
    uid, uname, pw_hash, salt, is_admin = row
    if verify_password(password, pw_hash, salt):
        return {"id": uid, "username": uname, "is_admin": bool(is_admin)}
    return None

def db_all_users():
    con = get_db()
    rows = con.execute(
        "SELECT id,username,is_admin,created_at FROM users ORDER BY id"
    ).fetchall()
    con.close()
    return [{"id": r[0], "username": r[1], "is_admin": bool(r[2]), "created_at": r[3]}
            for r in rows]

def db_delete_user(uid: int):
    con = get_db()
    con.executescript(f"""
        DELETE FROM users          WHERE id={uid};
        DELETE FROM account        WHERE user_id={uid};
        DELETE FROM portfolio      WHERE user_id={uid};
        DELETE FROM trades         WHERE user_id={uid};
        DELETE FROM equity_history WHERE user_id={uid};
        DELETE FROM watchlist      WHERE user_id={uid};
    """)
    con.commit()
    con.close()

def db_change_password(uid: int, new_password: str):
    pw_hash, salt = new_credentials(new_password)
    con = get_db()
    con.execute("UPDATE users SET password_hash=?,salt=? WHERE id=?", (pw_hash, salt, uid))
    con.commit()
    con.close()

# ── Per-user data ops ─────────────────────────────────────

def db_load_user_state(uid: int):
    con = get_db()
    row = con.execute(
        "SELECT cash,initial_cash,starting_capital FROM account WHERE user_id=?", (uid,)
    ).fetchone()
    cash, ic, sc = row if row else (10000.0, 10000.0, 10000.0)

    pos_rows = con.execute(
        "SELECT ticker,shares,avg_cost FROM portfolio WHERE user_id=?", (uid,)
    ).fetchall()
    positions = {r[0]: {"shares": r[1], "avg_cost": r[2]} for r in pos_rows}

    trade_rows = con.execute(
        "SELECT time,side,ticker,shares,price,notional FROM trades WHERE user_id=? ORDER BY id",
        (uid,)
    ).fetchall()
    trades_df = (pd.DataFrame(trade_rows,
                              columns=["Time", "Side", "Ticker", "Shares", "Price", "Notional"])
                 if trade_rows else
                 pd.DataFrame(columns=["Time", "Side", "Ticker", "Shares", "Price", "Notional"]))

    eq_rows = con.execute(
        "SELECT timestamp,equity,trade_count FROM equity_history WHERE user_id=? ORDER BY id",
        (uid,)
    ).fetchall()
    eq_hist = [{"timestamp": r[0], "equity": r[1], "trade_count": r[2]} for r in eq_rows]

    wl_rows = con.execute("SELECT ticker FROM watchlist WHERE user_id=?", (uid,)).fetchall()
    watchlist = [r[0] for r in wl_rows]

    con.close()
    return cash, ic, sc, positions, trades_df, eq_hist, watchlist or ["AAPL", "MSFT", "NVDA", "SPY"]

def db_save_trade(uid, time_str, side, ticker, shares, price, notional):
    con = get_db()
    con.execute(
        "INSERT INTO trades (user_id,time,side,ticker,shares,price,notional) VALUES (?,?,?,?,?,?,?)",
        (uid, time_str, side, ticker, shares, price, notional)
    )
    con.commit(); con.close()

def db_update_position(uid, ticker, shares, avg_cost):
    con = get_db()
    if shares <= 1e-9:
        con.execute("DELETE FROM portfolio WHERE user_id=? AND ticker=?", (uid, ticker))
    else:
        con.execute(
            "INSERT OR REPLACE INTO portfolio (user_id,ticker,shares,avg_cost) VALUES (?,?,?,?)",
            (uid, ticker, shares, avg_cost)
        )
    con.commit(); con.close()

def db_update_account(uid, cash, initial_cash, starting_capital):
    con = get_db()
    con.execute(
        "INSERT OR REPLACE INTO account (user_id,cash,initial_cash,starting_capital) VALUES (?,?,?,?)",
        (uid, cash, initial_cash, starting_capital)
    )
    con.commit(); con.close()

def db_add_equity(uid, equity, trade_count):
    con = get_db()
    con.execute(
        "INSERT INTO equity_history (user_id,timestamp,equity,trade_count) VALUES (?,?,?,?)",
        (uid, datetime.now().isoformat(), equity, trade_count)
    )
    con.commit(); con.close()

def db_update_watchlist(uid, tickers):
    con = get_db()
    con.execute("DELETE FROM watchlist WHERE user_id=?", (uid,))
    for t in tickers:
        con.execute(
            "INSERT INTO watchlist (user_id,ticker,added_at) VALUES (?,?,?)",
            (uid, t, datetime.now().isoformat())
        )
    con.commit(); con.close()

def db_reset_user(uid, starting_capital):
    con = get_db()
    con.execute("DELETE FROM trades          WHERE user_id=?", (uid,))
    con.execute("DELETE FROM portfolio       WHERE user_id=?", (uid,))
    con.execute("DELETE FROM equity_history  WHERE user_id=?", (uid,))
    con.execute(
        "INSERT OR REPLACE INTO account (user_id,cash,initial_cash,starting_capital) VALUES (?,?,?,?)",
        (uid, starting_capital, starting_capital, starting_capital)
    )
    con.commit(); con.close()

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Midas Capital Systems",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Midas Capital Systems v3.0 | Senior Project 2026"}
)

init_db()

# ============================================================
# CSS
# ============================================================

def _inject_html(html: str):
    try:
        st.html(html)
    except AttributeError:
        st.markdown(html, unsafe_allow_html=True)

_inject_html("""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700;900&display=swap" rel="stylesheet">
<style>
html,body,[class*="css"]{
    background:#000!important;
    color:#fff!important;
    font-family:'DM Sans',-apple-system,BlinkMacSystemFont,sans-serif!important;
}
.main>div{padding-top:0!important;}
[data-testid="stSidebar"]{background:#0a0a0a!important;border-right:1px solid #1a1a1a!important;}
[data-testid="stSidebar"] *{color:#ccc!important;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:#000!important;border-bottom:1px solid #1e1e1e!important;gap:0!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#666!important;border-radius:0!important;
    padding:14px 22px!important;font-weight:500!important;border-bottom:2px solid transparent!important;font-size:14px!important;}
.stTabs [aria-selected="true"]{background:transparent!important;color:#E63946!important;
    border-bottom:2px solid #E63946!important;font-weight:700!important;}
.stTabs [data-baseweb="tab"]:hover{color:#fff!important;}
.stTabs [data-baseweb="tab"] p,.stTabs [data-baseweb="tab"] span,.stTabs [data-baseweb="tab"] div{color:inherit!important;}

/* Buttons — Zima Blue, black text */
.stButton>button{
    background:#6FC3DF!important;
    color:#000!important;
    border:none!important;
    border-radius:24px!important;
    padding:10px 28px!important;
    font-weight:700!important;
    font-size:14px!important;
    transition:all 0.2s!important;
}
.stButton>button:hover{background:#5AB3CF!important;transform:scale(1.02)!important;}
/* Force black text even when sidebar overrides to #ccc */
[data-testid="stSidebar"] .stButton>button{color:#000!important;}

/* Inputs */
.stTextInput input,.stNumberInput input{background:#111!important;color:#fff!important;
    border:1px solid #2a2a2a!important;border-radius:8px!important;}
.stSelectbox>div>div{background:#111!important;border:1px solid #2a2a2a!important;border-radius:8px!important;}
.stRadio>div{gap:12px!important;}

/* Login card */
.login-wrap{
    display:flex;justify-content:center;align-items:flex-start;
    min-height:80vh;padding-top:60px;
}
.login-card{
    background:#111;border:1px solid #2a2a2a;border-radius:16px;
    padding:40px 44px;width:100%;max-width:420px;
}
.login-logo{text-align:center;margin-bottom:28px;}
.login-logo .brand{font-size:20px;font-weight:900;color:#fff;letter-spacing:0.5px;}
.login-logo .sub{font-size:11px;color:#3a3a3a;letter-spacing:2px;text-transform:uppercase;margin-top:3px;}
.login-title{font-size:24px;font-weight:900;color:#fff;margin-bottom:6px;}
.login-sub{font-size:13px;color:#555;margin-bottom:28px;}

/* Ticker tape */
.tape-wrap{background:#0a0a0a;border-bottom:1px solid #1a1a1a;padding:9px 0;
    overflow:hidden;white-space:nowrap;font-size:13px;font-weight:500;cursor:default;}
.tape-inner{display:inline-block;animation:tape 90s linear infinite;}
.tape-inner:hover{animation-play-state:paused;}
@keyframes tape{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
.tick{display:inline-block;margin:0 22px;}
.tick-sym{font-weight:700;color:#fff;margin-right:5px;}
.tick-px{color:#999;margin-right:4px;}

/* Status bar */
.sbar{background:#0a0a0a;padding:10px 20px;display:flex;align-items:center;
    gap:24px;border-bottom:1px solid #1a1a1a;font-size:13px;}

/* Metric cards */
.mcard{background:#111;border-radius:12px;padding:18px 20px;border-left:3px solid #2a2a2a;margin-bottom:4px;}
.mcard.up{border-left-color:#00C805;}
.mcard.dn{border-left-color:#FF5000;}
.mlbl{font-size:10px;color:#666;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;}
.mval{font-size:26px;font-weight:900;color:#fff;line-height:1.1;}
.msub{font-size:12px;margin-top:4px;}
.green{color:#00C805;}.red{color:#FF5000;}

/* Cards */
.card{background:#111;border:1px solid #1e1e1e;border-radius:10px;padding:14px 16px;}

/* Badges */
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;}
.badge-win{background:rgba(0,200,5,0.09);color:#00C805;border:1px solid rgba(0,200,5,0.25);}
.badge-loss{background:rgba(255,80,0,0.09);color:#FF5000;border:1px solid rgba(255,80,0,0.25);}
.badge-admin{background:rgba(212,160,23,0.12);color:#D4A017;border:1px solid rgba(212,160,23,0.3);
    padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700;}

/* Market status dot */
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px;
    animation:blink 2s ease-in-out infinite;}
.dot-g{background:#00C805;}.dot-y{background:#f59e0b;}.dot-p{background:#6366f1;}
.dot-x{background:#555;animation:none;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* Section headers */
.sh{font-size:18px;font-weight:700;color:#fff;margin:20px 0 10px;
    padding-bottom:6px;border-bottom:1px solid #1e1e1e;}

/* Alert */
.alert-y{background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.25);color:#f59e0b;
    padding:9px 14px;border-radius:8px;font-size:13px;margin:6px 0;}
.alert-r{background:rgba(255,80,0,0.07);border:1px solid rgba(255,80,0,0.25);color:#FF5000;
    padding:9px 14px;border-radius:8px;font-size:13px;margin:6px 0;}
.alert-g{background:rgba(0,200,5,0.07);border:1px solid rgba(0,200,5,0.25);color:#00C805;
    padding:9px 14px;border-radius:8px;font-size:13px;margin:6px 0;}

/* Admin panel */
.admin-row{display:flex;align-items:center;justify-content:space-between;
    padding:10px 14px;border-bottom:1px solid #1a1a1a;font-size:13px;}
.admin-row:last-child{border-bottom:none;}

hr{border-color:#1e1e1e!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:#000;}
::-webkit-scrollbar-thumb{background:#2a2a2a;border-radius:2px;}
</style>
""")

# ============================================================
# AUTH STATE HELPERS
# ============================================================

def logout():
    """Clear all user-scoped session keys and rerun to login screen."""
    preserve = {}   # nothing to preserve
    keys = list(st.session_state.keys())
    for k in keys:
        del st.session_state[k]
    st.rerun()

def is_authenticated() -> bool:
    return bool(st.session_state.get("user_id"))

# ============================================================
# LOGIN / REGISTER PAGE
# ============================================================

def show_auth_page():
    # Centered card layout
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        _inject_html("""
        <div style="text-align:center;padding:40px 0 28px;">
            <svg width="64" height="64" viewBox="0 0 72 72" xmlns="http://www.w3.org/2000/svg"
                 style="display:block;margin:0 auto 12px;">
                <defs>
                    <linearGradient id="goldRing" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%"   stop-color="#FFD700"/>
                        <stop offset="50%"  stop-color="#D4A017"/>
                        <stop offset="100%" stop-color="#7A5200"/>
                    </linearGradient>
                    <linearGradient id="greenLine" x1="0%" y1="100%" x2="100%" y2="0%">
                        <stop offset="0%"   stop-color="#007003"/>
                        <stop offset="100%" stop-color="#00C805"/>
                    </linearGradient>
                    <linearGradient id="bgFill" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%"   stop-color="#0f0f0f"/>
                        <stop offset="100%" stop-color="#1a1a1a"/>
                    </linearGradient>
                </defs>
                <polygon points="36,3 66,19 66,53 36,69 6,53 6,19"
                         fill="url(#bgFill)" stroke="url(#goldRing)" stroke-width="2.5"/>
                <polygon points="36,8 61,22 61,50 36,64 11,50 11,22"
                         fill="none" stroke="#D4A01730" stroke-width="1"/>
                <polyline points="16,50 16,24 36,42 56,24 56,50"
                          fill="none" stroke="url(#goldRing)"
                          stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
                <polyline points="13,56 22,47 31,51 44,37 59,41"
                          fill="none" stroke="url(#greenLine)"
                          stroke-width="2" stroke-linecap="round" opacity="0.9"/>
            </svg>
            <div style="font-size:22px;font-weight:900;color:#fff;letter-spacing:0.5px;">
                Midas Capital Systems
            </div>
            <div style="font-size:10px;color:#3a3a3a;letter-spacing:2px;text-transform:uppercase;margin-top:4px;">
                Paper Trading Platform
            </div>
        </div>
        """)

        # Tab switcher via radio
        auth_mode = st.radio(
            "", ["Sign In", "Create Account"],
            horizontal=True, key="auth_mode_radio",
            label_visibility="collapsed"
        )

        st.markdown("---")

        if auth_mode == "Sign In":
            st.markdown(
                "<div style='font-size:18px;font-weight:700;margin-bottom:4px;'>Welcome back</div>"
                "<div style='font-size:13px;color:#555;margin-bottom:20px;'>Sign in to your account</div>",
                unsafe_allow_html=True
            )
            login_user = st.text_input("Username", key="login_user", placeholder="Enter username")
            login_pass = st.text_input("Password", key="login_pass",
                                       placeholder="Enter password", type="password")

            if st.button("Sign In", use_container_width=True, key="btn_login"):
                if not login_user.strip() or not login_pass.strip():
                    st.error("Please enter both username and password.")
                else:
                    user = db_authenticate(login_user.strip(), login_pass)
                    if user:
                        # Load user data into session
                        st.session_state.user_id  = user["id"]
                        st.session_state.username = user["username"]
                        st.session_state.is_admin = user["is_admin"]
                        _load_user_into_session(user["id"])
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

            if db_user_count() == 0:
                _inject_html(
                    '<div class="alert-y" style="margin-top:16px;">'
                    '⚠ No accounts yet — use <b>Create Account</b> to register the first admin.'
                    '</div>'
                )

        else:  # Create Account
            is_first = db_user_count() == 0
            if is_first:
                _inject_html(
                    '<div class="alert-g" style="margin-bottom:16px;">'
                    '🔑 First account created will be the <b>Administrator</b>.'
                    '</div>'
                )

            st.markdown(
                "<div style='font-size:18px;font-weight:700;margin-bottom:4px;'>Create account</div>"
                "<div style='font-size:13px;color:#555;margin-bottom:20px;'>Start paper trading</div>",
                unsafe_allow_html=True
            )
            reg_user  = st.text_input("Choose a username", key="reg_user",  placeholder="e.g. trader_john")
            reg_pass  = st.text_input("Password",          key="reg_pass",  placeholder="Min 6 characters", type="password")
            reg_pass2 = st.text_input("Confirm password",  key="reg_pass2", placeholder="Repeat password",  type="password")

            if st.button("Create Account", use_container_width=True, key="btn_register"):
                u = reg_user.strip()
                p = reg_pass.strip()
                p2 = reg_pass2.strip()
                if not u or not p or not p2:
                    st.error("All fields are required.")
                elif len(u) < 3:
                    st.error("Username must be at least 3 characters.")
                elif len(p) < 6:
                    st.error("Password must be at least 6 characters.")
                elif p != p2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = db_create_user(u, p, is_admin=is_first)
                    if ok:
                        st.success(msg + " Please sign in.")
                        st.session_state.auth_mode_radio = "Sign In"
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown(
            "<div style='text-align:center;color:#2a2a2a;font-size:11px;padding:24px 0 0;'>"
            "Simulated trading only · Not financial advice</div>",
            unsafe_allow_html=True
        )

# ============================================================
# SESSION STATE — per-user load
# ============================================================

def _load_user_into_session(uid: int):
    cash, ic, sc, positions, trades, eq_hist, watchlist = db_load_user_state(uid)
    st.session_state.cash                   = cash
    st.session_state.initial_cash           = ic
    st.session_state.starting_capital_input = sc
    st.session_state.positions              = positions
    st.session_state.trades                 = trades
    st.session_state.equity_history         = eq_hist
    st.session_state.watchlist              = watchlist
    st.session_state.win_streak             = 0
    st.session_state.loss_streak            = 0
    st.session_state.last_buy_ts            = {}
    st.session_state.price_mode             = "Live (yfinance)"
    st.session_state.sim_seed               = 42
    st.session_state.sim_period             = "1 Year"
    st.session_state.auto_refresh           = True
    st.session_state.db_loaded              = True

def init_state():
    if not is_authenticated():
        return
    uid = st.session_state.user_id
    if not st.session_state.get("db_loaded"):
        _load_user_into_session(uid)

    for k, v in {
        "price_mode":    "Live (yfinance)",
        "sim_seed":      42,
        "sim_period":    "1 Year",
        "auto_refresh":  True,
        "win_streak":    0,
        "loss_streak":   0,
        "last_buy_ts":   {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ============================================================
# AUTH GATE
# ============================================================

if not is_authenticated():
    show_auth_page()
    st.stop()

init_state()

uid  = st.session_state.user_id
mode = st.session_state.price_mode
seed = int(st.session_state.sim_seed)

# ============================================================
# MARKET HOURS
# ============================================================

def market_status():
    if not PYTZ_AVAILABLE:
        return "open", "Market Open", "dot-g"
    try:
        now = datetime.now(pytz.timezone("US/Eastern"))
        wd  = now.weekday()
        t   = now.hour * 60 + now.minute
        if wd >= 5:
            return "closed", "Markets Closed", "dot-x"
        if 9*60+30 <= t < 16*60:
            return "open",   "Market Open",    "dot-g"
        if 4*60 <= t < 9*60+30:
            return "pre",    "Pre-Market",     "dot-y"
        if 16*60 <= t < 20*60:
            return "after",  "After-Hours",    "dot-p"
        return "closed", "Markets Closed", "dot-x"
    except Exception:
        return "open", "Market Open", "dot-g"

def ext_price(ticker):
    if not YF_AVAILABLE:
        return None, None
    try:
        info = yf.Ticker(ticker).fast_info
        code, _, _ = market_status()
        if code == "after":
            px = getattr(info, "post_market_price", None)
            return px, "After-Hrs"
        if code == "pre":
            px = getattr(info, "pre_market_price", None)
            return px, "Pre-Mkt"
        return None, None
    except Exception:
        return None, None

# ============================================================
# PRICE ENGINE — Live + Improved GBM Simulation
# ============================================================

def _ticker_sector(ticker: str) -> str:
    row = UNIVERSE[UNIVERSE["Ticker"] == ticker]
    return row["Sector"].iloc[0] if len(row) else "Unknown"

def _h(s):
    return abs(hash(s)) % (2**31 - 1)

@st.cache_data(show_spinner=False, ttl=60)
def _live(ticker, period="6mo"):
    if not YF_AVAILABLE:
        return pd.DataFrame(), False
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d")
        if df is None or df.empty:
            return pd.DataFrame(), False
        return df[["Close", "Volume"]].copy(), True
    except Exception:
        return pd.DataFrame(), False

@st.cache_data(show_spinner=False)
def _sim_gbm(ticker: str, days: int = 252, seed_base: int = 42) -> pd.Series:
    """
    Improved simulation using Geometric Brownian Motion with:
    - Sector-appropriate drift (μ) and volatility (σ)
    - Regime switching (bull / bear / sideways)
    - Left-skewed fat-tail shocks via Student-t residuals
    - Mean-reversion overlay (mild)
    - Volume simulation proportional to absolute returns
    """
    sector = _ticker_sector(ticker)
    params = SECTOR_SIM_PARAMS.get(sector, SECTOR_SIM_PARAMS["Unknown"])
    mu     = params["mu"]
    sigma  = params["sigma"]

    rng = np.random.default_rng((_h(ticker) + seed_base) % (2**31 - 1))

    # ── Regime switching ─────────────────────────────────────
    # Regimes: 0=Bull, 1=Sideways, 2=Bear (Markov chain)
    regime_trans = np.array([
        [0.97, 0.02, 0.01],   # Bull  → Bull / Sideways / Bear
        [0.03, 0.94, 0.03],   # Sideways
        [0.02, 0.03, 0.95],   # Bear
    ])
    regime_mu_adj   = [1.5, 0.2, -1.2]   # drift multipliers
    regime_sig_adj  = [0.8, 1.0,  1.6]   # vol multipliers

    regime = 0   # start bull
    regimes = []
    for _ in range(days):
        regime = rng.choice(3, p=regime_trans[regime])
        regimes.append(regime)

    # ── Returns (Student-t with ν=5 for fat tails) ───────────
    nu = 5
    t_shocks = rng.standard_t(nu, size=days) / np.sqrt(nu / (nu - 2))

    returns = np.array([
        (mu * regime_mu_adj[r] - 0.5 * (sigma * regime_sig_adj[r])**2)
        + sigma * regime_sig_adj[r] * t_shocks[i]
        for i, r in enumerate(regimes)
    ])

    # ── Mean-reversion overlay (mild) ────────────────────────
    # Pulls returns back toward long-run mean when too far off
    cumlog = np.cumsum(returns)
    long_mu = mu * days
    mr_strength = 0.003
    for i in range(1, days):
        deviation = cumlog[i-1] - long_mu * (i / days)
        returns[i] -= mr_strength * deviation

    # ── Reconstruct price path ────────────────────────────────
    start_price = ANCHOR_PRICES.get(ticker, rng.uniform(50, 500))
    prices = start_price * np.exp(np.cumsum(returns))

    # ── Volume (correlated with |return|, realistic) ──────────
    base_vol = rng.integers(500_000, 50_000_000)
    volume   = (base_vol * (1 + 5 * np.abs(returns) / sigma)
                * rng.lognormal(0, 0.3, days)).astype(int)

    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="B")
    df  = pd.DataFrame({"Close": prices, "Volume": volume}, index=idx)
    return df

def price_df(ticker, mode, seed, period_key="1 Year"):
    t = ticker.strip().upper()
    if not t:
        return pd.DataFrame(), False
    if mode == "Live (yfinance)":
        # Map period_key to yfinance period string
        yf_period_map = {
            "1 Month": "1mo", "3 Months": "3mo",
            "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y",
        }
        yf_p = yf_period_map.get(period_key, "6mo")
        if t == "DOGE":
            return _live("DOGE-USD", yf_p)
        return _live(t, yf_p)
    else:
        days = SIM_PERIODS.get(period_key, 252)
        df   = _sim_gbm(t, days=days, seed_base=seed)
        return df, True

def cur_price(ticker, mode, seed, period_key="1 Year"):
    df, ok = price_df(ticker, mode, seed, period_key)
    if not ok or df.empty:
        return float("nan")
    return float(df["Close"].iloc[-1])

def ticker_ok(ticker, mode, seed):
    df, ok = price_df(ticker.strip().upper(), mode, seed)
    return ok and not df.empty

# ============================================================
# ML ENGINE
# ============================================================

def tech_indicators(df):
    d = df.copy()
    d["SMA_5"]  = d["Close"].rolling(5).mean()
    d["SMA_20"] = d["Close"].rolling(20).mean()
    d["SMA_50"] = d["Close"].rolling(50).mean()
    d["EMA_12"] = d["Close"].ewm(span=12).mean()
    d["EMA_26"] = d["Close"].ewm(span=26).mean()
    d["MACD"]   = d["EMA_12"] - d["EMA_26"]
    d["MACD_Signal"] = d["MACD"].ewm(span=9).mean()
    delta = d["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d["RSI"]       = 100 - (100 / (1 + gain / loss))
    d["BB_Middle"] = d["Close"].rolling(20).mean()
    std = d["Close"].rolling(20).std()
    d["BB_Upper"] = d["BB_Middle"] + std * 2
    d["BB_Lower"] = d["BB_Middle"] - std * 2
    d["Momentum_5"]  = d["Close"] / d["Close"].shift(5) - 1
    d["Momentum_20"] = d["Close"] / d["Close"].shift(20) - 1
    d["Volatility"]  = d["Close"].rolling(20).std()
    return d

def run_model(df):
    df = tech_indicators(df).dropna()
    if len(df) < 60:
        return None, None
    df["Target"] = df["Close"].shift(-5)
    df = df.dropna()
    if len(df) < 30:
        return None, None

    X, y = df[FEATURE_COLS].values, df["Target"].values
    split = int(len(X) * 0.8)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[:split])
    Xte = scaler.transform(X[split:])

    m = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    m.fit(Xtr, y[:split])

    pred = m.predict(scaler.transform(df[FEATURE_COLS].iloc[-1:].values))[0]
    cur  = df["Close"].iloc[-1]

    last_row = df[FEATURE_COLS].iloc[-1]
    trend_score = int(sum([
        last_row["SMA_5"]       > last_row["SMA_20"],
        last_row["SMA_20"]      > last_row["SMA_50"],
        last_row["MACD"]        > last_row["MACD_Signal"],
        last_row["RSI"]         > 50,
    ]))
    trend_bias = (trend_score - 2) * 0.005
    pred = pred * (1 + trend_bias)

    mape = (np.mean(np.abs((y[split:] - m.predict(Xte)) / y[split:])) * 100
            if len(Xte) else 0)
    conf = max(0, min(100, 100 - mape))

    trend_labels = ["Strong Bearish", "Bearish", "Neutral", "Bullish", "Strong Bullish"]

    return {
        "prediction": pred,
        "current":    cur,
        "change_pct": (pred - cur) / cur * 100,
        "confidence": conf,
        "signal":     "BUY" if pred >= cur else "SELL",
        "trend_score": trend_score,
        "trend_label": trend_labels[trend_score],
        "importance":  dict(zip(FEATURE_COLS, m.feature_importances_))
    }, df

# ============================================================
# PORTFOLIO METRICS
# ============================================================

def portfolio_mv(mode, seed, period_key="1 Year"):
    return sum(
        p["shares"] * cur_price(t, mode, seed, period_key)
        for t, p in st.session_state.positions.items()
        if np.isfinite(cur_price(t, mode, seed, period_key))
    )

def unrealized(mode, seed, period_key="1 Year"):
    return sum(
        p["shares"] * (cur_price(t, mode, seed, period_key) - p["avg_cost"])
        for t, p in st.session_state.positions.items()
        if np.isfinite(cur_price(t, mode, seed, period_key))
    )

def get_metrics(mode, seed, period_key="1 Year"):
    mv     = portfolio_mv(mode, seed, period_key)
    equity = st.session_state.cash + mv
    ic     = st.session_state.initial_cash
    ret    = (equity - ic) / ic * 100 if ic > 0 else 0
    evals  = [e["equity"] for e in st.session_state.equity_history]
    sharpe = 0.0
    if len(evals) > 2:
        rets = np.diff(evals) / np.array(evals[:-1])
        if np.std(rets) > 0:
            sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252)
    return {
        "equity": equity, "mv": mv, "cash": st.session_state.cash,
        "ret": ret, "sharpe": sharpe,
        "upl": unrealized(mode, seed, period_key)
    }

# ============================================================
# TRADING ENGINE
# ============================================================

def apply_capital(amount, period_key="1 Year"):
    amount = float(amount)
    db_reset_user(uid, amount)
    st.session_state.cash                   = amount
    st.session_state.initial_cash           = amount
    st.session_state.starting_capital_input = amount
    st.session_state.positions              = {}
    st.session_state.trades = pd.DataFrame(
        columns=["Time", "Side", "Ticker", "Shares", "Price", "Notional"])
    st.session_state.equity_history = []
    st.session_state.win_streak  = 0
    st.session_state.loss_streak = 0

def place_order(side, ticker, shares, mode, seed, period_key="1 Year"):
    t = ticker.strip().upper()
    if shares <= 0:
        st.error("Shares must be > 0.")
        return False
    if not ticker_ok(t, mode, seed):
        st.error(f"'{t}' not found in current price mode.")
        return False

    px = cur_price(t, mode, seed, period_key)
    if not np.isfinite(px):
        st.error("Could not retrieve price.")
        return False

    notional = shares * px
    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if side == "BUY":
        if notional > st.session_state.cash + 1e-9:
            st.error("Insufficient cash.")
            return False
        st.session_state.cash -= notional
        if t not in st.session_state.positions:
            st.session_state.positions[t] = {"shares": 0.0, "avg_cost": 0.0}
        pos = st.session_state.positions[t]
        new_sh  = pos["shares"] + shares
        new_avg = (pos["shares"] * pos["avg_cost"] + shares * px) / new_sh
        pos["shares"], pos["avg_cost"] = float(new_sh), float(new_avg)
        db_update_position(uid, t, pos["shares"], pos["avg_cost"])
        st.session_state.last_buy_ts[t] = datetime.now()

    else:  # SELL
        if t not in st.session_state.positions or \
                st.session_state.positions[t]["shares"] < shares - 1e-9:
            st.error("Not enough shares to sell.")
            return False
        profit = (px - st.session_state.positions[t]["avg_cost"]) * shares
        if profit >= 0:
            st.session_state.win_streak  += 1
            st.session_state.loss_streak  = 0
        else:
            st.session_state.loss_streak += 1
            st.session_state.win_streak   = 0
            if st.session_state.loss_streak >= 3:
                st.info(random.choice(STOIC_QUOTES))
        st.session_state.cash += notional
        st.session_state.positions[t]["shares"] -= shares
        if st.session_state.positions[t]["shares"] <= 1e-9:
            del st.session_state.positions[t]
            db_update_position(uid, t, 0, 0)
        else:
            p = st.session_state.positions[t]
            db_update_position(uid, t, p["shares"], p["avg_cost"])

    db_save_trade(uid, now_str, side, t, shares, px, notional)
    row = pd.DataFrame([[now_str, side, t, shares, px, notional]],
                       columns=["Time", "Side", "Ticker", "Shares", "Price", "Notional"])
    st.session_state.trades = pd.concat(
        [st.session_state.trades, row], ignore_index=True)

    m  = get_metrics(mode, seed, period_key)
    tc = len(st.session_state.trades)
    st.session_state.equity_history.append(
        {"timestamp": now_str, "equity": m["equity"], "trade_count": tc})
    db_add_equity(uid, m["equity"], tc)
    db_update_account(uid, st.session_state.cash,
                      st.session_state.initial_cash,
                      st.session_state.starting_capital_input)

    st.success(f"Order confirmed: {side} {shares:g} × {t} @ ${px:,.2f}  |  ${notional:,.2f}")
    return True

# ============================================================
# UI HELPERS
# ============================================================

def ticker_tape(mode, seed, period_key="1 Year"):
    tickers = (st.session_state.watchlist +
               [t for t in ["AAPL","MSFT","NVDA","TSLA","AMZN","GOOGL","META","SPY"]
                if t not in st.session_state.watchlist])[:18]
    items = ""
    for t in tickers:
        px = cur_price(t, mode, seed, period_key)
        df_, ok_ = price_df(t, mode, seed, period_key)
        chg = 0.0
        if ok_ and len(df_) > 1:
            prev = df_["Close"].iloc[-2]
            chg  = (px - prev) / prev * 100 if np.isfinite(px) else 0
        col = RH_GREEN if chg >= 0 else RH_RED
        arr = "+" if chg >= 0 else "-"
        pxs = f"${px:,.2f}" if np.isfinite(px) else "—"
        items += (f'<span class="tick">'
                  f'<span class="tick-sym">{t}</span>'
                  f'<span class="tick-px">{pxs}</span>'
                  f'<span style="color:{col}">{arr}{abs(chg):.2f}%</span>'
                  f'</span>')
    _inject_html(
        f'<div class="tape-wrap"><div class="tape-inner">{items}&nbsp;&nbsp;&nbsp;&nbsp;{items}</div></div>'
    )

def status_bar(m, mode, period_key="1 Year"):
    code, label, dot_cls = market_status()
    admin_badge = (' &nbsp;<span class="badge-admin">ADMIN</span>'
                   if st.session_state.get("is_admin") else "")
    rc  = RH_GREEN if m["ret"] >= 0 else RH_RED
    arr = "+" if m["ret"] >= 0 else "-"
    ah_note = ""
    if code in ("pre", "after"):
        ah_note = (f"<span style='color:#f59e0b;font-size:11px;margin-left:16px;'>"
                   f"{'Pre-market' if code=='pre' else 'After-hours'} — prices reflect last close</span>")
    _inject_html(f"""
    <div class="sbar">
        <div style="font-size:16px;font-weight:900;color:#fff;">
            Midas Capital Systems
            <span style="font-size:12px;color:#444;font-weight:400;margin-left:10px;">
                {st.session_state.username}{admin_badge}
            </span>
        </div>
        <div style="margin-left:auto;display:flex;align-items:center;gap:28px;">
            <div>
                <div style="font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1px;">Portfolio</div>
                <div style="font-size:16px;font-weight:700;">${m['equity']:,.2f}
                    <span style="color:{rc};font-size:13px;">{arr}{abs(m['ret']):.2f}%</span>
                </div>
            </div>
            <div>
                <div style="font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1px;">Cash</div>
                <div style="font-size:16px;font-weight:700;">${m['cash']:,.2f}</div>
            </div>
            <div>
                <span class="dot {dot_cls}"></span>
                <span style="font-weight:600;">{label}</span>
                {ah_note}
            </div>
        </div>
    </div>
    """)

def mcard(label, value, sub=None, direction=None):
    cls     = "up" if direction == "up" else ("dn" if direction == "dn" else "")
    sub_cls = "green" if direction == "up" else ("red" if direction == "dn" else "")
    sub_html = f'<div class="msub {sub_cls}">{sub}</div>' if sub else ""
    _inject_html(f"""
    <div class="mcard {cls}">
        <div class="mlbl">{label}</div>
        <div class="mval">{value}</div>
        {sub_html}
    </div>""")

# ============================================================
# SIDEBAR
# ============================================================

period_key = st.session_state.get("sim_period", "1 Year")
m = get_metrics(mode, seed, period_key)

with st.sidebar:
    _inject_html("""
    <div style="padding:20px 0 14px;text-align:center;">
        <svg width="64" height="64" viewBox="0 0 72 72" xmlns="http://www.w3.org/2000/svg"
             style="display:block;margin:0 auto 10px;">
            <defs>
                <linearGradient id="gr" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#FFD700"/>
                    <stop offset="50%" stop-color="#D4A017"/>
                    <stop offset="100%" stop-color="#7A5200"/>
                </linearGradient>
                <linearGradient id="gl" x1="0%" y1="100%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#007003"/>
                    <stop offset="100%" stop-color="#00C805"/>
                </linearGradient>
                <linearGradient id="gb" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#0f0f0f"/>
                    <stop offset="100%" stop-color="#1a1a1a"/>
                </linearGradient>
            </defs>
            <polygon points="36,3 66,19 66,53 36,69 6,53 6,19"
                     fill="url(#gb)" stroke="url(#gr)" stroke-width="2.5"/>
            <polyline points="16,50 16,24 36,42 56,24 56,50"
                      fill="none" stroke="url(#gr)"
                      stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
            <polyline points="13,56 22,47 31,51 44,37 59,41"
                      fill="none" stroke="url(#gl)"
                      stroke-width="2" stroke-linecap="round" opacity="0.9"/>
        </svg>
        <div style="font-size:14px;font-weight:900;color:#fff;letter-spacing:0.5px;">
            Midas Capital Systems
        </div>
        <div style="font-size:9px;color:#3a3a3a;letter-spacing:2px;text-transform:uppercase;margin-top:3px;">
            Andrew Ignatius · 2026
        </div>
    </div>
    """)
    _inject_html("<hr>")

    # User info
    admin_html = ' <span class="badge-admin">ADMIN</span>' if st.session_state.get("is_admin") else ""
    _inject_html(f"""
    <div style="margin-bottom:14px;">
        <div style="font-size:13px;color:#888;">Signed in as</div>
        <div style="font-size:15px;font-weight:700;color:#fff;margin-top:2px;">
            {st.session_state.username}{admin_html}
        </div>
    </div>""")

    if st.button("Sign Out", use_container_width=True):
        logout()

    _inject_html("<hr>")

    # Market status
    code, lbl, dot_cls = market_status()
    mkt_col = {"open":"#00C805","pre":"#f59e0b","after":"#6366f1","closed":"#555"}.get(code,"#555")
    _inject_html(f"""
    <div style="text-align:center;margin-bottom:14px;">
        <span style="background:{mkt_col}18;color:{mkt_col};padding:4px 14px;
                     border-radius:20px;font-size:12px;font-weight:700;
                     border:1px solid {mkt_col}40;">● {lbl}</span>
    </div>""")

    # Account card
    rc  = RH_GREEN if m["ret"] >= 0 else RH_RED
    arr = "+" if m["ret"] >= 0 else "-"
    _inject_html(f"""
    <div class="card" style="margin-bottom:14px;">
        <div class="mlbl">Account Value</div>
        <div style="font-size:22px;font-weight:900;color:#fff;">${m['equity']:,.2f}</div>
        <div style="font-size:13px;color:{rc};margin-top:2px;">{arr}{abs(m['ret']):.2f}% all-time</div>
        <div style="font-size:11px;color:#444;margin-top:6px;">
            Cash ${m['cash']:,.2f} &middot; Held ${m['mv']:,.2f}
        </div>
    </div>""")

    # Data mode
    _inject_html('<div style="font-size:10px;color:#444;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Data Mode</div>')
    st.session_state.price_mode = st.radio(
        "mode", ["Live (yfinance)", "Simulated (Offline)"],
        index=0 if st.session_state.price_mode.startswith("Live") else 1,
        label_visibility="collapsed"
    )
    mode = st.session_state.price_mode

    if mode == "Simulated (Offline)":
        _inject_html('<div style="font-size:10px;color:#444;text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px;">Simulation Period</div>')
        st.session_state.sim_period = st.selectbox(
            "Period", list(SIM_PERIODS.keys()),
            index=list(SIM_PERIODS.keys()).index(
                st.session_state.get("sim_period", "1 Year")),
            label_visibility="collapsed"
        )
        _inject_html('<div style="font-size:10px;color:#444;text-transform:uppercase;letter-spacing:1px;margin:8px 0 4px;">Random Seed</div>')
        st.session_state.sim_seed = st.number_input(
            "Seed", 0, 999999, int(st.session_state.sim_seed), 1,
            label_visibility="collapsed",
            help="Change seed to generate different market scenarios."
        )
        _inject_html("""
        <div style="font-size:11px;color:#555;margin:6px 0 10px;line-height:1.5;">
            GBM simulation with regime switching, fat-tail shocks, and sector-calibrated parameters.
            Each ticker's behaviour matches its real-world sector characteristics.
        </div>""")

    period_key = st.session_state.get("sim_period", "1 Year")
    seed       = int(st.session_state.sim_seed)

    st.session_state.auto_refresh = st.checkbox(
        "Auto-refresh prices (60s)", value=st.session_state.auto_refresh)

    _inject_html("<hr>")

    # Starting capital
    _inject_html('<div style="font-size:10px;color:#444;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Starting Capital</div>')
    cap = st.number_input(
        "cap", min_value=1.0,
        value=float(st.session_state.starting_capital_input),
        step=1000.0, format="%.2f", label_visibility="collapsed"
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Apply", use_container_width=True):
            apply_capital(cap, period_key); st.rerun()
    with c2:
        if st.button("Reset", use_container_width=True):
            apply_capital(float(st.session_state.starting_capital_input), period_key); st.rerun()

    _inject_html("<hr>")

    # Streaks
    if st.session_state.win_streak >= 2:
        _inject_html(f'<span class="badge badge-win">{st.session_state.win_streak}-Trade Win Streak 🔥</span>')
    elif st.session_state.loss_streak >= 2:
        _inject_html(f'<span class="badge badge-loss">{st.session_state.loss_streak} Losses in a Row</span>')

# ============================================================
# MAIN CONTENT
# ============================================================

ticker_tape(mode, seed, period_key)
m = get_metrics(mode, seed, period_key)
status_bar(m, mode, period_key)

mkt_code, _, _ = market_status()
if mkt_code in ("pre", "after", "closed"):
    msgs = {
        "pre":    "Pre-market (4:00 AM – 9:30 AM ET). Prices reflect last regular-session close.",
        "after":  "After-hours (4:00 PM – 8:00 PM ET). Extended prices shown where available.",
        "closed": "Markets are closed. Prices reflect the last regular-session close.",
    }
    _inject_html(f'<div class="alert-y">{msgs[mkt_code]}</div>')

# ============================================================
# TABS
# ============================================================

tab_list = ["Dashboard", "AI Insights", "Trade", "Portfolio", "Performance", "Heatmap"]
if st.session_state.get("is_admin"):
    tab_list.append("Admin")

tabs = st.tabs(tab_list)
tab1, tab2, tab3, tab4, tab5, tab6 = tabs[:6]
tab_admin = tabs[6] if st.session_state.get("is_admin") else None

# ============================================================
# TAB 1 — DASHBOARD
# ============================================================

with tab1:
    _inject_html('<div class="sh">Overview</div>')
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        d = "up" if m["ret"] >= 0 else "dn"
        mcard("Total Equity", f"${m['equity']:,.2f}",
              f"{'+'if m['ret']>=0 else ''}{m['ret']:.2f}%", d)
    with c2:
        cp = m["cash"] / m["equity"] * 100 if m["equity"] > 0 else 0
        mcard("Cash Available", f"${m['cash']:,.2f}", f"{cp:.1f}% of equity")
    with c3:
        hp = m["mv"] / m["equity"] * 100 if m["equity"] > 0 else 0
        mcard("Holdings Value", f"${m['mv']:,.2f}", f"{hp:.1f}% of equity")
    with c4:
        upl  = m["upl"]
        ud   = "up" if upl >= 0 else "dn"
        uarr = "+" if upl >= 0 else "-"
        up_pct = upl / st.session_state.initial_cash * 100 if st.session_state.initial_cash > 0 else 0
        mcard("Unrealized P/L", f"${upl:,.2f}", f"{uarr}{abs(up_pct):.2f}%", ud)

    _inject_html("<br>")
    wl_col, stats_col = st.columns([3, 1])

    with wl_col:
        _inject_html('<div class="sh">Watchlist</div>')
        rows = []
        for t in st.session_state.watchlist:
            px   = cur_price(t, mode, seed, period_key)
            df_, ok_ = price_df(t, mode, seed, period_key)
            chg  = 0.0
            if ok_ and len(df_) > 1:
                prev = df_["Close"].iloc[-2]
                chg  = (px - prev) / prev * 100 if np.isfinite(px) else 0
            ep, el = ext_price(t) if mode == "Live (yfinance)" else (None, None)
            rows.append({
                "Ticker": t,
                "Price":  f"${px:,.2f}" if np.isfinite(px) else "—",
                "Change": f"{'+'if chg>=0 else ''}{chg:.2f}%",
                "Ext-Hrs": f"${ep:,.2f} ({el})" if ep else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        ca, cb = st.columns(2)
        with ca:
            nt = st.text_input("Add ticker", placeholder="e.g. AAPL", key="wl_add")
            if st.button("Add to Watchlist"):
                t_ = nt.strip().upper()
                if t_ and ticker_ok(t_, mode, seed):
                    if t_ not in st.session_state.watchlist:
                        st.session_state.watchlist.append(t_)
                        db_update_watchlist(uid, st.session_state.watchlist)
                        st.success(f"Added {t_}")
                        st.rerun()
                    else:
                        st.info(f"{t_} already in watchlist.")
                else:
                    st.error("Invalid ticker.")
        with cb:
            if st.session_state.watchlist:
                rm = st.selectbox("Remove", st.session_state.watchlist, key="wl_rm")
                if st.button("Remove from Watchlist"):
                    st.session_state.watchlist.remove(rm)
                    db_update_watchlist(uid, st.session_state.watchlist)
                    st.rerun()

    with stats_col:
        _inject_html('<div class="sh">Stats</div>')
        _inject_html(f"""
        <div class="card" style="margin-bottom:8px;">
            <div class="mlbl">Total Trades</div>
            <div style="font-size:22px;font-weight:900;">{len(st.session_state.trades)}</div>
        </div>
        <div class="card" style="margin-bottom:8px;">
            <div class="mlbl">Open Positions</div>
            <div style="font-size:22px;font-weight:900;">{len(st.session_state.positions)}</div>
        </div>
        <div class="card">
            <div class="mlbl">Sharpe Ratio</div>
            <div style="font-size:22px;font-weight:900;">{m['sharpe']:.2f}</div>
        </div>
        """)

# ============================================================
# TAB 2 — AI INSIGHTS
# ============================================================

with tab2:
    _inject_html('<div class="sh">Machine Learning Price Predictions</div>')
    st.caption("Random Forest · 14 technical features · 5-day horizon · Trend Alignment Score")

    ai_opts = sorted(set(st.session_state.watchlist) | set(ALL_TICKERS))
    ai_tkr  = st.selectbox("Select ticker", ai_opts)

    pred_col, mood_col = st.columns([2, 1])

    with pred_col:
        if st.button("Generate Prediction", type="primary"):
            with st.spinner("Training on historical data…"):
                df_, ok_ = price_df(ai_tkr, mode, seed, period_key)
                if ok_ and len(df_) > 60:
                    res, edf = run_model(df_)
                    if res:
                        r1, r2, r3 = st.columns(3)
                        with r1: mcard("Current Price", f"${res['current']:,.2f}")
                        with r2:
                            d_ = "up" if res["change_pct"] >= 0 else "dn"
                            mcard("5-Day Forecast", f"${res['prediction']:,.2f}",
                                  f"{'+'if res['change_pct']>=0 else ''}{res['change_pct']:.2f}%", d_)
                        with r3: mcard("Model Confidence", f"{res['confidence']:.0f}%")

                        sc_ = RH_GREEN if res["signal"] == "BUY" else RH_RED
                        _inject_html(f"""
                        <div style="background:{sc_}18;border:2px solid {sc_};color:{sc_};
                                    padding:14px;border-radius:10px;text-align:center;
                                    font-size:18px;font-weight:900;letter-spacing:2px;margin:12px 0;">
                            {"&#x2B06;" if res['signal']=='BUY' else "&#x2B07;"}
                            MODEL SIGNAL: {res['signal']}
                        </div>""")

                        # Trend Alignment panel
                        ta_score = res["trend_score"]
                        ta_label = res["trend_label"]
                        ta_col   = (RH_GREEN if ta_score >= 3 else
                                    (RH_RED if ta_score <= 1 else "#f59e0b"))
                        bar_fill = int(ta_score / 4 * 100)
                        _inject_html(f"""
                        <div class="card" style="margin:10px 0;">
                            <div class="mlbl">Trend Alignment Score</div>
                            <div style="display:flex;align-items:center;gap:12px;margin-top:6px;">
                                <div style="font-size:22px;font-weight:900;color:{ta_col};">
                                    {ta_score}/4
                                </div>
                                <div style="flex:1;">
                                    <div style="background:#1e1e1e;border-radius:4px;height:8px;">
                                        <div style="background:{ta_col};width:{bar_fill}%;height:8px;
                                                    border-radius:4px;"></div>
                                    </div>
                                    <div style="font-size:12px;color:{ta_col};margin-top:4px;
                                                font-weight:700;">{ta_label}</div>
                                </div>
                            </div>
                            <div style="font-size:11px;color:#555;margin-top:8px;">
                                SMA5&gt;SMA20 &nbsp;|&nbsp; SMA20&gt;SMA50
                                &nbsp;|&nbsp; MACD crossover &nbsp;|&nbsp; RSI &gt; 50
                            </div>
                        </div>""")

                        # Feature importance
                        imp = pd.DataFrame({
                            "Feature":    list(res["importance"].keys()),
                            "Importance": list(res["importance"].values())
                        }).sort_values("Importance").tail(10)
                        fig_i = go.Figure(go.Bar(
                            x=imp["Importance"], y=imp["Feature"],
                            orientation="h",
                            marker=dict(color=RH_GREEN)
                        ))
                        fig_i.update_layout(
                            title="Top 10 Predictive Features",
                            plot_bgcolor="#111111", paper_bgcolor="#111111",
                            font=dict(color="#fff"), height=320,
                            xaxis=dict(gridcolor="#2a2a2a"),
                            yaxis=dict(gridcolor="#2a2a2a"),
                            margin=dict(l=10,r=10,t=40,b=10)
                        )
                        st.plotly_chart(fig_i, use_container_width=True)

                        # Technical charts
                        fig_t = make_subplots(
                            rows=3, cols=1,
                            subplot_titles=("Price & Moving Averages","RSI","MACD"),
                            vertical_spacing=0.08,
                            row_heights=[0.5,0.25,0.25]
                        )
                        fig_t.add_trace(go.Scatter(x=edf.index,y=edf["Close"],name="Close",
                            line=dict(color=RH_GREEN,width=2)), row=1,col=1)
                        fig_t.add_trace(go.Scatter(x=edf.index,y=edf["SMA_20"],name="SMA20",
                            line=dict(color="#6366f1",dash="dash")), row=1,col=1)
                        fig_t.add_trace(go.Scatter(x=edf.index,y=edf["SMA_50"],name="SMA50",
                            line=dict(color="#f59e0b",dash="dash")), row=1,col=1)
                        fig_t.add_trace(go.Scatter(x=edf.index,y=edf["BB_Upper"],name="BB Upper",
                            line=dict(color="#555",width=1,dash="dot")), row=1,col=1)
                        fig_t.add_trace(go.Scatter(x=edf.index,y=edf["BB_Lower"],name="BB Lower",
                            line=dict(color="#555",width=1,dash="dot"),
                            fill="tonexty",fillcolor="rgba(111,195,223,0.18)"), row=1,col=1)
                        fig_t.add_trace(go.Scatter(x=edf.index,y=edf["RSI"],name="RSI",
                            line=dict(color="#ec4899")), row=2,col=1)
                        fig_t.add_hline(y=70,line_dash="dash",line_color=RH_RED,row=2,col=1)
                        fig_t.add_hline(y=30,line_dash="dash",line_color=RH_GREEN,row=2,col=1)
                        fig_t.add_trace(go.Scatter(x=edf.index,y=edf["MACD"],name="MACD",
                            line=dict(color="#06b6d4")), row=3,col=1)
                        fig_t.add_trace(go.Scatter(x=edf.index,y=edf["MACD_Signal"],name="Signal",
                            line=dict(color="#f97316")), row=3,col=1)
                        fig_t.update_layout(
                            height=780,
                            plot_bgcolor="#111111",paper_bgcolor="#111111",
                            font=dict(color="#fff"),
                            legend=dict(bgcolor="#111111",bordercolor="#2a2a2a"),
                        )
                        for r in range(1,4):
                            fig_t.update_xaxes(gridcolor="#2a2a2a",row=r,col=1)
                            fig_t.update_yaxes(gridcolor="#2a2a2a",row=r,col=1)
                        st.plotly_chart(fig_t, use_container_width=True)
                    else:
                        st.error("Insufficient data to train model.")
                else:
                    st.error("Not enough historical data for this ticker.")

    with mood_col:
        _inject_html('<div class="sh">Market Mood</div>')
        rsi_vals = []
        for t_ in st.session_state.watchlist[:8]:
            df_, ok_ = price_df(t_, mode, seed, period_key)
            if ok_ and len(df_) > 20:
                d  = df_["Close"].diff()
                g  = d.where(d>0,0).rolling(14).mean()
                l  = (-d.where(d<0,0)).rolling(14).mean()
                r  = 100 - 100/(1 + g/l)
                if np.isfinite(r.iloc[-1]):
                    rsi_vals.append(r.iloc[-1])
        avg_rsi  = np.nanmean(rsi_vals) if rsi_vals else 50
        mood_info = (
            ("GREED",    RH_RED)    if avg_rsi >= 70 else
            ("OPTIMISM", "#f59e0b") if avg_rsi >= 55 else
            ("NEUTRAL",  "#888")    if avg_rsi >= 45 else
            ("CAUTION",  "#6366f1") if avg_rsi >= 30 else
            ("FEAR",     RH_GREEN)
        )
        mood_lbl, mood_c = mood_info
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_rsi,
            title=dict(text="Avg RSI", font=dict(color="#888",size=12)),
            gauge=dict(
                axis=dict(range=[0,100],tickcolor="#444"),
                bar=dict(color=mood_c),
                bgcolor="#1a1a1a",bordercolor="#2a2a2a",
                steps=[
                    dict(range=[0,30],  color="rgba(0,200,5,0.06)"),
                    dict(range=[30,45], color="rgba(99,102,241,0.06)"),
                    dict(range=[45,55], color="rgba(136,136,136,0.06)"),
                    dict(range=[55,70], color="rgba(245,158,11,0.06)"),
                    dict(range=[70,100],color="rgba(255,80,0,0.06)"),
                ],
            ),
            number=dict(font=dict(color="#fff",size=28))
        ))
        fig_g.update_layout(
            height=230,paper_bgcolor="#111111",
            font=dict(color="#fff"),margin=dict(l=20,r=20,t=40,b=10)
        )
        st.plotly_chart(fig_g, use_container_width=True)
        _inject_html(f'<div style="text-align:center;font-size:20px;font-weight:900;color:{mood_c};margin-top:-10px;">{mood_lbl}</div>')

# ============================================================
# TAB 3 — TRADE
# ============================================================

with tab3:
    _inject_html('<div class="sh">Trading Interface</div>')
    l_, r_ = st.columns([1, 2])

    with l_:
        pick   = st.selectbox("Ticker", ALL_TICKERS)
        manual = st.text_input("Or type manually", placeholder="e.g. AAPL")
        tkr    = manual.strip().upper() if manual.strip() else pick

        side       = st.radio("Order type", ["BUY","SELL"], horizontal=True)
        shares_in  = st.number_input("Shares", min_value=0.0, value=1.0, step=1.0)

        px_   = cur_price(tkr, mode, seed, period_key)
        ep_, el_ = ext_price(tkr) if mode == "Live (yfinance)" else (None, None)

        if np.isfinite(px_):
            ext_html = (f"<div style='color:#f59e0b;font-size:11px;margin-top:4px;'>"
                        f"{el_}: ${ep_:,.2f}</div>") if ep_ else ""
            _inject_html(f"""
            <div class="card" style="margin:12px 0;">
                <div class="mlbl">Current Price</div>
                <div class="mval">${px_:,.2f}</div>
                {ext_html}
            </div>
            <div class="card">
                <div class="mlbl">Order Value</div>
                <div class="mval">${shares_in*px_:,.2f}</div>
            </div>""")

        _inject_html("<br>")
        if st.button(f"{'Buy' if side=='BUY' else 'Sell'} {tkr}",
                     use_container_width=True):
            if place_order(side, tkr, float(shares_in), mode, seed, period_key):
                st.rerun()

        if mkt_code in ("after","pre","closed") and mode == "Live (yfinance)":
            _inject_html('<div style="color:#f59e0b;font-size:11px;margin-top:8px;">Outside market hours — prices are last close</div>')

    with r_:
        df_, ok_ = price_df(tkr, mode, seed, period_key)
        if ok_ and not df_.empty:
            pnow   = df_["Close"].iloc[-1]
            pstart = df_["Close"].iloc[0]
            up     = pnow >= pstart
            lc     = RH_GREEN if up else RH_RED
            fc     = "rgba(0,200,5,0.07)" if up else "rgba(255,80,0,0.07)"
            fig_c  = go.Figure()
            fig_c.add_trace(go.Scatter(
                x=df_.index, y=df_["Close"],
                mode="lines", line=dict(color=lc,width=2),
                fill="tozeroy", fillcolor=fc,
                hovertemplate="%{x|%b %d}<br>$%{y:,.2f}<extra></extra>"
            ))
            fig_c.add_trace(go.Scatter(
                x=[df_.index[-1]], y=[pnow],
                mode="markers", marker=dict(size=8,color=lc)
            ))
            # Volume bar (secondary y) if available
            if "Volume" in df_.columns:
                fig_c.add_trace(go.Bar(
                    x=df_.index, y=df_["Volume"],
                    name="Volume",
                    marker_color="rgba(111,195,223,0.15)",
                    yaxis="y2"
                ))
                fig_c.update_layout(
                    yaxis2=dict(
                        overlaying="y", side="right",
                        showgrid=False, showticklabels=False
                    )
                )
            fig_c.update_layout(
                title=dict(text=f"{tkr} · ${pnow:,.2f} · {period_key}",
                           font=dict(color="#fff",size=18)),
                plot_bgcolor="#111111",paper_bgcolor="#111111",
                font=dict(color="#fff"),height=430,
                xaxis=dict(gridcolor="#2a2a2a"),
                yaxis=dict(gridcolor="#2a2a2a"),
                hovermode="x unified",showlegend=False,
                margin=dict(l=10,r=10,t=50,b=10)
            )
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.error("No price data available.")

# ============================================================
# TAB 4 — PORTFOLIO
# ============================================================

with tab4:
    _inject_html('<div class="sh">Open Positions</div>')

    if not st.session_state.positions:
        st.info("No open positions. Head to Trade to get started.")
    else:
        pos_rows = []
        for t_, p_ in st.session_state.positions.items():
            px_  = cur_price(t_, mode, seed, period_key)
            mv_  = p_["shares"] * px_
            upl_ = p_["shares"] * (px_ - p_["avg_cost"])
            cb_  = p_["shares"] * p_["avg_cost"]
            up_  = upl_ / cb_ * 100 if cb_ > 0 else 0
            sec_ = UNIVERSE[UNIVERSE["Ticker"]==t_]["Sector"]
            sec_ = sec_.iloc[0] if len(sec_) else "Unknown"
            pos_rows.append({
                "Ticker": t_, "Sector": sec_, "Shares": p_["shares"],
                "Avg Cost": f"${p_['avg_cost']:.2f}", "Price": f"${px_:.2f}",
                "Mkt Value": f"${mv_:,.2f}","P/L": f"${upl_:,.2f}",
                "Return": f"{up_:+.2f}%"
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

        if st.button("Suggest Equal-Weight Rebalance"):
            n   = len(st.session_state.positions)
            tv  = sum(p_["shares"] * cur_price(t_, mode, seed, period_key)
                      for t_, p_ in st.session_state.positions.items())
            tgt = tv / n
            sugg = []
            for t_, p_ in st.session_state.positions.items():
                cv   = p_["shares"] * cur_price(t_, mode, seed, period_key)
                px_  = cur_price(t_, mode, seed, period_key)
                diff = tgt - cv
                act  = "Buy" if diff > 0 else "Sell"
                sh   = abs(diff) / px_ if px_ > 0 else 0
                sugg.append(f"{'+'if diff>0 else '-'} {act} {sh:.2f} {t_}  (${abs(diff):,.0f})")
            _inject_html(
                '<div class="card"><div class="mlbl">Rebalance Preview — not executed</div>' +
                "".join(f"<div style='padding:3px 0;color:#ccc;font-size:13px;'>{s}</div>"
                        for s in sugg) + "</div>"
            )

        _inject_html("<br>")
        pc_, cc_ = st.columns(2)

        with pc_:
            _inject_html('<div class="sh">Sector Allocation</div>')
            sv = {}
            for t_, p_ in st.session_state.positions.items():
                px_  = cur_price(t_, mode, seed, period_key)
                sec_ = UNIVERSE[UNIVERSE["Ticker"]==t_]["Sector"]
                sec_ = sec_.iloc[0] if len(sec_) else "Unknown"
                sv[sec_] = sv.get(sec_, 0) + p_["shares"] * px_
            tv = sum(sv.values())
            fig_d = go.Figure(go.Pie(
                labels=list(sv.keys()), values=list(sv.values()), hole=0.55,
                marker=dict(colors=[SECTOR_COLORS.get(s,"#475569") for s in sv],
                            line=dict(color="#000",width=2)),
                textinfo="label+percent",
                textfont=dict(size=12,color="#fff"),
                hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
                direction="clockwise"
            ))
            fig_d.update_layout(
                paper_bgcolor="#111111",plot_bgcolor="#111111",
                font=dict(color="#fff"),height=380,showlegend=True,
                legend=dict(bgcolor="#111111",bordercolor="#2a2a2a",font=dict(color="#fff")),
                margin=dict(l=10,r=10,t=10,b=10),
                annotations=[dict(text=f"${tv:,.0f}",x=0.5,y=0.5,
                                  font=dict(size=16,color="#fff"),showarrow=False)]
            )
            st.plotly_chart(fig_d, use_container_width=True)

        with cc_:
            _inject_html('<div class="sh">Return Correlation</div>')
            held = list(st.session_state.positions.keys())
            if len(held) >= 2:
                cdata = {}
                for t_ in held:
                    df_, ok_ = price_df(t_, mode, seed, period_key)
                    if ok_ and len(df_) > 30:
                        cdata[t_] = df_["Close"].pct_change().dropna().tail(60)
                if len(cdata) >= 2:
                    cm = pd.DataFrame(cdata).dropna().corr()
                    fig_cr = go.Figure(go.Heatmap(
                        z=cm.values, x=cm.columns.tolist(), y=cm.index.tolist(),
                        colorscale=[[0,RH_RED],[0.5,"#2a2a2a"],[1,RH_GREEN]],
                        zmin=-1,zmax=1,
                        text=np.round(cm.values,2),texttemplate="%{text}",
                        textfont=dict(color="#fff",size=11),
                        hovertemplate="%{x} x %{y}: %{z:.2f}<extra></extra>"
                    ))
                    fig_cr.update_layout(
                        paper_bgcolor="#111111",plot_bgcolor="#111111",
                        font=dict(color="#fff"),height=380,
                        margin=dict(l=10,r=10,t=10,b=10),
                        xaxis=dict(color="#888"),yaxis=dict(color="#888")
                    )
                    st.plotly_chart(fig_cr, use_container_width=True)
                else:
                    st.info("Need price data for at least 2 held stocks.")
            else:
                st.info("Hold 2 or more positions to see correlation.")

# ============================================================
# TAB 5 — PERFORMANCE
# ============================================================

with tab5:
    _inject_html('<div class="sh">Performance Analytics</div>')
    eq = st.session_state.equity_history

    if len(eq) > 1:
        evals  = [e["equity"] for e in eq]
        tcount = list(range(1, len(evals)+1))
        ic     = st.session_state.initial_cash
        ret_   = [(v - ic) / ic * 100 for v in evals]
        is_up  = evals[-1] >= ic
        lc     = RH_GREEN if is_up else RH_RED

        mn, mx     = min(evals), max(evals)
        data_range = mx - mn
        pad        = max(data_range * 0.15, 10)

        # Equity curve
        fig_eq = go.Figure()
        fig_eq.add_hline(y=ic, line_dash="dot", line_color="#444",
                         annotation_text="Starting Capital",
                         annotation_font_color="#555",
                         annotation_position="right")
        fig_eq.add_trace(go.Scatter(
            x=tcount, y=evals, mode="lines+markers", name="Equity",
            line=dict(color=lc,width=3),
            marker=dict(size=7,color=lc,line=dict(width=1,color="#000")),
            fill="tozeroy",
            fillcolor=f"rgba({'0,200,5' if is_up else '255,80,0'},0.07)",
            hovertemplate="Trade #%{x}<br>$%{y:,.2f}<extra></extra>"
        ))
        fig_eq.update_layout(
            title=dict(
                text=f"Equity Curve · {'+'if is_up else ''}{ret_[-1]:.2f}% total return",
                font=dict(color=lc,size=16)
            ),
            xaxis=dict(title="Trade Number",gridcolor="#2a2a2a",color="#888",
                       tickmode="linear",dtick=max(1,len(tcount)//10)),
            yaxis=dict(title="Value ($)",gridcolor="#2a2a2a",color="#888",
                       range=[mn-pad, mx+pad]),
            plot_bgcolor="#111111",paper_bgcolor="#111111",
            font=dict(color="#fff"),height=400,
            hovermode="x unified",showlegend=False,
            margin=dict(l=10,r=10,t=50,b=10)
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # Return bars
        fig_r = go.Figure(go.Bar(
            x=tcount, y=ret_,
            marker_color=[RH_GREEN if r>=0 else RH_RED for r in ret_],
            hovertemplate="Trade #%{x}<br>%{y:+.2f}%<extra></extra>"
        ))
        fig_r.update_layout(
            title=dict(text="Cumulative Return % After Each Trade",
                       font=dict(color="#fff",size=14)),
            xaxis=dict(title="Trade #",gridcolor="#2a2a2a",color="#888"),
            yaxis=dict(title="Return (%)",gridcolor="#2a2a2a",color="#888"),
            plot_bgcolor="#111111",paper_bgcolor="#111111",
            font=dict(color="#fff"),height=220,
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig_r, use_container_width=True)

        # Monte Carlo
        if len(evals) >= 5:
            _inject_html('<div class="sh">Monte Carlo Projection — 30 Trading Days</div>')
            rets_arr = np.diff(evals) / np.array(evals[:-1])
            mu_  = np.mean(rets_arr)
            sig_ = np.std(rets_arr)
            np.random.seed(42)
            n_s, n_d = 500, 30
            sims = np.zeros((n_s, n_d+1))
            sims[:,0] = evals[-1]
            for d in range(1, n_d+1):
                sims[:,d] = sims[:,d-1] * (1 + np.random.normal(mu_,sig_,n_s))
            dx = list(range(n_d+1))
            p10,p25,p50,p75,p90 = [np.percentile(sims,p,axis=0) for p in [10,25,50,75,90]]

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(
                x=dx+dx[::-1], y=list(p90)+list(p10[::-1]),
                fill="toself",fillcolor="rgba(0,200,5,0.04)",
                line=dict(color="rgba(0,0,0,0)"),name="80% CI"))
            fig_mc.add_trace(go.Scatter(
                x=dx+dx[::-1], y=list(p75)+list(p25[::-1]),
                fill="toself",fillcolor="rgba(0,200,5,0.10)",
                line=dict(color="rgba(0,0,0,0)"),name="50% CI"))
            fig_mc.add_trace(go.Scatter(
                x=dx, y=p50, mode="lines", name="Median",
                line=dict(color=RH_GREEN,width=2)))
            fig_mc.add_hline(y=evals[-1],line_dash="dot",line_color="#444")
            fig_mc.update_layout(
                title=dict(text=f"Monte Carlo: 500 simulations — Median ${p50[-1]:,.0f}",
                           font=dict(color="#fff")),
                xaxis=dict(title="Days Forward",gridcolor="#2a2a2a",color="#888"),
                yaxis=dict(title="Portfolio Value ($)",gridcolor="#2a2a2a",color="#888"),
                plot_bgcolor="#111111",paper_bgcolor="#111111",
                font=dict(color="#fff"),height=340,
                legend=dict(bgcolor="#111111",bordercolor="#2a2a2a"),
                margin=dict(l=10,r=10,t=50,b=10)
            )
            st.plotly_chart(fig_mc, use_container_width=True)
    else:
        st.info("Execute trades to populate the performance chart.")

    _inject_html('<div class="sh">Trade History</div>')
    if not st.session_state.trades.empty:
        st.dataframe(
            st.session_state.trades.sort_values("Time",ascending=False).head(50),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No trades yet.")

# ============================================================
# TAB 6 — HEATMAP
# ============================================================

with tab6:
    _inject_html('<div class="sh">Portfolio Heatmap</div>')

    if st.session_state.positions:
        hm = []
        for t_, p_ in st.session_state.positions.items():
            px_  = cur_price(t_, mode, seed, period_key)
            mv_  = p_["shares"] * px_
            up_  = (px_ - p_["avg_cost"]) / p_["avg_cost"] * 100 if p_["avg_cost"] > 0 else 0
            sec_ = UNIVERSE[UNIVERSE["Ticker"]==t_]["Sector"]
            sec_ = sec_.iloc[0] if len(sec_) else "Unknown"
            hm.append({"Ticker":t_,"Sector":sec_,"Market Value":mv_,"Return %":round(up_,2)})
        hm_df = pd.DataFrame(hm)

        hm_sectors = hm_df["Sector"].unique().tolist()
        hm_labels  = ["Portfolio"] + hm_sectors + hm_df["Ticker"].tolist()
        hm_parents = [""] + ["Portfolio"]*len(hm_sectors) + hm_df["Sector"].tolist()
        hm_values  = [0] + [0]*len(hm_sectors) + hm_df["Market Value"].tolist()
        hm_colors  = (
            [0]
            + [hm_df[hm_df["Sector"]==s]["Return %"].mean() for s in hm_sectors]
            + hm_df["Return %"].tolist()
        )
        hm_custom  = np.array(
            [[0, 0]]
            + [[hm_df[hm_df["Sector"]==s]["Return %"].mean(),
                hm_df[hm_df["Sector"]==s]["Market Value"].sum()] for s in hm_sectors]
            + [[row["Return %"],row["Market Value"]] for _,row in hm_df.iterrows()]
        )
        fig_hm = go.Figure(go.Treemap(
            labels=hm_labels, parents=hm_parents, values=hm_values,
            customdata=hm_custom,
            texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%",
            hovertemplate="<b>%{label}</b><br>Value: $%{customdata[1]:,.2f}<br>Return: %{customdata[0]:+.2f}%<extra></extra>",
            marker=dict(
                colors=hm_colors,
                colorscale=[[0,RH_RED],[0.5,"#1a1a1a"],[1,RH_GREEN]],
                cmid=0, showscale=True,
                colorbar=dict(
                    title=dict(text="Return %",font=dict(color="#888")),
                    tickfont=dict(color="#888")
                )
            ),
            textfont=dict(color="#fff",size=14),
            branchvalues="remainder"
        ))
        fig_hm.update_layout(
            paper_bgcolor="#111111",font=dict(color="#fff"),
            height=460,margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No positions to display.")

    _inject_html('<div class="sh">S&P 500 Universe — Daily Change</div>')

    univ_rows = []
    for _, row in UNIVERSE.iterrows():
        px_      = cur_price(row["Ticker"], mode, seed, period_key)
        df_, ok_ = price_df(row["Ticker"], mode, seed, period_key)
        chg_     = 0.0
        if ok_ and len(df_) > 1:
            prev = df_["Close"].iloc[-2]
            chg_ = (px_ - prev) / prev * 100 if np.isfinite(px_) else 0
        univ_rows.append({
            "Ticker": row["Ticker"], "Sector": row["Sector"],
            "Price":  max(px_, 1) if np.isfinite(px_) else 1,
            "Change %": round(chg_, 2)
        })
    univ_df = pd.DataFrame(univ_rows)

    u_sectors = univ_df["Sector"].unique().tolist()
    u_labels  = ["Universe"] + u_sectors + univ_df["Ticker"].tolist()
    u_parents = [""] + ["Universe"]*len(u_sectors) + univ_df["Sector"].tolist()
    u_values  = [0] + [0]*len(u_sectors) + univ_df["Price"].tolist()
    u_colors  = (
        [0]
        + [univ_df[univ_df["Sector"]==s]["Change %"].mean() for s in u_sectors]
        + univ_df["Change %"].tolist()
    )
    u_custom = np.array(
        [[0]]
        + [[univ_df[univ_df["Sector"]==s]["Change %"].mean()] for s in u_sectors]
        + [[row["Change %"]] for _, row in univ_df.iterrows()]
    )
    fig_u = go.Figure(go.Treemap(
        labels=u_labels, parents=u_parents, values=u_values,
        customdata=u_custom,
        texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%",
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>Daily: %{customdata[0]:+.2f}%<extra></extra>",
        marker=dict(
            colors=u_colors,
            colorscale=[[0,RH_RED],[0.5,"#1a1a1a"],[1,RH_GREEN]],
            cmid=0, showscale=True,
        ),
        textfont=dict(color="#fff",size=13),
        branchvalues="remainder"
    ))
    fig_u.update_layout(
        paper_bgcolor="#111111",font=dict(color="#fff"),
        height=420,margin=dict(l=10,r=10,t=10,b=10)
    )
    st.plotly_chart(fig_u, use_container_width=True)

# ============================================================
# TAB 7 — ADMIN PANEL  (only shown to admins)
# ============================================================

if tab_admin is not None:
    with tab_admin:
        _inject_html('<div class="sh">Administrator Panel</div>')

        users = db_all_users()
        total_users = len(users)

        # Summary metrics
        a1, a2 = st.columns(2)
        with a1:
            _inject_html(f"""
            <div class="card" style="margin-bottom:12px;">
                <div class="mlbl">Total Accounts</div>
                <div style="font-size:28px;font-weight:900;">{total_users}</div>
            </div>""")
        with a2:
            admin_count = sum(1 for u in users if u["is_admin"])
            _inject_html(f"""
            <div class="card" style="margin-bottom:12px;">
                <div class="mlbl">Administrators</div>
                <div style="font-size:28px;font-weight:900;">{admin_count}</div>
            </div>""")

        _inject_html('<div class="sh">All Accounts</div>')

        # User table
        user_rows = []
        for u in users:
            con = get_db()
            tc  = con.execute("SELECT COUNT(*) FROM trades WHERE user_id=?", (u["id"],)).fetchone()[0]
            acc = con.execute("SELECT cash,initial_cash FROM account WHERE user_id=?", (u["id"],)).fetchone()
            con.close()
            cash_val = acc[0] if acc else 0
            ic_val   = acc[1] if acc else 0
            pnl = cash_val - ic_val
            user_rows.append({
                "ID": u["id"],
                "Username":  u["username"],
                "Role":      "Admin" if u["is_admin"] else "Trader",
                "Trades":    tc,
                "Cash":      f"${cash_val:,.2f}",
                "Created":   u["created_at"][:10],
            })
        st.dataframe(pd.DataFrame(user_rows), use_container_width=True, hide_index=True)

        _inject_html('<div class="sh">Account Actions</div>')
        action_col1, action_col2 = st.columns(2)

        with action_col1:
            _inject_html('<div style="font-size:13px;font-weight:700;color:#fff;margin-bottom:10px;">Delete Account</div>')
            other_users = [u for u in users if u["id"] != uid]
            if other_users:
                del_options = {f"{u['username']} (ID {u['id']})": u["id"] for u in other_users}
                del_choice  = st.selectbox("Select account to delete",
                                           list(del_options.keys()), key="admin_del")
                del_uid = del_options[del_choice]
                _inject_html('<div class="alert-r">⚠ This permanently deletes the account and all associated data.</div>')
                confirm = st.checkbox("I confirm this action is irreversible", key="del_confirm")
                if st.button("Delete Account", key="btn_delete") and confirm:
                    db_delete_user(del_uid)
                    st.success(f"Account deleted.")
                    st.rerun()
            else:
                st.info("No other accounts to delete.")

        with action_col2:
            _inject_html('<div style="font-size:13px;font-weight:700;color:#fff;margin-bottom:10px;">Reset Account Portfolio</div>')
            reset_options = {f"{u['username']} (ID {u['id']})": u["id"] for u in users}
            rst_choice    = st.selectbox("Select account to reset",
                                         list(reset_options.keys()), key="admin_rst")
            rst_uid       = reset_options[rst_choice]
            rst_capital   = st.number_input("Reset to capital ($)", min_value=1.0,
                                             value=10000.0, step=1000.0, key="rst_cap")
            if st.button("Reset Portfolio", key="btn_rst_port"):
                db_reset_user(rst_uid, rst_capital)
                # If resetting own account, reload session
                if rst_uid == uid:
                    _load_user_into_session(uid)
                st.success(f"Portfolio reset to ${rst_capital:,.2f}.")
                st.rerun()

        _inject_html('<div class="sh">Create New Account</div>')
        nc1, nc2, nc3, nc4 = st.columns(4)
        with nc1:
            new_uname = st.text_input("Username", key="admin_new_u", placeholder="username")
        with nc2:
            new_pw    = st.text_input("Password", key="admin_new_p",
                                      placeholder="password", type="password")
        with nc3:
            new_admin = st.checkbox("Admin privileges", key="admin_new_adm")
        with nc4:
            st.write("")
            st.write("")
            if st.button("Create Account", key="btn_admin_create"):
                if not new_uname.strip() or not new_pw.strip():
                    st.error("Username and password required.")
                elif len(new_uname.strip()) < 3:
                    st.error("Username must be ≥ 3 characters.")
                elif len(new_pw.strip()) < 6:
                    st.error("Password must be ≥ 6 characters.")
                else:
                    ok, msg = db_create_user(new_uname.strip(), new_pw.strip(), is_admin=new_admin)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        _inject_html('<div class="sh">Change Password</div>')
        cp1, cp2, cp3 = st.columns(3)
        with cp1:
            cp_options = {f"{u['username']} (ID {u['id']})": u["id"] for u in users}
            cp_choice  = st.selectbox("Account", list(cp_options.keys()), key="cp_sel")
            cp_uid     = cp_options[cp_choice]
        with cp2:
            new_cpw = st.text_input("New password", key="admin_cpw",
                                    placeholder="New password", type="password")
        with cp3:
            st.write("")
            st.write("")
            if st.button("Update Password", key="btn_cpw"):
                if len(new_cpw.strip()) < 6:
                    st.error("Password must be ≥ 6 characters.")
                else:
                    db_change_password(cp_uid, new_cpw.strip())
                    st.success("Password updated.")

# ============================================================
# AUTO-REFRESH
# ============================================================

if st.session_state.auto_refresh and mode == "Live (yfinance)":
    time.sleep(60)
    st.rerun()

# ============================================================
# FOOTER
# ============================================================

_inject_html("""
<hr>
<div style="text-align:center;color:#2a2a2a;font-size:11px;padding:16px 0;">
    &copy; 2026 Midas Capital Systems &middot; Andrew Ignatius &middot; Senior Capstone Project<br>
    <span style="font-size:10px;">Simulated trading only. Not financial advice.</span>
</div>
""")
