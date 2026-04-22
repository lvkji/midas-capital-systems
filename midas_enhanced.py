# ============================================================
# MIDAS CAPITAL SYSTEMS v2.0
# AI-Powered Paper Trading Platform
# Author: Andrew Ignatius | Senior Capstone Project 2026
# ============================================================

import os, time, sqlite3, random
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

DB_PATH = "midas_capital.db"

RH_GREEN = "#00C805"
RH_RED   = "#FF5000"
RH_GOLD  = "#D4A017"

UNIVERSE = pd.DataFrame([
    ("AAPL","Apple","Technology"),
    ("MSFT","Microsoft","Technology"),
    ("NVDA","NVIDIA","Technology"),
    ("GOOGL","Alphabet","Communication Services"),
    ("META","Meta","Communication Services"),
    ("AMZN","Amazon","Consumer Discretionary"),
    ("TSLA","Tesla","Consumer Discretionary"),
    ("HD","Home Depot","Consumer Discretionary"),
    ("JPM","JPMorgan Chase","Financials"),
    ("V","Visa","Financials"),
    ("BAC","Bank of America","Financials"),
    ("XOM","Exxon Mobil","Energy"),
    ("CVX","Chevron","Energy"),
    ("LLY","Eli Lilly","Health Care"),
    ("UNH","UnitedHealth","Health Care"),
    ("JNJ","Johnson & Johnson","Health Care"),
    ("KO","Coca-Cola","Consumer Staples"),
    ("PEP","PepsiCo","Consumer Staples"),
    ("WMT","Walmart","Consumer Staples"),
    ("DIS","Disney","Communication Services"),
    ("CAT","Caterpillar","Industrials"),
    ("BA","Boeing","Industrials"),
    ("SPY","SPDR S&P 500 ETF","ETF"),
], columns=["Ticker","Name","Sector"])

SECTOR_COLORS = {
    "Technology":"#6366f1","Communication Services":"#06b6d4",
    "Consumer Discretionary":"#f59e0b","Financials":"#10b981",
    "Energy":"#f97316","Health Care":"#ec4899",
    "Consumer Staples":"#84cc16","Industrials":"#8b5cf6",
    "ETF":"#94a3b8","Unknown":"#475569",
}

ALL_TICKERS = sorted(UNIVERSE["Ticker"].tolist())

STOIC_QUOTES = [
    '"The impediment to action advances action. What stands in the way becomes the way." — Marcus Aurelius',
    '"Wealth consists not in having great possessions, but in having few wants." — Epictetus',
    '"He who is brave is free." — Seneca',
    '"Never let the future disturb you." — Marcus Aurelius',
    '"It is not the man who has too little, but the man who craves more, that is poor." — Seneca',
]

FUND_ADJ  = ["Aggressive","Tactical","Sovereign","Apex","Quantum","Alpha","Strategic","Dynamic"]
FUND_NOUN = ["Penguin","Falcon","Rhino","Tiger","Condor","Panther","Jaguar","Phoenix"]

FEATURE_COLS = [
    "SMA_5","SMA_20","SMA_50","EMA_12","EMA_26",
    "MACD","MACD_Signal","RSI","BB_Middle","BB_Upper",
    "BB_Lower","Momentum_5","Momentum_20","Volatility"
]

# ============================================================
# DATABASE LAYER — SQLite Persistence
# ============================================================

def get_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = get_db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            time     TEXT NOT NULL,
            side     TEXT NOT NULL,
            ticker   TEXT NOT NULL,
            shares   REAL NOT NULL,
            price    REAL NOT NULL,
            notional REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio (
            ticker   TEXT PRIMARY KEY,
            shares   REAL NOT NULL,
            avg_cost REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS account (
            id              INTEGER PRIMARY KEY CHECK(id=1),
            cash            REAL NOT NULL,
            initial_cash    REAL NOT NULL,
            starting_capital REAL NOT NULL DEFAULT 10000.0
        );
        CREATE TABLE IF NOT EXISTS equity_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            equity      REAL NOT NULL,
            trade_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker   TEXT PRIMARY KEY,
            added_at TEXT NOT NULL
        );
    """)
    con.execute(
        "INSERT OR IGNORE INTO account (id,cash,initial_cash,starting_capital) VALUES (1,10000,10000,10000)"
    )
    for t in ["AAPL","MSFT","NVDA","SPY"]:
        con.execute("INSERT OR IGNORE INTO watchlist (ticker,added_at) VALUES (?,?)",
                    (t, datetime.now().isoformat()))
    con.commit()
    con.close()

def db_load_state():
    con = get_db()
    cur = con.cursor()

    cur.execute("SELECT cash,initial_cash,starting_capital FROM account WHERE id=1")
    row = cur.fetchone()
    cash, ic, sc = row if row else (10000.0, 10000.0, 10000.0)

    cur.execute("SELECT ticker,shares,avg_cost FROM portfolio")
    positions = {r[0]: {"shares": r[1], "avg_cost": r[2]} for r in cur.fetchall()}

    cur.execute("SELECT time,side,ticker,shares,price,notional FROM trades ORDER BY id")
    rows = cur.fetchall()
    trades_df = pd.DataFrame(rows, columns=["Time","Side","Ticker","Shares","Price","Notional"]) \
                if rows else pd.DataFrame(columns=["Time","Side","Ticker","Shares","Price","Notional"])

    cur.execute("SELECT timestamp,equity,trade_count FROM equity_history ORDER BY id")
    eq_hist = [{"timestamp": r[0], "equity": r[1], "trade_count": r[2]} for r in cur.fetchall()]

    cur.execute("SELECT ticker FROM watchlist")
    watchlist = [r[0] for r in cur.fetchall()]

    con.close()
    return cash, ic, sc, positions, trades_df, eq_hist, watchlist

def db_save_trade(time_str, side, ticker, shares, price, notional):
    con = get_db()
    con.execute("INSERT INTO trades (time,side,ticker,shares,price,notional) VALUES (?,?,?,?,?,?)",
                (time_str, side, ticker, shares, price, notional))
    con.commit(); con.close()

def db_update_position(ticker, shares, avg_cost):
    con = get_db()
    if shares <= 1e-9:
        con.execute("DELETE FROM portfolio WHERE ticker=?", (ticker,))
    else:
        con.execute("INSERT OR REPLACE INTO portfolio (ticker,shares,avg_cost) VALUES (?,?,?)",
                    (ticker, shares, avg_cost))
    con.commit(); con.close()

def db_update_account(cash, initial_cash, starting_capital):
    con = get_db()
    con.execute("INSERT OR REPLACE INTO account (id,cash,initial_cash,starting_capital) VALUES (1,?,?,?)",
                (cash, initial_cash, starting_capital))
    con.commit(); con.close()

def db_add_equity(equity, trade_count):
    con = get_db()
    con.execute("INSERT INTO equity_history (timestamp,equity,trade_count) VALUES (?,?,?)",
                (datetime.now().isoformat(), equity, trade_count))
    con.commit(); con.close()

def db_update_watchlist(tickers):
    con = get_db()
    con.execute("DELETE FROM watchlist")
    for t in tickers:
        con.execute("INSERT INTO watchlist (ticker,added_at) VALUES (?,?)",
                    (t, datetime.now().isoformat()))
    con.commit(); con.close()

def db_reset(starting_capital):
    con = get_db()
    con.executescript("DELETE FROM trades; DELETE FROM portfolio; DELETE FROM equity_history;")
    con.execute("INSERT OR REPLACE INTO account (id,cash,initial_cash,starting_capital) VALUES (1,?,?,?)",
                (starting_capital, starting_capital, starting_capital))
    con.commit(); con.close()

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Midas Capital Systems",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Midas Capital Systems v2.0 | Senior Project 2026"}
)

init_db()

# ============================================================
# CSS — ROBINHOOD DARK THEME
# ============================================================

# ── st.html() injects raw HTML without stripping <style> tags (Streamlit 1.31+)
# ── Falls back to st.markdown for older installs
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
.stTabs [aria-selected="true"]{background:transparent!important;color:#fff!important;
    border-bottom:2px solid #00C805!important;font-weight:700!important;}
.stTabs [data-baseweb="tab"]:hover{color:#fff!important;}
.stTabs [data-baseweb="tab"] p,.stTabs [data-baseweb="tab"] span,.stTabs [data-baseweb="tab"] div{color:inherit!important;}

/* Buttons */
.stButton>button{background:#00C805!important;color:#000!important;border:none!important;
    border-radius:24px!important;padding:10px 28px!important;font-weight:700!important;
    font-size:14px!important;transition:all 0.2s!important;}
.stButton>button:hover{background:#00a003!important;transform:scale(1.02)!important;}

/* Inputs */
.stTextInput input,.stNumberInput input{background:#111!important;color:#fff!important;
    border:1px solid #2a2a2a!important;border-radius:8px!important;}
.stSelectbox>div>div{background:#111!important;border:1px solid #2a2a2a!important;border-radius:8px!important;}

/* Radio */
.stRadio>div{gap:12px!important;}

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

/* Market status */
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px;
    animation:blink 2s ease-in-out infinite;}
.dot-g{background:#00C805;}.dot-y{background:#f59e0b;}.dot-p{background:#6366f1;}
.dot-x{background:#555;animation:none;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* God mode */
.godmode{background:linear-gradient(90deg,#b8860b,#FFD700,#b8860b);
    color:#000;padding:8px;text-align:center;font-weight:900;font-size:13px;
    letter-spacing:2px;animation:glow 2s ease-in-out infinite;}
@keyframes glow{0%,100%{opacity:1}50%{opacity:.75}}

/* Section headers */
.sh{font-size:18px;font-weight:700;color:#fff;margin:20px 0 10px;
    padding-bottom:6px;border-bottom:1px solid #1e1e1e;}

/* Alert */
.alert-y{background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.25);color:#f59e0b;
    padding:9px 14px;border-radius:8px;font-size:13px;margin:6px 0;}
hr{border-color:#1e1e1e!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:#000;}
::-webkit-scrollbar-thumb{background:#2a2a2a;border-radius:2px;}
</style>
""")

# ============================================================
# SESSION STATE — loads from SQLite on first run
# ============================================================

def init_state():
    if "db_loaded" not in st.session_state:
        cash, ic, sc, positions, trades, eq_hist, wl = db_load_state()
        st.session_state.cash                    = cash
        st.session_state.initial_cash            = ic
        st.session_state.starting_capital_input  = sc
        st.session_state.positions               = positions
        st.session_state.trades                  = trades
        st.session_state.equity_history          = eq_hist
        st.session_state.watchlist               = wl or ["AAPL","MSFT","NVDA","SPY"]
        st.session_state.db_loaded               = True

    for k, v in {
        "price_mode":    "Live (yfinance)",
        "sim_seed":      42,
        "auto_refresh":  True,
        "god_mode":      False,
        "win_streak":    0,
        "loss_streak":   0,
        "midas_shown":   False,
        "fund_name":     None,
        "last_buy_ts":   {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ============================================================
# MARKET HOURS
# ============================================================

def market_status():
    """Returns (code, label, dot_class) — code: open|pre|after|closed"""
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
    """Try to get pre/after-market price from yfinance info dict."""
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
# PRICE ENGINE
# ============================================================

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
        return df[["Close","Volume"]].copy(), True
    except Exception:
        return pd.DataFrame(), False

@st.cache_data(show_spinner=False)
def _sim(ticker, days=365, seed_base=42):
    rng = np.random.default_rng((_h(ticker) + seed_base) % (2**31-1))
    rets   = rng.normal(rng.normal(3e-4,2e-4), rng.uniform(0.01,0.03), days)
    prices = rng.uniform(50,300) * np.exp(np.cumsum(rets))
    idx    = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="B")
    return pd.Series(prices, index=idx, name="Close")

def price_df(ticker, mode, seed):
    t = ticker.strip().upper()
    if not t:
        return pd.DataFrame(), False
    # DOGE support
    if t == "DOGE" and mode == "Live (yfinance)":
        return _live("DOGE-USD")
    if mode == "Live (yfinance)":
        return _live(t)
    s = _sim(t, seed_base=seed)
    return pd.DataFrame({"Close": s}), True

def cur_price(ticker, mode, seed):
    df, ok = price_df(ticker, mode, seed)
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
    gain  = delta.where(delta>0,0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0)).rolling(14).mean()
    d["RSI"]       = 100 - (100/(1+gain/loss))
    d["BB_Middle"] = d["Close"].rolling(20).mean()
    std = d["Close"].rolling(20).std()
    d["BB_Upper"] = d["BB_Middle"] + std*2
    d["BB_Lower"] = d["BB_Middle"] - std*2
    d["Momentum_5"]  = d["Close"]/d["Close"].shift(5) - 1
    d["Momentum_20"] = d["Close"]/d["Close"].shift(20) - 1
    d["Volatility"]  = d["Close"].rolling(20).std()
    return d

def run_model(df, god=False):
    df = tech_indicators(df).dropna()
    if len(df) < 60:
        return None, None
    df["Target"] = df["Close"].shift(-5)
    df = df.dropna()
    if len(df) < 30:
        return None, None

    X, y = df[FEATURE_COLS].values, df["Target"].values
    split = int(len(X)*0.8)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[:split])
    Xte = scaler.transform(X[split:])

    m = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    m.fit(Xtr, y[:split])

    pred  = m.predict(scaler.transform(df[FEATURE_COLS].iloc[-1:].values))[0]
    cur   = df["Close"].iloc[-1]
    if god:
        pred = cur * 1.09

    mape = np.mean(np.abs((y[split:]-m.predict(Xte))/y[split:]))*100 if len(Xte) else 0
    conf = 99.0 if god else max(0, min(100, 100-mape))

    return {
        "prediction": pred, "current": cur,
        "change_pct": (pred-cur)/cur*100,
        "confidence": conf,
        "signal": "BUY",
        "importance": dict(zip(FEATURE_COLS, m.feature_importances_))
    }, df

# ============================================================
# PORTFOLIO METRICS
# ============================================================

def portfolio_mv(mode, seed):
    return sum(
        p["shares"] * cur_price(t, mode, seed)
        for t, p in st.session_state.positions.items()
        if np.isfinite(cur_price(t, mode, seed))
    )

def unrealized(mode, seed):
    return sum(
        p["shares"] * (cur_price(t, mode, seed) - p["avg_cost"])
        for t, p in st.session_state.positions.items()
        if np.isfinite(cur_price(t, mode, seed))
    )

def get_metrics(mode, seed):
    mv     = portfolio_mv(mode, seed)
    equity = st.session_state.cash + mv
    ic     = st.session_state.initial_cash
    ret    = (equity-ic)/ic*100 if ic > 0 else 0
    evals  = [e["equity"] for e in st.session_state.equity_history]
    sharpe = 0.0
    if len(evals) > 2:
        rets = np.diff(evals)/np.array(evals[:-1])
        if np.std(rets) > 0:
            sharpe = np.mean(rets)/np.std(rets)*np.sqrt(252)
    return {"equity": equity, "mv": mv, "cash": st.session_state.cash,
            "ret": ret, "sharpe": sharpe, "upl": unrealized(mode, seed)}

# ============================================================
# TRADING ENGINE
# ============================================================

def apply_capital(amount):
    amount = float(amount)
    db_reset(amount)
    st.session_state.cash   = amount
    st.session_state.initial_cash = amount
    st.session_state.starting_capital_input = amount
    st.session_state.positions = {}
    st.session_state.trades = pd.DataFrame(
        columns=["Time","Side","Ticker","Shares","Price","Notional"])
    st.session_state.equity_history = []
    st.session_state.win_streak = 0
    st.session_state.loss_streak = 0
    st.session_state.midas_shown = False

def place_order(side, ticker, shares, mode, seed):
    t = ticker.strip().upper()
    if shares <= 0:
        st.error("Shares must be > 0.")
        return False
    if not ticker_ok(t, mode, seed) and t not in ("DOGE",):
        st.error(f"'{t}' not found in current price mode.")
        return False

    px = cur_price(t, mode, seed)
    if not np.isfinite(px):
        st.error("Could not retrieve price.")
        return False

    notional = shares * px
    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Easter egg: Warren Buffett warning ──────────────────
    if side == "SELL" and t in st.session_state.last_buy_ts:
        secs = (datetime.now() - st.session_state.last_buy_ts[t]).total_seconds()
        if secs < 10:
            st.warning("⚠️ *Warren Buffett has been holding Coca-Cola since 1988. Just saying.*")

    if side == "BUY":
        if notional > st.session_state.cash + 1e-9:
            st.error("Insufficient cash.")
            return False
        st.session_state.cash -= notional
        if t not in st.session_state.positions:
            st.session_state.positions[t] = {"shares": 0.0, "avg_cost": 0.0}
        pos = st.session_state.positions[t]
        new_sh  = pos["shares"] + shares
        new_avg = (pos["shares"]*pos["avg_cost"] + shares*px) / new_sh
        pos["shares"], pos["avg_cost"] = float(new_sh), float(new_avg)
        db_update_position(t, pos["shares"], pos["avg_cost"])
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
            # ── Easter egg: Loss streak stoic quote ──────────
            if st.session_state.loss_streak >= 3:
                st.info(f"📜 {random.choice(STOIC_QUOTES)}")
        st.session_state.cash += notional
        st.session_state.positions[t]["shares"] -= shares
        if st.session_state.positions[t]["shares"] <= 1e-9:
            del st.session_state.positions[t]
            db_update_position(t, 0, 0)
        else:
            p = st.session_state.positions[t]
            db_update_position(t, p["shares"], p["avg_cost"])

    db_save_trade(now_str, side, t, shares, px, notional)
    row = pd.DataFrame([[now_str,side,t,shares,px,notional]],
                        columns=["Time","Side","Ticker","Shares","Price","Notional"])
    st.session_state.trades = pd.concat([st.session_state.trades, row], ignore_index=True)

    m = get_metrics(mode, seed)
    tc = len(st.session_state.trades)
    st.session_state.equity_history.append(
        {"timestamp": now_str, "equity": m["equity"], "trade_count": tc})
    db_add_equity(m["equity"], tc)
    db_update_account(st.session_state.cash, st.session_state.initial_cash,
                      st.session_state.starting_capital_input)

    # ── Easter egg: Penny stock mode ─────────────────────────
    if st.session_state.starting_capital_input < 100:
        st.success(f"🪙 Budget Edition — {side} {shares:g} × {t} @ ${px:,.4f}")
    else:
        st.success(f"✅ {side} {shares:g} × {t} @ ${px:,.2f}  |  ${notional:,.2f}")

    # ── Easter egg: Midas Touch ──────────────────────────────
    if not st.session_state.midas_shown and st.session_state.initial_cash > 0:
        if abs(m["equity"] - st.session_state.initial_cash * 2) < 1.0:
            st.balloons()
            _inject_html(
                '<div class="godmode">✨ YOU\'VE ACHIEVED THE MIDAS TOUCH ✨</div>'
            )
            st.session_state.midas_shown = True
    return True

# ============================================================
# UI HELPERS
# ============================================================

def ticker_tape(mode, seed):
    tickers = (st.session_state.watchlist +
               [t for t in ["AAPL","MSFT","NVDA","TSLA","AMZN","GOOGL","META","SPY"]
                if t not in st.session_state.watchlist])[:18]
    items = ""
    for t in tickers:
        px = cur_price(t, mode, seed)
        df, ok = price_df(t, mode, seed)
        chg = 0.0
        if ok and len(df) > 1:
            prev = df["Close"].iloc[-2]
            chg  = (px-prev)/prev*100 if np.isfinite(px) else 0
        col = RH_GREEN if chg >= 0 else RH_RED
        arr = "▲" if chg >= 0 else "▼"
        pxs = f"${px:,.2f}" if np.isfinite(px) else "—"
        items += (f'<span class="tick">'
                  f'<span class="tick-sym">{t}</span>'
                  f'<span class="tick-px">{pxs}</span>'
                  f'<span style="color:{col}">{arr}{abs(chg):.2f}%</span>'
                  f'</span>')
    _inject_html(
        f'<div class="tape-wrap"><div class="tape-inner">{items}&nbsp;&nbsp;&nbsp;&nbsp;{items}</div></div>'
    )

def status_bar(m, mode):
    code, label, dot_cls = market_status()
    title = "Midas Capital Systems"
    if st.session_state.starting_capital_input < 100:
        title += " 🪙 Budget Edition"
    god_badge = " &nbsp;<span style='color:#D4A017;font-weight:900;'>👑 GOD MODE</span>" \
                if st.session_state.god_mode else ""
    rc = RH_GREEN if m["ret"] >= 0 else RH_RED
    arr = "▲" if m["ret"] >= 0 else "▼"
    # After-hours note
    ah_note = ""
    if code in ("pre","after"):
        ah_note = (f"<span style='color:#f59e0b;font-size:11px;margin-left:16px;'>"
                   f"⏰ {'Pre-market' if code=='pre' else 'After-hours'} — prices reflect last close</span>")
    _inject_html(f"""
    <div class="sbar">
        <div style="font-size:16px;font-weight:900;color:#fff;">{title}{god_badge}</div>
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
    cls = ("up" if direction == "up" else "dn" if direction == "dn" else "")
    sub_cls = ("green" if direction == "up" else "red" if direction == "dn" else "")
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

mode = st.session_state.price_mode
seed = int(st.session_state.sim_seed)
m    = get_metrics(mode, seed)

with st.sidebar:
    # Logo
    _inject_html("""
    <div style="padding:20px 0 14px;text-align:center;">

        <!-- Custom SVG Logo -->
        <svg width="72" height="72" viewBox="0 0 72 72" xmlns="http://www.w3.org/2000/svg"
             style="display:block;margin:0 auto 10px;">
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
                <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
                    <feGaussianBlur stdDeviation="1.5" result="blur"/>
                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
                <clipPath id="hexClip">
                    <polygon points="36,3 66,19 66,53 36,69 6,53 6,19"/>
                </clipPath>
            </defs>

            <!-- Hex background -->
            <polygon points="36,3 66,19 66,53 36,69 6,53 6,19"
                     fill="url(#bgFill)" stroke="url(#goldRing)" stroke-width="2.5"/>

            <!-- Inner hex subtle glow ring -->
            <polygon points="36,8 61,22 61,50 36,64 11,50 11,22"
                     fill="none" stroke="#D4A01730" stroke-width="1"/>

            <!-- M letterform — two diagonal strokes meeting in centre -->
            <polyline points="16,50 16,24 36,42 56,24 56,50"
                      fill="none" stroke="url(#goldRing)"
                      stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>

            <!-- Upward trend line (green, glowing) -->
            <polyline points="13,56 22,47 31,51 44,37 59,41"
                      fill="none" stroke="url(#greenLine)"
                      stroke-width="2" stroke-linecap="round"
                      filter="url(#glow)" opacity="0.9"/>

            <!-- Dot at trend peak -->
            <circle cx="44" cy="37" r="2.2" fill="#00C805" filter="url(#glow)"/>
        </svg>

        <!-- Wordmark -->
        <div style="font-size:15px;font-weight:900;color:#fff;letter-spacing:0.5px;
                    font-family:'DM Sans',sans-serif;line-height:1.2;">
            Midas Capital Systems
        </div>
        <div style="font-size:9px;color:#3a3a3a;letter-spacing:2px;
                    text-transform:uppercase;margin-top:3px;font-family:'DM Sans',sans-serif;">
            Developed by Andrew Ignatius
        </div>
    </div>
    """)
    _inject_html('<hr>')

    # Market status
    code, lbl, dot_cls = market_status()
    mkt_col = {"open":"#00C805","pre":"#f59e0b","after":"#6366f1","closed":"#555"}.get(code,"#555")
    _inject_html(f"""
    <div style="text-align:center;margin-bottom:14px;">
        <span style="background:{mkt_col}18;color:{mkt_col};padding:4px 14px;
                     border-radius:20px;font-size:12px;font-weight:700;
                     border:1px solid {mkt_col}40;">● {lbl}</span>
    </div>""")

    # Account summary card
    rc = RH_GREEN if m["ret"] >= 0 else RH_RED
    arr = "▲" if m["ret"] >= 0 else "▼"
    _inject_html(f"""
    <div class="card" style="margin-bottom:14px;">
        <div class="mlbl">Account Value</div>
        <div style="font-size:22px;font-weight:900;color:#fff;">${m['equity']:,.2f}</div>
        <div style="font-size:13px;color:{rc};margin-top:2px;">{arr}{abs(m['ret']):.2f}% all-time</div>
        <div style="font-size:11px;color:#444;margin-top:6px;">
            Cash ${m['cash']:,.2f} · Held ${m['mv']:,.2f}
        </div>
    </div>""")

    # Price mode
    _inject_html('<div style="font-size:10px;color:#444;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Data Mode</div>')
    st.session_state.price_mode = st.radio(
        "mode", ["Live (yfinance)", "Simulated (Offline)"],
        index=0 if st.session_state.price_mode.startswith("Live") else 1,
        label_visibility="collapsed"
    )
    mode = st.session_state.price_mode

    if mode == "Simulated (Offline)":
        st.session_state.sim_seed = st.number_input(
            "Sim seed", 0, 999999, int(st.session_state.sim_seed), 1,
            label_visibility="collapsed"
        )
    seed = int(st.session_state.sim_seed)
    st.session_state.auto_refresh = st.checkbox("Auto-refresh prices (60s)",
                                                 value=st.session_state.auto_refresh)

    _inject_html('<hr>')

    # Capital
    _inject_html('<div style="font-size:10px;color:#444;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Starting Capital</div>')
    cap = st.number_input("cap", min_value=1.0,
                          value=float(st.session_state.starting_capital_input),
                          step=1000.0, format="%.2f", label_visibility="collapsed")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Apply", use_container_width=True):
            apply_capital(cap); st.rerun()
    with c2:
        if st.button("Reset", use_container_width=True):
            apply_capital(cap); st.rerun()

    _inject_html('<hr>')

    # Streaks
    if st.session_state.win_streak >= 2:
        _inject_html(f'<span class="badge badge-win">🔥 {st.session_state.win_streak}-Trade Win Streak</span>')
    elif st.session_state.loss_streak >= 2:
        _inject_html(f'<span class="badge badge-loss">❄️ {st.session_state.loss_streak} Losses in a Row</span>')

    # Fund name generator
    if st.button("🎲 Name My Fund", use_container_width=True):
        st.session_state.fund_name = f"{random.choice(FUND_ADJ)} {random.choice(FUND_NOUN)} Capital Management"
    if st.session_state.fund_name:
        _inject_html(f'<div style="text-align:center;color:#555;font-size:11px;font-style:italic;margin-top:6px;">"{st.session_state.fund_name}"</div>')

    _inject_html('<hr>')

    # ── Easter egg: secret input (type KONAMI or GODMODE) ───
    code_in = st.text_input("", placeholder="secret…", label_visibility="collapsed",
                             key="secret_input")
    if code_in.strip().upper() in ("KONAMI","GODMODE","MIDAS","UNLIMITED POWER"):
        st.session_state.god_mode = not st.session_state.god_mode
        st.session_state["secret_input"] = ""
        if st.session_state.god_mode:
            st.success("👑 GOD MODE ACTIVATED")
        else:
            st.info("God mode deactivated.")

    if st.session_state.god_mode:
        _inject_html('<div class="godmode">👑 GOD MODE ACTIVE 👑</div>')

# ============================================================
# MAIN CONTENT
# ============================================================

if st.session_state.god_mode:
    _inject_html('<div class="godmode">👑 ALL ML SIGNALS BULLISH · GOD MODE ENGAGED 👑</div>')

ticker_tape(mode, seed)
m = get_metrics(mode, seed)
status_bar(m, mode)

# After-hours notice
mkt_code, _, _ = market_status()
if mkt_code in ("pre","after","closed"):
    msgs = {
        "pre":    "Pre-market trading (4:00 AM – 9:30 AM ET). Prices reflect last regular-session close.",
        "after":  "After-hours trading (4:00 PM – 8:00 PM ET). Extended prices shown where available.",
        "closed": "Markets are closed. Prices reflect the last regular-session close.",
    }
    _inject_html(f'<div class="alert-y">⏰ {msgs[mkt_code]}</div>')

# ── Penny stock header ────────────────────────────────────────
if st.session_state.starting_capital_input < 100:
    _inject_html('<div style="text-align:center;color:#f59e0b;font-size:13px;padding:6px 0;">🪙 Budget Edition — All prices shown to 4 decimal places</div>')

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", "🤖 AI Insights", "⚡ Trade",
    "📁 Portfolio", "📈 Performance", "🗺️ Heatmap"
])

# ============================================================
# ── TAB 1: DASHBOARD ─────────────────────────────────────────
# ============================================================

with tab1:
    _inject_html('<div class="sh">Overview</div>')

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        d = "up" if m["ret"] >= 0 else "dn"
        arr = "▲" if m["ret"] >= 0 else "▼"
        mcard("Total Equity", f"${m['equity']:,.2f}",
              f"{arr}{abs(m['ret']):.2f}%", d)
    with c2:
        cp = m["cash"]/m["equity"]*100 if m["equity"] > 0 else 0
        mcard("Cash Available", f"${m['cash']:,.2f}", f"{cp:.1f}% of equity")
    with c3:
        hp = m["mv"]/m["equity"]*100 if m["equity"] > 0 else 0
        mcard("Holdings Value", f"${m['mv']:,.2f}", f"{hp:.1f}% of equity")
    with c4:
        upl = m["upl"]
        ud = "up" if upl >= 0 else "dn"
        ua = "▲" if upl >= 0 else "▼"
        up_pct = upl/st.session_state.initial_cash*100 if st.session_state.initial_cash > 0 else 0
        mcard("Unrealized P/L", f"${upl:,.2f}", f"{ua}{abs(up_pct):.2f}%", ud)

    _inject_html("<br>")
    wl_col, stats_col = st.columns([3, 1])

    with wl_col:
        _inject_html('<div class="sh">Watchlist</div>')
        rows = []
        for t in st.session_state.watchlist:
            px = cur_price(t, mode, seed)
            df_, ok_ = price_df(t, mode, seed)
            chg = 0.0
            if ok_ and len(df_) > 1:
                p = df_["Close"].iloc[-2]
                chg = (px-p)/p*100 if np.isfinite(px) else 0
            ep, el = ext_price(t) if mode == "Live (yfinance)" else (None, None)
            rows.append({
                "Ticker": t,
                "Price": f"${px:,.2f}" if np.isfinite(px) else "—",
                "Change": f"{'▲' if chg>=0 else '▼'}{abs(chg):.2f}%",
                "Ext-Hrs": f"${ep:,.2f} ({el})" if ep else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        ca, cb = st.columns(2)
        with ca:
            nt = st.text_input("Add ticker", placeholder="e.g. AAPL", key="wl_add")
            if st.button("Add"):
                t_ = nt.strip().upper()
                if t_ == "DOGE":
                    if t_ not in st.session_state.watchlist:
                        st.session_state.watchlist.append(t_)
                        db_update_watchlist(st.session_state.watchlist)
                        st.info("🐕 Much watchlist. Very crypto. Such wow.")
                        st.rerun()
                elif t_ and ticker_ok(t_, mode, seed):
                    if t_ not in st.session_state.watchlist:
                        st.session_state.watchlist.append(t_)
                        db_update_watchlist(st.session_state.watchlist)
                        st.success(f"Added {t_}")
                        st.rerun()
                    else:
                        st.info(f"{t_} already in watchlist.")
                else:
                    st.error("Invalid ticker.")
        with cb:
            if st.session_state.watchlist:
                rm = st.selectbox("Remove", st.session_state.watchlist, key="wl_rm")
                if st.button("Remove"):
                    st.session_state.watchlist.remove(rm)
                    db_update_watchlist(st.session_state.watchlist)
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
# ── TAB 2: AI INSIGHTS ───────────────────────────────────────
# ============================================================

with tab2:
    _inject_html('<div class="sh">Machine Learning Price Predictions</div>')
    st.caption("Random Forest ensemble · 14 technical indicator features · 5-day horizon")

    ai_opts = sorted(set(st.session_state.watchlist) | set(ALL_TICKERS))
    ai_tkr  = st.selectbox("Select ticker", ai_opts)

    pred_col, mood_col = st.columns([2, 1])

    with pred_col:
        if st.button("Generate Prediction", type="primary"):
            with st.spinner("Training on historical data…"):
                df_, ok_ = price_df(ai_tkr, mode, seed)
                if ok_ and len(df_) > 60:
                    res, edf = run_model(df_, god=st.session_state.god_mode)
                    if res:
                        r1, r2, r3 = st.columns(3)
                        with r1: mcard("Current Price", f"${res['current']:,.2f}")
                        with r2:
                            d_ = "up" if res["change_pct"] >= 0 else "dn"
                            mcard("5-Day Forecast", f"${res['prediction']:,.2f}",
                                  f"{'▲' if res['change_pct']>=0 else '▼'}{abs(res['change_pct']):.2f}%", d_)
                        with r3: mcard("Confidence", f"{res['confidence']:.0f}%")

                        sc_ = RH_GREEN if res["signal"] == "BUY" else RH_RED
                        gm  = " 👑 (GOD MODE)" if st.session_state.god_mode else ""
                        _inject_html(f"""
                        <div style="background:{sc_}18;border:2px solid {sc_};color:{sc_};
                                    padding:14px;border-radius:10px;text-align:center;
                                    font-size:18px;font-weight:900;letter-spacing:2px;margin:12px 0;">
                            {"⬆" if res["signal"]=="BUY" else "⬇"} MODEL SIGNAL: {res["signal"]}{gm}
                        </div>""")

                        # Feature importance — explicit bgcolor fixes transparency
                        imp = pd.DataFrame({
                            "Feature": list(res["importance"].keys()),
                            "Importance": list(res["importance"].values())
                        }).sort_values("Importance").tail(10)
                        fig_i = go.Figure(go.Bar(
                            x=imp["Importance"], y=imp["Feature"],
                            orientation="h",
                            marker=dict(color=RH_GREEN, opacity=1.0)
                        ))
                        fig_i.update_layout(
                            title="Top 10 Predictive Features",
                            plot_bgcolor="#111111",
                            paper_bgcolor="#111111",
                            font=dict(color="#fff"),
                            height=320,
                            xaxis=dict(gridcolor="#2a2a2a"),
                            yaxis=dict(gridcolor="#2a2a2a"),
                            margin=dict(l=10,r=10,t=40,b=10)
                        )
                        st.plotly_chart(fig_i, use_container_width=True)

                        # Technical charts — explicit bgcolor
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
                        fig_t.add_trace(go.Scatter(x=edf.index,y=edf["BB_Upper"],name="BB+",
                            line=dict(color="#333",width=1,dash="dot")), row=1,col=1)
                        fig_t.add_trace(go.Scatter(x=edf.index,y=edf["BB_Lower"],name="BB-",
                            line=dict(color="#333",width=1,dash="dot"),
                            fill="tonexty",fillcolor="rgba(100,100,100,0.08)"), row=1,col=1)
                        fig_t.add_trace(go.Scatter(x=edf.index,y=edf["RSI"],name="RSI",
                            line=dict(color="#ec4899")), row=2,col=1)
                        fig_t.add_hline(y=70,line_dash="dash",line_color="#FF5000",row=2,col=1)
                        fig_t.add_hline(y=30,line_dash="dash",line_color=RH_GREEN,row=2,col=1)
                        fig_t.add_trace(go.Scatter(x=edf.index,y=edf["MACD"],name="MACD",
                            line=dict(color="#06b6d4")), row=3,col=1)
                        fig_t.add_trace(go.Scatter(x=edf.index,y=edf["MACD_Signal"],name="Signal",
                            line=dict(color="#f97316")), row=3,col=1)
                        fig_t.update_layout(
                            height=780,
                            plot_bgcolor="#111111",
                            paper_bgcolor="#111111",
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
                    st.error("Insufficient historical data.")

    with mood_col:
        _inject_html('<div class="sh">Market Mood</div>')
        rsi_vals = []
        for t_ in st.session_state.watchlist[:8]:
            df_, ok_ = price_df(t_, mode, seed)
            if ok_ and len(df_) > 20:
                d = df_["Close"].diff()
                g = d.where(d>0,0).rolling(14).mean()
                l = (-d.where(d<0,0)).rolling(14).mean()
                r = 100 - 100/(1+g/l)
                if np.isfinite(r.iloc[-1]):
                    rsi_vals.append(r.iloc[-1])
        avg_rsi = np.nanmean(rsi_vals) if rsi_vals else 50
        mood_info = (
            ("GREED",    RH_RED)    if avg_rsi >= 70 else
            ("OPTIMISM", "#f59e0b") if avg_rsi >= 55 else
            ("NEUTRAL",  "#888")    if avg_rsi >= 45 else
            ("CAUTION",  "#6366f1") if avg_rsi >= 30 else
            ("FEAR",     RH_GREEN)
        )
        mood_lbl, mood_col_ = mood_info
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_rsi,
            title=dict(text="Avg RSI", font=dict(color="#888",size=12)),
            gauge=dict(
                axis=dict(range=[0,100],tickcolor="#444"),
                bar=dict(color=mood_col_),
                bgcolor="#1a1a1a",
                bordercolor="#2a2a2a",
                steps=[
                    dict(range=[0,30],   color="rgba(0,200,5,0.06)"),
                    dict(range=[30,45],  color="rgba(99,102,241,0.06)"),
                    dict(range=[45,55],  color="rgba(136,136,136,0.06)"),
                    dict(range=[55,70],  color="rgba(245,158,11,0.06)"),
                    dict(range=[70,100], color="rgba(255,80,0,0.06)"),
                ],
            ),
            number=dict(font=dict(color="#fff",size=28))
        ))
        fig_g.update_layout(
            height=230, paper_bgcolor="#111111",
            font=dict(color="#fff"), margin=dict(l=20,r=20,t=40,b=10)
        )
        st.plotly_chart(fig_g, use_container_width=True)
        _inject_html(f'<div style="text-align:center;font-size:20px;font-weight:900;color:{mood_col_};margin-top:-10px;">{mood_lbl}</div>')

# ============================================================
# ── TAB 3: TRADE ─────────────────────────────────────────────
# ============================================================

with tab3:
    _inject_html('<div class="sh">Trading Interface</div>')
    l_, r_ = st.columns([1, 2])

    with l_:
        pick = st.selectbox("Ticker", ALL_TICKERS)
        manual = st.text_input("Or type manually", placeholder="e.g. AAPL, DOGE")
        tkr = manual.strip().upper() if manual.strip() else pick

        if tkr == "DOGE":
            _inject_html('<div style="color:#f59e0b;font-size:13px;">🐕 Much wow. Very trade. Such Dogecoin.</div>')

        side = st.radio("Order type", ["BUY","SELL"], horizontal=True)
        shares_in = st.number_input("Shares", min_value=0.0, value=1.0, step=1.0)

        px_ = cur_price(tkr, mode, seed)
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
        btn_lbl = f"{'Buy' if side=='BUY' else 'Sell'} {tkr}"
        if st.button(btn_lbl, use_container_width=True):
            if place_order(side, tkr, float(shares_in), mode, seed):
                st.rerun()

        if mkt_code in ("after","pre","closed") and mode == "Live (yfinance)":
            _inject_html('<div style="color:#f59e0b;font-size:11px;margin-top:8px;">⏰ Outside market hours — prices are last close</div>')

    with r_:
        df_, ok_ = price_df(tkr, mode, seed)
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
            fig_c.update_layout(
                title=dict(text=f"{tkr} · ${pnow:,.2f}", font=dict(color="#fff",size=18)),
                plot_bgcolor="#111111", paper_bgcolor="#111111",
                font=dict(color="#fff"), height=430,
                xaxis=dict(gridcolor="#2a2a2a",showgrid=True),
                yaxis=dict(gridcolor="#2a2a2a",showgrid=True),
                hovermode="x unified", showlegend=False,
                margin=dict(l=10,r=10,t=50,b=10)
            )
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.error("No price data available.")

# ============================================================
# ── TAB 4: PORTFOLIO ─────────────────────────────────────────
# ============================================================

with tab4:
    _inject_html('<div class="sh">Open Positions</div>')

    if not st.session_state.positions:
        st.info("No open positions. Head to Trade to get started.")
    else:
        pos_rows = []
        for t_, p_ in st.session_state.positions.items():
            px_ = cur_price(t_, mode, seed)
            mv_ = p_["shares"] * px_
            upl_= p_["shares"] * (px_ - p_["avg_cost"])
            cb_ = p_["shares"] * p_["avg_cost"]
            up_ = upl_/cb_*100 if cb_ > 0 else 0
            sec_= UNIVERSE[UNIVERSE["Ticker"]==t_]["Sector"]
            sec_= sec_.iloc[0] if len(sec_) else "Unknown"
            pos_rows.append({
                "Ticker":t_,"Sector":sec_,"Shares":p_["shares"],
                "Avg Cost":f"${p_['avg_cost']:.2f}","Price":f"${px_:.2f}",
                "Mkt Value":f"${mv_:,.2f}","P/L":f"${upl_:,.2f}","Return":f"{up_:+.2f}%"
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

        # Rebalance suggester
        if st.button("⚖️ Suggest Equal-Weight Rebalance"):
            n  = len(st.session_state.positions)
            tv = sum(p_["shares"]*cur_price(t_,mode,seed) for t_,p_ in st.session_state.positions.items())
            tgt= tv / n
            sugg = []
            for t_,p_ in st.session_state.positions.items():
                cv = p_["shares"] * cur_price(t_,mode,seed)
                px_ = cur_price(t_,mode,seed)
                diff= tgt - cv
                act = "Buy" if diff > 0 else "Sell"
                sh  = abs(diff)/px_ if px_ > 0 else 0
                sugg.append(f"{'▲' if diff>0 else '▼'} {act} {sh:.2f} {t_}  (${abs(diff):,.0f})")
            _inject_html('<div class="card"><div class="mlbl">Rebalance Preview — not executed</div>' +
                        "".join(f"<div style='padding:3px 0;color:#ccc;font-size:13px;'>{s}</div>" for s in sugg) +
                        "</div>")

        _inject_html("<br>")
        pc_, cc_ = st.columns(2)

        with pc_:
            _inject_html('<div class="sh">Sector Allocation</div>')
            sv = {}
            for t_,p_ in st.session_state.positions.items():
                px_ = cur_price(t_,mode,seed)
                sec_= UNIVERSE[UNIVERSE["Ticker"]==t_]["Sector"]
                sec_= sec_.iloc[0] if len(sec_) else "Unknown"
                sv[sec_] = sv.get(sec_,0) + p_["shares"]*px_
            tv = sum(sv.values())

            fig_d = go.Figure(go.Pie(
                labels=list(sv.keys()),
                values=list(sv.values()),
                hole=0.55,
                marker=dict(colors=[SECTOR_COLORS.get(s,"#475569") for s in sv],
                            line=dict(color="#000",width=2)),
                textinfo="label+percent",
                textfont=dict(size=12,color="#fff"),
                hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
                direction="clockwise"
            ))
            fig_d.update_layout(
                paper_bgcolor="#111111",
                plot_bgcolor="#111111",
                font=dict(color="#fff"),
                height=380, showlegend=True,
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
                    df_, ok_ = price_df(t_,mode,seed)
                    if ok_ and len(df_) > 30:
                        cdata[t_] = df_["Close"].pct_change().dropna().tail(60)
                if len(cdata) >= 2:
                    cm = pd.DataFrame(cdata).dropna().corr()
                    fig_cr = go.Figure(go.Heatmap(
                        z=cm.values,
                        x=cm.columns.tolist(),
                        y=cm.index.tolist(),
                        colorscale=[[0,RH_RED],[0.5,"#2a2a2a"],[1,RH_GREEN]],
                        zmin=-1,zmax=1,
                        text=np.round(cm.values,2),
                        texttemplate="%{text}",
                        textfont=dict(color="#fff",size=11),
                        hovertemplate="%{x} × %{y}: %{z:.2f}<extra></extra>"
                    ))
                    fig_cr.update_layout(
                        paper_bgcolor="#111111", plot_bgcolor="#111111",
                        font=dict(color="#fff"), height=380,
                        margin=dict(l=10,r=10,t=10,b=10),
                        xaxis=dict(color="#888"),yaxis=dict(color="#888")
                    )
                    st.plotly_chart(fig_cr, use_container_width=True)
                else:
                    st.info("Need price data for at least 2 held stocks.")
            else:
                st.info("Hold ≥ 2 positions to see correlation.")

# ============================================================
# ── TAB 5: PERFORMANCE ───────────────────────────────────────
# ============================================================

with tab5:
    _inject_html('<div class="sh">Performance Analytics</div>')
    eq = st.session_state.equity_history

    if len(eq) > 1:
        evals = [e["equity"] for e in eq]
        tcount= list(range(1, len(evals)+1))
        ic    = st.session_state.initial_cash
        ret_  = [(v-ic)/ic*100 for v in evals]
        is_up = evals[-1] >= ic
        lc    = RH_GREEN if is_up else RH_RED

        mn, mx = min(evals), max(evals)
        pad    = max((mx-mn)*0.18, ic*0.03)

        # ── Equity curve — X = trade count, tight Y-axis ──────
        fig_eq = go.Figure()
        fig_eq.add_hline(y=ic, line_dash="dot", line_color="#444",
                         annotation_text="Starting Capital",
                         annotation_font_color="#555",
                         annotation_position="right")
        fig_eq.add_trace(go.Scatter(
            x=tcount, y=evals,
            mode="lines+markers",
            name="Equity",
            line=dict(color=lc,width=3),
            marker=dict(size=7,color=lc,line=dict(width=1,color="#000")),
            fill="tozeroy",
            fillcolor=f"rgba({'0,200,5' if is_up else '255,80,0'},0.07)",
            hovertemplate="Trade #%{x}<br>$%{y:,.2f}<extra></extra>"
        ))
        fig_eq.update_layout(
            title=dict(
                text=f"Equity Curve · {'▲' if is_up else '▼'}{abs(ret_[-1]):.2f}% total return",
                font=dict(color=lc,size=16)
            ),
            xaxis=dict(title="Trade Number",gridcolor="#2a2a2a",color="#888",
                       tickmode="linear",dtick=max(1,len(tcount)//10)),
            yaxis=dict(title="Value ($)",gridcolor="#2a2a2a",color="#888",
                       range=[mn-pad, mx+pad]),
            plot_bgcolor="#111111", paper_bgcolor="#111111",
            font=dict(color="#fff"), height=400,
            hovermode="x unified", showlegend=False,
            margin=dict(l=10,r=10,t=50,b=10)
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # ── Cumulative return bar chart ───────────────────────
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
            plot_bgcolor="#111111", paper_bgcolor="#111111",
            font=dict(color="#fff"), height=220,
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig_r, use_container_width=True)

        # ── Monte Carlo ───────────────────────────────────────
        if len(evals) >= 5:
            _inject_html('<div class="sh">Monte Carlo Projection — 30 Trading Days</div>')
            rets_arr = np.diff(evals)/np.array(evals[:-1])
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
                fill="toself", fillcolor="rgba(0,200,5,0.04)",
                line=dict(color="rgba(0,0,0,0)"), name="80% CI"))
            fig_mc.add_trace(go.Scatter(
                x=dx+dx[::-1], y=list(p75)+list(p25[::-1]),
                fill="toself", fillcolor="rgba(0,200,5,0.10)",
                line=dict(color="rgba(0,0,0,0)"), name="50% CI"))
            fig_mc.add_trace(go.Scatter(
                x=dx, y=p50, mode="lines", name="Median",
                line=dict(color=RH_GREEN,width=2)))
            fig_mc.add_hline(y=evals[-1], line_dash="dot", line_color="#444")
            fig_mc.update_layout(
                title=dict(
                    text=f"Monte Carlo: 500 simulations · Median ${p50[-1]:,.0f}",
                    font=dict(color="#fff")),
                xaxis=dict(title="Days Forward",gridcolor="#2a2a2a",color="#888"),
                yaxis=dict(title="Portfolio Value ($)",gridcolor="#2a2a2a",color="#888"),
                plot_bgcolor="#111111", paper_bgcolor="#111111",
                font=dict(color="#fff"), height=340,
                legend=dict(bgcolor="#111111",bordercolor="#2a2a2a"),
                margin=dict(l=10,r=10,t=50,b=10)
            )
            st.plotly_chart(fig_mc, use_container_width=True)
    else:
        st.info("Execute trades to populate the performance chart.")

    _inject_html('<div class="sh">Trade History</div>')
    if not st.session_state.trades.empty:
        st.dataframe(
            st.session_state.trades.sort_values("Time", ascending=False).head(50),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No trades yet.")

# ============================================================
# ── TAB 6: HEATMAP ───────────────────────────────────────────
# ============================================================

with tab6:
    _inject_html('<div class="sh">Portfolio Heatmap</div>')

    if st.session_state.positions:
        hm = []
        for t_,p_ in st.session_state.positions.items():
            px_ = cur_price(t_,mode,seed)
            mv_ = p_["shares"]*px_
            up_ = (px_-p_["avg_cost"])/p_["avg_cost"]*100 if p_["avg_cost"] > 0 else 0
            sec_= UNIVERSE[UNIVERSE["Ticker"]==t_]["Sector"]
            sec_= sec_.iloc[0] if len(sec_) else "Unknown"
            hm.append({"Ticker":t_,"Sector":sec_,"Market Value":mv_,"Return %":round(up_,2)})
        hm_df = pd.DataFrame(hm)

        fig_hm = go.Figure(go.Treemap(
            labels=hm_df["Ticker"],
            parents=hm_df["Sector"],
            values=hm_df["Market Value"],
            customdata=np.stack([hm_df["Return %"], hm_df["Market Value"]],axis=-1),
            texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%",
            hovertemplate="<b>%{label}</b><br>Mkt Value: $%{customdata[1]:,.2f}<br>Return: %{customdata[0]:+.2f}%<extra></extra>",
            marker=dict(
                colors=hm_df["Return %"],
                colorscale=[[0,RH_RED],[0.5,"#1a1a1a"],[1,RH_GREEN]],
                cmid=0, showscale=True,
                colorbar=dict(
                    title=dict(text="Return %", font=dict(color="#888")),
                    tickfont=dict(color="#888")
                )
            ),
            textfont=dict(color="#fff",size=14)
        ))
        fig_hm.update_layout(
            paper_bgcolor="#111111", font=dict(color="#fff"),
            height=460, margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No positions to display.")

    _inject_html('<div class="sh">S&P 500 Universe — Daily Change</div>')

    univ_rows = []
    for _,row in UNIVERSE.iterrows():
        px_ = cur_price(row["Ticker"],mode,seed)
        df_,ok_ = price_df(row["Ticker"],mode,seed)
        chg_ = 0.0
        if ok_ and len(df_) > 1:
            prev = df_["Close"].iloc[-2]
            chg_ = (px_-prev)/prev*100 if np.isfinite(px_) else 0
        univ_rows.append({"Ticker":row["Ticker"],"Sector":row["Sector"],
                           "Price":max(px_,1) if np.isfinite(px_) else 1,
                           "Change %":round(chg_,2)})
    univ_df = pd.DataFrame(univ_rows)

    fig_u = go.Figure(go.Treemap(
        labels=univ_df["Ticker"],
        parents=univ_df["Sector"],
        values=univ_df["Price"],
        customdata=univ_df["Change %"],
        texttemplate="<b>%{label}</b><br>%{customdata:+.2f}%",
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>Daily: %{customdata:+.2f}%<extra></extra>",
        marker=dict(
            colors=univ_df["Change %"],
            colorscale=[[0,RH_RED],[0.5,"#1a1a1a"],[1,RH_GREEN]],
            cmid=0, showscale=True,
        ),
        textfont=dict(color="#fff",size=13)
    ))
    fig_u.update_layout(
        paper_bgcolor="#111111", font=dict(color="#fff"),
        height=420, margin=dict(l=10,r=10,t=10,b=10)
    )
    st.plotly_chart(fig_u, use_container_width=True)

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
    © 2026 Midas Capital Systems · Andrew Ignatius · Senior Capstone Project<br>
    <span style="font-size:10px;">Simulated trading only. Not financial advice.</span>
</div>
""")