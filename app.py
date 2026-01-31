import time
import math
import random
import datetime as dt
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# -------------------------
# Optional autorefresh
# -------------------------
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# -------------------------
# yfinance
# -------------------------
try:
    import yfinance as yf
except Exception:
    yf = None


# =========================
# Streamlit config (ONLY ONCE)
# =========================
st.set_page_config(page_title="Pro Multi-Strike Options Dashboard (NSE+YF)", layout="wide")


# =========================
# Time helpers (IST logic)
# =========================
IST_OFFSET = dt.timedelta(hours=5, minutes=30)
IST_TZ = dt.timezone(IST_OFFSET)

def now_ist() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).astimezone(IST_TZ)

def is_market_open_ist(t: Optional[dt.datetime] = None) -> bool:
    """
    NSE cash/index hours (approx): 09:15 to 15:30 IST, Mon-Fri.
    (Does NOT include holidays.)
    """
    t = t or now_ist()
    if t.weekday() >= 5:
        return False
    start = t.replace(hour=9, minute=15, second=0, microsecond=0)
    end   = t.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= t <= end

def next_market_open_ist(t: Optional[dt.datetime] = None) -> dt.datetime:
    """
    Best-effort next opening time (IST).
    - If before 09:15 weekday → today 09:15
    - Else next weekday 09:15
    - Weekend → next Monday 09:15
    (No holiday calendar.)
    """
    t = t or now_ist()
    start_today = t.replace(hour=9, minute=15, second=0, microsecond=0)
    end_today   = t.replace(hour=15, minute=30, second=0, microsecond=0)

    # Weekend -> Monday
    if t.weekday() == 5:
        return (t + dt.timedelta(days=2)).replace(hour=9, minute=15, second=0, microsecond=0)
    if t.weekday() == 6:
        return (t + dt.timedelta(days=1)).replace(hour=9, minute=15, second=0, microsecond=0)

    if t < start_today:
        return start_today
    if t > end_today:
        nxt = t + dt.timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += dt.timedelta(days=1)
        return nxt.replace(hour=9, minute=15, second=0, microsecond=0)

    # During market hours -> tomorrow
    nxt = t + dt.timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += dt.timedelta(days=1)
    return nxt.replace(hour=9, minute=15, second=0, microsecond=0)


# =========================
# yfinance normalization (CRITICAL FIX)
# =========================
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes yfinance output on Streamlit Cloud where columns can be MultiIndex.
    Ensures columns: Open, High, Low, Close, Volume and datetime index.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Flatten MultiIndex columns
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    out.columns = [str(c) for c in out.columns]

    # Rename variants
    ren = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
        "Adj Close": "Close", "adj close": "Close"
    }
    out = out.rename(columns=ren)

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    out = out[keep].copy()

    out.index = pd.to_datetime(out.index, errors="coerce")
    if "Close" in out.columns:
        out["Close"] = pd.to_numeric(out["Close"], errors="coerce")

    out = out.dropna(subset=["Close"]).sort_index()
    return out


# =========================
# NSE Client (Options Chain)
# =========================
class NSEWebClient:
    BASE = "https://www.nseindia.com"
    _lock = threading.Lock()
    _last_req_ts = 0.0
    _cooldown_until = 0.0

    def __init__(self, min_interval_sec: float = 1.2):
        self.s = requests.Session()
        self.min_interval_sec = float(min_interval_sec)
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": self.BASE,
            "Connection": "keep-alive",
        }

    @classmethod
    def _wait_for_slot(cls, min_interval: float):
        with cls._lock:
            now = time.time()
            if now < cls._cooldown_until:
                time.sleep(min(5.0, max(0.25, cls._cooldown_until - now)))
                now = time.time()

            elapsed = now - cls._last_req_ts
            if elapsed < min_interval:
                time.sleep((min_interval - elapsed) + random.uniform(0.05, 0.15))

            cls._last_req_ts = time.time()

    @classmethod
    def _enter_cooldown(cls, seconds: float):
        with cls._lock:
            cls._cooldown_until = max(cls._cooldown_until, time.time() + seconds)

    def warmup(self):
        try:
            self._wait_for_slot(self.min_interval_sec)
            self.s.get(self.BASE, headers=self.headers, timeout=10)
        except Exception:
            pass

    def get_json(self, path: str, params: Optional[dict] = None, retries: int = 3) -> dict:
        url = f"{self.BASE}{path}"
        backoff = 1.1

        for attempt in range(retries):
            try:
                self.warmup()
                self._wait_for_slot(self.min_interval_sec)
                r = self.s.get(url, headers=self.headers, params=params, timeout=15)

                if r.status_code in (403, 429, 500, 502, 503, 504):
                    cooldown = min(120, (2 ** attempt) * 10) + random.uniform(1, 4)
                    self._enter_cooldown(cooldown)
                    time.sleep(min(10, backoff + random.uniform(0.5, 1.5)))
                    backoff *= 1.7
                    continue

                r.raise_for_status()
                return r.json()

            except Exception:
                cooldown = min(90, (2 ** attempt) * 7) + random.uniform(1, 3)
                self._enter_cooldown(cooldown)
                time.sleep(min(10, backoff + random.uniform(0.5, 1.5)))
                backoff *= 1.5

        return {}

    def option_chain_indices(self, symbol: str) -> dict:
        return self.get_json("/api/option-chain-indices", params={"symbol": symbol})


# =========================
# Indicators (No ta lib)
# =========================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_c = c.shift(1)
    return pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df["High"], df["Low"], df["Close"])
    return tr.ewm(alpha=1/period, adjust=False).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "Volume" not in df.columns or df["Volume"].isna().all():
        return pd.Series(index=df.index, dtype=float)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = (tp * df["Volume"]).cumsum()
    vv = df["Volume"].cumsum().replace(0, np.nan)
    return pv / vv


# =========================
# yfinance fetchers (CACHED)
# =========================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday_5m_yf(ticker: str, days: int = 5) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        raw = yf.download(
            ticker,
            period=f"{days}d",
            interval="5m",
            progress=False,
            auto_adjust=False,
            threads=False
        )
        return normalize_ohlcv(raw)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily_yf(ticker: str, days: int = 180) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        raw = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False
        )
        out = normalize_ohlcv(raw)
        return out.tail(days) if not out.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_option_chain_cached(nse_symbol: str) -> dict:
    return NSEWebClient(min_interval_sec=1.2).option_chain_indices(nse_symbol)


# =========================
# Options chain parsing
# =========================
def pick_nearest_expiry(oc: dict) -> Optional[str]:
    try:
        expiries = oc.get("records", {}).get("expiryDates", [])
        return expiries[0] if expiries else None
    except Exception:
        return None

def option_chain_table(oc: dict, expiry: str) -> pd.DataFrame:
    recs = oc.get("records", {}).get("data", []) if isinstance(oc, dict) else []
    rows = []
    for d in recs:
        if d.get("expiryDate") != expiry:
            continue
        strike = d.get("strikePrice")
        ce = d.get("CE", {}) or {}
        pe = d.get("PE", {}) or {}
        rows.append({
            "Strike": strike,
            "CE_LTP": ce.get("lastPrice", np.nan),
            "PE_LTP": pe.get("lastPrice", np.nan),
            "CE_OI": ce.get("openInterest", np.nan),
            "PE_OI": pe.get("openInterest", np.nan),
            "CE_ChgOI": ce.get("changeinOpenInterest", np.nan),
            "PE_ChgOI": pe.get("changeinOpenInterest", np.nan),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["Strike"])
    for c in df.columns:
        if c != "Strike":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
    return df.sort_values("Strike").reset_index(drop=True)


# =========================
# Trade Plan
# =========================
@dataclass
class TradePlan:
    bias: str
    entry: float
    stop: float
    target: float
    confidence: str
    reason: str

def confidence_tag(close: float, ema20: float, ema50: float, vwap_last: float, atr_last: float) -> str:
    score = 0
    if ema50 and abs(ema20/ema50 - 1) > 0.0025:
        score += 1
    if vwap_last and abs(close/vwap_last - 1) > 0.0015:
        score += 1
    if atr_last > 0:
        score += 1
    return "HIGH" if score >= 3 else "MEDIUM" if score == 2 else "LOW"

def compute_trade_plan(df5: pd.DataFrame) -> Optional[TradePlan]:
    if df5 is None or df5.empty or len(df5) < 50:
        return None

    df = df5.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["VWAP"]  = vwap(df)
    df["ATR14"] = atr(df, 14)

    latest = df.iloc[-1]
    close = float(latest["Close"])
    open_ = float(latest["Open"])
    high  = float(latest["High"])
    low   = float(latest["Low"])
    ema20_v = float(latest["EMA20"])
    ema50_v = float(latest["EMA50"])
    vwap_v  = float(latest["VWAP"]) if not math.isnan(float(latest["VWAP"])) else close
    atr_v   = float(latest["ATR14"]) if not math.isnan(float(latest["ATR14"])) else 0.0

    # ORB: first 30 minutes (6 candles of 5m)
    orb = df.iloc[:6]
    orb_high = float(orb["High"].max())
    orb_low  = float(orb["Low"].min())

    long_ok  = (close > vwap_v) and (ema20_v > ema50_v) and (close > open_) and (close > orb_high)
    short_ok = (close < vwap_v) and (ema20_v < ema50_v) and (close < open_) and (close < orb_low)

    conf = confidence_tag(close, ema20_v, ema50_v, vwap_v, atr_v)

    if long_ok:
        entry = close
        stop = min(low, entry - 1.1 * atr_v) if atr_v > 0 else low
        risk = max(entry - stop, 1e-6)
        tgt = entry + 2 * risk
        return TradePlan("BULLISH", entry, stop, tgt, conf,
                         "ORB breakout + EMA20>EMA50 + Price>VWAP + Bull candle")

    if short_ok:
        entry = close
        stop = max(high, entry + 1.1 * atr_v) if atr_v > 0 else high
        risk = max(stop - entry, 1e-6)
        tgt = entry - 2 * risk
        return TradePlan("BEARISH", entry, stop, tgt, conf,
                         "ORB breakdown + EMA20<EMA50 + Price<VWAP + Bear candle")

    return None


# =========================
# Multi-strike option signals
# =========================
def build_multi_strike_table(
    strikes: List[int],
    oc_df: pd.DataFrame,
    plan: Optional[TradePlan],
    capital: float,
    risk_percent: float,
    force_null: bool = False
) -> pd.DataFrame:
    """
    force_null=True: show strikes but keep all option values NULL (market closed behavior).
    """
    risk_amount = capital * (risk_percent / 100.0)
    rows = []

    for strike in strikes:
        row = {"Strike": int(strike)}

        if force_null:
            row.update({
                "CALL_Signal": None, "CALL_Premium": None, "CALL_Entry": None, "CALL_SL": None, "CALL_TGT": None, "CALL_Qty": None,
                "PUT_Signal": None,  "PUT_Premium": None,  "PUT_Entry": None,  "PUT_SL": None,  "PUT_TGT": None,  "PUT_Qty": None,
                "CE_OI": None, "PE_OI": None, "CE_ChgOI": None, "PE_ChgOI": None
            })
            rows.append(row)
            continue

        # Read from option chain
        ce_ltp = pe_ltp = ce_oi = pe_oi = ce_chg = pe_chg = np.nan

        if oc_df is not None and not oc_df.empty:
            m = oc_df[oc_df["Strike"] == strike]
            if not m.empty:
                r = m.iloc[0]
                ce_ltp = float(r["CE_LTP"]) if not pd.isna(r["CE_LTP"]) else np.nan
                pe_ltp = float(r["PE_LTP"]) if not pd.isna(r["PE_LTP"]) else np.nan
                ce_oi  = float(r["CE_OI"]) if not pd.isna(r["CE_OI"]) else np.nan
                pe_oi  = float(r["PE_OI"]) if not pd.isna(r["PE_OI"]) else np.nan
                ce_chg = float(r["CE_ChgOI"]) if not pd.isna(r["CE_ChgOI"]) else np.nan
                pe_chg = float(r["PE_ChgOI"]) if not pd.isna(r["PE_ChgOI"]) else np.nan

        row.update({
            "CALL_Signal": "NO TRADE",
            "CALL_Premium": (round(ce_ltp, 2) if not math.isnan(ce_ltp) else None),
            "CALL_Entry": None, "CALL_SL": None, "CALL_TGT": None, "CALL_Qty": None,

            "PUT_Signal": "NO TRADE",
            "PUT_Premium": (round(pe_ltp, 2) if not math.isnan(pe_ltp) else None),
            "PUT_Entry": None, "PUT_SL": None, "PUT_TGT": None, "PUT_Qty": None,

            "CE_OI": (int(ce_oi) if not math.isnan(ce_oi) else None),
            "PE_OI": (int(pe_oi) if not math.isnan(pe_oi) else None),
            "CE_ChgOI": (int(ce_chg) if not math.isnan(ce_chg) else None),
            "PE_ChgOI": (int(pe_chg) if not math.isnan(pe_chg) else None),
        })

        if plan is None:
            rows.append(row)
            continue

        premium_mult = 0.15  # tune 0.10–0.25
        idx_risk_points = abs(plan.entry - plan.stop)
        prem_risk = max(idx_risk_points * premium_mult, 1.0)

        if plan.bias == "BULLISH" and not math.isnan(ce_ltp):
            entry = ce_ltp
            sl = max(entry - prem_risk, 0.05)
            tgt = entry + 2 * (entry - sl)
            qty = int(risk_amount / max(entry - sl, 1e-6))
            row.update({
                "CALL_Signal": "BUY (CALL)",
                "CALL_Entry": round(entry, 2),
                "CALL_SL": round(sl, 2),
                "CALL_TGT": round(tgt, 2),
                "CALL_Qty": qty
            })

        if plan.bias == "BEARISH" and not math.isnan(pe_ltp):
            entry = pe_ltp
            sl = max(entry - prem_risk, 0.05)
            tgt = entry + 2 * (entry - sl)
            qty = int(risk_amount / max(entry - sl, 1e-6))
            row.update({
                "PUT_Signal": "BUY (PUT)",
                "PUT_Entry": round(entry, 2),
                "PUT_SL": round(sl, 2),
                "PUT_TGT": round(tgt, 2),
                "PUT_Qty": qty
            })

        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# Last Trading Day Snapshot (DAILY)
# =========================
def last_trading_day_summary(daily: pd.DataFrame) -> Optional[Dict[str, object]]:
    """
    Uses:
      last_day = last available daily candle
      prev_day = candle before that
    Works even when market is closed (weekend).
    """
    if daily is None or daily.empty or len(daily) < 2:
        return None

    d = daily.copy().dropna(subset=["Close"])
    if len(d) < 2:
        return None

    last_day = d.iloc[-1]
    prev_day = d.iloc[-2]

    o = float(last_day["Open"])
    h = float(last_day["High"])
    l = float(last_day["Low"])
    c = float(last_day["Close"])
    p_close = float(prev_day["Close"])

    pct = ((c / p_close) - 1) * 100.0 if p_close else np.nan
    gap = ((o / p_close) - 1) * 100.0 if p_close else np.nan
    rng = h - l
    body = abs(c - o)
    body_pct = (body / rng * 100.0) if rng else np.nan
    close_pos = ((c - l) / rng * 100.0) if rng else np.nan

    direction = "Bullish" if c > o else "Bearish" if c < o else "Doji/Flat"
    volatility = "High" if (rng / c) > 0.015 else "Medium" if (rng / c) > 0.008 else "Low"

    midpoint = (h + l) / 2.0
    held_mid = "YES" if c >= midpoint else "NO"

    date_val = d.index[-1].date()
    return {
        "date": date_val,
        "open": o, "high": h, "low": l, "close": c,
        "pct": pct, "gap": gap, "range": rng,
        "body_pct": body_pct, "close_pos": close_pos,
        "direction": direction, "volatility": volatility, "held_mid": held_mid
    }


# =========================
# UI
# =========================
st.title("📈 Live NIFTY/BANKNIFTY Multi-Strike Options Dashboard")
st.caption("Fixed version: yfinance MultiIndex handled + no st.stop() + manual SCAN + last trading day snapshot + market-closed NULL options.")

# -------- Controls --------
ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])

with ctrl1:
    inst = st.selectbox("Select Instrument", ["NIFTY", "BANKNIFTY"])
with ctrl2:
    refresh_sec = st.slider("Auto-refresh (seconds)", 10, 120, 15)
with ctrl3:
    scan_now = st.button("🔁 SCAN NOW", use_container_width=True)

capital = st.number_input("Enter Capital (₹)", value=8000, min_value=1000, step=500)
risk_percent = st.slider("Risk per Trade (%)", 1, 5, 2)

# Auto-refresh
if st_autorefresh is not None:
    st_autorefresh(interval=refresh_sec * 1000, key="refresh")

# Manual refresh clears cache and reruns
if scan_now:
    st.cache_data.clear()
    st.rerun()

# Market status
t_ist = now_ist()
market_open = is_market_open_ist(t_ist)
next_open = next_market_open_ist(t_ist)

if not market_open:
    st.warning(
        f"🛑 Market is **CLOSED** (IST: {t_ist.strftime('%a %d %b %Y %H:%M')}). "
        f"Next opening: **{next_open.strftime('%a %d %b %Y, 09:15 IST')}**. "
        f"(Holiday accuracy needs NSE holiday calendar.)"
    )
else:
    st.success(f"✅ Market is **OPEN** (IST: {t_ist.strftime('%a %d %b %Y %H:%M')})")

yf_ticker = "^NSEI" if inst == "NIFTY" else "^NSEBANK"
nse_symbol = "NIFTY" if inst == "NIFTY" else "BANKNIFTY"

# Fetch intraday
df = fetch_intraday_5m_yf(yf_ticker, days=5)
if df.empty:
    st.error("No intraday data fetched (yfinance). Showing dashboard with placeholders.")

# Compute indicators safely
if not df.empty:
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["VWAP"]  = vwap(df)
    df["ATR14"] = atr(df, 14)
    latest = df.iloc[-1]
    close = float(latest["Close"])
else:
    latest = pd.Series(dtype="object")
    close = float("nan")

# ATM and strikes
step = 50 if inst == "NIFTY" else 100
atm = int(round(close / step) * step) if not math.isnan(close) else 0
strikes = [atm - 2*step, atm - step, atm, atm + step, atm + 2*step] if atm else [0, 0, 0, 0, 0]

# NSE option chain (best effort)
oc = fetch_option_chain_cached(nse_symbol) if market_open else {}
expiry = pick_nearest_expiry(oc) if oc else None
oc_df = option_chain_table(oc, expiry) if (oc and expiry) else pd.DataFrame()

# Plan
plan = compute_trade_plan(df) if not df.empty else None

# Debug expander (VERY useful on streamlit.app)
with st.expander("🧪 Debug / Data Health (open if something is blank)", expanded=False):
    st.write("Instrument:", inst)
    st.write("yfinance ticker:", yf_ticker)
    st.write("Intraday rows:", 0 if df is None else len(df))
    st.write("Intraday columns:", [] if df is None else list(df.columns))
    if df is not None and not df.empty:
        st.write("Intraday last time:", str(df.index[-1]))

    daily_dbg = fetch_daily_yf(yf_ticker, days=180)
    st.write("Daily rows:", 0 if daily_dbg is None else len(daily_dbg))
    st.write("Daily columns:", [] if daily_dbg is None else list(daily_dbg.columns))
    if daily_dbg is not None and not daily_dbg.empty:
        st.write("Daily last date:", str(daily_dbg.index[-1].date()))

    st.write("NSE option chain expiry:", expiry)
    st.write("Option chain rows:", 0 if oc_df is None else len(oc_df))


# Tabs
tabs = st.tabs(["Live Chart", "Last Trading Day Snapshot", "Option Signals", "Latest Candle"])

# ---------------- Live Chart ----------------
with tabs[0]:
    st.subheader("📊 Live Intraday Chart (5m)")

    if df.empty:
        st.info("No intraday candles available right now.")
    else:
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Candles"
        )])
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50"))
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"],  name="VWAP"))

        fig.add_trace(go.Scatter(
            x=[df.index[-1]], y=[close], mode="markers+text",
            text=[f"ATM {atm}"], textposition="top center", name="ATM"
        ))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 Intraday Trade Plan (AUTO)")
    if not market_open:
        st.info("Market closed → Trade plan shown for reference using last available intraday ticks.")
    if plan is None:
        st.warning("No clean setup right now (ORB + EMA + VWAP filters) or not enough candles.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Bias", plan.bias)
        c2.metric("Entry (Index)", f"{plan.entry:.2f}")
        c3.metric("Stop (Index)", f"{plan.stop:.2f}")
        c4.metric("Target (Index)", f"{plan.target:.2f}")
        c5.metric("Confidence", plan.confidence)
        st.info(plan.reason)

# ---------------- Last Trading Day Snapshot ----------------
with tabs[1]:
    st.subheader("🕯️ Last Trading Day Snapshot (Open/Close + Key Points)")

    daily = fetch_daily_yf(yf_ticker, days=180)
    summ = last_trading_day_summary(daily)

    if summ is None:
        st.warning("Not enough daily candles to compute last trading day snapshot.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Open", f"{summ['open']:.2f}")
        m2.metric("Close", f"{summ['close']:.2f}", f"{summ['pct']:.2f}%")
        m3.metric("Gap vs Prev Close", f"{summ['gap']:.2f}%")
        m4.metric("Range (H-L)", f"{summ['range']:.2f}")

        st.markdown("### Key Points")
        st.write(f"• Date: **{summ['date'].strftime('%d %b %Y')}**")
        st.write(f"• Candle: **{summ['direction']}** | Volatility: **{summ['volatility']}**")
        st.write(f"• Body/Range: **{summ['body_pct']:.0f}%** (bigger = stronger conviction)")
        st.write(f"• Close Position: **{summ['close_pos']:.0f}%** of day range (≥70 strong, ≤30 weak)")
        st.write(f"• Held Midpoint: **{summ['held_mid']}**")

        show = daily.tail(10).copy()
        if not show.empty:
            figd = go.Figure(data=[go.Candlestick(
                x=show.index, open=show["Open"], high=show["High"], low=show["Low"], close=show["Close"], name="Daily"
            )])
            figd.add_vline(
                x=pd.Timestamp(summ["date"]),
                line_dash="dot",
                annotation_text="Last Trading Day",
                annotation_position="top left"
            )
            figd.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                               title=f"{inst} — Last 10 Daily Candles")
            st.plotly_chart(figd, use_container_width=True)

# ---------------- Option Signals ----------------
with tabs[2]:
    st.subheader("🎯 Multi-Strike Option Signals")

    if not market_open:
        st.info("Market closed → showing strikes but keeping all option values NULL (as requested).")
        signals_df = build_multi_strike_table(
            strikes=strikes,
            oc_df=pd.DataFrame(),
            plan=None,
            capital=capital,
            risk_percent=risk_percent,
            force_null=True
        )
        st.dataframe(signals_df, use_container_width=True, hide_index=True)
        st.caption("When market opens, NSE premiums/OI will populate (subject to NSE access).")
    else:
        if oc_df.empty:
            st.warning("Option chain not available (NSE blocked/rate limit). Premium/OI may be blank.")
        signals_df = build_multi_strike_table(
            strikes=strikes,
            oc_df=oc_df,
            plan=plan,
            capital=capital,
            risk_percent=risk_percent,
            force_null=False
        )
        st.dataframe(signals_df, use_container_width=True, hide_index=True)
        if expiry:
            st.caption(f"Option Chain Expiry used: {expiry}")

# ---------------- Latest Candle ----------------
with tabs[3]:
    st.subheader("📌 Latest Market Candle (5m)")
    if df.empty:
        st.info("No intraday candle available.")
    else:
        st.write(df.iloc[-1])

st.caption("Note: NSE endpoints are unofficial and can fail on cloud. Intraday/Daily candles are from yfinance.")
