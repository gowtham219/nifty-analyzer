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

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

try:
    import yfinance as yf
except Exception:
    yf = None


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
# yfinance fetchers
# =========================
def fetch_intraday_5m_yf(ticker: str, days: int = 5) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(ticker, period=f"{days}d", interval="5m", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}).copy()
        df.index = pd.to_datetime(df.index)
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()

def fetch_daily_yf(ticker: str, days: int = 30) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}).copy()
        df.index = pd.to_datetime(df.index)
        df = df.dropna(subset=["Close"])
        return df.tail(days)
    except Exception:
        return pd.DataFrame()


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
    df = pd.DataFrame(rows).dropna(subset=["Strike"])
    for c in df.columns:
        if c != "Strike":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
    return df.sort_values("Strike").reset_index(drop=True)


# =========================
# Code-2 style plan (Index)
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
    if ema50 and abs(ema20/ema50 - 1) > 0.0025: score += 1
    if vwap_last and abs(close/vwap_last - 1) > 0.0015: score += 1
    if atr_last > 0: score += 1
    return "HIGH" if score >= 3 else "MEDIUM" if score == 2 else "LOW"

def compute_trade_plan(df5: pd.DataFrame) -> Optional[TradePlan]:
    if df5 is None or df5.empty or len(df5) < 50:
        return None

    df = df5.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["VWAP"] = vwap(df)
    df["ATR14"] = atr(df, 14)

    latest = df.iloc[-1]
    close = float(latest["Close"])
    open_ = float(latest["Open"])
    high = float(latest["High"])
    low  = float(latest["Low"])
    ema20_v = float(latest["EMA20"])
    ema50_v = float(latest["EMA50"])
    vwap_v  = float(latest["VWAP"]) if not math.isnan(latest["VWAP"]) else close
    atr_v   = float(latest["ATR14"]) if not math.isnan(latest["ATR14"]) else 0.0

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
        return TradePlan("BULLISH", entry, stop, tgt, conf, "ORB breakout + EMA20>EMA50 + Price>VWAP + Bull candle")

    if short_ok:
        entry = close
        stop = max(high, entry + 1.1 * atr_v) if atr_v > 0 else high
        risk = max(stop - entry, 1e-6)
        tgt = entry - 2 * risk
        return TradePlan("BEARISH", entry, stop, tgt, conf, "ORB breakdown + EMA20<EMA50 + Price<VWAP + Bear candle")

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
) -> pd.DataFrame:
    risk_amount = capital * (risk_percent / 100.0)
    rows = []

    for strike in strikes:
        row = {"Strike": int(strike)}

        ce_ltp = np.nan
        pe_ltp = np.nan
        m = oc_df[oc_df["Strike"] == strike]
        if not m.empty:
            ce_ltp = float(m.iloc[0]["CE_LTP"]) if not pd.isna(m.iloc[0]["CE_LTP"]) else np.nan
            pe_ltp = float(m.iloc[0]["PE_LTP"]) if not pd.isna(m.iloc[0]["PE_LTP"]) else np.nan

        premium_mult = 0.15  # tune 0.10–0.25

        # Defaults
        row.update({
            "CALL_Signal": "NO TRADE",
            "CALL_Premium": (round(ce_ltp, 2) if not math.isnan(ce_ltp) else None),
            "CALL_Entry": None, "CALL_SL": None, "CALL_TGT": None, "CALL_Qty": None,
            "PUT_Signal": "NO TRADE",
            "PUT_Premium": (round(pe_ltp, 2) if not math.isnan(pe_ltp) else None),
            "PUT_Entry": None, "PUT_SL": None, "PUT_TGT": None, "PUT_Qty": None,
        })

        if plan is None:
            rows.append(row)
            continue

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
                "CALL_Qty": qty,
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
                "PUT_Qty": qty,
            })

        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# Yesterday snapshot (daily)
# =========================
def yesterday_summary(daily: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Needs at least 3 daily candles to compute yesterday & day-before.
    """
    if daily is None or daily.empty or len(daily) < 3:
        return None

    d = daily.copy().dropna(subset=["Close"])
    if len(d) < 3:
        return None

    y = d.iloc[-2]
    p = d.iloc[-3]

    y_open = float(y["Open"])
    y_high = float(y["High"])
    y_low  = float(y["Low"])
    y_close= float(y["Close"])
    p_close= float(p["Close"])

    y_pct = ((y_close / p_close) - 1) * 100.0 if p_close else np.nan
    gap = ((y_open / p_close) - 1) * 100.0 if p_close else np.nan
    rng = y_high - y_low
    body = abs(y_close - y_open)
    body_pct = (body / rng * 100.0) if rng else np.nan
    close_pos = ((y_close - y_low) / rng * 100.0) if rng else np.nan

    direction = "Bullish" if y_close > y_open else "Bearish" if y_close < y_open else "Doji/Flat"
    volatility = "High" if (rng / y_close) > 0.015 else "Medium" if (rng / y_close) > 0.008 else "Low"

    midpoint = (y_high + y_low) / 2.0
    held_mid = "YES" if y_close >= midpoint else "NO"

    y_date = d.index[-2].date()
    return {
        "date": y_date,
        "open": y_open, "high": y_high, "low": y_low, "close": y_close,
        "pct": y_pct, "gap": gap, "range": rng,
        "body_pct": body_pct, "close_pos": close_pos,
        "direction": direction, "volatility": volatility, "held_mid": held_mid
    }


# =========================
# UI
# =========================
st.set_page_config(page_title="Pro Multi-Strike Options Dashboard (NSE+YF)", layout="wide")
st.title("📈 Live NIFTY/BANKNIFTY Multi-Strike Options Dashboard")
st.caption("Code 1 UI + Code 2 analytics + NSE option chain + yfinance candles + Yesterday snapshot.")

inst = st.selectbox("Select Instrument", ["NIFTY", "BANKNIFTY"])
capital = st.number_input("Enter Capital (₹)", value=8000, min_value=1000, step=500)
risk_percent = st.slider("Risk per Trade (%)", 1, 5, 2)
refresh_sec = st.slider("Refresh every (seconds)", 10, 60, 15)

st.write(f"Dashboard auto-refreshes every {refresh_sec} seconds")
if st_autorefresh is not None:
    st_autorefresh(interval=refresh_sec * 1000, key="refresh")

yf_ticker = "^NSEI" if inst == "NIFTY" else "^NSEBANK"
nse_symbol = "NIFTY" if inst == "NIFTY" else "BANKNIFTY"

# Fetch intraday 5m
df = fetch_intraday_5m_yf(yf_ticker, days=5)
if df.empty:
    st.error("No intraday data fetched (yfinance). Try again later.")
    st.stop()

# Indicators
df["EMA20"] = ema(df["Close"], 20)
df["EMA50"] = ema(df["Close"], 50)
df["VWAP"] = vwap(df)
df["ATR14"] = atr(df, 14)

latest = df.iloc[-1]
close = float(latest["Close"])
atm = int(round(close / 50.0) * 50)

# Strikes (auto)
step = 50 if inst == "NIFTY" else 100
strikes = [atm - 2*step, atm - step, atm, atm + step, atm + 2*step]

# NSE option chain
oc = NSEWebClient(min_interval_sec=1.2).option_chain_indices(nse_symbol)
expiry = pick_nearest_expiry(oc) if oc else None
oc_df = option_chain_table(oc, expiry) if (oc and expiry) else pd.DataFrame()

# Plan
plan = compute_trade_plan(df)

# Layout tabs
tabs = st.tabs(["Live Chart", "Yesterday Snapshot", "Option Signals", "Latest Candle"])

# ---------------- Live Chart ----------------
with tabs[0]:
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Candles"
    )])
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP"))

    fig.add_trace(go.Scatter(
        x=[df.index[-1]], y=[close],
        mode="markers+text",
        text=[f"ATM {atm}"],
        textposition="top center",
        name="ATM",
    ))

    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 Intraday Trade Plan (AUTO)")
    if plan is None:
        st.warning("No clean setup right now (ORB + EMA + VWAP filters).")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Bias", plan.bias)
        c2.metric("Entry (Index)", f"{plan.entry:.2f}")
        c3.metric("Stop (Index)", f"{plan.stop:.2f}")
        c4.metric("Target (Index)", f"{plan.target:.2f}")
        c5.metric("Confidence", plan.confidence)
        st.info(plan.reason)

# ---------------- Yesterday Snapshot ----------------
with tabs[1]:
    st.subheader("🕯️ Yesterday Snapshot (Open/Close + Key Points)")

    daily = fetch_daily_yf(yf_ticker, days=30)
    ysum = yesterday_summary(daily)

    if ysum is None:
        st.warning("Not enough daily data to compute yesterday snapshot.")
    else:
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Yesterday Open", f"{ysum['open']:.2f}")
        m2.metric("Yesterday Close", f"{ysum['close']:.2f}", f"{ysum['pct']:.2f}%")
        m3.metric("Gap vs Prev Close", f"{ysum['gap']:.2f}%")
        m4.metric("Range (H-L)", f"{ysum['range']:.2f}")

        # Key points bullets
        st.markdown("### Key Points")
        st.write(f"• Date: **{ysum['date'].strftime('%d %b %Y')}**")
        st.write(f"• Candle: **{ysum['direction']}** | Volatility: **{ysum['volatility']}**")
        st.write(f"• Body/Range: **{ysum['body_pct']:.0f}%** (bigger = stronger conviction)")
        st.write(f"• Close Position: **{ysum['close_pos']:.0f}%** of day range (≥70 strong, ≤30 weak)")
        st.write(f"• Held Midpoint: **{ysum['held_mid']}** (close above midpoint supports bullish bias)")

        # Chart: last 10 daily candles, highlight yesterday
        show = daily.tail(10).copy()
        figd = go.Figure(data=[go.Candlestick(
            x=show.index,
            open=show["Open"], high=show["High"], low=show["Low"], close=show["Close"],
            name="Daily"
        )])

        # vertical line on yesterday
        y_date = pd.Timestamp(ysum["date"])
        figd.add_vline(x=y_date, line_dash="dot", annotation_text="Yesterday", annotation_position="top left")

        figd.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), title=f"{inst} — Last 10 Daily Candles (Yesterday marked)")
        st.plotly_chart(figd, use_container_width=True)

# ---------------- Option Signals ----------------
with tabs[2]:
    st.subheader("🎯 Multi-Strike Option Signals (Premiums from NSE Option Chain)")

    if oc_df.empty:
        st.error("Option chain not available (NSE blocked / rate limit). Premium columns will be blank.")

    signals_df = build_multi_strike_table(strikes, oc_df, plan, capital, risk_percent)
    st.dataframe(signals_df, use_container_width=True, hide_index=True)

    if expiry:
        st.caption(f"Option Chain Expiry used: {expiry}")

# ---------------- Latest Candle ----------------
with tabs[3]:
    st.subheader("📊 Latest Market Candle (5m)")
    st.write(latest)

st.caption("Note: NSE endpoints are unofficial and can fail. Intraday/daily candles come from yfinance for reliability.")
