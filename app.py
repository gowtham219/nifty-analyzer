import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="NIFTY Intraday Signal Scanner", layout="wide")

# -------------------- Indicator helpers -------------------- #
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def crossover(a: pd.Series, b: pd.Series) -> bool:
    # last candle cross
    if len(a) < 2 or len(b) < 2:
        return False
    return (a.iloc[-2] <= b.iloc[-2]) and (a.iloc[-1] > b.iloc[-1])

def crossunder(a: pd.Series, b: pd.Series) -> bool:
    if len(a) < 2 or len(b) < 2:
        return False
    return (a.iloc[-2] >= b.iloc[-2]) and (a.iloc[-1] < b.iloc[-1])

# -------------------- Data fetch -------------------- #
@st.cache_data(ttl=60)
def fetch_intraday(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    return df

def plot_chart(df: pd.DataFrame, title: str) -> go.Figure:
    xcol = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df[xcol],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candles"
    ))

    fig.add_trace(go.Scatter(x=df[xcol], y=df["EMA_FAST"], mode="lines", name="EMA Fast"))
    fig.add_trace(go.Scatter(x=df[xcol], y=df["EMA_SLOW"], mode="lines", name="EMA Slow"))

    # markers for the latest signal
    if df["CALL_FLAG"].any():
        calls = df[df["CALL_FLAG"]]
        fig.add_trace(go.Scatter(
            x=calls[xcol], y=calls["Close"],
            mode="markers", name="CALL",
            marker=dict(symbol="triangle-up", size=12)
        ))
    if df["PUT_FLAG"].any():
        puts = df[df["PUT_FLAG"]]
        fig.add_trace(go.Scatter(
            x=puts[xcol], y=puts["Close"],
            mode="markers", name="PUT",
            marker=dict(symbol="triangle-down", size=12)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=650
    )
    return fig

# -------------------- Signal logic -------------------- #
def build_trade_now_signal(df: pd.DataFrame) -> dict:
    """
    Returns a single 'trade now' plan based on the most recent candle.
    Strategy: EMA9/EMA21 + RSI filter + ATR based SL/targets.
    """
    # Parameters (kept fixed since you want minimal UI)
    EMA_FAST_LEN = 9
    EMA_SLOW_LEN = 21
    RSI_LEN = 14
    RSI_CALL_MIN = 55
    RSI_PUT_MAX = 45
    ATR_LEN = 14

    SL_ATR = 1.5
    T1_ATR = 2.0
    T2_ATR = 3.0

    df = df.copy()
    df["EMA_FAST"] = ema(df["Close"], EMA_FAST_LEN)
    df["EMA_SLOW"] = ema(df["Close"], EMA_SLOW_LEN)
    df["RSI"] = rsi(df["Close"], RSI_LEN)
    df["ATR"] = atr(df, ATR_LEN)

    # Latest cross check
    call_cross = crossover(df["EMA_FAST"], df["EMA_SLOW"]) and (df["RSI"].iloc[-1] >= RSI_CALL_MIN)
    put_cross  = crossunder(df["EMA_FAST"], df["EMA_SLOW"]) and (df["RSI"].iloc[-1] <= RSI_PUT_MAX)

    # Mark only the last candle as signal marker (for chart)
    df["CALL_FLAG"] = False
    df["PUT_FLAG"] = False
    if call_cross:
        df.loc[df.index[-1], "CALL_FLAG"] = True
    if put_cross:
        df.loc[df.index[-1], "PUT_FLAG"] = True

    last_close = float(df["Close"].iloc[-1])
    last_atr = float(df["ATR"].iloc[-1]) if not np.isnan(df["ATR"].iloc[-1]) else 0.0
    last_rsi = float(df["RSI"].iloc[-1]) if not np.isnan(df["RSI"].iloc[-1]) else np.nan

    if call_cross:
        side = "CALL"
        entry = last_close
        sl = entry - (last_atr * SL_ATR)
        t1 = entry + (last_atr * T1_ATR)
        t2 = entry + (last_atr * T2_ATR)
        confidence = "Trend up + momentum"
    elif put_cross:
        side = "PUT"
        entry = last_close
        sl = entry + (last_atr * SL_ATR)
        t1 = entry - (last_atr * T1_ATR)
        t2 = entry - (last_atr * T2_ATR)
        confidence = "Trend down + momentum"
    else:
        side = "NO TRADE"
        entry = last_close
        sl = np.nan
        t1 = np.nan
        t2 = np.nan
        confidence = "No valid EMA cross + RSI filter"

    plan = {
        "Side": side,
        "Entry": round(entry, 2),
        "StopLoss": (round(sl, 2) if not np.isnan(sl) else None),
        "Target1": (round(t1, 2) if not np.isnan(t1) else None),
        "Target2": (round(t2, 2) if not np.isnan(t2) else None),
        "RSI": round(last_rsi, 2) if not np.isnan(last_rsi) else None,
        "ATR": round(last_atr, 2),
        "Note": confidence,
    }

    return plan, df

# -------------------- UI -------------------- #
st.title("📌 NIFTY Intraday Scanner (Trade Now)")

col1, col2 = st.columns([2, 1])
with col1:
    market = st.selectbox("Select", ["NIFTY 50", "NIFTY BANK"])
with col2:
    scan = st.button("🔍 Scan Now", use_container_width=True)

symbol = "^NSEI" if market == "NIFTY 50" else "^NSEBANK"

# Only run analysis when Scan Now is clicked
if scan:
    st.cache_data.clear()  # force fresh fetch on manual scan

    with st.spinner("Scanning…"):
        df = fetch_intraday(symbol, period="5d", interval="5m")

    if df.empty:
        st.error("No data received. Try again later or change interval/provider.")
        st.stop()

    plan, df2 = build_trade_now_signal(df)

    # Chart
    fig = plot_chart(df2, f"{market} ({symbol}) • 5m candles • EMA9/EMA21 + RSI filter")
    st.plotly_chart(fig, use_container_width=True)

    # Trade Now card
    st.subheader("✅ Trade Signal (Now)")
    if plan["Side"] == "NO TRADE":
        st.warning(f"**NO TRADE** — {plan['Note']} | RSI: {plan['RSI']} | ATR: {plan['ATR']}")
    else:
        st.success(
            f"**{plan['Side']}** | Entry: **{plan['Entry']}** | SL: **{plan['StopLoss']}** | "
            f"T1: **{plan['Target1']}** | T2: **{plan['Target2']}**  \n"
            f"RSI: {plan['RSI']} • ATR: {plan['ATR']} • {plan['Note']}"
        )

    # Below table: last 10 candles + indicators for quick verification
    st.subheader("📊 Recent Data (quick verification)")
    xcol = "Datetime" if "Datetime" in df2.columns else ("Date" if "Date" in df2.columns else df2.columns[0])
    view = df2[[xcol, "Open", "High", "Low", "Close", "EMA_FAST", "EMA_SLOW", "RSI", "ATR"]].tail(15).copy()
    view = view.rename(columns={xcol: "Time"})
    st.dataframe(view, use_container_width=True, height=320)

else:
    st.info("Select NIFTY 50 / NIFTY BANK and press **Scan Now**.")
