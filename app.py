import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="NIFTY Intraday Signal Scanner", layout="wide")


# ===================== INDICATORS ===================== #
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / length, adjust=False).mean()


def crossover(a: pd.Series, b: pd.Series) -> bool:
    # last candle crossover only
    if len(a) < 2 or len(b) < 2:
        return False
    return (a.iloc[-2] <= b.iloc[-2]) and (a.iloc[-1] > b.iloc[-1])


def crossunder(a: pd.Series, b: pd.Series) -> bool:
    if len(a) < 2 or len(b) < 2:
        return False
    return (a.iloc[-2] >= b.iloc[-2]) and (a.iloc[-1] < b.iloc[-1])


# ===================== DATA ===================== #
@st.cache_data(ttl=60)
def fetch_intraday(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """
    Fetch intraday OHLC from Yahoo via yfinance.
    """
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # Make sure we have a clean time column
    if "Datetime" in df.columns:
        pass
    elif "Date" in df.columns:
        pass
    else:
        # rename first column to Datetime if unknown
        df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)

    # Make sure required columns exist
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    # Ensure numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    return df


# ===================== SIGNAL + TRADE PLAN ===================== #
def compute_trade_now(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Strategy (fixed params for minimal UI):
    - EMA 9 / EMA 21 cross
    - RSI filter: CALL >= 55, PUT <= 45
    - ATR-based SL and targets
    """
    EMA_FAST = 9
    EMA_SLOW = 21
    RSI_LEN = 14
    RSI_CALL_MIN = 55
    RSI_PUT_MAX = 45
    ATR_LEN = 14

    SL_ATR = 1.5
    T1_ATR = 2.0
    T2_ATR = 3.0

    df = df.copy()
    df["EMA_FAST"] = ema(df["Close"], EMA_FAST)
    df["EMA_SLOW"] = ema(df["Close"], EMA_SLOW)
    df["RSI"] = rsi(df["Close"], RSI_LEN)
    df["ATR"] = atr(df, ATR_LEN)

    # default flags
    df["CALL_FLAG"] = False
    df["PUT_FLAG"] = False

    last_rsi = df["RSI"].iloc[-1]
    last_atr = df["ATR"].iloc[-1]
    last_close = df["Close"].iloc[-1]

    call_ok = crossover(df["EMA_FAST"], df["EMA_SLOW"]) and (last_rsi >= RSI_CALL_MIN)
    put_ok = crossunder(df["EMA_FAST"], df["EMA_SLOW"]) and (last_rsi <= RSI_PUT_MAX)

    plan = {
        "Side": "NO TRADE",
        "Entry": round(float(last_close), 2),
        "StopLoss": None,
        "Target1": None,
        "Target2": None,
        "RSI": None if np.isnan(last_rsi) else round(float(last_rsi), 2),
        "ATR": None if np.isnan(last_atr) else round(float(last_atr), 2),
        "Note": "No valid signal",
    }

    if call_ok and not np.isnan(last_atr):
        df.loc[df.index[-1], "CALL_FLAG"] = True

        entry = float(last_close)
        sl = entry - (float(last_atr) * SL_ATR)
        t1 = entry + (float(last_atr) * T1_ATR)
        t2 = entry + (float(last_atr) * T2_ATR)

        plan.update(
            {
                "Side": "CALL",
                "StopLoss": round(sl, 2),
                "Target1": round(t1, 2),
                "Target2": round(t2, 2),
                "Note": "EMA bullish cross + RSI support",
            }
        )

    elif put_ok and not np.isnan(last_atr):
        df.loc[df.index[-1], "PUT_FLAG"] = True

        entry = float(last_close)
        sl = entry + (float(last_atr) * SL_ATR)
        t1 = entry - (float(last_atr) * T1_ATR)
        t2 = entry - (float(last_atr) * T2_ATR)

        plan.update(
            {
                "Side": "PUT",
                "StopLoss": round(sl, 2),
                "Target1": round(t1, 2),
                "Target2": round(t2, 2),
                "Note": "EMA bearish cross + RSI support",
            }
        )

    return plan, df


# ===================== PLOT ===================== #
def plot_chart(df: pd.DataFrame, title: str) -> go.Figure:
    time_col = "Datetime" if "Datetime" in df.columns else "Date"

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df[time_col],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candles",
        )
    )

    fig.add_trace(go.Scatter(x=df[time_col], y=df["EMA_FAST"], mode="lines", name="EMA 9"))
    fig.add_trace(go.Scatter(x=df[time_col], y=df["EMA_SLOW"], mode="lines", name="EMA 21"))

    calls = df[df["CALL_FLAG"]]
    puts = df[df["PUT_FLAG"]]

    if not calls.empty:
        fig.add_trace(
            go.Scatter(
                x=calls[time_col],
                y=calls["Close"],
                mode="markers",
                name="CALL",
                marker=dict(symbol="triangle-up", size=12),
            )
        )

    if not puts.empty:
        fig.add_trace(
            go.Scatter(
                x=puts[time_col],
                y=puts["Close"],
                mode="markers",
                name="PUT",
                marker=dict(symbol="triangle-down", size=12),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=650,
    )
    return fig


# ===================== UI ===================== #
st.title("📈 NIFTY Intraday Signal Scanner (Manual Scan)")

c1, c2 = st.columns([2, 1])

with c1:
    market = st.selectbox("Select Index", ["NIFTY 50", "NIFTY BANK"])

with c2:
    scan = st.button("🔍 Scan Now", use_container_width=True)

symbol = "^NSEI" if market == "NIFTY 50" else "^NSEBANK"

if not scan:
    st.info("Select index and click **Scan Now** to run analysis.")
    st.stop()

# Manual refresh
st.cache_data.clear()

with st.spinner("Scanning latest candles…"):
    df = fetch_intraday(symbol, period="5d", interval="5m")

if df.empty:
    st.error("No data received. Try again later (free sources can fail sometimes).")
    st.stop()

plan, df2 = compute_trade_now(df)

# Chart
fig = plot_chart(df2, f"{market} • 5m Candles • EMA9/EMA21 + RSI + ATR (Trade Now)")
st.plotly_chart(fig, use_container_width=True)

# Signal card
st.subheader("✅ Trade Signal (Take Now)")
if plan["Side"] == "NO TRADE":
    st.warning(f"**NO TRADE** — {plan['Note']} | RSI: {plan['RSI']} | ATR: {plan['ATR']}")
else:
    st.success(
        f"**{plan['Side']}** | Entry: **{plan['Entry']}** | SL: **{plan['StopLoss']}** | "
        f"T1: **{plan['Target1']}** | T2: **{plan['Target2']}**\n\n"
        f"RSI: {plan['RSI']} • ATR: {plan['ATR']} • {plan['Note']}"
    )

# Recent verification table
st.subheader("📊 Recent Candles (Verification)")
time_col = "Datetime" if "Datetime" in df2.columns else "Date"
view = df2[[time_col, "Open", "High", "Low", "Close", "EMA_FAST", "EMA_SLOW", "RSI", "ATR"]].tail(15).copy()
view.rename(columns={time_col: "Time"}, inplace=True)
st.dataframe(view, use_container_width=True, height=320)
