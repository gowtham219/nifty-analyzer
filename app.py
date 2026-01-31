import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="NIFTY Intraday Signal Analyzer", layout="wide")

# -------------------- Helpers -------------------- #

@st.cache_data(ttl=60)
def fetch_ohlc(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch OHLCV using yfinance.
    Note: Intraday availability depends on Yahoo rules; may be delayed/unreliable.
    """
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
    # Normalize column names
    df.columns = [c.replace(" ", "_") for c in df.columns]
    # Ensure expected columns exist
    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(set(df.columns)):
        # Sometimes yfinance returns multiindex columns; attempt flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(x) for x in tup if x]) for tup in df.columns]
    return df

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

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)

def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) >= b.shift(1)) & (a < b)

def build_signals(df: pd.DataFrame,
                  ema_fast_len: int,
                  ema_slow_len: int,
                  rsi_len: int,
                  rsi_call_min: float,
                  rsi_put_max: float,
                  atr_len: int,
                  sl_atr_mult: float,
                  t1_atr_mult: float,
                  t2_atr_mult: float) -> pd.DataFrame:

    out = df.copy()
    out["EMA_FAST"] = ema(out["Close"], ema_fast_len)
    out["EMA_SLOW"] = ema(out["Close"], ema_slow_len)
    out["RSI"] = rsi(out["Close"], rsi_len)
    out["ATR"] = atr(out, atr_len)

    out["CALL_SIGNAL"] = crossover(out["EMA_FAST"], out["EMA_SLOW"]) & (out["RSI"] >= rsi_call_min)
    out["PUT_SIGNAL"]  = crossunder(out["EMA_FAST"], out["EMA_SLOW"]) & (out["RSI"] <= rsi_put_max)

    # Signal rows
    signals = out.loc[out["CALL_SIGNAL"] | out["PUT_SIGNAL"], [
        "Datetime" if "Datetime" in out.columns else ("Date" if "Date" in out.columns else out.columns[0]),
        "Open","High","Low","Close","Volume","EMA_FAST","EMA_SLOW","RSI","ATR","CALL_SIGNAL","PUT_SIGNAL"
    ]].copy()

    if signals.empty:
        return signals

    time_col = signals.columns[0]
    signals = signals.rename(columns={time_col: "Time"})

    # Build trade plan
    side = np.where(signals["CALL_SIGNAL"], "CALL", "PUT")
    signals["Side"] = side
    signals["Entry"] = signals["Close"]

    # SL / targets using ATR
    signals["StopLoss"] = np.where(
        signals["Side"] == "CALL",
        signals["Entry"] - signals["ATR"] * sl_atr_mult,
        signals["Entry"] + signals["ATR"] * sl_atr_mult
    )

    signals["Target1"] = np.where(
        signals["Side"] == "CALL",
        signals["Entry"] + signals["ATR"] * t1_atr_mult,
        signals["Entry"] - signals["ATR"] * t1_atr_mult
    )

    signals["Target2"] = np.where(
        signals["Side"] == "CALL",
        signals["Entry"] + signals["ATR"] * t2_atr_mult,
        signals["Entry"] - signals["ATR"] * t2_atr_mult
    )

    # Risk/Reward snapshots
    signals["Risk"] = (signals["Entry"] - signals["StopLoss"]).abs()
    signals["R:R_T1"] = (signals["Target1"] - signals["Entry"]).abs() / signals["Risk"].replace(0, np.nan)
    signals["R:R_T2"] = (signals["Target2"] - signals["Entry"]).abs() / signals["Risk"].replace(0, np.nan)

    # Nice rounding
    price_cols = ["Entry","StopLoss","Target1","Target2","Risk"]
    signals[price_cols] = signals[price_cols].round(2)
    signals["RSI"] = signals["RSI"].round(2)
    signals["ATR"] = signals["ATR"].round(2)
    signals["R:R_T1"] = signals["R:R_T1"].round(2)
    signals["R:R_T2"] = signals["R:R_T2"].round(2)

    return signals.sort_values("Time", ascending=False)


def plot_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    xcol = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])

    fig.add_trace(go.Candlestick(
        x=df[xcol],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candles"
    ))

    if "EMA_FAST" in df.columns:
        fig.add_trace(go.Scatter(x=df[xcol], y=df["EMA_FAST"], mode="lines", name="EMA Fast"))
    if "EMA_SLOW" in df.columns:
        fig.add_trace(go.Scatter(x=df[xcol], y=df["EMA_SLOW"], mode="lines", name="EMA Slow"))

    # Mark signals
    if "CALL_SIGNAL" in df.columns:
        calls = df[df["CALL_SIGNAL"]]
        fig.add_trace(go.Scatter(
            x=calls[xcol],
            y=calls["Close"],
            mode="markers",
            name="CALL",
            marker=dict(symbol="triangle-up", size=10)
        ))
    if "PUT_SIGNAL" in df.columns:
        puts = df[df["PUT_SIGNAL"]]
        fig.add_trace(go.Scatter(
            x=puts[xcol],
            y=puts["Close"],
            mode="markers",
            name="PUT",
            marker=dict(symbol="triangle-down", size=10)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=650
    )
    return fig


# -------------------- UI -------------------- #

st.title("📈 NIFTY Intraday Signal Analyzer (CALL/PUT)")

with st.expander("⚠️ Disclaimer", expanded=True):
    st.write(
        "This app is for educational/analysis purposes only and is not financial advice. "
        "Intraday data from free sources may be delayed or unreliable. Always verify with your broker terminal."
    )

left, right = st.columns([1, 3], gap="large")

with left:
    st.subheader("Settings")

    market = st.selectbox("Market", ["NIFTY 50", "NIFTY BANK"])
    symbol = "^NSEI" if market == "NIFTY 50" else "^NSEBANK"

    interval = st.selectbox("Interval", ["1m", "2m", "5m", "15m", "30m", "60m"], index=2)
    period = st.selectbox("Period", ["1d", "5d", "7d", "1mo"], index=1)

    st.divider()
    st.caption("Signal Strategy (EMA Cross + RSI filter + ATR SL/Targets)")

    ema_fast_len = st.slider("EMA Fast", 5, 50, 9)
    ema_slow_len = st.slider("EMA Slow", 10, 200, 21)

    rsi_len = st.slider("RSI Length", 5, 30, 14)
    rsi_call_min = st.slider("RSI min for CALL", 40, 80, 55)
    rsi_put_max  = st.slider("RSI max for PUT", 20, 60, 45)

    atr_len = st.slider("ATR Length", 5, 30, 14)
    sl_atr_mult = st.slider("StopLoss ATR x", 0.5, 5.0, 1.5, step=0.1)
    t1_atr_mult = st.slider("Target1 ATR x", 0.5, 10.0, 2.0, step=0.1)
    t2_atr_mult = st.slider("Target2 ATR x", 0.5, 15.0, 3.0, step=0.1)

    show_rsi = st.checkbox("Show RSI panel (table)", value=True)
    auto_refresh = st.checkbox("Auto refresh (every 60s)", value=False)

    if auto_refresh:
        st.cache_data.clear()
        st.rerun()

with right:
    st.subheader(f"{market} • {symbol} • {interval} • {period}")

    df = fetch_ohlc(symbol, period=period, interval=interval)

    if df.empty:
        st.error(
            "No data received from the data source.\n\n"
            "If you are hosting on Streamlit Cloud, some sources can fail intermittently. "
            "Try a bigger interval (5m/15m/60m) or period (5d/1mo)."
        )
        st.stop()

    # Identify time column name from yfinance
    if "Datetime" not in df.columns and "Date" not in df.columns:
        # After reset_index, yfinance often gives "Datetime" or "Date"
        # If not, keep first column as time-like
        pass

    # Ensure numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    # Add indicators + signal flags to the main df for plotting
    df["EMA_FAST"] = ema(df["Close"], ema_fast_len)
    df["EMA_SLOW"] = ema(df["Close"], ema_slow_len)
    df["RSI"] = rsi(df["Close"], rsi_len)
    df["ATR"] = atr(df, atr_len)

    df["CALL_SIGNAL"] = crossover(df["EMA_FAST"], df["EMA_SLOW"]) & (df["RSI"] >= rsi_call_min)
    df["PUT_SIGNAL"]  = crossunder(df["EMA_FAST"], df["EMA_SLOW"]) & (df["RSI"] <= rsi_put_max)

    fig = plot_chart(df, f"{market} Intraday Chart + Signals")
    st.plotly_chart(fig, use_container_width=True)

    # Latest snapshot box
    last = df.iloc[-1]
    trend = "BULLISH (CALL bias)" if last["EMA_FAST"] > last["EMA_SLOW"] else "BEARISH (PUT bias)"
    st.info(
        f"**Latest Close:** {last['Close']:.2f}  •  **RSI:** {last['RSI']:.2f}  •  **ATR:** {last['ATR']:.2f}  •  **Trend:** {trend}"
    )

    # Signal table
    signals = build_signals(
        df=df,
        ema_fast_len=ema_fast_len,
        ema_slow_len=ema_slow_len,
        rsi_len=rsi_len,
        rsi_call_min=rsi_call_min,
        rsi_put_max=rsi_put_max,
        atr_len=atr_len,
        sl_atr_mult=sl_atr_mult,
        t1_atr_mult=t1_atr_mult,
        t2_atr_mult=t2_atr_mult
    )

    st.markdown("### ✅ Signals (latest first)")
    if signals.empty:
        st.write("No signals detected for the selected window/settings.")
    else:
        show_cols = ["Time","Side","Entry","StopLoss","Target1","Target2","RSI","ATR","R:R_T1","R:R_T2","Volume"]
        st.dataframe(signals[show_cols], use_container_width=True, height=320)

        csv = signals.to_csv(index=False).encode("utf-8")
        st.download_button("Download signals CSV", data=csv, file_name="signals.csv", mime="text/csv")

    if show_rsi:
        st.markdown("### 📊 Latest Indicator Rows")
        st.dataframe(df.tail(25)[[
            ("Datetime" if "Datetime" in df.columns else "Date"),
            "Close","EMA_FAST","EMA_SLOW","RSI","ATR","CALL_SIGNAL","PUT_SIGNAL"
        ]], use_container_width=True)
