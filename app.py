with tabs[1]:
    st.subheader("🕯️ Last Trading Day (Open–Close Insights)")

    daily = fetch_daily_yf(yf_ticker, days=200)

    if daily is None or daily.empty:
        st.error("Daily candles are not available from yfinance right now (API/rate-limit).")
        st.info("Press SCAN NOW after 30–60 seconds. Streamlit Cloud can sometimes get empty responses.")

        st.dataframe(pd.DataFrame(columns=[
            "Day","Date","Open","High","Low","Close",
            "%Chg vs PrevClose","Gap% vs PrevClose",
            "Range(H-L)","Body%","ClosePos%","Candle"
        ]), use_container_width=True, hide_index=True)

    else:
        table = build_last_trading_day_table(daily)

        if table.empty:
            st.warning("yfinance returned less than 2 daily candles. Showing last rows for debug:")
            st.dataframe(daily.tail(5), use_container_width=True)
        else:
            st.dataframe(table, use_container_width=True, hide_index=True)

            st.markdown("### Key Points (Auto)")
            r = table.iloc[0]
            st.write(f"• Date: **{r['Date']}**")
            st.write(f"• Candle: **{r['Candle']}** | Body%: **{r['Body%']}** | ClosePos%: **{r['ClosePos%']}**")
            st.write(f"• %Chg vs PrevClose: **{r.get('%Chg vs PrevClose', None)}** | Gap%: **{r.get('Gap% vs PrevClose', None)}**")

            show = daily.tail(15).copy()
            figd = go.Figure(data=[go.Candlestick(
                x=show.index,
                open=show["Open"], high=show["High"], low=show["Low"], close=show["Close"],
                name="Daily"
            )])
            figd.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=40, b=10),
                title=f"{inst} — Last 15 Daily Candles"
            )
            st.plotly_chart(figd, use_container_width=True)
