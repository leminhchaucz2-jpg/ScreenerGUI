"""Streamlit UI for the MACD + RSI divergence screener MVP."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from screener.config import (
    DEFAULT_ENABLED_DIVERGENCE_TYPES,
    DEFAULT_ENABLED_INDICATORS,
    MACD_FAST,
    MA_REGIME_FILTER_MODE,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    TIMEFRAMES,
    USE_ADJUSTED_PRICES,
    USE_MA_REGIME_FILTER,
    USE_STRICT_INDICATOR_PIVOTS,
)
from screener.data import fetch_candles, resample_ohlcv
import screener.indicators as screener_indicators
import screener.scanner as screener_scanner

scan_universe = screener_scanner.scan_universe
scan_universe_history = getattr(screener_scanner, "scan_universe_history", screener_scanner.scan_universe)
compute_macd = screener_indicators.compute_macd
compute_rsi = screener_indicators.compute_rsi
compute_sma = getattr(
    screener_indicators,
    "compute_sma",
    lambda close, period: close.rolling(window=period, min_periods=period).mean(),
)


@st.cache_data(ttl=1800)
def _load_price_cached(
    symbol: str,
    timeframe: str,
    source_interval: str,
    period: str,
    resample_rule: str | None,
    session_aware_resample: bool,
    session_timezone: str | None,
    session_start: str,
    session_end: str,
) -> pd.DataFrame:
    candles = fetch_candles(symbol, source_interval, period, auto_adjust=USE_ADJUSTED_PRICES)
    if resample_rule:
        candles = resample_ohlcv(
            candles,
            resample_rule,
            session_aware=session_aware_resample,
            session_timezone=session_timezone,
            session_start=session_start,
            session_end=session_end,
        )
    return candles


def load_price(symbol: str, timeframe: str) -> pd.DataFrame:
    conf = TIMEFRAMES[timeframe]
    return _load_price_cached(
        symbol=symbol,
        timeframe=timeframe,
        source_interval=conf.source_interval,
        period=conf.period,
        resample_rule=conf.resample_rule,
        session_aware_resample=conf.session_aware_resample,
        session_timezone=conf.session_timezone,
        session_start=conf.session_start,
        session_end=conf.session_end,
    )


def _as_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(value)


def _add_divergence_overlays(
    fig: go.Figure,
    rsi: pd.Series,
    macd: pd.DataFrame,
    signal_row: pd.Series,
    *,
    emphasize: bool = False,
) -> None:
    def _is_bullish(div_type: str) -> bool:
        return div_type in {"regular_bullish", "hidden_bullish"}

    a_time = _as_timestamp(signal_row["pivot_a_time"])
    b_time = _as_timestamp(signal_row["pivot_b_time"])
    a_price = float(signal_row["pivot_a_price"])
    b_price = float(signal_row["pivot_b_price"])
    div_type = str(signal_row["divergence_type"])

    price_color = "#2ca02c" if _is_bullish(div_type) else "#d62728"
    line_width = 4 if emphasize else 2
    marker_size = 10 if emphasize else 7
    marker_symbol = "diamond-open" if emphasize else "circle-open"
    legend_suffix = "(selected)" if emphasize else ""
    fig.add_trace(
        go.Scatter(
            x=[a_time, b_time],
            y=[a_price, b_price],
            mode="lines+markers+text",
            text=["A", "B"],
            textposition="top center",
            name=f"{div_type} {legend_suffix}".strip(),
            line={"color": price_color, "width": line_width},
            marker={"size": marker_size, "color": price_color, "symbol": marker_symbol},
        ),
        row=1,
        col=1,
    )
    fig.add_vline(x=a_time, line_dash="dot", line_color=price_color, opacity=0.25 if not emphasize else 0.45)
    fig.add_vline(x=b_time, line_dash="dot", line_color=price_color, opacity=0.45 if not emphasize else 0.75)

    if signal_row["indicator"] == "rsi":
        indicator_series = rsi
        indicator_row = 2
    else:
        indicator_series = macd["macd_hist"]
        indicator_row = 3

    indicator_a_time = _as_timestamp(signal_row["indicator_a_time"]) if "indicator_a_time" in signal_row else a_time
    indicator_b_time = _as_timestamp(signal_row["indicator_b_time"]) if "indicator_b_time" in signal_row else b_time
    indicator_a = float(signal_row["indicator_a"]) if "indicator_a" in signal_row else float(indicator_series.loc[a_time])
    indicator_b = float(signal_row["indicator_b"]) if "indicator_b" in signal_row else float(indicator_series.loc[b_time])

    fig.add_trace(
        go.Scatter(
            x=[indicator_a_time, indicator_b_time],
            y=[indicator_a, indicator_b],
            mode="lines+markers",
            name=f"{signal_row['indicator'].upper()} divergence {legend_suffix}".strip(),
            line={"color": price_color, "width": line_width, "dash": "dash"},
            marker={"size": max(marker_size - 1, 5), "color": price_color},
        ),
        row=indicator_row,
        col=1,
    )


def _add_sma_cross_markers(fig: go.Figure, candles: pd.DataFrame, sma_50: pd.Series, sma_200: pd.Series) -> None:
    cross_df = pd.DataFrame({"sma_50": sma_50, "sma_200": sma_200}).dropna()
    if len(cross_df) < 2:
        return

    diff = cross_df["sma_50"] - cross_df["sma_200"]
    prev = diff.shift(1)

    golden_events = cross_df[(prev <= 0) & (diff > 0)]
    death_events = cross_df[(prev >= 0) & (diff < 0)]

    for event_time in golden_events.index:
        if event_time not in candles.index:
            continue
        close_price = float(candles.loc[event_time, "close"])
        fig.add_trace(
            go.Scatter(
                x=[event_time],
                y=[close_price],
                mode="markers+text",
                text=["Golden Cross"],
                textposition="top center",
                name="Golden Cross",
                marker={"symbol": "x", "size": 13, "color": "#2ca02c", "line": {"width": 2, "color": "#2ca02c"}},
            ),
            row=1,
            col=1,
        )

    for event_time in death_events.index:
        if event_time not in candles.index:
            continue
        close_price = float(candles.loc[event_time, "close"])
        fig.add_trace(
            go.Scatter(
                x=[event_time],
                y=[close_price],
                mode="markers+text",
                text=["Death Cross"],
                textposition="top center",
                name="Death Cross",
                marker={
                    "symbol": "triangle-down",
                    "size": 13,
                    "color": "#d62728",
                    "line": {"width": 1, "color": "#d62728"},
                },
            ),
            row=1,
            col=1,
        )


def render_technical_chart(
    symbol: str,
    timeframe: str,
    signal_row: pd.Series | None = None,
    all_signal_rows: pd.DataFrame | None = None,
    max_chart_bars: int = 800,
) -> None:
    full_candles = load_price(symbol, timeframe)
    if full_candles.empty:
        st.info("No chart data available.")
        return

    candles = full_candles.tail(max_chart_bars) if len(full_candles) > max_chart_bars else full_candles
    visible_start = candles.index.min()

    visible_signal_rows = all_signal_rows
    if all_signal_rows is not None and not all_signal_rows.empty:
        visible_signal_rows = all_signal_rows[pd.to_datetime(all_signal_rows["pivot_b_time"]) >= visible_start]

    st.caption(
        f"Loaded {len(full_candles)} candles from {full_candles.index.min().date()} to {full_candles.index.max().date()}"
        f". Displaying last {len(candles)} candles for performance."
    )

    close = candles["close"]
    rsi = compute_rsi(close, period=RSI_PERIOD)
    macd = compute_macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    sma_50 = compute_sma(close, period=50)
    sma_200 = compute_sma(close, period=200)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.20, 0.22],
        subplot_titles=(f"{symbol} - {timeframe}", "RSI", "MACD"),
    )
    fig.add_trace(
        go.Candlestick(
            x=candles.index,
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=candles.index, y=sma_50, mode="lines", name="SMA 50", line={"color": "#17becf", "width": 1.6}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=candles.index, y=sma_200, mode="lines", name="SMA 200", line={"color": "#111111", "width": 1.8}),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=candles.index, y=rsi, mode="lines", name="RSI", line={"color": "#1f77b4", "width": 1.5}),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#d62728", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#2ca02c", row=2, col=1)

    hist_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in macd["macd_hist"]]
    fig.add_trace(
        go.Bar(x=candles.index, y=macd["macd_hist"], name="MACD Hist", marker_color=hist_colors, opacity=0.6),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=candles.index, y=macd["macd"], mode="lines", name="MACD", line={"color": "#ff7f0e", "width": 1.5}),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=candles.index,
            y=macd["macd_signal"],
            mode="lines",
            name="Signal",
            line={"color": "#9467bd", "width": 1.3, "dash": "dot"},
        ),
        row=3,
        col=1,
    )

    _add_sma_cross_markers(fig, candles, sma_50, sma_200)

    if visible_signal_rows is not None and not visible_signal_rows.empty:
        for _, row in visible_signal_rows.iterrows():
            is_selected = (
                signal_row is not None
                and pd.Timestamp(row["pivot_a_time"]) == pd.Timestamp(signal_row["pivot_a_time"])
                and pd.Timestamp(row["pivot_b_time"]) == pd.Timestamp(signal_row["pivot_b_time"])
                and row["indicator"] == signal_row["indicator"]
                and row["divergence_type"] == signal_row["divergence_type"]
            )
            _add_divergence_overlays(fig, rsi, macd, row, emphasize=is_selected)
    elif signal_row is not None:
        _add_divergence_overlays(fig, rsi, macd, signal_row, emphasize=True)

    fig.update_layout(
        title=f"{symbol} - {timeframe} Technical View",
        xaxis_rangeslider_visible=False,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        height=780,
        dragmode="pan",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1.0},
    )
    fig.update_xaxes(range=[candles.index.min(), candles.index.max()])
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["zoom2d", "pan2d", "resetScale2d"],
        },
    )


def _build_backtest_frame(history_results: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    backtest_rows: list[dict[str, object]] = []

    if history_results.empty:
        return pd.DataFrame(backtest_rows)

    price_cache: dict[tuple[str, str], pd.DataFrame] = {}
    unique_pairs = history_results[["symbol", "timeframe"]].drop_duplicates()
    for _, pair in unique_pairs.iterrows():
        key = (str(pair["symbol"]), str(pair["timeframe"]))
        price_cache[key] = load_price(key[0], key[1])

    for _, row in history_results.iterrows():
        key = (str(row["symbol"]), str(row["timeframe"]))
        candles = price_cache.get(key, pd.DataFrame())
        if candles.empty:
            continue

        signal_time = pd.Timestamp(row["pivot_b_time"])
        signal_loc = candles.index.get_indexer([signal_time])
        if signal_loc.size == 0 or signal_loc[0] < 0:
            continue

        start_idx = int(signal_loc[0])
        end_idx = start_idx + int(horizon_bars)
        if end_idx >= len(candles):
            continue

        signal_close = float(candles.iloc[start_idx]["close"])
        future_close = float(candles.iloc[end_idx]["close"])
        raw_return = (future_close / signal_close) - 1.0
        direction = 1.0 if row["divergence_type"] == "regular_bullish" else -1.0
        if str(row["divergence_type"]).startswith("hidden_"):
            direction = 1.0 if str(row["divergence_type"]).endswith("bullish") else -1.0
        strategy_return = raw_return * direction

        backtest_rows.append(
            {
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "divergence_type": row["divergence_type"],
                "indicator": row["indicator"],
                "signal_time": signal_time,
                "forward_return_pct": raw_return * 100.0,
                "strategy_return_pct": strategy_return * 100.0,
                "win": strategy_return > 0,
            }
        )

    return pd.DataFrame(backtest_rows)


def main() -> None:
    st.set_page_config(page_title="MACD + RSI Divergence Screener", layout="wide")
    st.title("MACD + RSI Divergence Screener")
    st.caption("Scanner for regular + hidden divergence with causal scoring and MA regime filtering")

    col1, col2 = st.columns([2, 1])
    with col1:
        raw_symbols = st.text_input("Symbols (comma separated)", value="AAPL,MSFT,TSLA,NVDA")
    with col2:
        selected_tfs = st.multiselect(
            "Timeframes",
            options=list(TIMEFRAMES.keys()),
            default=["1h", "4h", "1d", "1w"],
        )
    divergence_types = st.multiselect(
        "Divergence types",
        options=["regular_bullish", "regular_bearish", "hidden_bullish", "hidden_bearish"],
        default=list(DEFAULT_ENABLED_DIVERGENCE_TYPES),
    )
    selected_indicators = st.multiselect(
        "Indicators",
        options=["rsi", "macd_hist"],
        default=list(DEFAULT_ENABLED_INDICATORS),
    )
    use_ma_regime_filter = st.checkbox("Use MA regime filter (daily/weekly)", value=USE_MA_REGIME_FILTER)
    ma_regime_filter_mode = st.selectbox(
        "MA regime mode",
        options=["soft", "hard"],
        index=0 if MA_REGIME_FILTER_MODE == "soft" else 1,
        help="Soft: reject only direct regime opposition. Hard: require exact regime alignment.",
    )
    strict_indicator_pivots = st.checkbox(
        "Strict indicator pivots (require indicator swing pivots)",
        value=USE_STRICT_INDICATOR_PIVOTS,
    )
    if st.button("Clear price cache"):
        _load_price_cached.clear()
        st.success("Price cache cleared. Run Scan again to reload full history.")
    if st.button("Clear scanner cache"):
        if hasattr(screener_scanner, "clear_candle_cache"):
            screener_scanner.clear_candle_cache()
        st.success("Scanner candle cache cleared. Run Scan again to reload fresh candles.")
    backtest_enabled = st.checkbox("Show historical backtest", value=True)
    backtest_horizon = st.number_input(
        "Backtest horizon (bars)",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )
    chart_max_bars = st.number_input(
        "Chart bars to display",
        min_value=200,
        max_value=5000,
        value=800,
        step=100,
    )

    if "scan_ready" not in st.session_state:
        st.session_state.scan_ready = False
        st.session_state.scan_results = pd.DataFrame()
        st.session_state.scan_coverage = pd.DataFrame()
        st.session_state.scan_history_results = pd.DataFrame()
        st.session_state.scan_backtest_enabled = True
        st.session_state.scan_backtest_horizon = 10

    run_scan = st.button("Run Scan", type="primary")

    if run_scan:
        symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
        if not symbols:
            st.warning("Please provide at least one symbol.")
            return
        if not selected_tfs:
            st.warning("Please select at least one timeframe.")
            return

        with st.spinner("Scanning symbols..."):
            results = scan_universe(
                symbols,
                selected_tfs,
                divergence_types=divergence_types,
                indicators=selected_indicators,
                use_ma_regime_filter=use_ma_regime_filter,
                ma_regime_filter_mode=ma_regime_filter_mode,
                strict_indicator_pivots=strict_indicator_pivots,
            )
            history_results = (
                scan_universe_history(
                    symbols,
                    selected_tfs,
                    divergence_types=divergence_types,
                    indicators=selected_indicators,
                    use_ma_regime_filter=use_ma_regime_filter,
                    ma_regime_filter_mode=ma_regime_filter_mode,
                    strict_indicator_pivots=strict_indicator_pivots,
                )
                if backtest_enabled
                else pd.DataFrame()
            )

        coverage = pd.DataFrame({"symbol": symbols})
        coverage["signal_count"] = coverage["symbol"].map(results["symbol"].value_counts()).fillna(0).astype(int)
        coverage["status"] = coverage["signal_count"].apply(lambda n: "signal_found" if n > 0 else "no_signal_found")

        st.session_state.scan_ready = True
        st.session_state.scan_results = results
        st.session_state.scan_coverage = coverage
        st.session_state.scan_history_results = history_results
        st.session_state.scan_backtest_enabled = backtest_enabled
        st.session_state.scan_backtest_horizon = int(backtest_horizon)
        st.session_state.scan_chart_max_bars = int(chart_max_bars)

    if st.session_state.scan_ready:
        results = st.session_state.scan_results
        coverage = st.session_state.scan_coverage
        history_results = st.session_state.scan_history_results
        backtest_enabled = st.session_state.scan_backtest_enabled
        backtest_horizon = st.session_state.scan_backtest_horizon
        chart_max_bars = st.session_state.get("scan_chart_max_bars", int(chart_max_bars))

        st.subheader("Scan Coverage")
        st.dataframe(coverage, width="stretch")

        st.subheader("Signals")
        if results.empty:
            st.info("No divergences found with current settings.")
        else:
            st.dataframe(results, width="stretch")

            st.subheader("Chart Preview")
            options = list(range(len(results)))
            if not options:
                return

            if "selected_signal_row" not in st.session_state or st.session_state.selected_signal_row not in options:
                st.session_state.selected_signal_row = 0

            selected_idx = st.selectbox(
                "Signal row",
                options=options,
                key="selected_signal_row",
                format_func=lambda i: (
                    f"{i} | {results.iloc[i]['symbol']} | {results.iloc[i]['timeframe']} | "
                    f"{results.iloc[i]['divergence_type']} | {results.iloc[i]['indicator']}"
                ),
            )

            row = results.iloc[int(selected_idx)]
            st.subheader("Signal Diagnostics")
            diagnostics = pd.DataFrame(
                [
                    {
                        "symbol": row.get("symbol"),
                        "timeframe": row.get("timeframe"),
                        "divergence_type": row.get("divergence_type"),
                        "indicator": row.get("indicator"),
                        "price_move_pct": row.get("price_move_pct"),
                        "indicator_move": row.get("indicator_move"),
                        "pivot_gap_bars": row.get("pivot_gap_bars"),
                        "ma_regime": row.get("ma_regime"),
                        "ma_regime_timeframe": row.get("ma_regime_timeframe"),
                        "score": row.get("score"),
                    }
                ]
            )
            st.dataframe(diagnostics, width="stretch")
            st.caption("Score component breakdown")
            st.json(row.get("score_components", {}), expanded=False)

            same_asset_tf = results[
                (results["symbol"] == row["symbol"]) & (results["timeframe"] == row["timeframe"])
            ].reset_index(drop=True)
            st.caption(
                f"Showing {len(same_asset_tf)} signal(s) for {row['symbol']} on {row['timeframe']} in one chart."
            )
            render_technical_chart(
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                signal_row=row,
                all_signal_rows=same_asset_tf,
                max_chart_bars=int(chart_max_bars),
            )

        if backtest_enabled:
            st.subheader("Historical Backtest")
            if history_results.empty:
                st.info("No historical signals were found for the selected symbols and timeframes.")
            else:
                backtest_df = _build_backtest_frame(history_results, int(backtest_horizon))
                if backtest_df.empty:
                    st.info("Historical signals exist, but not enough future bars were available to evaluate them.")
                else:
                    summary = pd.DataFrame(
                        [
                            {
                                "signals": len(backtest_df),
                                "win_rate_pct": round(backtest_df["win"].mean() * 100.0, 2),
                                "avg_strategy_return_pct": round(backtest_df["strategy_return_pct"].mean(), 2),
                                "median_strategy_return_pct": round(backtest_df["strategy_return_pct"].median(), 2),
                            }
                        ]
                    )
                    st.dataframe(summary, width="stretch")
                    st.dataframe(backtest_df, width="stretch")


if __name__ == "__main__":
    main()
