"""Streamlit UI for the MACD + RSI divergence screener MVP."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
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
    PIVOT_RIGHT_BARS,
    RSI_PERIOD,
    TIMEFRAMES,
    TIMEFRAME_SCAN_RULES,
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


@st.cache_data(ttl=3600)
def search_symbol_suggestions(query: str, max_results: int = 8) -> list[dict[str, str]]:
    query = query.strip()
    if len(query) < 2:
        return []

    try:
        search = yf.Search(query, max_results=max_results)
    except Exception:
        return []

    suggestions: list[dict[str, str]] = []
    for item in (search.quotes or []):
        if item.get("quoteType") != "EQUITY":
            continue
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        name = str(item.get("longname") or item.get("shortname") or "Unknown")
        exch = str(item.get("exchDisp") or item.get("exchange") or "")
        label = f"{symbol} - {name}" if not exch else f"{symbol} - {name} ({exch})"
        suggestions.append({"symbol": symbol, "label": label})

    # Deduplicate by symbol while preserving rank.
    deduped: dict[str, dict[str, str]] = {}
    for suggestion in suggestions:
        deduped.setdefault(suggestion["symbol"], suggestion)
    return list(deduped.values())


def _as_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(value)


def _humanize_text(value: object) -> str:
    text = str(value)
    text = text.replace("_", " ")
    text = text.replace(" pct", " %")
    text = text.replace("Pct", "%")
    return text


DISPLAY_COLUMN_ALIASES: dict[str, str] = {
    "pivot_a_time": "start pivot time",
    "pivot_b_time": "end pivot time",
    "pivot_a_price": "start pivot price",
    "pivot_b_price": "end pivot price",
    "indicator_a_time": "indicator start time",
    "indicator_b_time": "indicator end time",
    "indicator_a": "indicator value at start pivot",
    "indicator_b": "indicator value at end pivot",
    "price_move_pct": "price move between pivots pct",
    "indicator_move": "indicator move between pivots",
    "pivot_gap_bars": "bars between pivots",
    "ma_regime": "MA regime",
    "ma_regime_timeframe": "MA regime timeframe",
    "signal_count": "signals found",
    "trade_side": "trade side",
    "entry_time": "entry time (confirmed)",
    "confirmation_lag_bars": "confirmation lag bars",
    "forward_return_pct": "forward return pct",
    "strategy_return_pct": "strategy return pct",
}


def _pretty_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    pretty = frame.copy()
    pretty = pretty.rename(
        columns={
            col: DISPLAY_COLUMN_ALIASES.get(str(col), _humanize_text(col))
            for col in pretty.columns
        }
    )

    for col in ["divergence type", "indicator", "ma regime", "ma regime timeframe", "sma cross"]:
        if col in pretty.columns:
            pretty[col] = pretty[col].astype(str).map(_humanize_text)
    return pretty


def _format_pct_values(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    formatted = frame.copy()
    for col in formatted.columns:
        col_name = str(col).lower()
        if "pct" not in col_name and "%" not in col_name:
            continue
        formatted[col] = formatted[col].apply(
            lambda v: f"{float(v):.2f}%" if pd.notna(v) else v
        )
    return formatted


def _format_currency_values(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    formatted = frame.copy()

    def _to_currency(value: object) -> object:
        if pd.isna(value):
            return value
        try:
            return f"${float(value):,.2f}"
        except (TypeError, ValueError):
            return value

    for col in formatted.columns:
        col_name = str(col).lower()
        is_price_value_col = "price" in col_name and "pct" not in col_name and "%" not in col_name
        if is_price_value_col:
            formatted[col] = formatted[col].apply(_to_currency)
    return formatted


def _style_score_column(frame: pd.DataFrame, score_col: str = "score"):
    if frame.empty or score_col not in frame.columns:
        return frame

    scores = pd.to_numeric(frame[score_col], errors="coerce")
    min_score, max_score = scores.min(), scores.max()
    span = max(max_score - min_score, 1e-9)

    def _color(value: object) -> str:
        score = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(score):
            return ""
        intensity = (score - min_score) / span
        alpha = 0.15 + 0.55 * intensity
        return f"background-color: rgba(46, 163, 96, {alpha:.2f})"

    return frame.style.map(_color, subset=[score_col])


def _build_market_rangebreaks(candles: pd.DataFrame, timeframe: str, *, hide_non_trading_gaps: bool) -> list[dict[str, object]]:
    if not hide_non_trading_gaps:
        return []

    if candles.empty or len(candles.index) < 3:
        return []

    # Intraday compression is handled with a category x-axis in render_technical_chart.
    if timeframe in {"1h", "4h"}:
        return []

    if timeframe in {"1d", "1w"}:
        return [{"bounds": ["sat", "mon"]}]

    return []


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
    display_div_type = _humanize_text(div_type)

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
            name=f"{display_div_type} {legend_suffix}".strip(),
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
            name=f"{_humanize_text(str(signal_row['indicator']).upper())} divergence {legend_suffix}".strip(),
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
    hide_non_trading_gaps: bool = True,
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
        go.Scatter(
            x=candles.index,
            y=sma_200,
            mode="lines",
            name="SMA 200",
            line={"color": "#6d28d9", "width": 2.3},
        ),
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
        margin={"l": 10, "r": 10, "t": 50, "b": 130},
        height=780,
        dragmode="pan",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "left",
            "x": 0.0,
        },
    )
    if hide_non_trading_gaps and timeframe in {"1h", "4h"}:
        # Category axis removes timeline gaps without relying on timezone-sensitive
        # datetime rangebreak filtering.
        fig.update_xaxes(type="category")
    else:
        fig.update_xaxes(
            range=[candles.index.min(), candles.index.max()],
            rangebreaks=_build_market_rangebreaks(candles, timeframe, hide_non_trading_gaps=hide_non_trading_gaps),
        )
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


_DIRECTION_BY_DIVERGENCE_TYPE: dict[str, float] = {
    "regular_bullish": 1.0,
    "regular_bearish": -1.0,
    "hidden_bullish": 1.0,
    "hidden_bearish": -1.0,
}


def _confirmation_lag_bars(timeframe: str) -> int:
    # A pivot at pivot_b_time is only knowable `right` bars later, once the pivot
    # detector has confirmed it as a local extreme (see _find_pivots in divergence.py).
    # Entering a backtest trade at pivot_b_time's own close assumes a fill price that
    # was not actually tradable yet, and captures part of the very reversal used to
    # confirm the pivot - this look-ahead inflates win rate and average return.
    rule = TIMEFRAME_SCAN_RULES.get(timeframe)
    return rule.pivot_right_bars if rule is not None else PIVOT_RIGHT_BARS


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

        confirmation_lag = _confirmation_lag_bars(str(row["timeframe"]))
        entry_idx = int(signal_loc[0]) + confirmation_lag
        end_idx = entry_idx + int(horizon_bars)
        if entry_idx >= len(candles) or end_idx >= len(candles):
            continue

        entry_time = candles.index[entry_idx]
        entry_close = float(candles.iloc[entry_idx]["close"])
        future_close = float(candles.iloc[end_idx]["close"])
        raw_return = (future_close / entry_close) - 1.0
        direction = _DIRECTION_BY_DIVERGENCE_TYPE.get(str(row["divergence_type"]), 1.0)
        strategy_return = raw_return * direction
        trade_side = "long" if direction > 0 else "short"

        backtest_rows.append(
            {
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "divergence_type": row["divergence_type"],
                "indicator": row["indicator"],
                "trade_side": trade_side,
                "signal_time": signal_time,
                "entry_time": entry_time,
                "confirmation_lag_bars": confirmation_lag,
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

    if "symbol_input" not in st.session_state:
        st.session_state.symbol_input = "AAPL"

    with st.sidebar:
        st.header("Scan Settings")

        with st.expander("Divergence & Indicators", expanded=True):
            st.caption(
                "Divergence is when price and an indicator move in different directions — "
                "often an early sign of a reversal or continuation."
            )
            divergence_types = st.multiselect(
                "Divergence types",
                options=["regular_bullish", "regular_bearish", "hidden_bullish", "hidden_bearish"],
                default=list(DEFAULT_ENABLED_DIVERGENCE_TYPES),
                format_func=_humanize_text,
                help="Which divergence patterns to scan for. See the guide below for what each means.",
            )
            selected_indicators = st.multiselect(
                "Indicators",
                options=["rsi", "macd_hist"],
                default=list(DEFAULT_ENABLED_INDICATORS),
                format_func=_humanize_text,
                help="Which indicator(s) to compare against price when looking for divergence.",
            )
            st.markdown(
                "- **Regular bullish**: price makes a lower low while the indicator makes a "
                "higher low (possible upside reversal).\n"
                "- **Regular bearish**: price makes a higher high while the indicator makes a "
                "lower high (possible downside reversal).\n"
                "- **Hidden bullish**: price makes a higher low while the indicator makes a "
                "lower low (bullish continuation bias).\n"
                "- **Hidden bearish**: price makes a lower high while the indicator makes a "
                "higher high (bearish continuation bias)."
            )

        with st.expander("MA Regime Filter"):
            use_ma_regime_filter = st.checkbox(
                "Use MA regime filter (daily/weekly)",
                value=USE_MA_REGIME_FILTER,
                help=(
                    "Only keep signals that agree with the broader trend (50/200 moving "
                    "averages on daily and weekly candles). Reduces reversal signals taken "
                    "against a strong prevailing trend."
                ),
            )
            ma_regime_filter_mode = st.selectbox(
                "MA regime mode",
                options=["soft", "hard"],
                index=0 if MA_REGIME_FILTER_MODE == "soft" else 1,
                help="Soft: reject only direct regime opposition. Hard: require exact regime alignment.",
            )

        with st.expander("Signal Precision"):
            strict_indicator_pivots = st.checkbox(
                "Strict indicator pivots (require indicator swing pivots)",
                value=USE_STRICT_INDICATOR_PIVOTS,
                help=(
                    "When on, the indicator (RSI or MACD histogram) must form its own "
                    "independent swing high/low at the same point as the price pivot. Filters "
                    "out weaker setups, but finds fewer signals overall."
                ),
            )

        with st.expander("Backtest"):
            backtest_enabled = st.checkbox("Show historical backtest", value=True)
            backtest_horizon = st.number_input(
                "Backtest horizon (bars)",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
            )

        with st.expander("Chart Display"):
            chart_max_bars = st.number_input(
                "Chart bars to display",
                min_value=200,
                max_value=5000,
                value=800,
                step=100,
            )

        with st.expander("Cache Management"):
            if st.button("Clear price cache"):
                _load_price_cached.clear()
                st.success("Price cache cleared. Run Scan again to reload full history.")
            if st.button("Clear scanner cache"):
                if hasattr(screener_scanner, "clear_candle_cache"):
                    screener_scanner.clear_candle_cache()
                st.success("Scanner candle cache cleared. Run Scan again to reload fresh candles.")

    col1, col2 = st.columns([2, 1])
    with col1:
        symbol_query = st.text_input(
            "Symbol or company",
            value=st.session_state.symbol_input,
            help="Type a ticker or company name (for example: AAPL, Apple, Microsoft).",
            key="symbol_query",
        )

        suggestions = search_symbol_suggestions(symbol_query)
        selected_symbol = symbol_query.strip().upper()
        if suggestions:
            options = [s["label"] for s in suggestions]
            label_to_symbol = {s["label"]: s["symbol"] for s in suggestions}
            default_label = next(
                (s["label"] for s in suggestions if s["symbol"] == selected_symbol),
                options[0],
            )
            picked_label = st.selectbox(
                "Suggestions",
                options=options,
                index=options.index(default_label),
                help="Pick the symbol to scan.",
            )
            selected_symbol = label_to_symbol[picked_label]

        st.session_state.symbol_input = selected_symbol
    with col2:
        selected_tfs = st.multiselect(
            "Timeframes",
            options=list(TIMEFRAMES.keys()),
            default=["1h", "4h", "1d", "1w"],
        )

    if "scan_ready" not in st.session_state:
        st.session_state.scan_ready = False
        st.session_state.scan_results = pd.DataFrame()
        st.session_state.scan_coverage = pd.DataFrame()
        st.session_state.scan_history_results = pd.DataFrame()
        st.session_state.scan_backtest_enabled = True
        st.session_state.scan_backtest_horizon = 10

    run_scan = st.button("Run Scan", type="primary", use_container_width=True)

    if run_scan:
        symbol = st.session_state.symbol_input.strip().upper()
        if not symbol:
            st.warning("Please provide a symbol.")
            return
        symbols = [symbol]
        if not selected_tfs:
            st.warning("Please select at least one timeframe.")
            return

        with st.spinner(f"Looking up {symbol}..."):
            try:
                symbol_has_data = any(not load_price(symbol, tf).empty for tf in selected_tfs)
            except Exception:
                st.session_state.scan_ready = False
                st.error(
                    f"Couldn't reach the market data provider while looking up '{symbol}'. "
                    "This is usually a temporary network issue — please try again in a moment."
                )
                return

        if not symbol_has_data:
            st.session_state.scan_ready = False
            st.error(
                f"No market data found for '{symbol}'. Double-check the ticker spelling, or type "
                "a company name and pick a match from the Suggestions dropdown above. If the symbol "
                "looks right, the data provider may be temporarily unavailable — try again shortly."
            )
            return

        with st.spinner("Scanning symbols..."):
            try:
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
            except Exception as exc:
                st.session_state.scan_ready = False
                st.error(
                    "Something went wrong while scanning. Try again, or clear the caches in the "
                    "sidebar (Cache Management) if the problem persists."
                )
                with st.expander("Technical details"):
                    st.code(str(exc))
                return

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
        st.dataframe(_pretty_dataframe(_format_currency_values(coverage)), width="stretch")

        st.subheader("Signals")
        if results.empty:
            st.info("No divergences found with current settings.")
        else:
            st.caption("Sorted by score — strongest setups first.")
            pretty_results = _pretty_dataframe(_format_pct_values(_format_currency_values(results)))
            st.dataframe(_style_score_column(pretty_results), width="stretch")

            st.subheader("Chart Preview")
            hide_non_trading_gaps = st.checkbox(
                "Hide non-trading gaps",
                value=st.session_state.get("chart_hide_non_trading_gaps", True),
                key="chart_hide_non_trading_gaps",
                help="Toggle to compare compressed market-time view vs continuous timeline.",
            )
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
                    f"{_humanize_text(results.iloc[i]['divergence_type'])} | "
                    f"{_humanize_text(results.iloc[i]['indicator'])}"
                ),
            )

            row = results.iloc[int(selected_idx)]
            st.subheader("Signal Diagnostics")
            diagnostics = pd.DataFrame(
                [
                    {
                        "symbol": row.get("symbol"),
                        "timeframe": row.get("timeframe"),
                        "divergence type": _humanize_text(row.get("divergence_type")),
                        "indicator": _humanize_text(row.get("indicator")),
                        "start pivot price": row.get("pivot_a_price"),
                        "end pivot price": row.get("pivot_b_price"),
                        "price move %": row.get("price_move_pct"),
                        "indicator move": row.get("indicator_move"),
                        "pivot gap bars": row.get("pivot_gap_bars"),
                        "ma regime": _humanize_text(row.get("ma_regime")),
                        "ma regime timeframe": _humanize_text(row.get("ma_regime_timeframe")),
                        "score": row.get("score"),
                    }
                ]
            )
            st.dataframe(_format_pct_values(_format_currency_values(diagnostics)), width="stretch")
            st.caption("Score component breakdown")
            score_components = row.get("score_components", {})
            if isinstance(score_components, dict) and score_components:
                components_df = pd.DataFrame(
                    [{"component": _humanize_text(k), "points": v} for k, v in score_components.items()]
                ).sort_values("points", ascending=False).reset_index(drop=True)
                st.dataframe(components_df, width="stretch", hide_index=True)
            else:
                st.caption("No score component breakdown available.")
            st.caption(
                "Score is a ranking strength metric. Higher score means stronger multi-factor confirmation, "
                "not guaranteed profit. It combines timeframe weight, indicator confirmation, recent price impulse, "
                "price/indicator strength, swing compactness, and MA regime alignment."
            )

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
                hide_non_trading_gaps=hide_non_trading_gaps,
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
                                "win rate pct": round(backtest_df["win"].mean() * 100.0, 2),
                                "avg strategy return pct": round(backtest_df["strategy_return_pct"].mean(), 2),
                                "median strategy return pct": round(backtest_df["strategy_return_pct"].median(), 2),
                            }
                        ]
                    )
                    st.dataframe(_pretty_dataframe(_format_pct_values(_format_currency_values(summary))), width="stretch")
                    st.caption(
                        "Entry is priced at the pivot's confirmation bar (pivot time + the timeframe's "
                        "right-side pivot bars), not the pivot bar itself, since a pivot can't be identified "
                        "as such until those confirmation bars have printed. Forward return is raw price change "
                        "from entry over the backtest horizon. Strategy return applies signal direction: long "
                        "keeps sign, short flips sign."
                    )
                    st.dataframe(_pretty_dataframe(_format_pct_values(_format_currency_values(backtest_df))), width="stretch")
    else:
        st.info(
            "Enter a symbol above, pick your timeframes, and click **Run Scan** to see divergence "
            "signals, chart overlays, and a historical backtest here. (Divergence type explanations "
            "are in the sidebar under **Divergence & Indicators**.)"
        )


if __name__ == "__main__":
    main()
