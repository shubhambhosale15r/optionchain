"""
Relative CE/PE ATP-LTP Implied Level
-------------------------------------

For ATM-1, ATM and ATM+1:

    CE deviation =
        CE_LTP - CE_ATP

    PE deviation =
        PE_LTP - PE_ATP

    Relative deviation =
        PE deviation - CE deviation

    Level_K =
        K + Relative deviation

Therefore:

    Level_K =
        K
        + (PE_LTP - PE_ATP)
        - (CE_LTP - CE_ATP)

Final level:

    Average Level =
        mean(
            Level_ATM-1,
            Level_ATM,
            Level_ATM+1
        )

Signal:

    Signal =
        Average Level - Spot

Important:
    This is NOT standard synthetic spot.

    It measures the relative position of
    CE and PE LTP versus their own Fyers
    Average Traded Price (ATP).

Examples:

    CE above ATP by 10
    PE above ATP by 5

    Relative deviation = 5 - 10 = -5

    So:

        Level = K - 5

This remains valid even when BOTH CE and PE
are above ATP or BOTH are below ATP.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from fyers_apiv3 import fyersModel


# ==============================================================
# Optional autorefresh
# ==============================================================

try:
    from streamlit_autorefresh import st_autorefresh

    HAS_AUTOREFRESH = True

except ImportError:
    HAS_AUTOREFRESH = False


# ==============================================================
# Fyers
# ==============================================================

def get_fyers(client_id, access_token):

    return fyersModel.FyersModel(
        client_id=client_id,
        token=access_token,
        is_async=False,
        log_path=""
    )


# ==============================================================
# Option chain
# ==============================================================

@st.cache_data(ttl=5, show_spinner=False)
def fetch_chain(
    client_id,
    access_token,
    symbol,
    strikecount,
    timestamp=""
):

    fyers = get_fyers(
        client_id,
        access_token
    )

    return fyers.optionchain(
        data={
            "symbol": symbol,
            "strikecount": int(strikecount),
            "timestamp": timestamp,
            "greeks": "1"
        }
    )


# ==============================================================
# Spot
# ==============================================================

@st.cache_data(ttl=5, show_spinner=False)
def fetch_spot(
    client_id,
    access_token,
    symbol
):

    fyers = get_fyers(
        client_id,
        access_token
    )

    response = fyers.quotes(
        data={
            "symbols": symbol
        }
    )

    if (
        response.get("s") != "ok"
        or not response.get("d")
        or response["d"][0].get("s") != "ok"
    ):
        raise RuntimeError(
            f"Quotes API error while fetching spot: {response}"
        )

    return float(
        response["d"][0]["v"]["lp"]
    )


# ==============================================================
# Quotes API
# ==============================================================
#
# Fetch CE/PE quotes together.
#
# We need:
#
# ATM-1 CE
# ATM-1 PE
# ATM CE
# ATM PE
# ATM+1 CE
# ATM+1 PE
#
# From the same Quotes API response:
#
#     lp  = current LTP
#     atp = Average Traded Price
#
# ==============================================================
@st.cache_data(ttl=5, show_spinner=False)
def fetch_option_quotes(
    client_id,
    access_token,
    symbols_tuple
):

    symbols = list(symbols_tuple)

    if not symbols:

        raise RuntimeError(
            "No option symbols supplied to Quotes API."
        )

    if len(symbols) > 50:

        raise RuntimeError(
            f"Quotes API supports max 50 symbols. "
            f"Received {len(symbols)}."
        )

    fyers = get_fyers(
        client_id,
        access_token
    )

    response = fyers.quotes(
        data={
            "symbols": ",".join(symbols)
        }
    )

    if response.get("s") != "ok":

        raise RuntimeError(
            f"Option Quotes API error: {response}"
        )

    result = {}

    for item in response.get("d", []):

        if item.get("s") != "ok":
            continue

        symbol = (
            item.get("n")
            or (item.get("v") or {}).get("symbol")
        )

        values = item.get("v") or {}

        if not symbol:
            continue

        try:
            lp = float(values.get("lp"))
        except (TypeError, ValueError):
            lp = np.nan

        try:
            atp = float(values.get("atp"))
        except (TypeError, ValueError):
            atp = np.nan

        result[symbol] = {

            "lp": lp,
            "atp": atp,

            "bid": values.get("bid"),
            "ask": values.get("ask"),

            "volume": values.get("volume"),

            "symbol": symbol,
        }

    return result


# ==============================================================
# Build option map
# ==============================================================

def build_option_map(chain):
    """
    Creates:

        strike -> {
            CE: option symbol,
            PE: option symbol
        }
    """

    result = {}

    for row in chain:

        option_type = row.get("option_type")

        if option_type not in ("CE", "PE"):
            continue

        strike = row.get("strike_price")

        if strike is None:
            continue

        symbol = (
            row.get("symbol")
            or row.get("name")
            or row.get("original_name")
        )

        if not symbol:
            continue

        if strike not in result:
            result[strike] = {}

        result[strike][option_type] = symbol

    return result


# ==============================================================
# Select ATM-1 / ATM / ATM+1
# ==============================================================

def select_three_strikes(
    option_map,
    spot
):

    strikes = sorted(
        option_map.keys()
    )

    if not strikes:

        raise RuntimeError(
            "No strikes available in option chain."
        )

    # ----------------------------------------------------------
    # ATM = strike closest to actual NIFTY spot
    # ----------------------------------------------------------

    atm = min(
        strikes,
        key=lambda x: abs(x - spot)
    )

    atm_index = strikes.index(atm)

    if atm_index == 0:

        raise RuntimeError(
            "ATM has no lower strike available for ATM-1."
        )

    if atm_index == len(strikes) - 1:

        raise RuntimeError(
            "ATM has no higher strike available for ATM+1."
        )

    lower = strikes[atm_index - 1]
    upper = strikes[atm_index + 1]

    selected = [
        lower,
        atm,
        upper
    ]

    # ----------------------------------------------------------
    # Make sure CE and PE both exist
    # ----------------------------------------------------------

    for strike in selected:

        if strike not in option_map:

            raise RuntimeError(
                f"Strike {strike} missing from option map."
            )

        if "CE" not in option_map[strike]:

            raise RuntimeError(
                f"CE missing for strike {strike}."
            )

        if "PE" not in option_map[strike]:

            raise RuntimeError(
                f"PE missing for strike {strike}."
            )

    return (
        lower,
        atm,
        upper
    )


# ==============================================================
# Calculate relative ATP deviation
# ==============================================================

def calculate_levels(
    selected_strikes,
    option_map,
    quotes
):

    rows = []

    for strike in selected_strikes:

        ce_symbol = option_map[strike]["CE"]
        pe_symbol = option_map[strike]["PE"]

        ce_quote = quotes.get(
            ce_symbol
        )

        pe_quote = quotes.get(
            pe_symbol
        )

        if ce_quote is None:

            raise RuntimeError(
                f"No Quotes API response for CE: {ce_symbol}"
            )

        if pe_quote is None:

            raise RuntimeError(
                f"No Quotes API response for PE: {pe_symbol}"
            )

        # ------------------------------------------------------
        # Current prices
        # ------------------------------------------------------

        ce_ltp = ce_quote["lp"]
        pe_ltp = pe_quote["lp"]

        # ------------------------------------------------------
        # Average traded prices
        # ------------------------------------------------------

        ce_atp = ce_quote["atp"]
        pe_atp = pe_quote["atp"]

        values = [
            ce_ltp,
            ce_atp,
            pe_ltp,
            pe_atp
        ]

        if any(
            pd.isna(x) or x <= 0
            for x in values
        ):

            raise RuntimeError(
                f"Invalid quote data for strike {strike}: "
                f"CE={ce_quote}, PE={pe_quote}"
            )

        # ======================================================
        # Individual deviations from ATP
        # ======================================================

        # Positive = LTP above ATP
        # Negative = LTP below ATP

        ce_deviation = (
            ce_ltp - ce_atp
        )

        pe_deviation = (
            pe_ltp - pe_atp
        )

        # ======================================================
        # Relative CE/PE deviation
        # ======================================================
        #
        # This is the important metric.
        #
        # PE deviation - CE deviation
        #
        # Equivalent to:
        #
        # (PE_LTP - PE_ATP)
        # -
        # (CE_LTP - CE_ATP)
        #
        # ======================================================

        relative_deviation = (
            pe_deviation
            - ce_deviation
        )

        # ======================================================
        # Convert relative deviation into a price level
        # ======================================================

        level = (
            strike
            + relative_deviation
        )

        rows.append({

            "Strike": strike,

            # ------------------------------
            # CE
            # ------------------------------

            "CE Symbol": ce_symbol,

            "CE ATP": ce_atp,

            "CE LTP": ce_ltp,

            "CE LTP-ATP": ce_deviation,

            # ------------------------------
            # PE
            # ------------------------------

            "PE Symbol": pe_symbol,

            "PE LTP": pe_ltp,

            "PE ATP": pe_atp,

            "PE LTP-ATP": pe_deviation,

            # ------------------------------
            # Relative metric
            # ------------------------------

            "PE Dev - CE Dev": relative_deviation,

            # ------------------------------
            # Implied level
            # ------------------------------

            "Calculated Level": level,
        })

    result = pd.DataFrame(
        rows
    )

    result = (
        result
        .sort_values("Strike")
        .reset_index(drop=True)
    )

    # ==========================================================
    # Average of ATM-1 / ATM / ATM+1
    # ==========================================================

    average_level = (
        result["Calculated Level"].mean()
    )

    average_relative_deviation = (
        result["PE Dev - CE Dev"].mean()
    )

    return (
        result,
        float(average_level),
        float(average_relative_deviation)
    )


# ==============================================================
# Auto refresh
# ==============================================================

def ms_until_next_minute():

    now = datetime.now()

    seconds = (
        now.second
        + now.microsecond / 1_000_000
    )

    ms = int(
        round(
            (60.0 - seconds) * 1000
        )
    )

    if ms < 250:

        ms += 60_000

    return ms


def install_autorefresh(
    enabled,
    mode,
    fixed_seconds
):

    if not enabled:
        return None

    if not HAS_AUTOREFRESH:
        return None

    if mode == "Align to minute boundary":

        interval_ms = (
            ms_until_next_minute()
        )

    else:

        interval_ms = (
            max(
                int(fixed_seconds),
                1
            ) * 1000
        )

    st_autorefresh(
        interval=interval_ms,
        key="auto_refresh_tick"
    )

    return interval_ms


# ==============================================================
# Main
# ==============================================================

def main():

    st.set_page_config(
        page_title="Relative CE/PE ATP Signal",
        layout="wide"
    )

    st.title(
        "Relative CE / PE ATP Signal"
    )

    st.caption(
        "(PE LTP − PE ATP) − (CE LTP − CE ATP)"
    )

    # ==========================================================
    # Sidebar
    # ==========================================================

    with st.sidebar:

        client_id = st.text_input(
            "Client ID",
            placeholder="XXXXXXX-100"
        )

        access_token = st.text_input(
            "Access token",
            type="password"
        )

        symbol = st.text_input(
            "Underlying",
            "NSE:NIFTY50-INDEX"
        )

        strikecount = st.number_input(
            "Strikes fetched each side of ATM",
            min_value=3,
            max_value=50,
            value=6
        )

        st.divider()

        st.subheader(
            "Auto-refresh"
        )

        refresh_enabled = st.checkbox(
            "Enable auto-refresh",
            value=True
        )

        refresh_mode = st.radio(
            "Refresh mode",
            [
                "Align to minute boundary",
                "Fixed interval"
            ],
            index=0
        )

        fixed_seconds = 60

        if refresh_mode == "Fixed interval":

            fixed_seconds = st.number_input(
                "Refresh every (seconds)",
                min_value=5,
                max_value=3600,
                value=60,
                step=5
            )

        if not HAS_AUTOREFRESH:

            st.warning(
                "Install streamlit-autorefresh:\n\n"
                "`pip install streamlit-autorefresh`"
            )

    # ==========================================================
    # Credentials
    # ==========================================================

    if not (
        client_id
        and access_token
    ):

        st.info(
            "Enter your Fyers Client ID and "
            "Access Token in the sidebar."
        )

        return

    # ==========================================================
    # Auto refresh
    # ==========================================================

    interval_ms = install_autorefresh(
        refresh_enabled,
        refresh_mode,
        fixed_seconds
    )

    # ==========================================================
    # Fetch chain
    # ==========================================================

    fetch_error = None
    resp = None

    try:

        resp = fetch_chain(
            client_id,
            access_token,
            symbol,
            strikecount
        )

        if resp.get("s") != "ok":

            fetch_error = (
                f"API error: {resp}"
            )

            resp = None

        else:

            expiry_data = (
                resp["data"]
                .get("expiryData", [])
            )

            if expiry_data:

                labels = {
                    e["date"]: e["expiry"]
                    for e in expiry_data
                }

                chosen = st.selectbox(
                    "Expiry",
                    list(labels)
                )

                if (
                    chosen
                    != expiry_data[0]["date"]
                ):

                    resp = fetch_chain(
                        client_id,
                        access_token,
                        symbol,
                        strikecount,
                        labels[chosen]
                    )

                    if resp.get("s") != "ok":

                        fetch_error = (
                            f"API error: {resp}"
                        )

                        resp = None

        if resp is None:

            raise RuntimeError(
                fetch_error
                or
                "Option-chain request failed."
            )

        chain = (
            resp["data"]["optionsChain"]
        )

        # ------------------------------------------------------
        # Spot
        # ------------------------------------------------------

        spot = fetch_spot(
            client_id,
            access_token,
            symbol
        )

        # ------------------------------------------------------
        # Option symbols
        # ------------------------------------------------------

        option_map = build_option_map(
            chain
        )

        # ------------------------------------------------------
        # ATM-1 / ATM / ATM+1
        # ------------------------------------------------------

        lower, atm, upper = (
            select_three_strikes(
                option_map,
                spot
            )
        )

        selected_strikes = [
            lower,
            atm,
            upper
        ]

        # ------------------------------------------------------
        # Six option symbols
        # ------------------------------------------------------

        quote_symbols = []

        for strike in selected_strikes:

            quote_symbols.append(
                option_map[strike]["CE"]
            )

            quote_symbols.append(
                option_map[strike]["PE"]
            )

        quote_symbols = list(
            dict.fromkeys(
                quote_symbols
            )
        )

        # ------------------------------------------------------
        # Quotes API
        # ------------------------------------------------------

        quotes = fetch_option_quotes(
            client_id,
            access_token,
            tuple(quote_symbols)
        )

        # ------------------------------------------------------
        # Calculate
        # ------------------------------------------------------

        (
            calculation_df,
            average_level,
            average_relative_deviation
        ) = calculate_levels(
            selected_strikes,
            option_map,
            quotes
        )

        # ------------------------------------------------------
        # Final signal
        # ------------------------------------------------------

        signal = (
            average_level
            - spot
        )

        st.session_state[
            "last_refresh"
        ] = datetime.now()

    except Exception as e:

        fetch_error = (
            f"Failed: {e}"
        )

    # ==========================================================
    # Header
    # ==========================================================

    last_refresh = (
        st.session_state.get(
            "last_refresh"
        )
    )

    hdr = st.columns(
        [3, 2, 2]
    )

    if last_refresh is not None:

        hdr[0].caption(
            "🕒 Last refreshed: "
            f"**{last_refresh.strftime('%H:%M:%S')}**"
        )

    else:

        hdr[0].caption(
            "🕒 Last refreshed: —"
        )

    if fetch_error:

        st.error(
            fetch_error
        )

        return

    hdr[1].caption(
        "Auto-refresh: "
        f"**{'ON' if refresh_enabled and HAS_AUTOREFRESH else 'OFF'}**"
        + (
            f" · every {interval_ms / 1000:.0f}s"
            if interval_ms
            else ""
        )
    )

    if (
        refresh_enabled
        and refresh_mode
        == "Align to minute boundary"
    ):

        hdr[2].caption(
            "Next tick: next minute boundary"
        )

    if st.button(
        "🔄 Refresh now"
    ):

        st.cache_data.clear()

        st.rerun()

    # ==========================================================
    # Main metrics
    # ==========================================================

    m = st.columns(5)

    m[0].metric(
        "Spot",
        f"{spot:,.2f}"
    )

    m[1].metric(
        "ATM",
        f"{atm:,.0f}"
    )

    m[2].metric(
        "Average Relative Dev.",
        f"{average_relative_deviation:+,.2f}"
    )

    m[3].metric(
        "Average Level",
        f"{average_level:,.2f}"
    )

    m[4].metric(
        "Signal vs Spot",
        f"{signal:+,.2f}"
    )

    # ==========================================================
    # Signal interpretation
    # ==========================================================

    if signal > 0:

        st.success(
            f"Average calculated level is "
            f"**{signal:+,.2f} points above spot**."
        )

    elif signal < 0:

        st.warning(
            f"Average calculated level is "
            f"**{signal:+,.2f} points below spot**."
        )

    else:

        st.info(
            "Average calculated level equals spot."
        )

    # ==========================================================
    # Main calculation table
    # ==========================================================

    st.subheader(
        "ATM-1 / ATM / ATM+1"
    )

    display_df = calculation_df[
        [
            "Strike",

            "CE ATP",
            "CE LTP",
            "CE LTP-ATP",

            "PE LTP",
            "PE ATP",
            "PE LTP-ATP",

            "PE Dev - CE Dev",

            "Calculated Level",
        ]
    ].copy()

    st.dataframe(
        display_df.style.format(
            {
                "Strike": "{:,.0f}",

                "CE ATP": "{:,.2f}",
                "CE LTP": "{:,.2f}",
                "CE LTP-ATP": "{:+,.2f}",

                "PE LTP": "{:,.2f}",
                "PE ATP": "{:,.2f}",
                "PE LTP-ATP": "{:+,.2f}",

                "PE Dev - CE Dev": "{:+,.2f}",

                "Calculated Level": "{:,.2f}",
            },
            na_rep="–"
        ),
        hide_index=True,
        use_container_width=True
    )

    # ==========================================================
    # Explicit calculation
    # ==========================================================

    st.subheader(
        "Calculation"
    )

    for _, row in calculation_df.iterrows():

        strike = row["Strike"]

        ce_atp = row["CE ATP"]
        ce_ltp = row["CE LTP"]

        pe_ltp = row["PE LTP"]
        pe_atp = row["PE ATP"]

        ce_dev = row["CE LTP-ATP"]
        pe_dev = row["PE LTP-ATP"]

        relative_dev = row[
            "PE Dev - CE Dev"
        ]

        level = row[
            "Calculated Level"
        ]

        st.write(
            f"**{strike:,.0f}:**  "
            f"CE: {ce_ltp:,.2f} − {ce_atp:,.2f} "
            f"= **{ce_dev:+,.2f}**  |  "
            f"PE: {pe_ltp:,.2f} − {pe_atp:,.2f} "
            f"= **{pe_dev:+,.2f}**  |  "
            f"PE Dev − CE Dev = **{relative_dev:+,.2f}**  |  "
            f"Level = **{level:,.2f}**"
        )

    st.divider()

    # ==========================================================
    # Final calculation
    # ==========================================================

    levels_text = " + ".join(
        f"{x:,.2f}"
        for x in calculation_df[
            "Calculated Level"
        ]
    )

    st.write(
        f"**Average Level:** "
        f"({levels_text}) / 3 "
        f"= **{average_level:,.2f}**"
    )

    st.write(
        f"**Signal:** "
        f"{average_level:,.2f} − "
        f"{spot:,.2f} "
        f"= **{signal:+,.2f}**"
    )

    # ==========================================================
    # Raw Quotes API
    # ==========================================================

    with st.expander(
        "Quotes API details",
        expanded=False
    ):

        quote_rows = []

        for strike in selected_strikes:

            ce_symbol = option_map[
                strike
            ]["CE"]

            pe_symbol = option_map[
                strike
            ]["PE"]

            ce = quotes.get(
                ce_symbol,
                {}
            )

            pe = quotes.get(
                pe_symbol,
                {}
            )

            quote_rows.append({

                "Strike": strike,

                "CE Symbol": ce_symbol,
                "CE LP": ce.get("lp"),
                "CE ATP": ce.get("atp"),

                "PE Symbol": pe_symbol,
                "PE LP": pe.get("lp"),
                "PE ATP": pe.get("atp"),
            })

        quote_df = pd.DataFrame(
            quote_rows
        )

        st.dataframe(
            quote_df.style.format(
                {
                    "Strike": "{:,.0f}",

                    "CE LP": "{:,.2f}",
                    "CE ATP": "{:,.2f}",

                    "PE LP": "{:,.2f}",
                    "PE ATP": "{:,.2f}",
                },
                na_rep="–"
            ),
            hide_index=True,
            use_container_width=True
        )

    # ==========================================================
    # Formula explanation
    # ==========================================================

    with st.expander(
        "Formula and interpretation",
        expanded=False
    ):

        st.latex(
            r"""
            CE_{dev}=CE_{LTP}-CE_{ATP}
            """
        )

        st.latex(
            r"""
            PE_{dev}=PE_{LTP}-PE_{ATP}
            """
        )

        st.latex(
            r"""
            D_K=PE_{dev}-CE_{dev}
            """
        )

        st.latex(
            r"""
            L_K=K+D_K
            """
        )

        st.latex(
            r"""
            L_{avg}
            =
            \frac{
            L_{ATM-1}
            +
            L_{ATM}
            +
            L_{ATM+1}
            }{3}
            """
        )

        st.latex(
            r"""
            Signal=L_{avg}-Spot
            """
        )

        st.markdown(
            """
### Why use the relative deviation?

The important point is that an option being above or below
its ATP **by itself does not tell us the direction**.

For example:

**Case 1 — both above ATP**

CE = +10 above ATP

PE = +5 above ATP

Therefore:

`PE deviation - CE deviation = 5 - 10 = -5`

The CE is relatively stronger versus its ATP.

---

**Case 2 — both below ATP**

CE = -8 below ATP

PE = -3 below ATP

Therefore:

`PE deviation - CE deviation = -3 - (-8) = +5`

Again, we are comparing the two sides relative to their
own average traded prices.

---

**Case 3 — CE above, PE below**

CE = +10

PE = -5

Therefore:

`-5 - (+10) = -15`

---

**Case 4 — CE below, PE above**

CE = -5

PE = +10

Therefore:

`+10 - (-5) = +15`

So the metric does NOT require one option to be above ATP
and the other to be below ATP.

It measures the **relative displacement** between CE and PE.
"""
        )

        st.warning(
            """
Do not assume yet that positive signal = bullish or
negative signal = bearish.

First test the historical relationship between:

    Signal(t)

and:

    Spot(t+n) - Spot(t)

for your chosen forward horizon.

The code is calculating the metric correctly; the direction
of its predictive relationship needs to come from the data.
"""
        )


# ==============================================================
# Run
# ==============================================================

if __name__ == "__main__":
    main()
