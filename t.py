"""
VWAP / ATP - LTP Implied Level
--------------------------------

For ATM-1, ATM and ATM+1:

    Level_K =
        K + (CE_ATP - CE_LTP)
          - (PE_ATP - PE_LTP)

where:

    CE_ATP = Fyers Quotes API `atp`
    PE_ATP = Fyers Quotes API `atp`
    CE_LTP = Fyers Quotes API `lp`
    PE_LTP = Fyers Quotes API `lp`

Final level:

    Average Level =
        mean(Level_ATM-1, Level_ATM, Level_ATM+1)

Signal:

    Signal = Average Level - Spot

Interpretation:

    Signal > 0 -> calculated level above actual spot
    Signal < 0 -> calculated level below actual spot
    Signal ~ 0 -> calculated level near actual spot

NOTE:
This is NOT standard put-call parity synthetic spot.

It is a VWAP/ATP-vs-LTP CE/PE relative metric.
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
# Fyers Quotes API supports up to 50 symbols.
#
# We only need 6 option symbols:
#
# ATM-1 CE
# ATM-1 PE
# ATM CE
# ATM PE
# ATM+1 CE
# ATM+1 PE
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

        symbol = item.get("n")

        values = item.get("v") or {}

        if not symbol:
            continue

        lp = values.get("lp")
        atp = values.get("atp")

        try:
            lp = float(lp)
        except (TypeError, ValueError):
            lp = np.nan

        try:
            atp = float(atp)
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
# Extract option-chain structure
# ==============================================================

def build_option_map(chain):
    """
    Create:

        strike -> {
            CE: option symbol,
            PE: option symbol
        }

    from the option-chain response.
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

    strikes = sorted(option_map.keys())

    if not strikes:
        raise RuntimeError(
            "No strikes available in option chain."
        )

    # ATM = strike nearest actual spot

    atm = min(
        strikes,
        key=lambda x: abs(x - spot)
    )

    atm_index = strikes.index(atm)

    if atm_index == 0:
        raise RuntimeError(
            "ATM has no lower strike available "
            "for ATM-1."
        )

    if atm_index == len(strikes) - 1:
        raise RuntimeError(
            "ATM has no higher strike available "
            "for ATM+1."
        )

    lower = strikes[atm_index - 1]
    upper = strikes[atm_index + 1]

    selected = [
        lower,
        atm,
        upper
    ]

    # Make sure all six option symbols exist.

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
# Calculate signal
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

        ce_quote = quotes.get(ce_symbol)
        pe_quote = quotes.get(pe_symbol)

        if ce_quote is None:
            raise RuntimeError(
                f"No Quotes API response for CE: {ce_symbol}"
            )

        if pe_quote is None:
            raise RuntimeError(
                f"No Quotes API response for PE: {pe_symbol}"
            )

        ce_ltp = ce_quote["lp"]
        ce_atp = ce_quote["atp"]

        pe_ltp = pe_quote["lp"]
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

        # ------------------------------------------------------
        # Your formula
        #
        # K + (CE ATP - CE LTP)
        #   - (PE ATP - PE LTP)
        # ------------------------------------------------------

        ce_deviation = (
            ce_atp - ce_ltp
        )

        pe_deviation = (
            pe_atp - pe_ltp
        )

        level = (
            strike
            + ce_deviation
            - pe_deviation
        )

        rows.append({

            "Strike": strike,

            "CE Symbol": ce_symbol,
            "CE ATP": ce_atp,
            "CE LTP": ce_ltp,
            "CE ATP-LTP": ce_deviation,

            "PE Symbol": pe_symbol,
            "PE LTP": pe_ltp,
            "PE ATP": pe_atp,
            "PE ATP-LTP": pe_deviation,

            "Calculated Level": level,
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "Strike"
    ).reset_index(drop=True)

    average_level = (
        result["Calculated Level"].mean()
    )

    return result, float(average_level)


# ==============================================================
# Styling
# ==============================================================

def style_chain(
    df,
    selected_strikes,
    atm
):

    selected_style = (
        "background-color: rgba(46,160,67,0.25)"
    )

    atm_style = (
        "background-color: rgba(227,179,65,0.45); "
        "font-weight: 600"
    )

    def paint(d):

        css = pd.DataFrame(
            "",
            index=d.index,
            columns=d.columns
        )

        css.loc[
            d["Strike"].isin(selected_strikes),
            :
        ] = selected_style

        css.loc[
            d["Strike"] == atm,
            "Strike"
        ] = atm_style

        return css

    fmt = {

        "Strike": "{:,.0f}",

        "CE ATP": "{:,.2f}",
        "CE LTP": "{:,.2f}",
        "CE ATP-LTP": "{:+,.2f}",

        "PE LTP": "{:,.2f}",
        "PE ATP": "{:,.2f}",
        "PE ATP-LTP": "{:+,.2f}",

        "Calculated Level": "{:,.2f}",
    }

    return (
        df.style
        .apply(
            paint,
            axis=None
        )
        .format(
            fmt,
            na_rep="–"
        )
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
        page_title="VWAP-LTP Implied Level",
        layout="wide"
    )

    st.title(
        "VWAP / ATP-LTP Implied Level"
    )

    st.caption(
        "K + (CE ATP − CE LTP) − (PE ATP − PE LTP)"
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
    # Fetch option chain
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
                fetch_error or
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
        # Build strike -> CE/PE symbol map
        # ------------------------------------------------------

        option_map = build_option_map(
            chain
        )

        # ------------------------------------------------------
        # Determine ATM / ATM-1 / ATM+1
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
        # Get the six option symbols
        # ------------------------------------------------------

        quote_symbols = []

        for strike in selected_strikes:

            quote_symbols.append(
                option_map[strike]["CE"]
            )

            quote_symbols.append(
                option_map[strike]["PE"]
            )

        # Remove duplicates while preserving order

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

        calculation_df, average_level = (
            calculate_levels(
                selected_strikes,
                option_map,
                quotes
            )
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
    # Headline metrics
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
        "Average calculated level",
        f"{average_level:,.2f}"
    )

    m[3].metric(
        "Signal",
        f"{signal:+,.2f}"
    )

    m[4].metric(
        "Strikes",
        (
            f"{lower:,.0f} / "
            f"{atm:,.0f} / "
            f"{upper:,.0f}"
        )
    )

    # ==========================================================
    # Signal interpretation
    # ==========================================================

    if signal > 0:

        st.success(
            f"Calculated level is "
            f"**{signal:+,.2f} points above spot**."
        )

    elif signal < 0:

        st.warning(
            f"Calculated level is "
            f"**{signal:+,.2f} points below spot**."
        )

    else:

        st.info(
            "Calculated level equals spot."
        )

    # ==========================================================
    # Individual calculation
    # ==========================================================

    st.subheader(
        "ATM-1 / ATM / ATM+1"
    )

    display_df = calculation_df[
        [
            "Strike",

            "CE ATP",
            "CE LTP",
            "CE ATP-LTP",

            "PE LTP",
            "PE ATP",
            "PE ATP-LTP",

            "Calculated Level",
        ]
    ].copy()

    st.dataframe(
        display_df.style.format(
            {
                "Strike": "{:,.0f}",

                "CE ATP": "{:,.2f}",
                "CE LTP": "{:,.2f}",
                "CE ATP-LTP": "{:+,.2f}",

                "PE LTP": "{:,.2f}",
                "PE ATP": "{:,.2f}",
                "PE ATP-LTP": "{:+,.2f}",

                "Calculated Level": "{:,.2f}",
            },
            na_rep="–"
        ),
        hide_index=True,
        use_container_width=True
    )

    # ==========================================================
    # Explicit formulas
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

        level = row["Calculated Level"]

        st.write(
            f"**{strike:,.0f}:** "
            f"{strike:,.0f} + "
            f"({ce_atp:,.2f} − {ce_ltp:,.2f}) − "
            f"({pe_atp:,.2f} − {pe_ltp:,.2f}) "
            f"= **{level:,.2f}**"
        )

    st.divider()

    levels_text = " + ".join(
        f"{x:,.2f}"
        for x in calculation_df[
            "Calculated Level"
        ]
    )

    st.write(
        f"**Average calculated level:** "
        f"({levels_text}) / "
        f"{len(calculation_df)} "
        f"= **{average_level:,.2f}**"
    )

    st.write(
        f"**Signal:** "
        f"{average_level:,.2f} − "
        f"{spot:,.2f} "
        f"= **{signal:+,.2f}**"
    )

    # ==========================================================
    # Raw Quotes API diagnostic
    # ==========================================================

    with st.expander(
        "Quotes API details",
        expanded=False
    ):

        quote_rows = []

        for strike in selected_strikes:

            ce_symbol = option_map[strike]["CE"]
            pe_symbol = option_map[strike]["PE"]

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
    # Formula / notes
    # ==========================================================

    with st.expander(
        "Formula and interpretation",
        expanded=False
    ):

        st.latex(
            r"""
            L_K =
            K+
            (CE_{ATP}-CE_{LTP})
            -
            (PE_{ATP}-PE_{LTP})
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
            Signal =
            L_{avg}-Spot
            """
        )

        st.markdown(
            """
### What the code is doing

For each of the three strikes:

`ATM-1, ATM, ATM+1`

it gets:

- CE `lp` from Quotes API
- CE `atp` from Quotes API
- PE `lp` from Quotes API
- PE `atp` from Quotes API

Then:

`K + (CE ATP - CE LTP) - (PE ATP - PE LTP)`

The three resulting levels are averaged.

Finally:

`Average Level - NIFTY Spot`

is displayed as the signal.

### Important

`atp` is the Fyers **Average Traded Price**.

This metric is **not** the usual synthetic spot:

`K + CE - PE`

Instead, it measures the relationship between each option's current traded price and its accumulated average traded price, with the CE/PE difference converted into a level around the strike.
"""
        )

        st.markdown(
            """
### Example

Suppose the three levels are:

`22528`

`22534`

`22531`

Then:

`Average = 22531`

If spot is:

`22521`

then:

`Signal = 22531 - 22521 = +10`

So the dashboard reports:

**+10 points**
"""
        )


# ==============================================================
# Run
# ==============================================================

if __name__ == "__main__":
    main()
