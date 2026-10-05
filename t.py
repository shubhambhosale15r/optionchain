import datetime
import numpy as np
import pandas as pd
import streamlit as st

from fyers_apiv3 import fyersModel

# Optional auto refresh
try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except ImportError:
    AUTO_REFRESH_AVAILABLE = False


# ============================================================
# FYERS CONNECTION
# ============================================================

def get_fyers(client_id, access_token):
    return fyersModel.FyersModel(
        client_id=client_id,
        token=access_token,
        is_async=False,
        log_path=""
    )


# ============================================================
# OPTION CHAIN
# ============================================================

@st.cache_data(ttl=5)
def fetch_chain(client_id, access_token, underlying, strikecount, expiry_date):
    fyers = get_fyers(client_id, access_token)

    data = {
        "symbol": underlying,
        "strikecount": strikecount,
        "timestamp": "",
        "option_type": "all",
        "expiry": expiry_date,
        "greeks": "1"
    }

    response = fyers.optionchain(data=data)

    if response.get("s") != "ok":
        st.error(f"Option chain error: {response}")
        return None

    return response


# ============================================================
# SPOT PRICE
# ============================================================

@st.cache_data(ttl=5)
def fetch_spot(client_id, access_token, underlying):
    fyers = get_fyers(client_id, access_token)

    data = {
        "symbols": underlying
    }

    response = fyers.quotes(data=data)

    if response.get("s") != "ok":
        st.error(f"Spot quote error: {response}")
        return None

    try:
        return float(response["d"][0]["v"]["lp"])
    except Exception as e:
        st.error(f"Could not read spot price: {e}")
        return None


# ============================================================
# OPTION QUOTES
#
# Fetches:
#   lp  = LTP
#   atp = Average Traded Price
# ============================================================

@st.cache_data(ttl=5)
def fetch_option_quotes(client_id, access_token, symbols):

    if not symbols:
        return {}

    if len(symbols) > 50:
        st.error("Quotes API supports maximum 50 symbols per request.")
        return {}

    fyers = get_fyers(client_id, access_token)

    data = {
        "symbols": ",".join(symbols)
    }

    response = fyers.quotes(data=data)

    if response.get("s") != "ok":
        st.error(f"Quotes API error: {response}")
        return {}

    result = {}

    for item in response.get("d", []):

        symbol = item.get("n")

        if not symbol:
            symbol = item.get("v", {}).get("symbol")

        if not symbol:
            continue

        v = item.get("v", {})

        try:
            result[symbol] = {
                "lp": float(v.get("lp", np.nan)),
                "atp": float(v.get("atp", np.nan)),
                "bid": float(v.get("bid", np.nan)),
                "ask": float(v.get("ask", np.nan)),
                "volume": float(v.get("volume", np.nan))
            }

        except Exception:
            continue

    return result


# ============================================================
# BUILD OPTION MAP
#
# strike -> CE / PE symbol
# ============================================================

def build_option_map(chain_response):

    option_map = {}

    if not chain_response:
        return option_map

    chain = chain_response.get("data", {}).get("optionsChain", [])

    for row in chain:

        strike = row.get("strike")

        if strike is None:
            continue

        try:
            strike = float(strike)
        except Exception:
            continue

        option_type = str(
            row.get("option_type")
            or row.get("type")
            or ""
        ).upper()

        symbol = (
            row.get("symbol")
            or row.get("name")
            or row.get("original_name")
        )

        if not symbol:
            continue

        if option_type in ["CE", "CALL"]:
            option_type = "CE"

        elif option_type in ["PE", "PUT"]:
            option_type = "PE"

        else:
            continue

        if strike not in option_map:
            option_map[strike] = {}

        option_map[strike][option_type] = symbol

    return option_map


# ============================================================
# SELECT ATM-1 / ATM / ATM+1
# ============================================================

def select_three_strikes(option_map, spot):

    strikes = sorted(option_map.keys())

    if not strikes:
        return None, None

    # Find closest strike to spot
    atm_index = min(
        range(len(strikes)),
        key=lambda i: abs(strikes[i] - spot)
    )

    # Need one strike below and one above
    if atm_index == 0 or atm_index == len(strikes) - 1:
        return None, None

    atm_minus_1 = strikes[atm_index - 1]
    atm = strikes[atm_index]
    atm_plus_1 = strikes[atm_index + 1]

    selected = [
        atm_minus_1,
        atm,
        atm_plus_1
    ]

    # Validate both CE and PE exist
    for strike in selected:

        if "CE" not in option_map[strike]:
            return None, None

        if "PE" not in option_map[strike]:
            return None, None

    return selected, atm


# ============================================================
# CALCULATE CE / PE DEVIATIONS
# ============================================================

def calculate_deviations(selected_strikes, option_map, quotes):

    rows = []

    for strike in selected_strikes:

        ce_symbol = option_map[strike]["CE"]
        pe_symbol = option_map[strike]["PE"]

        ce_quote = quotes.get(ce_symbol)
        pe_quote = quotes.get(pe_symbol)

        if ce_quote is None or pe_quote is None:
            continue

        ce_ltp = ce_quote["lp"]
        ce_atp = ce_quote["atp"]

        pe_ltp = pe_quote["lp"]
        pe_atp = pe_quote["atp"]

        # ----------------------------------------------------
        # CE deviation
        #
        # Positive = CE LTP above its ATP
        # Negative = CE LTP below its ATP
        # ----------------------------------------------------

        ce_dev = ce_ltp - ce_atp

        # ----------------------------------------------------
        # PE deviation
        #
        # Positive = PE LTP above its ATP
        # Negative = PE LTP below its ATP
        # ----------------------------------------------------

        pe_dev = pe_ltp - pe_atp

        rows.append({
            "Strike": strike,

            "CE LTP": ce_ltp,
            "CE ATP": ce_atp,
            "CE Dev": ce_dev,

            "PE LTP": pe_ltp,
            "PE ATP": pe_atp,
            "PE Dev": pe_dev
        })

    if not rows:
        return None, None, None

    df = pd.DataFrame(rows)

    # ========================================================
    # AVERAGES ACROSS ATM-1 / ATM / ATM+1
    # ========================================================

    avg_ce_dev = df["CE Dev"].mean()
    avg_pe_dev = df["PE Dev"].mean()

    return df, avg_ce_dev, avg_pe_dev


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="NIFTY CE / PE ATP Deviation",
    layout="wide"
)

st.title("NIFTY CE / PE ATP Deviation")

st.caption(
    "Measures how far CE and PE LTP are from their own Average Traded Price (ATP)."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Settings")

client_id = st.sidebar.text_input(
    "Fyers Client ID"
)

access_token = st.sidebar.text_input(
    "Access Token",
    type="password"
)

underlying = st.sidebar.text_input(
    "Underlying",
    value="NSE:NIFTY50-INDEX"
)

strikecount = st.sidebar.number_input(
    "Strike Count",
    min_value=3,
    max_value=20,
    value=5,
    step=1
)

refresh_seconds = st.sidebar.number_input(
    "Auto Refresh Seconds",
    min_value=0,
    max_value=300,
    value=5,
    step=1
)


# ============================================================
# AUTO REFRESH
# ============================================================

if AUTO_REFRESH_AVAILABLE and refresh_seconds > 0:

    st_autorefresh(
        interval=refresh_seconds * 1000,
        key="nifty_atp_refresh"
    )


# ============================================================
# VALIDATE LOGIN
# ============================================================

if not client_id or not access_token:

    st.info("Enter Fyers Client ID and Access Token.")
    st.stop()


# ============================================================
# FETCH SPOT
# ============================================================

spot = fetch_spot(
    client_id,
    access_token,
    underlying
)

if spot is None:
    st.stop()


# ============================================================
# EXPIRY
# ============================================================

st.sidebar.markdown("---")

expiry_date = st.sidebar.text_input(
    "Expiry Date",
    value=""
)

st.sidebar.caption(
    "Leave blank if your option-chain implementation handles expiry automatically."
)


# ============================================================
# FETCH OPTION CHAIN
# ============================================================

chain_response = fetch_chain(
    client_id,
    access_token,
    underlying,
    int(strikecount),
    expiry_date
)

if chain_response is None:
    st.stop()


# ============================================================
# BUILD OPTION MAP
# ============================================================

option_map = build_option_map(chain_response)

if not option_map:
    st.error("No option symbols found in option chain.")
    st.stop()


# ============================================================
# SELECT ATM-1 / ATM / ATM+1
# ============================================================

selected_strikes, atm_strike = select_three_strikes(
    option_map,
    spot
)

if selected_strikes is None:

    st.error(
        "Could not find valid ATM-1 / ATM / ATM+1 strikes "
        "with both CE and PE."
    )

    st.stop()


# ============================================================
# COLLECT SIX OPTION SYMBOLS
# ============================================================

symbols = []

for strike in selected_strikes:

    symbols.append(
        option_map[strike]["CE"]
    )

    symbols.append(
        option_map[strike]["PE"]
    )


# ============================================================
# FETCH QUOTES
# ============================================================

quotes = fetch_option_quotes(
    client_id,
    access_token,
    tuple(symbols)
)

if not quotes:

    st.error("No option quotes received.")
    st.stop()


# ============================================================
# CALCULATE
# ============================================================

df, avg_ce_dev, avg_pe_dev = calculate_deviations(
    selected_strikes,
    option_map,
    quotes
)

if df is None:

    st.error(
        "Could not calculate CE/PE deviations."
    )

    st.stop()


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "NIFTY Spot",
        f"{spot:,.2f}"
    )

with col2:
    st.metric(
        "ATM Strike",
        f"{atm_strike:,.0f}"
    )

with col3:
    st.metric(
        "Avg CE Dev",
        f"{avg_ce_dev:+.2f}"
    )

with col4:
    st.metric(
        "Avg PE Dev",
        f"{avg_pe_dev:+.2f}"
    )


# ============================================================
# MAIN TABLE
# ============================================================

st.subheader("CE / PE ATP Deviation")

display_df = df.copy()

display_df["Strike"] = display_df["Strike"].map(
    lambda x: f"{x:,.0f}"
)

for col in [
    "CE LTP",
    "CE ATP",
    "CE Dev",
    "PE LTP",
    "PE ATP",
    "PE Dev"
]:

    display_df[col] = display_df[col].map(
        lambda x: f"{x:.2f}"
    )

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# EXPLICIT CALCULATION
# ============================================================

st.subheader("Calculation")

for _, row in df.iterrows():

    strike = row["Strike"]

    ce_ltp = row["CE LTP"]
    ce_atp = row["CE ATP"]
    ce_dev = row["CE Dev"]

    pe_ltp = row["PE LTP"]
    pe_atp = row["PE ATP"]
    pe_dev = row["PE Dev"]

    st.write(
        f"**{strike:,.0f}:** "
        f"CE: {ce_ltp:.2f} − {ce_atp:.2f} = "
        f"**{ce_dev:+.2f}** | "
        f"PE: {pe_ltp:.2f} − {pe_atp:.2f} = "
        f"**{pe_dev:+.2f}**"
    )


# ============================================================
# AVERAGE CALCULATION
# ============================================================

st.subheader("Average Across ATM-1 / ATM / ATM+1")

ce_values = df["CE Dev"].tolist()
pe_values = df["PE Dev"].tolist()

ce_formula = " + ".join(
    f"({x:.2f})" for x in ce_values
)

pe_formula = " + ".join(
    f"({x:.2f})" for x in pe_values
)

st.write(
    f"**Average CE Dev:** "
    f"({ce_formula}) / {len(ce_values)} "
    f"= **{avg_ce_dev:+.2f}**"
)

st.write(
    f"**Average PE Dev:** "
    f"({pe_formula}) / {len(pe_values)} "
    f"= **{avg_pe_dev:+.2f}**"
)

