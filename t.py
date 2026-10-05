import math
import pandas as pd
import streamlit as st
from fyers_apiv3 import fyersModel
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="NIFTY CE/PE ATP Deviation", layout="wide")

def get_fyers(client_id, access_token):
    return fyersModel.FyersModel(client_id=client_id, token=access_token, is_async=False, log_path="")

@st.cache_data(ttl=5, show_spinner=False)
def fetch_chain(_fyers, symbol, strikecount, timestamp):
    response = _fyers.optionchain(data={
        "symbol": symbol,
        "strikecount": int(strikecount),
        "timestamp": int(timestamp),
        "greeks": "1"
    })
    return response

@st.cache_data(ttl=5, show_spinner=False)
def fetch_spot(_fyers, symbol):
    response = _fyers.quotes({"symbols": symbol})
    if response.get("s") != "ok":
        raise RuntimeError(f"Spot quote failed: {response}")
    data = response.get("d", [])
    if not data:
        raise RuntimeError(f"Spot quote returned no data: {response}")
    item = data[0]
    if item.get("s") != "ok":
        raise RuntimeError(f"Spot quote failed: {item}")
    value = item.get("v", {})
    lp = value.get("lp")
    if lp is None or pd.isna(lp):
        raise RuntimeError(f"Spot LTP unavailable: {item}")
    return float(lp)

@st.cache_data(ttl=5, show_spinner=False)
def fetch_option_quotes(_fyers, symbols):
    symbols = tuple(dict.fromkeys(symbols))
    if not symbols:
        return {}
    if len(symbols) > 50:
        raise ValueError("Quotes API accepts a maximum of 50 symbols per request.")
    response = _fyers.quotes({"symbols": ",".join(symbols)})
    if response.get("s") != "ok":
        raise RuntimeError(f"Option quotes failed: {response}")
    quotes = {}
    for item in response.get("d", []):
        if item.get("s") != "ok":
            continue
        value = item.get("v", {})
        symbol = item.get("n") or value.get("symbol")
        if not symbol:
            continue
        quotes[symbol] = {
            "symbol": symbol,
            "lp": value.get("lp"),
            "atp": value.get("atp"),
            "bid": value.get("bid"),
            "ask": value.get("ask"),
            "volume": value.get("volume")
        }
    return quotes

def get_expiry_data(chain_response):
    data = chain_response.get("data", {})
    expiry_data = data.get("expiryData", [])
    if not expiry_data:
        return []
    return expiry_data

def get_option_chain_rows(chain_response):
    data = chain_response.get("data", {})
    options_chain = data.get("optionsChain", [])
    if not options_chain:
        return []
    return options_chain

def format_expiry(expiry_item):
    expiry_date = expiry_item.get("expiryDate")
    expiry_epoch = expiry_item.get("expiry")
    if expiry_date:
        return str(expiry_date)
    if expiry_epoch:
        return str(expiry_epoch)
    return "Unknown"

def get_expiry_value(expiry_item):
    if expiry_item.get("expiry") is not None:
        return int(expiry_item["expiry"])
    if expiry_item.get("expiryDate") is not None:
        return expiry_item["expiryDate"]
    return None

def build_option_map(chain):
    option_map = {}
    for row in chain:
        option_type = row.get("option_type")
        strike_price = row.get("strike_price")
        symbol = row.get("symbol") or row.get("name") or row.get("original_name")
        if option_type not in ("CE", "PE"):
            continue
        if strike_price is None or symbol is None:
            continue
        try:
            strike = float(strike_price)
        except (TypeError, ValueError):
            continue
        if strike not in option_map:
            option_map[strike] = {}
        option_map[strike][option_type] = symbol
    return option_map

def select_three_strikes(option_map, spot):
    strikes = sorted(option_map.keys())
    if not strikes:
        raise RuntimeError("No option strikes found in option chain.")
    atm_index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
    if atm_index == 0 or atm_index == len(strikes) - 1:
        raise RuntimeError("Unable to select ATM-1, ATM and ATM+1 from the available strikes.")
    selected = strikes[atm_index - 1:atm_index + 2]
    for strike in selected:
        if "CE" not in option_map[strike] or "PE" not in option_map[strike]:
            raise RuntimeError(f"Both CE and PE are not available for strike {strike}.")
    return selected, strikes[atm_index]

def valid_number(value):
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def calculate_deviations(selected_strikes, option_map, quotes):
    rows = []
    for strike in selected_strikes:
        ce_symbol = option_map[strike]["CE"]
        pe_symbol = option_map[strike]["PE"]
        ce_quote = quotes.get(ce_symbol)
        pe_quote = quotes.get(pe_symbol)
        if not ce_quote:
            raise RuntimeError(f"CE quote missing: {ce_symbol}")
        if not pe_quote:
            raise RuntimeError(f"PE quote missing: {pe_symbol}")
        ce_ltp = ce_quote.get("lp")
        ce_atp = ce_quote.get("atp")
        pe_ltp = pe_quote.get("lp")
        pe_atp = pe_quote.get("atp")
        if not all(valid_number(x) for x in [ce_ltp, ce_atp, pe_ltp, pe_atp]):
            raise RuntimeError(
                f"Invalid quote data for strike {strike}: "
                f"CE LTP={ce_ltp}, CE ATP={ce_atp}, PE LTP={pe_ltp}, PE ATP={pe_atp}"
            )
        ce_ltp = float(ce_ltp)
        ce_atp = float(ce_atp)
        pe_ltp = float(pe_ltp)
        pe_atp = float(pe_atp)
        ce_deviation = ce_ltp - ce_atp
        pe_deviation = pe_ltp - pe_atp
        rows.append({
            "Strike": strike,
            "CE Symbol": ce_symbol,
            "CE ATP": ce_atp,
            "CE LTP": ce_ltp,
            "CE LTP-ATP": ce_deviation,
            "PE Symbol": pe_symbol,
            "PE LTP": pe_ltp,
            "PE ATP": pe_atp,
            "PE LTP-ATP": pe_deviation
        })
    result = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
    average_ce_deviation = result["CE LTP-ATP"].mean()
    average_pe_deviation = result["PE LTP-ATP"].mean()
    return result, float(average_ce_deviation), float(average_pe_deviation)

def ms_until_next_minute():
    now = pd.Timestamp.now()
    next_minute = now.ceil("min")
    milliseconds = int((next_minute - now).total_seconds() * 1000)
    return max(milliseconds, 100)

def install_autorefresh(enabled, mode, fixed_seconds):
    if not enabled:
        return
    if mode == "Next minute":
        interval = ms_until_next_minute()
    else:
        interval = max(int(fixed_seconds * 1000), 1000)
    st_autorefresh(interval=interval, key="nifty_atp_refresh")

st.title("NIFTY CE/PE ATP Deviation")

with st.sidebar:
    st.header("Settings")
    client_id = st.text_input("Client ID", type="default")
    access_token = st.text_input("Access Token", type="password")
    underlying = st.text_input("Underlying", value="NSE:NIFTY50-INDEX")
    strikecount = st.number_input("Strike Count", min_value=1, max_value=50, value=10, step=1)
    autorefresh_enabled = st.checkbox("Auto Refresh", value=True)
    refresh_mode = st.selectbox("Refresh Mode", ["Next minute", "Fixed seconds"])
    fixed_seconds = st.number_input("Fixed Seconds", min_value=1, max_value=3600, value=5, step=1)

install_autorefresh(autorefresh_enabled, refresh_mode, fixed_seconds)

if not client_id or not access_token:
    st.info("Enter Client ID and Access Token in the sidebar.")
    st.stop()

try:
    fyers = get_fyers(client_id, access_token)
except Exception as e:
    st.error(f"Fyers initialization failed: {e}")
    st.stop()

try:
    initial_chain = fetch_chain(fyers, underlying, int(strikecount), 0)
except Exception as e:
    st.error(f"Option chain fetch failed: {e}")
    st.stop()

expiry_data = get_expiry_data(initial_chain)

if not expiry_data:
    st.error("No expiry data returned by the option chain.")
    st.stop()

expiry_labels = []
expiry_values = []

for item in expiry_data:
    label = format_expiry(item)
    value = get_expiry_value(item)
    expiry_labels.append(label)
    expiry_values.append(value)

expiry_index = st.sidebar.selectbox(
    "Expiry",
    range(len(expiry_labels)),
    format_func=lambda i: expiry_labels[i]
)

selected_expiry = expiry_values[expiry_index]

try:
    chain_response = fetch_chain(
        fyers,
        underlying,
        int(strikecount),
        selected_expiry
    )
except Exception as e:
    st.error(f"Option chain fetch failed: {e}")
    st.stop()

try:
    spot = fetch_spot(fyers, underlying)
except Exception as e:
    st.error(f"Spot fetch failed: {e}")
    st.stop()

chain_rows = get_option_chain_rows(chain_response)

if not chain_rows:
    st.error("No option chain rows returned.")
    st.stop()

option_map = build_option_map(chain_rows)

try:
    selected_strikes, atm = select_three_strikes(option_map, spot)
except Exception as e:
    st.error(str(e))
    st.stop()

symbols = []

for strike in selected_strikes:
    symbols.append(option_map[strike]["CE"])
    symbols.append(option_map[strike]["PE"])

try:
    quotes = fetch_option_quotes(fyers, tuple(symbols))
except Exception as e:
    st.error(f"Option quotes fetch failed: {e}")
    st.stop()

try:
    calculation_df, average_ce_deviation, average_pe_deviation = calculate_deviations(
        selected_strikes,
        option_map,
        quotes
    )
except Exception as e:
    st.error(f"Deviation calculation failed: {e}")
    st.stop()

m = st.columns(4)
m[0].metric("Spot", f"{spot:,.2f}")
m[1].metric("ATM", f"{atm:,.0f}")
m[2].metric("Average CE Dev.", f"{average_ce_deviation:+,.2f}")
m[3].metric("Average PE Dev.", f"{average_pe_deviation:+,.2f}")

st.subheader("CE / PE LTP − ATP")

display_df = calculation_df[
    [
        "Strike",
        "CE ATP",
        "CE LTP",
        "CE LTP-ATP",
        "PE LTP",
        "PE ATP",
        "PE LTP-ATP"
    ]
].copy()

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.subheader("Calculation")

for _, row in calculation_df.iterrows():
    st.write(
        f"**{row['Strike']:,.0f}** → "
        f"CE: {row['CE LTP']:,.2f} − {row['CE ATP']:,.2f} = "
        f"**{row['CE LTP-ATP']:+,.2f}** | "
        f"PE: {row['PE LTP']:,.2f} − {row['PE ATP']:,.2f} = "
        f"**{row['PE LTP-ATP']:+,.2f}**"
    )

ce_values = calculation_df["CE LTP-ATP"].tolist()
pe_values = calculation_df["PE LTP-ATP"].tolist()

ce_values_text = " + ".join(f"({x:+,.2f})" for x in ce_values)
pe_values_text = " + ".join(f"({x:+,.2f})" for x in pe_values)

st.write(
    f"**Average CE Dev:** "
    f"({ce_values_text}) / {len(ce_values)} = "
    f"**{average_ce_deviation:+,.2f}**"
)

st.write(
    f"**Average PE Dev:** "
    f"({pe_values_text}) / {len(pe_values)} = "
    f"**{average_pe_deviation:+,.2f}**"
)

st.subheader("Selected Strikes")

strike_table = []

for strike in selected_strikes:
    ce_symbol = option_map[strike]["CE"]
    pe_symbol = option_map[strike]["PE"]
    strike_table.append({
        "Strike": strike,
        "CE Symbol": ce_symbol,
        "PE Symbol": pe_symbol
    })

st.dataframe(
    pd.DataFrame(strike_table),
    use_container_width=True,
    hide_index=True
)

with st.expander("Raw Quotes API Data"):
    raw_rows = []
    for symbol in symbols:
        quote = quotes.get(symbol, {})
        raw_rows.append({
            "Symbol": symbol,
            "LTP": quote.get("lp"),
            "ATP": quote.get("atp"),
            "Bid": quote.get("bid"),
            "Ask": quote.get("ask"),
            "Volume": quote.get("volume")
        })
    st.dataframe(
        pd.DataFrame(raw_rows),
        use_container_width=True,
        hide_index=True
    )

with st.expander("Formula"):
    st.latex(r"CE_{dev}=CE_{LTP}-CE_{ATP}")
    st.latex(r"PE_{dev}=PE_{LTP}-PE_{ATP}")
    st.latex(
        r"AvgCEDev=\frac{CEDev_{ATM-1}+CEDev_{ATM}+CEDev_{ATM+1}}{3}"
    )
    st.latex(
        r"AvgPEDev=\frac{PEDev_{ATM-1}+PEDev_{ATM}+PEDev_{ATM+1}}{3}"
    )
    st.write(
        "CE and PE deviations are kept separate. No combined directional metric is calculated."
    )

if st.button("Manual Refresh"):
    st.cache_data.clear()
    st.rerun()
