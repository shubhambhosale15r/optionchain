"""
Call vs Put decay per unit of delta (|theta| / |delta|) on the live Fyers option chain.

Layout: proper option-chain view with LTP included
    CE LTP | CE Theta | CE Δ | CE |θ|/|Δ| | Strike | PE |θ|/|Δ| | PE Δ | PE Theta | PE LTP
plus a paired strike-by-strike comparison and per-side sums.
"""
import numpy as np
import pandas as pd
import streamlit as st
from fyers_apiv3 import fyersModel


# ---------------------------------------------------------------- Fyers calls
def get_fyers(client_id, access_token):
    return fyersModel.FyersModel(client_id=client_id, token=access_token,
                                 is_async=False, log_path="")


@st.cache_data(ttl=5, show_spinner=False)
def fetch_chain(client_id, access_token, symbol, strikecount, timestamp=""):
    return get_fyers(client_id, access_token).optionchain(
        data={"symbol": symbol, "strikecount": int(strikecount),
              "timestamp": timestamp, "greeks": "1"}
    )


@st.cache_data(ttl=5, show_spinner=False)
def fetch_spot(client_id, access_token, symbol):
    resp = get_fyers(client_id, access_token).quotes(data={"symbols": symbol})
    if resp.get("s") != "ok" or not resp.get("d") or resp["d"][0].get("s") != "ok":
        raise RuntimeError(f"Quotes API error: {resp}")
    return float(resp["d"][0]["v"]["lp"])


# ---------------------------------------------------------------- pure logic
def _price(r):
    """Bid/ask mid when both quoted, else LTP."""
    b, a = r.get("bid") or 0, r.get("ask") or 0
    return (b + a) / 2 if b > 0 and a > 0 else r["ltp"]


def synthetic_forward(chain, spot, n=3):
    """F = K + C - P at the n strikes nearest spot (median)."""
    ce = {r["strike_price"]: _price(r) for r in chain if r.get("option_type") == "CE"}
    pe = {r["strike_price"]: _price(r) for r in chain if r.get("option_type") == "PE"}
    common = sorted(set(ce) & set(pe), key=lambda k: abs(k - spot))[:n]
    if not common:
        raise RuntimeError("No strike with both CE and PE to derive the forward")
    return float(np.median([k + ce[k] - pe[k] for k in common]))


CHAIN_COLS = [
    "CE LTP", "CE Theta", "CE Δ", "CE |θ|/|Δ|",
    "Strike",
    "PE |θ|/|Δ|", "PE Δ", "PE Theta", "PE LTP",
]


def build_chain_table(chain):
    """One row per strike: CE block | Strike | PE block."""
    rows = {}
    for r in chain:
        t = r.get("option_type")
        if t not in ("CE", "PE"):
            continue
        k = r["strike_price"]
        g = r.get("greeks") or {}
        theta = g.get("theta")
        delta = g.get("delta")
        ratio = (abs(theta) / abs(delta)) if (theta and delta and delta != 0) else np.nan

        rows.setdefault(k, {"Strike": k}).update(
            {
                f"{t} LTP": r["ltp"],
                f"{t} Theta": theta,
                f"{t} Δ": delta,
                f"{t} |θ|/|Δ|": ratio,
            }
        )

    df = pd.DataFrame(list(rows.values())).sort_values("Strike").reset_index(drop=True)
    return df.reindex(columns=CHAIN_COLS)


def band_compare(df, ref, n):
    """Pair the k-th OTM call above ATM with the k-th OTM put below ATM."""
    atm = df.Strike.iloc[(df.Strike - ref).abs().argmin()]
    ce = df[(df.Strike > atm) & df["CE |θ|/|Δ|"].notna()].sort_values("Strike").head(n).reset_index(drop=True)
    pe = df[(df.Strike < atm) & df["PE |θ|/|Δ|"].notna()].sort_values("Strike", ascending=False).head(n).reset_index(drop=True)
    m = min(len(ce), len(pe))
    ce, pe = ce.head(m), pe.head(m)

    pairs = pd.DataFrame({
        "Step": np.arange(1, m + 1),
        "Call Strike": ce["Strike"].values,
        "Call LTP": ce["CE LTP"].values,
        "Call Δ": ce["CE Δ"].values,
        "Call |θ|/|Δ|": ce["CE |θ|/|Δ|"].values,
        "Put Strike": pe["Strike"].values,
        "Put LTP": pe["PE LTP"].values,
        "Put Δ": pe["PE Δ"].values,
        "Put |θ|/|Δ|": pe["PE |θ|/|Δ|"].values,
    })
    pairs["Put advantage %"] = (
        (pairs["Put |θ|/|Δ|"] / pairs["Call |θ|/|Δ|"] - 1) * 100
    ).round(1)

    sc = ce["CE |θ|/|Δ|"].sum()
    sp = pe["PE |θ|/|Δ|"].sum()
    return atm, ce, pe, pairs, sc, sp


def style_chain(df, atm, ce_strikes, pe_strikes):
    """Green for rows entering the sums; amber for ATM."""
    green = "background-color: rgba(46,160,67,0.30)"
    amber = "background-color: rgba(227,179,65,0.45); font-weight: 600"

    def paint(d):
        css = pd.DataFrame("", index=d.index, columns=d.columns)
        css.loc[d.Strike.isin(ce_strikes), "CE |θ|/|Δ|"] = green
        css.loc[d.Strike.isin(pe_strikes), "PE |θ|/|Δ|"] = green
        css.loc[d.Strike == atm, "Strike"] = amber
        return css

    fmt = {c: "{:,.3f}" for c in CHAIN_COLS}
    fmt.update({
        "Strike": "{:,.0f}",
        "CE LTP": "{:,.2f}", "PE LTP": "{:,.2f}",
        "CE Δ": "{:.2f}", "PE Δ": "{:.2f}",
        "CE Theta": "{:,.2f}", "PE Theta": "{:,.2f}",
        "CE |θ|/|Δ|": "{:,.3f}", "PE |θ|/|Δ|": "{:,.3f}",
    })
    return df.style.apply(paint, axis=None).format(fmt, na_rep="–")


# ---------------------------------------------------------------- UI
def main():
    st.set_page_config(page_title="Theta / Delta: Calls vs Puts", layout="wide")
    st.title("Decay per unit of delta — |θ| / |Δ|")

    with st.sidebar:
        client_id = st.text_input("Client ID", placeholder="XXXXXXX-100")
        access_token = st.text_input("Access token", type="password")
        symbol = st.text_input("Underlying", "NSE:NIFTY50-INDEX")
        strikecount = st.number_input("Strikes fetched each side of ATM", 3, 50, 10)
        n = st.slider("Strikes compared each side (OTM)", 1, 15, 3)

    if not (client_id and access_token):
        st.info("Enter your Fyers Client ID and Access Token in the sidebar.")
        return

    # ---- fetch
    try:
        resp = fetch_chain(client_id, access_token, symbol, strikecount)
        if resp.get("s") != "ok":
            st.error(f"API error: {resp}")
            return

        exp = resp["data"].get("expiryData", [])
        if exp:
            labels = {e["date"]: e["expiry"] for e in exp}
            chosen = st.selectbox("Expiry", list(labels))
            if chosen != exp[0]["date"]:
                resp = fetch_chain(client_id, access_token, symbol, strikecount, labels[chosen])
                if resp.get("s") != "ok":
                    st.error(f"API error: {resp}")
                    return

        chain = resp["data"]["optionsChain"]
        spot = fetch_spot(client_id, access_token, symbol)
        fwd = synthetic_forward(chain, spot)
    except Exception as e:
        st.error(f"Failed: {e}")
        return

    df = build_chain_table(chain)

    if df.empty:
        st.error("No greeks returned (theta/delta missing). Check greeks=1 availability.")
        return

    if df["CE Theta"].dropna().nunique() <= 1 and df["PE Theta"].dropna().nunique() <= 1:
        st.warning("Theta is constant across strikes — looks like placeholder data. Do not trust results.")

    # ---- compare
    atm, ce, pe, pairs, sc, sp = band_compare(df, fwd, int(n))
    adv = (sp / sc - 1) * 100 if sc else np.nan

    # ---- headline metrics
    m = st.columns(5)
    m[0].metric("Spot", f"{spot:,.2f}")
    m[1].metric("Synthetic forward", f"{fwd:,.2f}", f"{fwd - spot:+,.2f} basis")
    m[2].metric("ATM strike", f"{atm:,.0f}")
    m[3].metric("Strikes compared", f"{len(pairs)} per side")
    m[4].metric("Put advantage", f"{adv:+.1f}%")

    s = st.columns(3)
    s[0].metric("Σ CE |θ|/|Δ|", f"{sc:,.3f}")
    s[1].metric("Σ PE |θ|/|Δ|", f"{sp:,.3f}")
    s[2].metric("Put ÷ Call", f"{(sp / sc if sc else float('nan')):.2f}×")

    if not np.isnan(adv):
        if abs(adv) < 5:
            st.info("Gap under 5% — inside delta-rounding noise, treat as no clear difference.")
        else:
            side = "Puts" if adv > 0 else "Calls"
            st.success(f"{side} pay more decay per unit of delta by {abs(adv):.1f}%.")

    # ---- paired table
    st.subheader("Strike-by-strike — k-th OTM call vs k-th OTM put")
    st.dataframe(
        pairs.style.format({
            "Call Strike": "{:,.0f}", "Put Strike": "{:,.0f}",
            "Call LTP": "{:,.2f}", "Put LTP": "{:,.2f}",
            "Call Δ": "{:.2f}", "Put Δ": "{:.2f}",
            "Call |θ|/|Δ|": "{:,.3f}", "Put |θ|/|Δ|": "{:,.3f}",
            "Put advantage %": "{:+.1f}%",
        }),
        hide_index=True, use_container_width=True,
    )

    # ---- full chain in proper layout
    st.subheader("Full chain — |θ| / |Δ| by strike")
    st.caption(
        f"Calls summed: {', '.join(f'{k:,.0f}' for k in ce.Strike)}  |  "
        f"Puts summed: {', '.join(f'{k:,.0f}' for k in pe.Strike[::-1])}"
    )
    st.dataframe(
        style_chain(df, atm, set(ce.Strike), set(pe.Strike)),
        hide_index=True, use_container_width=True,
    )

    # ---- notes
    with st.expander("Formulas and caveats", expanded=False):
        st.markdown(
            r"""
- **Forward** F = K + C − P (put-call parity, nearest strikes, median). **ATM** = strike nearest F.
- **|θ| / |Δ|** = |daily time decay| ÷ |delta| — decay earned per unit of directional exposure.
- Calls summed = first N strikes above ATM; puts = first N below. Strike *k* above is paired with strike *k* below.
- **Put advantage** = Σ PE ratio ÷ Σ CE ratio − 1. This is a **skew** measure — what the market charges per unit of delta — not a profitability signal.
- The ratio rises as you go further OTM, so use it to compare **sides at a matched distance**, not to pick how far out to sell. It ignores gamma and tail risk.
- Delta is rounded to 2 decimals by the feed, so gaps under ~5% are noise.
            """
        )


if __name__ == "__main__":
    main()