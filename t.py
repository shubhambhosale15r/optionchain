"""
Call vs Put decay per unit of delta (|theta| / |delta|) on the live Fyers option chain.

Layout:
    CE LTP | CE Theta | CE Δ | CE |θ|/|Δ| | Strike | PE |θ|/|Δ| | PE Δ | PE Theta | PE LTP

Comparison metric (paired OTM ratio, geometric mean):
    R_i     = (|θ|/|Δ|)_CE,i / (|θ|/|Δ|)_PE,i        i = 1..N  (i-th OTM call vs i-th OTM put)
    R_final = (R_1 · R_2 · ... · R_N)^(1/N)
    R_final > 1  -> CE has more theta per unit of delta
    R_final < 1  -> PE has more theta per unit of delta
    R_final ≈ 1  -> balanced

Auto-refresh:
    - align to the next minute boundary (10:00, 10:01, 10:02, ...)  OR
    - fixed interval in seconds
Last refreshed time is shown at the top.
"""
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from fyers_apiv3 import fyersModel

# ---- optional autorefresh package
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# Delta is rounded to 2 decimals by the feed, so gaps below this are treated as "balanced".
BALANCE_BAND = 0.05


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


def geometric_mean(values):
    """Geometric mean of the positive, finite values. NaN if there are none."""
    v = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    v = v[v > 0]
    return float(np.exp(np.log(v).mean())) if len(v) else np.nan


def band_compare(df, ref, n):
    """Pair the i-th OTM call above ATM with the i-th OTM put below ATM.

    Returns (atm, ce, pe, pairs, sum_ce, sum_pe, r_final) where
        R_i     = CE ratio_i / PE ratio_i
        R_final = geometric mean of R_1..R_N
    """
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
    pairs["R = CE ÷ PE"] = pairs["Call |θ|/|Δ|"] / pairs["Put |θ|/|Δ|"]

    sc = ce["CE |θ|/|Δ|"].sum()
    sp = pe["PE |θ|/|Δ|"].sum()
    r_final = geometric_mean(pairs["R = CE ÷ PE"])
    return atm, ce, pe, pairs, sc, sp, r_final


def interpret(r_final):
    """('success'|'info'|'warning', message) for R_final, with a noise band around 1."""
    if np.isnan(r_final):
        return "warning", "Not enough valid OTM pairs to compute R_final."
    if r_final > 1 + BALANCE_BAND:
        return "success", f"R_final = {r_final:.3f} (> 1): **CE** has more theta per unit of delta, by {(r_final - 1) * 100:.1f}%."
    if r_final < 1 / (1 + BALANCE_BAND):
        return "success", f"R_final = {r_final:.3f} (< 1): **PE** has more theta per unit of delta, by {(1 / r_final - 1) * 100:.1f}%."
    return "info", (f"R_final = {r_final:.3f} (≈ 1): balanced. Inside ±{BALANCE_BAND * 100:.0f}% "
                    "delta-rounding noise, so no clear difference.")


def style_chain(df, atm, ce_strikes, pe_strikes):
    """Green for rows entering the comparison; amber for ATM."""
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


# ---------------------------------------------------------------- auto-refresh
def ms_until_next_minute():
    """Milliseconds from now until the next minute boundary (e.g. 10:00 → 10:01)."""
    now = datetime.now()
    sec_into_min = now.second + now.microsecond / 1_000_000
    ms = int(round((60.0 - sec_into_min) * 1000))
    # guard against a tight loop right at the boundary
    if ms < 250:
        ms += 60_000
    return ms


def install_autorefresh(enabled, mode, fixed_seconds):
    """Schedule the next rerun. Returns the interval used (ms) or None."""
    if not enabled:
        return None
    if not HAS_AUTOREFRESH:
        return None

    if mode == "Align to minute boundary":
        interval_ms = ms_until_next_minute()
    else:
        interval_ms = max(int(fixed_seconds), 1) * 1000

    st_autorefresh(interval=interval_ms, key="auto_refresh_tick")
    return interval_ms


# ---------------------------------------------------------------- UI
def main():
    st.set_page_config(page_title="Theta / Delta: Calls vs Puts", layout="wide")
    st.title("Decay per unit of delta — |θ| / |Δ|")

    with st.sidebar:
        client_id = st.text_input("Client ID", placeholder="XXXXXXX-100")
        access_token = st.text_input("Access token", type="password")
        symbol = st.text_input("Underlying", "NSE:NIFTY50-INDEX")
        strikecount = st.number_input("Strikes fetched each side of ATM", 3, 50, 6)
        n = st.slider("Strikes compared each side (OTM)", 1, 15, 3)

        st.divider()
        st.subheader("Auto-refresh")
        refresh_enabled = st.checkbox("Enable auto-refresh", value=True)

        refresh_mode = st.radio(
            "Refresh mode",
            ["Align to minute boundary", "Fixed interval"],
            index=0,
            help="Minute-boundary mode reruns right on 10:00, 10:01, 10:02, ...",
        )

        fixed_seconds = 60
        if refresh_mode == "Fixed interval":
            fixed_seconds = st.number_input(
                "Refresh every (seconds)", 5, 3600, 60, 5,
            )

        if not HAS_AUTOREFRESH:
            st.warning("Install `streamlit-autorefresh` for auto-refresh:\n\n`pip install streamlit-autorefresh`")

    if not (client_id and access_token):
        st.info("Enter your Fyers Client ID and Access Token in the sidebar.")
        return

    # ---- schedule the next rerun BEFORE the fetch so the boundary is tight
    interval_ms = install_autorefresh(refresh_enabled, refresh_mode, fixed_seconds)

    # ---- fetch
    fetch_error = None
    try:
        resp = fetch_chain(client_id, access_token, symbol, strikecount)
        if resp.get("s") != "ok":
            fetch_error = f"API error: {resp}"
            resp = None
        else:
            exp = resp["data"].get("expiryData", [])
            if exp:
                labels = {e["date"]: e["expiry"] for e in exp}
                chosen = st.selectbox("Expiry", list(labels))
                if chosen != exp[0]["date"]:
                    resp = fetch_chain(client_id, access_token, symbol, strikecount, labels[chosen])
                    if resp.get("s") != "ok":
                        fetch_error = f"API error: {resp}"
                        resp = None

        if resp is not None:
            chain = resp["data"]["optionsChain"]
            spot = fetch_spot(client_id, access_token, symbol)
            fwd = synthetic_forward(chain, spot)
            st.session_state["last_refresh"] = datetime.now()
    except Exception as e:
        fetch_error = f"Failed: {e}"

    # ---- last refreshed line
    last_refresh = st.session_state.get("last_refresh")
    hdr = st.columns([3, 2, 2])
    if last_refresh is not None:
        hdr[0].caption(f"🕒 Last refreshed: **{last_refresh.strftime('%H:%M:%S')}**")
    else:
        hdr[0].caption("🕒 Last refreshed: —")

    if fetch_error:
        st.error(fetch_error)
        return

    hdr[1].caption(
        f"Auto-refresh: **{'ON' if refresh_enabled and HAS_AUTOREFRESH else 'OFF'}**"
        + (f" · every {interval_ms/1000:.0f}s" if interval_ms else "")
    )
    if refresh_enabled and refresh_mode == "Align to minute boundary":
        hdr[2].caption("Next tick: on the next minute boundary")

    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

    df = build_chain_table(chain)

    if df.empty:
        st.error("No greeks returned (theta/delta missing). Check greeks=1 availability.")
        return

    if df["CE Theta"].dropna().nunique() <= 1 and df["PE Theta"].dropna().nunique() <= 1:
        st.warning("Theta is constant across strikes — looks like placeholder data. Do not trust results.")

    # ---- compare
    atm, ce, pe, pairs, sc, sp, r_final = band_compare(df, fwd, int(n))

    # ---- headline metrics
    m = st.columns(5)
    m[0].metric("Spot", f"{spot:,.2f}")
    m[1].metric("Synthetic forward", f"{fwd:,.2f}", f"{fwd - spot:+,.2f} basis")
    m[2].metric("ATM strike", f"{atm:,.0f}")
    m[3].metric("Strikes compared", f"{len(pairs)} per side")
    m[4].metric("R_final  (CE ÷ PE)", f"{r_final:.3f}" if not np.isnan(r_final) else "–")

    s = st.columns(3)
    s[0].metric("Σ CE |θ|/|Δ|", f"{sc:,.3f}")
    s[1].metric("Σ PE |θ|/|Δ|", f"{sp:,.3f}")
    s[2].metric("1 ÷ R_final  (PE ÷ CE)", f"{1 / r_final:.3f}" if not np.isnan(r_final) else "–")

    kind, msg = interpret(r_final)
    getattr(st, kind)(msg)

    # ---- paired table
    st.subheader("Strike-by-strike — i-th OTM call vs i-th OTM put")
    st.dataframe(
        pairs.style.format({
            "Call Strike": "{:,.0f}", "Put Strike": "{:,.0f}",
            "Call LTP": "{:,.2f}", "Put LTP": "{:,.2f}",
            "Call Δ": "{:.2f}", "Put Δ": "{:.2f}",
            "Call |θ|/|Δ|": "{:,.3f}", "Put |θ|/|Δ|": "{:,.3f}",
            "R = CE ÷ PE": "{:.3f}",
        }),
        hide_index=True, use_container_width=True,
    )

    # ---- full chain
    st.subheader("Full chain — |θ| / |Δ| by strike")
    st.caption(
        f"Calls compared: {', '.join(f'{k:,.0f}' for k in ce.Strike)}  |  "
        f"Puts compared: {', '.join(f'{k:,.0f}' for k in pe.Strike[::-1])}"
    )
    st.dataframe(
        style_chain(df, atm, set(ce.Strike), set(pe.Strike)),
        hide_index=True, use_container_width=True,
    )

    # ---- notes
    with st.expander("Formulas and caveats", expanded=False):
        st.latex(r"R_i=\frac{(|\Theta|/|\Delta|)_{CE,i}}{(|\Theta|/|\Delta|)_{PE,i}}"
                 r"\qquad R_{final}=\Big(\prod_{i=1}^{N}R_i\Big)^{1/N}")
        st.markdown(
            r"""
- **Forward** F = K + C − P (put-call parity, nearest strikes, median). **ATM** = strike nearest F.
- **|θ| / |Δ|** = |daily time decay| ÷ |delta|, decay earned per unit of directional exposure.
- Calls = first N strikes above ATM; puts = first N below. The i-th call is paired with the i-th put, so each pair sits at about the same distance from ATM.
- **R_final > 1**: CE has more theta per unit of delta. **< 1**: PE has more. **≈ 1**: balanced (within ±5%).
- The geometric mean is the right average for ratios: swapping CE and PE just flips R_final to 1/R_final. It also weights every OTM step equally, whereas a plain sum is dominated by the strikes nearest ATM, which have the largest values.
- This is a **skew** measure, i.e. what the market charges per unit of delta. It is not a profitability signal, and it ignores gamma and tail risk.
- Delta is rounded to 2 decimals by the feed, so gaps under ~5% are noise.

**Auto-refresh**
- Minute-boundary mode computes milliseconds until the next minute (e.g. 10:00:37 → 23 s → rerun at 10:01:00) and re-arms each run, so ticks land on 10:00, 10:01, 10:02, ...
- Fixed-interval mode reruns every *N* seconds.
- `fetch_chain` is cached at `ttl=5s`, so each rerun hits the API, not the cache.
- Click **🔄 Refresh now** to force a manual rerun (clears cache first).
            """
        )


if __name__ == "__main__":
    main()
