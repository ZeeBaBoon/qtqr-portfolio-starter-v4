"""
QT/QR — From-scratch, single-file research pipeline (monthly rebalance + realistic costs/margin)

You asked to:
- Use monthly rebalance (recommend date) and implement realism features:
  1) Trading costs charged on TRUE notional traded (SPY+TLT), including leverage changes + monthly rebalances
  2) Borrow costs charged on negative cash (financing)
  3) Path-dependent margin realism with maintenance-triggered forced deleveraging
  4) One-day execution lag (signals computed on t-1, trades executed at start of day t)
- Keep clean experiment structure:
  (A) Base EW (no leverage)
  (B) Headline: Vol-target only, NO costs, NO margin
  (C) Realism ladder on vol-target only + table vs base
  (D) MA CAP sweep across MA={200,300,400} × band={0..0.30} under realism ON
  (E) Best option (auto) under realism ON + plot/table compare Base vs Headline realism ON vs Best
  (F) Conclusions block in report

Monthly rebalance date recommendation:
- Rebalance on the FIRST trading day of each month (at the start of that day, using information up to prior close).
  This avoids lookahead and is a standard institutional convention.

Dependencies:
  pip install numpy pandas matplotlib yfinance tabulate

Run:
  python qtqr_pipeline.py

Outputs:
  outputs/scenarios_YYYY-MM-DD_HHMMSS/
    figures/
    tables/
    series/
    report.md
"""

from __future__ import annotations

import os
import math
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


def progress(i: int, n: int, label: str = "", every: int = 1, t0: float | None = None):
    if n <= 0:
        return
    if (i == 0) or ((i + 1) % every == 0) or (i + 1 == n):
        msg = f"{label} {i+1}/{n} ({(i+1)/n:.1%})"
        if t0 is not None and i > 0:
            elapsed = time.perf_counter() - t0
            rate = elapsed / (i + 1)
            eta = rate * (n - (i + 1))
            msg += f" | elapsed {elapsed:.1f}s | ETA {eta:.1f}s"
        logging.info(msg)

# =========================
# Data
# =========================
def download_prices_yf(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    t0 = time.perf_counter() 
    logging.info(f"Downloading prices from yfinance | tickers={tickers} | start={start} | end={end}")
    import yfinance as yf  # type: ignore
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    logging.info(f"yfinance raw shape: {df.shape}")

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            px = df["Close"].copy()
        else:
            field0 = df.columns.levels[0][0]
            px = df[field0].copy()
    else:
        px = df.copy()

    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])

    px = px.dropna(how="all").dropna()
    px = px[[c for c in tickers if c in px.columns]]

    dt = time.perf_counter() - t0  # NEW
    logging.info(f"Download complete | final px shape={px.shape} | took {dt:.2f}s")  # NEW
 
    return px


def compute_returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna()


# =========================
# Metrics
# =========================
def annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return float("nan")
    total = float(equity.iloc[-1] / equity.iloc[0])
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return float("nan")
    return total ** (1 / years) - 1


def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=0) * math.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    rf_daily = (1 + rf_annual) ** (1 / periods_per_year) - 1
    ex = returns - rf_daily
    vol = ex.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float((ex.mean() / vol) * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


# =========================
# Config
# =========================
@dataclass(frozen=True)
class Universe:
    tickers: Tuple[str, str] = ("SPY", "TLT")
    base_weights: Tuple[float, float] = (0.50, 0.50)


@dataclass(frozen=True)
class VolTarget:
    target_vol_annual: float = 0.10
    vol_lookback: int = 20
    max_leverage: float = 3.0
    min_leverage: float = 0.0
    band: float = 0.00  # do NOT guess early; sweep later


@dataclass(frozen=True)
class Costs:
    borrow_annual: float = 0.06
    trading_bps: float = 10.0
    periods_per_year: int = 252

    @property
    def borrow_daily(self) -> float:
        return (1 + self.borrow_annual) ** (1 / self.periods_per_year) - 1

    @property
    def trading_cost_per_notional(self) -> float:
        return self.trading_bps / 10000.0


@dataclass(frozen=True)
class Margin:
    enabled: bool = True
    m_init: float = 0.50
    m_maint: float = 0.30
    maint_buffer: float = 0.05


@dataclass(frozen=True)
class Regime:
    mode: str = "cap"
    ma_lookback: int = 300


# =========================
# Signals
# =========================
def compute_ma_gate(spy_px: pd.Series, lookback: int) -> Tuple[pd.Series, pd.Series]:
    ma = spy_px.rolling(lookback, min_periods=lookback).mean()
    gate = (spy_px >= ma).astype(int).fillna(0).astype(int)
    return gate, ma


def vol_target_leverage(port_ret: pd.Series, rules: VolTarget) -> pd.Series:
    vol = port_ret.rolling(rules.vol_lookback, min_periods=rules.vol_lookback).std(ddof=0)
    vol_annual = vol * math.sqrt(252)
    lev = rules.target_vol_annual / vol_annual
    lev = lev.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lev = lev.clip(lower=rules.min_leverage, upper=rules.max_leverage)
    return lev


def apply_regime_cap(lev_raw: pd.Series, gate: pd.Series) -> pd.Series:
    lev = lev_raw.copy()
    off = gate == 0
    lev.loc[off] = np.minimum(1.0, lev.loc[off])  # no borrowing in risk-off
    return lev


def apply_no_trade_band_series(target: pd.Series, band: float) -> Tuple[pd.Series, pd.Series]:
    """
    No-trade band on leverage target (series-level). We still do monthly rebalance for weights,
    but leverage scaling trades follow this banded series.
    """
    lev = pd.Series(np.nan, index=target.index)
    active = pd.Series(0, index=target.index, dtype=int)
    cur = float(target.dropna().iloc[0]) if target.notna().any() else 0.0
    for t in target.index:
        tgt = float(target.loc[t])
        if np.isnan(tgt):
            lev.loc[t] = cur
            continue
        if abs(tgt - cur) > band:
            cur = tgt
            active.loc[t] = 1
        lev.loc[t] = cur
    return lev, active


def first_trading_day_flags(index: pd.DatetimeIndex) -> pd.Series:
    """
    Monthly rebalance schedule: first trading day of each month.
    Trades execute at start of day t using signals up to t-1.
    """
    months = pd.Series(index.to_period("M"), index=index)
    is_first = months != months.shift(1)
    is_first.iloc[0] = True
    return is_first.astype(int)

def compute_ma_gate_hysteresis(spy_px: pd.Series, lookback: int, enter_buffer: float, exit_buffer: float) -> Tuple[pd.Series, pd.Series]:
    ma = spy_px.rolling(lookback, min_periods=lookback).mean()
    gate = pd.Series(0, index=spy_px.index, dtype=int)

    state = 0
    for t in spy_px.index:
        p = float(spy_px.loc[t])
        m = float(ma.loc[t]) if not pd.isna(ma.loc[t]) else float("nan")
        if not np.isfinite(m) or m <= 0:
            gate.loc[t] = 0
            state = 0
            continue

        enter = p > m * (1.0 + enter_buffer)
        exit_ = p < m * (1.0 - exit_buffer)

        if state == 0 and enter:
            state = 1
        elif state == 1 and exit_:
            state = 0

        gate.loc[t] = state

    return gate, ma

# =========================
# Simulator (realism)
# =========================
def simulate_account(
    rets: pd.DataFrame,
    tradable_cols: List[str],
    base_weights: np.ndarray,
    lev_target_series: pd.Series,
    monthly_rebalance: bool,
    costs: Costs,
    margin: Margin,
    charge_costs: bool,
    apply_margin: bool,
    label: str,
) -> Dict[str, object]:
    """
    Account model with:
    - One-day execution lag: lev_target_series is aligned to dates; lev applied on that day is computed from t-1 info.
      (Caller should build lev_target_series from signals; we internally shift by 1.)
    - Positions in SPY/TLT + cash (can be negative -> borrow).
    - Trading costs on actual notional traded.
    - Borrow costs on negative cash.
    - Monthly rebalance to base_weights (first trading day of month), plus daily leverage scaling trades when lev changes.
    - Path-dependent maintenance margin: after returns and after costs, if leverage breaches maintenance threshold,
      force delever to the max allowed and pay trading costs on that forced trade.

    Conventions:
    - At start of day t: we trade to target holdings using equity from end of day t-1.
    - Then we apply day t returns to holdings.
    - Then we charge borrow interest on negative cash (financing for that day).
    - Then maintenance check and forced delever if needed (end-of-day style).
    """
    idx = rets.index
    r = rets[tradable_cols].copy()
    t0 = time.perf_counter()
    n = len(idx)
    logging.info(f"[simulate_account START] {label} | days={n} | monthly_rebalance={monthly_rebalance} | charge_costs={charge_costs} | apply_margin={apply_margin}")

    # One-day lag on leverage target
    lev_exec = lev_target_series.shift(1).reindex(idx).fillna(0.0)

    # Margin caps (pre-trade initial margin) and maintenance caps (end-of-day)
    max_init_lev = (1.0 / margin.m_init) if (apply_margin and margin.enabled) else float("inf")
    max_maint_lev = (1.0 / (margin.m_maint + margin.maint_buffer)) if (apply_margin and margin.enabled) else float("inf")

    # Monthly rebalance flags
    reb_flag = first_trading_day_flags(idx) if monthly_rebalance else pd.Series(0, index=idx, dtype=int)

    # State
    equity = 1.0
    cash = 1.0
    holdings = pd.Series(0.0, index=tradable_cols)  # dollar holdings in each asset
    last_target_lev = 0.0

    # Logs
    eq_series = []
    ret_net_series = []
    ret_gross_series = []
    cash_series = []
    lev_series = []
    borrowed_series = []
    trading_cost_series = []
    borrow_cost_series = []
    turnover_notional_series = []
    forced_series = []
    band_active_series = []  # passed in via lev_target_series? we approximate by change in lev_exec

    prev_exec_lev = float(lev_exec.iloc[0]) if len(lev_exec) else 0.0

    for i, t in enumerate(idx):
        if (i == 0) or ((i + 1) % 252 == 0) or (i + 1 == n):  # logs ~ once per year, plus first/last
        elapsed = time.perf_counter() - t0
        rate = elapsed / max(i + 1, 1)
        eta = rate * (n - (i + 1))
        logging.info(f"[simulate_account] {label} {i+1}/{n} ({(i+1)/n:.1%}) | elapsed {elapsed:.1f}s | ETA {eta:.1f}s")

        # --- Start of day: trade to target ---
        lev_t = float(lev_exec.loc[t])
        lev_t = max(0.0, min(lev_t, max_init_lev))  # initial margin cap pre-trade

        # Determine whether leverage changed materially (for diagnostics)
        band_active = 1 if abs(lev_t - prev_exec_lev) > 1e-12 else 0
        prev_exec_lev = lev_t

        # Desired gross exposure = lev_t * equity
        target_gross = lev_t * equity

        # If monthly rebalance: reset weights to base_weights at target_gross
        # Else: keep drifted weights but scale whole sleeve to target_gross (proportional scaling)
        current_gross = float(holdings.abs().sum())
        if current_gross < 1e-12:
            # initialize holdings by base weights
            target_holdings = pd.Series(target_gross * base_weights, index=tradable_cols)
        else:
            if reb_flag.loc[t] == 1:
                target_holdings = pd.Series(target_gross * base_weights, index=tradable_cols)
            else:
                scale = (target_gross / current_gross) if current_gross > 0 else 0.0
                target_holdings = holdings * scale

        trades = target_holdings - holdings
        traded_notional = float(trades.abs().sum())

        trading_cost = costs.trading_cost_per_notional * traded_notional if charge_costs else 0.0

        # Execute trades: update cash and holdings (cost reduces cash/equity)
        cash -= float(trades.sum())  # buying uses cash, selling frees cash
        cash -= trading_cost
        holdings = target_holdings.copy()

        # --- Apply returns for the day ---
        day_ret_vec = r.loc[t]
        pnl = float((holdings * day_ret_vec).sum())
        holdings = holdings * (1.0 + day_ret_vec)

        # Gross return on equity before financing/costs (approx)
        ret_gross = pnl / equity if equity > 0 else 0.0

        # Update equity with PnL and trading cost already taken via cash
        equity = float(holdings.sum() + cash)

        # --- Borrow cost on negative cash (financing) ---
        borrowed = max(-cash, 0.0)
        borrow_cost = costs.borrow_daily * borrowed if (charge_costs and borrowed > 0) else 0.0
        cash -= borrow_cost
        equity = float(holdings.sum() + cash)

        # --- Maintenance margin check (path-dependent) ---
        forced = 0
        gross_exposure = float(holdings.abs().sum())
        lev_realized = (gross_exposure / equity) if equity > 1e-12 else float("inf")

        if apply_margin and margin.enabled and equity > 1e-12 and lev_realized > max_maint_lev + 1e-12:
            # Force delever to maintenance max: scale down holdings so gross_exposure = max_maint_lev * equity
            forced = 1
            target_gross2 = max_maint_lev * equity
            scale2 = target_gross2 / gross_exposure if gross_exposure > 0 else 0.0
            target_holdings2 = holdings * scale2
            trades2 = target_holdings2 - holdings
            traded_notional2 = float(trades2.abs().sum())
            trading_cost2 = costs.trading_cost_per_notional * traded_notional2 if charge_costs else 0.0

            cash -= float(trades2.sum())
            cash -= trading_cost2
            holdings = target_holdings2.copy()

            equity = float(holdings.sum() + cash)
            # (optionally) borrow cost again after forced trade; we skip to avoid double-charging within same day.

            trading_cost += trading_cost2
            traded_notional += traded_notional2

            gross_exposure = float(holdings.abs().sum())
            lev_realized = (gross_exposure / equity) if equity > 1e-12 else float("inf")

        # Net return on equity (post all costs already applied)
        ret_net = (equity / (eq_series[-1] if eq_series else 1.0) - 1.0) if eq_series else (equity - 1.0)

        # Log
        eq_series.append(equity)
        ret_net_series.append(ret_net)
        ret_gross_series.append(ret_gross)
        cash_series.append(cash)
        lev_series.append(lev_realized if np.isfinite(lev_realized) else max_maint_lev)
        borrowed_series.append(borrowed)
        trading_cost_series.append(trading_cost)
        borrow_cost_series.append(borrow_cost)
        turnover_notional_series.append(traded_notional)
        forced_series.append(forced)
        band_active_series.append(band_active)

        last_target_lev = lev_t

    eq = pd.Series(eq_series, index=idx, name="equity")
    ret_net = pd.Series(ret_net_series, index=idx, name="ret_net")
    ret_gross = pd.Series(ret_gross_series, index=idx, name="ret_gross")

    out = {
        "label": label,
        "equity": eq,
        "returns_net": ret_net,
        "returns_gross": ret_gross,
        "cash": pd.Series(cash_series, index=idx, name="cash"),
        "lev_realized": pd.Series(lev_series, index=idx, name="lev_realized"),
        "borrowed": pd.Series(borrowed_series, index=idx, name="borrowed"),
        "trading_cost": pd.Series(trading_cost_series, index=idx, name="trading_cost"),
        "borrow_cost": pd.Series(borrow_cost_series, index=idx, name="borrow_cost"),
        "traded_notional": pd.Series(turnover_notional_series, index=idx, name="traded_notional"),
        "forced": pd.Series(forced_series, index=idx, name="forced"),
        "band_active": pd.Series(band_active_series, index=idx, name="band_active"),
        "rebalance_flag": reb_flag,
        "lev_exec": lev_exec,
    }

    logging.info(f"[simulate_account DONE ] {label} | took {time.perf_counter() - t0:.2f}s")

    return out


def summarize_sim(sim: Dict[str, object]) -> Dict[str, float]:
    eq = sim["equity"]  # type: ignore
    rn = sim["returns_net"]  # type: ignore
    forced = sim["forced"]  # type: ignore
    lev = sim["lev_realized"]  # type: ignore
    borrowed = sim["borrowed"]  # type: ignore
    traded = sim["traded_notional"]  # type: ignore
    band_active = sim["band_active"]  # type: ignore

    # Convert traded notional to "turnover-ish" by scaling to equity (approx). We use mean(traded/equity_prev).
    eq_prev = eq.shift(1).fillna(eq.iloc[0])
    turn_proxy = (traded / eq_prev).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return {
        "AnnReturn": annualized_return(eq),
        "AnnVol": annualized_vol(rn),
        "Sharpe": sharpe_ratio(rn),
        "MaxDD": max_drawdown(eq),
        "ForcedLiqDays": float(pd.Series(forced).sum()),
        "AvgLev": float(pd.Series(lev).replace([np.inf, -np.inf], np.nan).dropna().mean()),
        "AvgBorrowed": float(pd.Series(borrowed).mean()),
        "AvgTurnover": float(turn_proxy.mean()),
        "BandActive": float(pd.Series(band_active).mean()),
    }


# =========================
# Plot/report helpers
# =========================
def ensure_outdir(root: str = "outputs") -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = os.path.join(root, f"scenarios_{ts}")
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out, "series"), exist_ok=True)
    return out


def save_equity_plot(path: str, title: str, curves: List[Tuple[str, pd.Series]]):
    plt.figure(figsize=(12, 6))
    for name, eq in curves:
        plt.plot(eq.index, eq.values, label=name, linewidth=1.7)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_heatmap(path: str, title: str, df: pd.DataFrame, value_col: str, ma_values: List[int], bands: List[float]):
    pivot = df.pivot(index="band", columns="ma", values=value_col).reindex(index=bands, columns=ma_values)
    plt.figure(figsize=(9, 5.5))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label=value_col)
    plt.title(title)
    plt.xticks(range(len(ma_values)), [f"MA{m}" for m in ma_values])
    plt.yticks(range(len(bands)), [f"{b:.2f}" for b in bands])
    plt.xlabel("MA lookback")
    plt.ylabel("no_trade_band")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_scatter(path: str, title: str, df: pd.DataFrame, x: str, y: str, hue: str):
    plt.figure(figsize=(10, 6))
    for key, g in df.groupby(hue):
        plt.scatter(g[x], g[y], s=36, alpha=0.85, label=str(key))
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def md_image(rel_path: str, caption: str) -> str:
    return f"![{caption}]({rel_path})\n\n*{caption}*\n"


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(float(v)))
    try:
        from tabulate import tabulate  # type: ignore
        return tabulate(df2.reset_index(), headers="keys", tablefmt="github", showindex=False)
    except Exception:
        return df2.to_markdown()


# =========================
# Main pipeline (A–F)
# =========================
def main():
    # ---- Your parameters ----
    uni = Universe(tickers=("SPY", "TLT"), base_weights=(0.50, 0.50))
    vt_base = VolTarget(target_vol_annual=0.10, vol_lookback=20, max_leverage=3.0, min_leverage=0.0, band=0.00)
    realistic_costs = Costs(borrow_annual=0.06, trading_bps=10.0)
    realistic_margin = Margin(enabled=True, m_init=0.50, m_maint=0.30, maint_buffer=0.05)

    ma_list = [200, 300, 400]
    bands = [round(x, 2) for x in np.linspace(0.00, 0.30, 16)]
    tol_near_base = 0.005

    # Data window
    start = "2004-01-01"
    end = None

    # Output
    outdir = ensure_outdir()
    figdir = os.path.join(outdir, "figures")
    tdir = os.path.join(outdir, "tables")
    sdir = os.path.join(outdir, "series")

    tradable = list(uni.tickers)
    base_w = np.array(uni.base_weights, dtype=float)

    # Load data
    px = download_prices_yf(list(uni.tickers), start=start, end=end)
    rets = compute_returns(px)

    # Base EW equity (A)
    base_ret = (rets[tradable] * base_w).sum(axis=1)
    base_eq = (1.0 + base_ret).cumprod()
    base_metrics = {
        "AnnReturn": annualized_return(base_eq),
        "AnnVol": annualized_vol(base_ret),
        "Sharpe": sharpe_ratio(base_ret),
        "MaxDD": max_drawdown(base_eq),
    }
    save_equity_plot(os.path.join(figdir, "equity_base.png"), "Base EW (SPY/TLT 50/50) — no leverage", [("Base EW", base_eq)])

    # Signals: MA gates
    spy_px = px["SPY"].reindex(rets.index)
    gates: Dict[int, pd.Series] = {}
    mas: Dict[int, pd.Series] = {}
    for m in ma_list:
        g, ma = compute_ma_gate(spy_px, m)
        gates[m] = g.reindex(rets.index).fillna(0).astype(int)
        mas[m] = ma.reindex(rets.index)

    # Hysteresis gates (keyed only by MA lookback to avoid key/hash issues)
    enter_buf = 0.01  # 1% enter threshold above MA
    exit_buf  = 0.01  # 1% exit threshold below MA

    gates_hyst: Dict[int, pd.Series] = {}
    for m in ma_list:
        gh, _ = compute_ma_gate_hysteresis(spy_px, m, enter_buf, exit_buf)
        gates_hyst[m] = gh.reindex(rets.index).fillna(0).astype(int)
        

    # Build leverage targets (raw vol targeting, then optional regime cap, then band)
    def build_lev_target(
        band: float,
        regime: Optional[Regime],
        gate_mode: str = "simple",  # "simple" or "hyst"
    ) -> Tuple[pd.Series, pd.Series]:
        vt = VolTarget(
            target_vol_annual=vt_base.target_vol_annual,
            vol_lookback=vt_base.vol_lookback,
            max_leverage=vt_base.max_leverage,
            min_leverage=vt_base.min_leverage,
            band=band,
        )

        lev_raw = vol_target_leverage(base_ret, vt).reindex(rets.index).fillna(0.0)
        lev_adj = lev_raw.copy()

        if regime is not None:
            if gate_mode == "simple":
                gate = gates[regime.ma_lookback]
            elif gate_mode == "hyst":
                gate = gates_hyst[regime.ma_lookback]
            else:
                raise ValueError("gate_mode must be 'simple' or 'hyst'")

            lev_adj = apply_regime_cap(lev_raw, gate)

        lev_banded, band_active = apply_no_trade_band_series(lev_adj, band)
        return lev_banded, band_active


    # (B) Headline: vol-target only, NO costs, NO margin (monthly rebalance ON)
    lev_headline, _ = build_lev_target(band=0.00, regime=None)
    sim_headline = simulate_account(
        rets=rets,
        tradable_cols=tradable,
        base_weights=base_w,
        lev_target_series=lev_headline,
        monthly_rebalance=True,
        costs=Costs(borrow_annual=0.0, trading_bps=0.0),
        margin=Margin(enabled=False),
        charge_costs=False,
        apply_margin=False,
        label="Headline: Vol-target only (band=0.00) — NO COSTS, NO MARGIN",
    )
    met_headline = summarize_sim(sim_headline)
    save_equity_plot(
        os.path.join(figdir, "equity_headline_vs_base.png"),
        "Base EW vs Headline vol-target (NO COSTS, NO MARGIN)",
        [("Base EW", base_eq), (sim_headline["label"], sim_headline["equity"])],  # type: ignore
    )

    # (C) Realism ladder on vol-target only
    ladder_rows = []

    def run_ladder_case(name: str, borrow_annual: float, trading_bps: float, margin_on: bool) -> pd.Series:
        lev, _ = build_lev_target(band=0.00, regime=None)
        sim = simulate_account(
            rets=rets,
            tradable_cols=tradable,
            base_weights=base_w,
            lev_target_series=lev,
            monthly_rebalance=True,
            costs=Costs(borrow_annual=borrow_annual, trading_bps=trading_bps),
            margin=(realistic_margin if margin_on else Margin(enabled=False)),
            charge_costs=True,
            apply_margin=margin_on,
            label=name,
        )
        met = summarize_sim(sim)
        # daily average drags
        rg = pd.Series(sim["returns_gross"])  # type: ignore
        rn = pd.Series(sim["returns_net"])  # type: ignore
        tc = pd.Series(sim["trading_cost"])  # type: ignore
        bc = pd.Series(sim["borrow_cost"])  # type: ignore
        eq = pd.Series(sim["equity"])  # type: ignore
        eq_prev = eq.shift(1).fillna(eq.iloc[0])
        met["AvgGrossDaily"] = float(rg.mean())
        met["AvgNetDaily"] = float(rn.mean())
        met["AvgTradingCostDaily"] = float((tc / eq_prev).replace([np.inf, -np.inf], np.nan).fillna(0.0).mean())
        met["AvgBorrowCostDaily"] = float((bc / eq_prev).replace([np.inf, -np.inf], np.nan).fillna(0.0).mean())
        return pd.Series(met, name=name)

    ladder_rows.append(run_ladder_case("A) NO COSTS, NO MARGIN", 0.0, 0.0, margin_on=False))
    ladder_rows.append(run_ladder_case("B) Trading only, NO MARGIN", 0.0, realistic_costs.trading_bps, margin_on=False))
    ladder_rows.append(run_ladder_case("C) Borrow only, NO MARGIN", realistic_costs.borrow_annual, 0.0, margin_on=False))
    ladder_rows.append(run_ladder_case("D) Trading+Borrow, NO MARGIN", realistic_costs.borrow_annual, realistic_costs.trading_bps, margin_on=False))
    ladder_rows.append(run_ladder_case("E) Trading+Borrow, MARGIN ON", realistic_costs.borrow_annual, realistic_costs.trading_bps, margin_on=True))

    cost_df = pd.DataFrame(ladder_rows)
    cost_df["ΔAnnReturn_vs_A"] = cost_df["AnnReturn"] - float(cost_df.loc["A) NO COSTS, NO MARGIN", "AnnReturn"])
    cost_df.to_csv(os.path.join(tdir, "realism_ladder_vol_target_only.csv"))

    realism_vs_base = pd.DataFrame(
        [
            {"Variant": "Base EW", **base_metrics},
            {
                "Variant": "Headline realism ON (vol-target only, band=0.00)",
                "AnnReturn": float(cost_df.loc["E) Trading+Borrow, MARGIN ON", "AnnReturn"]),
                "AnnVol": float(cost_df.loc["E) Trading+Borrow, MARGIN ON", "AnnVol"]),
                "Sharpe": float(cost_df.loc["E) Trading+Borrow, MARGIN ON", "Sharpe"]),
                "MaxDD": float(cost_df.loc["E) Trading+Borrow, MARGIN ON", "MaxDD"]),
            },
        ]
    ).set_index("Variant")
    realism_vs_base.to_csv(os.path.join(tdir, "headline_realism_on_vs_base.csv"))

    # (D) MA CAP sweep across MA/band under realism ON
    sweep_rows = []
    for ma in ma_list:
        for band in bands:
            # --- Simple gate ---
            lev_s, _ = build_lev_target(band=band, regime=Regime(mode="cap", ma_lookback=ma), gate_mode="simple")
            sim_s = simulate_account(
                rets=rets,
                tradable_cols=tradable,
                base_weights=base_w,
                lev_target_series=lev_s,
                monthly_rebalance=True,
                costs=realistic_costs,
                margin=realistic_margin,
                charge_costs=True,
                apply_margin=True,
                label=f"MA{ma} CAP band={band:.2f} simple",
            )
            met_s = summarize_sim(sim_s)
            met_s.update({"ma": ma, "band": band, "gate_mode": "simple"})
            sweep_rows.append(met_s)

            # --- Hysteresis gate ---
            lev_h, _ = build_lev_target(band=band, regime=Regime(mode="cap", ma_lookback=ma), gate_mode="hyst")
            sim_h = simulate_account(
                rets=rets,
                tradable_cols=tradable,
                base_weights=base_w,
                lev_target_series=lev_h,
                monthly_rebalance=True,
                costs=realistic_costs,
                margin=realistic_margin,
                charge_costs=True,
                apply_margin=True,
                label=f"MA{ma} CAP band={band:.2f} hyst",
            )
            met_h = summarize_sim(sim_h)
            met_h.update({"ma": ma, "band": band, "gate_mode": "hyst"})
            sweep_rows.append(met_h)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(os.path.join(tdir, "ma_cap_band_sweep_realism_on.csv"), index=False)

    # --- Split datasets ---
    sweep_df_simple = sweep_df[sweep_df["gate_mode"] == "simple"].copy()
    sweep_df_hyst   = sweep_df[sweep_df["gate_mode"] == "hyst"].copy()

    # --- Heatmaps: SIMPLE ---
    save_heatmap(
        os.path.join(figdir, "heatmap_annreturn_simple.png"),
        "AnnReturn by MA and band (CAP, realism ON, SIMPLE gate)",
        sweep_df_simple,
        "AnnReturn",
        ma_values=ma_list,
        bands=bands,
    )
    save_heatmap(
        os.path.join(figdir, "heatmap_maxdd_simple.png"),
        "MaxDD by MA and band (CAP, realism ON, SIMPLE gate)",
        sweep_df_simple,
        "MaxDD",
        ma_values=ma_list,
        bands=bands,
    )

    # --- Heatmaps: HYST ---
    save_heatmap(
        os.path.join(figdir, "heatmap_annreturn_hyst.png"),
        f"AnnReturn by MA and band (CAP, realism ON, HYST enter={enter_buf:.2%}, exit={exit_buf:.2%})",
        sweep_df_hyst,
        "AnnReturn",
        ma_values=ma_list,
        bands=bands,
    )
    save_heatmap(
        os.path.join(figdir, "heatmap_maxdd_hyst.png"),
        f"MaxDD by MA and band (CAP, realism ON, HYST enter={enter_buf:.2%}, exit={exit_buf:.2%})",
        sweep_df_hyst,
        "MaxDD",
        ma_values=ma_list,
        bands=bands,
    )

    # --- Two clean scatters (not the 6-group one) ---
    save_scatter(
        os.path.join(figdir, "scatter_return_vs_dd_simple.png"),
        "Return vs MaxDD across MA and band (CAP, realism ON, SIMPLE gate)",
        sweep_df_simple.assign(MA=sweep_df_simple["ma"].map(lambda x: f"MA{x}")),
        x="MaxDD",
        y="AnnReturn",
        hue="MA",
    )
    save_scatter(
        os.path.join(figdir, "scatter_return_vs_dd_hyst.png"),
        f"Return vs MaxDD across MA and band (CAP, realism ON, HYST enter={enter_buf:.2%}, exit={exit_buf:.2%})",
        sweep_df_hyst.assign(MA=sweep_df_hyst["ma"].map(lambda x: f"MA{x}")),
        x="MaxDD",
        y="AnnReturn",
        hue="MA",
    )



    # Robustness summaries
    rob = sweep_df.copy()
    rob["NearBaseReturn"] = rob["AnnReturn"] >= (base_metrics["AnnReturn"] - tol_near_base)
    rob["BetterDDThanBase"] = rob["MaxDD"] > base_metrics["MaxDD"]
    rob["MeetsRobustness"] = rob["NearBaseReturn"] & rob["BetterDDThanBase"] & (rob["ForcedLiqDays"] == 0)

    rob_summary = (
        rob.groupby("ma")
        .agg(
            n_total=("band", "count"),
            n_meets=("MeetsRobustness", "sum"),
            annret_mean=("AnnReturn", "mean"),
            annret_std=("AnnReturn", "std"),
            annret_min=("AnnReturn", "min"),
            annret_max=("AnnReturn", "max"),
            maxdd_mean=("MaxDD", "mean"),
            maxdd_std=("MaxDD", "std"),
            maxdd_best=("MaxDD", "max"),
            maxdd_worst=("MaxDD", "min"),
        )
        .reset_index()
    )
    rob_summary.to_csv(os.path.join(tdir, "robustness_summary_by_ma.csv"), index=False)

    # Heatmaps + scatter
    sweep_plot = sweep_df.copy()
    sweep_plot["MA"] = sweep_plot["ma"].map(lambda x: f"MA{x}")

    # Heatmaps for SIMPLE gate
    sweep_df_simple = sweep_df[sweep_df["gate_mode"] == "simple"].copy()
    save_heatmap(
        os.path.join(figdir, "heatmap_annreturn_simple.png"),
        "AnnReturn by MA and band (CAP, realism ON, SIMPLE gate)",
        sweep_df_simple,
        "AnnReturn",
        ma_values=ma_list,
        bands=bands,
    )
    save_heatmap(
        os.path.join(figdir, "heatmap_maxdd_simple.png"),
        "MaxDD by MA and band (CAP, realism ON, SIMPLE gate)",
        sweep_df_simple,
        "MaxDD",
        ma_values=ma_list,
        bands=bands,
    )

    # Heatmaps for HYSTERESIS gate
    save_heatmap(
        os.path.join(figdir, "heatmap_annreturn_hyst.png"),
        f"AnnReturn by MA and band (CAP, realism ON, HYST gate enter={enter_buf:.2%}, exit={exit_buf:.2%})",
        sweep_df_hyst,
        "AnnReturn",
        ma_values=ma_list,
        bands=bands,
    )
    save_heatmap(
        os.path.join(figdir, "heatmap_maxdd_hyst.png"),
        f"MaxDD by MA and band (CAP, realism ON, HYST gate enter={enter_buf:.2%}, exit={exit_buf:.2%})",
        sweep_df_hyst,
        "MaxDD",
        ma_values=ma_list,
        bands=bands,
    )

    # Combined scatter: simple + hysteresis (distinguish by MA + gate_mode)
    sweep_scatter = sweep_df.copy()
    sweep_scatter["Group"] = sweep_scatter.apply(
        lambda r: f"MA{int(r['ma'])} {r['gate_mode']}",
        axis=1
    )

    save_scatter(
        os.path.join(figdir, "scatter_return_vs_dd_simple_and_hyst.png"),
        "Return vs MaxDD across MA and band (CAP, realism ON) — SIMPLE vs HYST",
        sweep_scatter,
        x="MaxDD",
        y="AnnReturn",
        hue="Group",
    ) 

    # (E) Best option auto (under realism ON)
    rob = sweep_df[sweep_df["gate_mode"] == "hyst"].copy()
    DD_LIMIT = -0.25

    cand = rob[(rob["ForcedLiqDays"] == 0) & (rob["MaxDD"] >= DD_LIMIT)].copy()

    if len(cand) == 0:
        # fallback: closest to constraint by MaxDD, then highest AnnReturn
        cand = (
            rob[rob["ForcedLiqDays"] == 0]
            .copy()
            .sort_values(["MaxDD", "AnnReturn"], ascending=[False, False])
            .head(1)
        )
        best_row = cand.iloc[0]
        best_reason = f"No config met MaxDD >= {DD_LIMIT:.2f}; selected closest MaxDD with highest AnnReturn."
    else:
        best_row = cand.sort_values(["AnnReturn", "MaxDD"], ascending=[False, False]).iloc[0]
        best_reason = f"Selected max AnnReturn subject to MaxDD >= {DD_LIMIT:.2f} and ForcedLiqDays=0."

    best_ma = int(best_row["ma"])
    best_band = float(best_row["band"])
    best_gate_mode = str(best_row.get("gate_mode", "simple"))


    lev_best, _ = build_lev_target(band=best_band, regime=Regime(mode="cap", ma_lookback=best_ma),gate_mode=best_gate_mode,)
    sim_best = simulate_account(
        rets=rets,
        tradable_cols=tradable,
        base_weights=base_w,
        lev_target_series=lev_best,
        monthly_rebalance=True,
        costs=realistic_costs,
        margin=realistic_margin,
        charge_costs=True,
        apply_margin=True,
        label=f"Best (auto): MA{best_ma} CAP band={best_band:.2f} {best_gate_mode} (realism ON)"
    )
    met_best = summarize_sim(sim_best)

    # Headline realism ON equity curve (vol-target only, realism ON)
    lev_hro, _ = build_lev_target(band=0.00, regime=None)
    sim_headline_realism_on = simulate_account(
        rets=rets,
        tradable_cols=tradable,
        base_weights=base_w,
        lev_target_series=lev_hro,
        monthly_rebalance=True,
        costs=realistic_costs,
        margin=realistic_margin,
        charge_costs=True,
        apply_margin=True,
        label="Headline realism ON: Vol-target only (band=0.00)",
    )
    met_headline_realism_on = summarize_sim(sim_headline_realism_on)

    # ============================================================
    # Borrow-rate sensitivity (AUTO-SELECTED BEST OPTION) vs Base EW
    # ============================================================
    borrow_grid = [0.02, 0.04, 0.06, 0.08]  # plausible annual borrow rates

    rows = []
    for br in borrow_grid:
        # Rebuild leverage target using the SAME best parameters
        lev_br, _ = build_lev_target(
            band=best_band,
            regime=Regime(mode="cap", ma_lookback=best_ma),
            gate_mode=best_gate_mode,
        )

        sim_br = simulate_account(
            rets=rets,
            tradable_cols=tradable,
            base_weights=base_w,
            lev_target_series=lev_br,
            monthly_rebalance=True,
            costs=Costs(borrow_annual=br, trading_bps=realistic_costs.trading_bps),
            margin=realistic_margin,
            charge_costs=True,
            apply_margin=True,
            label=f"Best: MA{best_ma} {best_gate_mode} band={best_band:.2f} borrow={br:.2%}",
        )
        met = summarize_sim(sim_br)

        rows.append({
            "BorrowAnnual": br,
            "AnnReturn": met["AnnReturn"],
            "AnnVol": met["AnnVol"],
            "Sharpe": met["Sharpe"],
            "MaxDD": met["MaxDD"],
            "ForcedLiqDays": met["ForcedLiqDays"],
            "AvgLev": met["AvgLev"],
            "AvgBorrowed": met["AvgBorrowed"],
            "AvgTurnover": met["AvgTurnover"],
            "ΔAnnReturn_vs_Base": met["AnnReturn"] - base_metrics["AnnReturn"],
            "ΔMaxDD_vs_Base": met["MaxDD"] - base_metrics["MaxDD"],
        })

    best_borrow_sens_df = pd.DataFrame(rows).sort_values("BorrowAnnual").set_index("BorrowAnnual")
    best_borrow_sens_df.to_csv(os.path.join(tdir, "borrow_rate_sensitivity_best_option_vs_base.csv"))

    save_equity_plot(
        os.path.join(figdir, "equity_final_compare.png"),
        "Final comparison: Base vs Headline (realism ON) vs Best MA CAP (realism ON)",
        [
            ("Base EW", base_eq),
            (sim_headline_realism_on["label"], sim_headline_realism_on["equity"]),  # type: ignore
            (sim_best["label"], sim_best["equity"]),  # type: ignore
        ],
    )

    
    compare_final = pd.DataFrame(
        [
            {"Variant": "Base EW (no leverage)", **base_metrics},
            {"Variant": "Headline realism ON: Vol-target only (band=0.00)", **{k: met_headline_realism_on[k] for k in ["AnnReturn", "AnnVol", "Sharpe", "MaxDD"]}},
            {"Variant": f"Best (auto): MA{best_ma} CAP (band={best_band:.2f}) realism ON", **{k: met_best[k] for k in ["AnnReturn", "AnnVol", "Sharpe", "MaxDD"]}},
        ]
    ).set_index("Variant")
    compare_final.to_csv(os.path.join(tdir, "final_comparison_base_headlineRealism_best.csv"))

    # Save key series
    pd.DataFrame(
        {
            "base_eq": base_eq,
            "headline_nocost_eq": sim_headline["equity"],  # type: ignore
            "headline_realism_on_eq": sim_headline_realism_on["equity"],  # type: ignore
            "best_eq": sim_best["equity"],  # type: ignore
            "best_lev_realized": sim_best["lev_realized"],  # type: ignore
            "best_borrowed": sim_best["borrowed"],  # type: ignore
            "best_trading_cost": sim_best["trading_cost"],  # type: ignore
            "best_borrow_cost": sim_best["borrow_cost"],  # type: ignore
            "spy_px": spy_px,
            f"ma{best_ma}": mas[best_ma],
            f"gate_ma{best_ma}": gates[best_ma],
        }
    ).to_csv(os.path.join(sdir, "key_series.csv"))

   # (F) Report — polished narrative
    report = []

    report.append("# QT/QR — Vol Targeting + MA CAP + Realistic Costs & Margin (Monthly Rebalance)\n")
    report.append(f"Run time: `{datetime.now().isoformat(timespec='seconds')}`  \n")
    report.append(f"Output folder: `{os.path.basename(outdir)}`\n")

    # ---------------- Executive summary ----------------
    report.append("## Executive summary\n")
    report.append(textwrap.dedent(f"""
    This project builds a **from-scratch backtesting pipeline** for a levered volatility-target overlay on a 2-asset portfolio and evaluates it under increasing realism:
    execution lag, real notional trading costs, financing costs via negative cash, and path-dependent maintenance margin deleveraging.

    The main result is a **trade-off frontier**: leverage overlays can reduce tail risk (MaxDD) via de-risking rules (MA CAP), but **net returns are sensitive to financing and turnover**, and margin constraints cap achievable leverage.

    This report provides:
    - A best-case benchmark (no costs, no margin) to show the “physics” of the signal.
    - A realism ladder to quantify how each friction compresses AnnReturn.
    - A sweep across **MA200/300/400** and **no-trade band 0.00–0.30**, comparing **SIMPLE vs HYST** regime gating.
    - An auto-selected “best” configuration and a borrow-rate sensitivity table.
    """).strip() + "\n")

    # ---------------- Design choices ----------------
    report.append("## Design choices & conventions\n")
    report.append(
        textwrap.dedent(
            f"""
            **Monthly rebalance date:** first trading day of each month (trades executed at start of day *t* using information through *t−1*).  
            **Execution lag:** leverage targets are shifted by one day to avoid lookahead.  
            **Trading costs:** charged on **actual traded notional** across SPY and TLT (rebalance + leverage scaling + forced delever).  
            **Financing (borrow) costs:** charged on **negative cash** (leveraged financing).  
            **Margin realism:** initial margin caps leverage pre-trade; maintenance margin triggers **path-dependent forced deleveraging**.
            """
        ).strip()
        + "\n"
    )

    # ---------------- Parameters ----------------
    report.append("## Parameters\n")
    report.append(
        textwrap.dedent(
            f"""
            **Universe:** {uni.tickers} with base weights {uni.base_weights}  
            **Vol-target:** target_vol={vt_base.target_vol_annual:.2f}, lookback={vt_base.vol_lookback}d, max_leverage={vt_base.max_leverage:.1f}  
            **Realistic costs:** borrow_annual={realistic_costs.borrow_annual:.2%}, trading_bps={realistic_costs.trading_bps:.1f}  
            **Margin:** m_init={realistic_margin.m_init:.2f}, m_maint={realistic_margin.m_maint:.2f}, buffer={realistic_margin.maint_buffer:.2f}  
            **MA sweep:** {ma_list}, band grid {bands[0]:.2f}..{bands[-1]:.2f}
            """
        ).strip()
        + "\n"
    )

    # ---------------- (A) Base ----------------
    report.append("## (A) Base benchmark (no leverage)\n")
    report.append(textwrap.dedent(f"""
    Base EW (equal-weighted in the sense of fixed weights {uni.base_weights}) is the reference portfolio.
    - AnnReturn: **{base_metrics['AnnReturn']:.4f}**
    - AnnVol: **{base_metrics['AnnVol']:.4f}**
    - Sharpe: **{base_metrics['Sharpe']:.4f}**
    - MaxDD: **{base_metrics['MaxDD']:.4f}**
    """).strip() + "\n\n")
    report.append(md_image("figures/equity_base.png", "Base EW equity curve"))

    # ---------------- (B) Best-case headline ----------------
    report.append("## (B) Best-case benchmark: vol-target only (no costs, no margin)\n")
    report.append(textwrap.dedent(f"""
    This is a **best-case upper bound** for the overlay: it ignores financing, trading frictions, and margin constraints.
    It is included to show what the signal could achieve *before* real-world implementation drag.

    - AnnReturn: **{met_headline['AnnReturn']:.4f}**
    - AnnVol: **{met_headline['AnnVol']:.4f}**
    - Sharpe: **{met_headline['Sharpe']:.4f}**
    - MaxDD: **{met_headline['MaxDD']:.4f}**
    """).strip() + "\n\n")
    report.append(md_image("figures/equity_headline_vs_base.png", "Base EW vs vol-target (NO costs, NO margin)"))

    # ---------------- (C) Realism ladder ----------------
    report.append("## (C) Realism ladder: where returns get compressed\n")
    report.append(textwrap.dedent("""
    We progressively add realism on the *same vol-target signal* to isolate drivers of performance drag:

    - **Trading drag:** scales with actual notional turnover (monthly rebalance + leverage scaling + forced trades).
    - **Financing drag:** scales with negative cash (borrowed funding).
    - **Margin caps & forced deleveraging:** cap achievable leverage and can lock-in losses during drawdowns.
    """).strip() + "\n\n")
    report.append(df_to_md(cost_df) + "\n\n")

    report.append("### Realism ON vs Base (quick read)\n\n")
    report.append(df_to_md(realism_vs_base) + "\n\n")

    # ---------------- (D) MA sweeps ----------------
    report.append("## (D) Regime overlays: MA CAP sweeps across MA and no-trade band\n")
    report.append(textwrap.dedent("""
    We evaluate a regime overlay that prevents borrowing in risk-off regimes (MA CAP).  
    We compare two gating rules:

    - **SIMPLE gate:** risk-on if SPY ≥ MA, risk-off otherwise.
    - **HYST gate:** uses entry/exit buffers around MA to reduce boundary churn (fewer flips).

    The band parameter controls how aggressively leverage target changes are executed:
    higher bands reduce turnover (lower trading costs) but can increase lag.
    """).strip() + "\n\n")

    report.append("### SIMPLE gate (realism ON)\n")
    report.append(md_image("figures/heatmap_annreturn_simple.png", "AnnReturn heatmap (SIMPLE gate, realism ON)"))
    report.append(md_image("figures/heatmap_maxdd_simple.png", "MaxDD heatmap (SIMPLE gate, realism ON)"))
    report.append(md_image("figures/scatter_return_vs_dd_simple.png", "Return vs MaxDD scatter (SIMPLE gate, realism ON)"))

    report.append("### Hysteresis gate (realism ON)\n")
    report.append(md_image("figures/heatmap_annreturn_hyst.png", "AnnReturn heatmap (HYST gate, realism ON)"))
    report.append(md_image("figures/heatmap_maxdd_hyst.png", "MaxDD heatmap (HYST gate, realism ON)"))
    report.append(md_image("figures/scatter_return_vs_dd_hyst.png", "Return vs MaxDD scatter (HYST gate, realism ON)"))

    report.append("### Robustness summary (stability across bands)\n")
    report.append(textwrap.dedent(f"""
    We label a configuration as “robust” if it simultaneously:
    - achieves near-base return: **AnnReturn ≥ BaseAnnReturn − {tol_near_base:.3f}**
    - improves tail risk: **MaxDD > BaseMaxDD**
    - avoids margin events: **ForcedLiqDays = 0**

    The table summarizes how many (band) choices satisfy that for each MA lookback.
    Higher counts imply the result is less parameter-fragile.
    """).strip() + "\n\n")
    report.append(df_to_md(rob_summary.set_index("ma")) + "\n\n")

    # ---------------- (E) Best option ----------------
    report.append("## (E) Auto-selected best configuration + final comparison\n")
    report.append(textwrap.dedent(f"""
    Selection note: {best_reason}

    Selected best (realism ON):
    - **MA{best_ma} CAP**
    - **gate_mode:** {best_gate_mode}
    - **band:** {best_band:.2f}

    Final comparison below uses **headline realism ON** (not the no-cost benchmark), since that is the relevant baseline for implementable performance.
    """).strip() + "\n\n")

    report.append(md_image("figures/equity_final_compare.png", "Base vs Headline (realism ON) vs Best MA CAP (realism ON)"))
    report.append(df_to_md(compare_final) + "\n\n")

    # ---------------- Borrow sensitivity ----------------
    report.append("## Borrow-rate sensitivity (best option vs Base EW)\n")
    report.append(textwrap.dedent(f"""
    We hold the auto-selected best configuration fixed and vary only the annual borrow rate.
    This isolates the dependence of net performance on financing conditions.

    Held fixed:
    - MA lookback: **{best_ma}**
    - gate_mode: **{best_gate_mode}**
    - band: **{best_band:.2f}**
    """).strip() + "\n\n")
    report.append(df_to_md(best_borrow_sens_df.reset_index()) + "\n\n")

    # ---------------- Conclusions ----------------
    report.append("## (F) Conclusions\n")
    report.append(textwrap.dedent("""
    1) **Best-case vs implementable reality:** the no-cost/no-margin benchmark can look strong, but it is not tradable as-is.  
    2) **Why returns compress under realism:** trading drag (turnover), financing drag (negative cash), and margin caps/forced deleveraging are first-order effects under leverage.  
    3) **What MA CAP is doing:** it improves tail behavior by removing borrowed exposure in sustained risk-off regimes, but this reduces participation during recoveries (a structural trade-off).  
    4) **Robustness matters:** the sweep + robustness counts show whether improvements persist across reasonable parameter settings rather than a single tuned point.  
    5) **Practical takeaway:** treat this as a *risk overlay research framework*. Any claim of “outperformance” must be evaluated under realistic financing and execution assumptions, and should be stress-tested across borrow/trading cost regimes.
    """).strip() + "\n")

    report_path = os.path.join(outdir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    # Console summary
    print("\n=== DONE ===")
    print(f"Output folder: {outdir}")
    print(f"Report: {report_path}")
    print("\nBase metrics:", {k: round(v, 4) for k, v in base_metrics.items()})
    print("Headline (no cost/no margin):", {k: round(met_headline[k], 4) for k in ["AnnReturn", "AnnVol", "Sharpe", "MaxDD"]})
    print("Headline realism ON:", {k: round(met_headline_realism_on[k], 4) for k in ["AnnReturn", "AnnVol", "Sharpe", "MaxDD"]})
    print(f"Best (auto) realism ON: MA{best_ma} band={best_band:.2f} ->", {k: round(met_best[k], 4) for k in ["AnnReturn", "AnnVol", "Sharpe", "MaxDD"]})
    print("Best reason:", best_reason)


if __name__ == "__main__":
    main()
