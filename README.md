# QT/QR — Vol Targeting + MA Regime CAP (Realistic Costs & Margin)

This repo contains a **from-scratch research pipeline** for a levered volatility-target overlay on a 2-asset portfolio (default: SPY/TLT 50/50), with an emphasis on **implementation realism** rather than idealized backtests.

The script produces a full `report.md` with plots and tables, including:
- Base benchmark (no leverage)
- Best-case benchmark (no costs, no margin)
- A realism ladder (trading costs → borrow costs → margin constraints)
- MA regime CAP sweeps (MA200/MA300/MA400 × no-trade band 0.00–0.30)
- SIMPLE vs HYSTERESIS gating comparison
- Auto-selected “best” configuration under constraints
- Borrow-rate sensitivity analysis

## Why this project
Leverage strategies often look good until you model:
- **Execution lag** (no lookahead)
- **True notional turnover costs** (not just |Δleverage|)
- **Financing drag** via negative cash
- **Margin constraints** including path-dependent maintenance deleveraging

This repo is a compact template to test whether an overlay survives those frictions.

## How to run

### Install
```bash
pip install numpy pandas matplotlib yfinance tabulate

## Reusability: changing tickers, weights, and the regime “decision driver”

This pipeline is reusable for other **2‑asset universes**.

To change the universe, edit the `Universe` configuration in `qtqr_pipeline.py`:
- `tickers`: a tuple of two tickers, e.g. `("QQQ", "TLT")`
- `base_weights`: the corresponding fixed weights, e.g. `(0.60, 0.40)`

**Important:** the script treats the **first ticker** as the **regime decision driver** for the MA gate (risk-on/risk-off).  
Example:
- `tickers=("SPY","TLT")` → SPY drives the MA regime gate
- `tickers=("QQQ","TLT")` → QQQ drives the MA regime gate

If you change tickers, make sure the first ticker is the asset whose trend you want to define regime state. Financing rates, liquidity, and broker margin requirements can vary by instrument; update `Costs(...)` and `Margin(...)` accordingly when modeling other assets.
