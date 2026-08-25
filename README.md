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
```
### Run
```bash
py lev_vol_target.py
```

## Reusability: change tickers + weights (and the regime driver)

This pipeline is designed to run on any **2‑asset universe**.

To change inputs, edit the `Universe` config in `lev_vol_target.py`:
- `tickers`: a tuple of two tickers (example: `("QQQ", "TLT")`)
- `base_weights`: matching weights that sum to 1 (example: `(0.60, 0.40)`)

**Regime driver:** the script uses the **first ticker** as the MA regime signal (risk-on/risk-off).  
Examples:
- `("SPY", "TLT")` → SPY drives the regime gate  
- `("QQQ", "TLT")` → QQQ drives the regime gate  

When changing assets, consider adjusting `Costs(...)` and `Margin(...)` to reflect instrument-specific financing, liquidity, and broker rules.
