# BTC Auto

BTCUSDT futures strategy research and paper-trading tools. The default portfolio combines
the tactical `trend` strategy with a six-hour `timeseries_trend` sleeve. Experimental event
modules are not promoted automatically.

The active v2 shadow profile requires a 0.30% fast/slow EMA spread before the six-hour
trend sleeve changes direction and targets 11% annualized volatility. This hysteresis reduces
small crossover reversals and remains shadow-only until prospective promotion gates pass.

## Reproducible Data

Freeze public Binance futures candles and funding rates:

```powershell
python scripts\download_market_snapshot.py `
  --start-utc 2020-01-01T00:00:00Z `
  --end-utc 2026-06-28T15:00:00Z `
  --output data\snapshots\btcusdt_20200101_20260628.json.gz
```

The command refuses to overwrite an existing snapshot unless `--force` is passed and prints
the snapshot SHA-256.

## Backtests

Current default portfolio:

```powershell
python scripts\simulate_range_swing.py --days 365 --max-drawdown-stop-pct 0
```

The default trend entry uses a near-touch limit at `0.05 ATR` from the signal close. Override it
with `--trend-entry-pullback-atr` when reproducing older runs.
The time-series sleeve uses `24/120` EMAs on six-hour candles.

Tactical trend only:

```powershell
python scripts\simulate_range_swing.py --days 365 `
  --strategy-modes trend --portfolio-mode single `
  --max-drawdown-stop-pct 0
```

Six-hour time-series trend only:

```powershell
python scripts\simulate_range_swing.py --days 365 `
  --strategy-modes timeseries_trend --portfolio-mode single `
  --max-drawdown-stop-pct 0
```

Research-only delta-neutral funding carry with synchronized spot/perpetual basis:

```powershell
python scripts\download_spot_snapshot.py `
  --start-utc 2020-01-01T00:00:00Z `
  --end-utc 2026-06-28T15:00:00Z `
  --output data\snapshots\btcusdt_spot_1h_20200101_20260628.json.gz

python scripts\backtest_funding_carry.py `
  --days 2000 `
  --notional-fraction 0.50 `
  --rebalance-margin-buffer-pct 40 `
  --spot-snapshot data\snapshots\btcusdt_spot_1h_20200101_20260628.json.gz `
  --double-cost `
  --output-json data\validation\funding_carry_basis_2000d_double_cost.json

python scripts\validate_carry_trend_portfolio.py `
  --carry-report data\validation\funding_carry_basis_2000d_double_cost.json `
  --carry-weight 0.85 `
  --double-cost `
  --output-json data\validation\carry_trend_portfolio_double_cost.json
```

The carry screen models equal-quantity long BTC spot and short BTC USD-M perpetual
positions, following the perpetual-futures arbitrage studied by He, Manela, Ross, and
von Wachter in *Fundamentals of Perpetual Futures* (SSRN 4301150). It marks the observed
spot/perpetual basis at every funding event, charges both-leg fees and slippage, applies
an adverse terminal-basis stress, and reports isolated futures-margin breaches. The
validated research allocation keeps 85% in a carry subaccount and 15% in the frozen
trend satellite; those balances must remain segregated for the margin model to hold.
Custody, transfers, tax, and order-book impact beyond the configured stress remain
unverified. Neither module is connected to paper or live order execution.

Research runs disable the permanent drawdown halt and evaluate drawdown as an acceptance
metric. Paper trading keeps the 12% halt and requires `--resume-after-drawdown`.

## Validation

Run the preregistered risk grid, quarterly walk-forward folds, block bootstrap, and doubled-cost
stress test:

```powershell
python scripts\validate_strategies.py `
  --data-snapshot data\snapshots\btcusdt_20200101_20260628.json.gz `
  --output-json data\validation\strategy_validation.json `
  --output-csv data\validation\strategy_validation.csv
```

Exit code `0` means every historical acceptance gate passed. A nonzero exit keeps the current
default and records the nearest diagnostic candidate without promoting it.

The risk-controlled strategy frozen on 2026-07-11 has a complete config hash and is validated
without refitting. It uses 1.5% tactical risk, a 12% volatility target, a 2x gross leverage cap,
and a 12% drawdown halt:

```powershell
python scripts\validate_frozen_strategy.py `
  --manifest config\frozen_strategy_20260711.json `
  --bootstrap-samples 2000
```

Historical rolling folds are explicitly reported as post-selection pseudo-OOS. Only observations
after the manifest freeze time are genuinely out of sample.

## Macro risk overlay

Freeze daily S&P 500 and Nasdaq futures, VIX, U.S. Dollar Index, gold futures, and silver
futures into a separate point-in-time snapshot. Daily closes are delayed by 24 hours before
they can affect a BTC decision, preventing same-day close lookahead:

```powershell
python scripts\download_macro_snapshot.py `
  --start-utc 2019-10-01T00:00:00Z `
  --end-utc 2026-06-28T15:00:00Z `
  --output data\snapshots\macro_20191001_20260628.json.gz
```

Run a continuous, non-resetting ablation against the active frozen portfolio:

```powershell
python scripts\validate_macro_overlay.py `
  --manifest config\frozen_strategy_active_20260720.json `
  --macro-snapshot data\snapshots\macro_20191001_20260628.json.gz `
  --output-json data\validation\macro_overlay_20260720.json
```

The macro overlay is research-only and never increases the original position size. The snapshot
also contains the Alternative.me Crypto Fear & Greed Index. A nonzero validation exit means the
candidate remains unpromoted.

Validate the no-range shadow candidate with continuous tiered drawdown control, block bootstrap,
and doubled execution costs:

```powershell
python scripts\validate_candidate_portfolio.py `
  --manifest config\frozen_strategy_active_20260720.json `
  --macro-snapshot data\snapshots\macro_20191001_20260628.json.gz `
  --output-json data\validation\candidate_portfolio_20260720.json

python scripts\verify_shadow_candidate.py
```

The frozen shadow profile starts reducing new position sizes after an 8% drawdown and permanently
blocks new entries at 15%. It does not resume automatically.

## Shadow Mode

The time-series strategy has a separate shadow state and never places orders:

```powershell
python scripts\paper_trade_timeseries_trend.py --loop --poll-seconds 300
```

After a 10% drawdown halt, start a new shadow generation explicitly:

```powershell
python scripts\paper_trade_timeseries_trend.py --resume-after-drawdown
```

Track the exact frozen portfolio prospectively without placing orders:

```powershell
python scripts\paper_trade_frozen_portfolio.py --loop --poll-seconds 300
```

Enable the research macro overlay in shadow mode explicitly:

```powershell
python scripts\download_macro_snapshot.py `
  --start-utc 2019-10-01T00:00:00Z `
  --output data\snapshots\macro_shadow_latest.json.gz `
  --force

python scripts\paper_trade_frozen_portfolio.py `
  --manifest config\frozen_strategy_active_20260720.json `
  --state-path data\paper_trading\macro_candidate_v2_state.json `
  --report-path data\paper_trading\macro_candidate_v2_report.json `
  --trades-path data\paper_trading\macro_candidate_v2_trades.csv `
  --strategy-modes-override trend,timeseries_trend `
  --macro-snapshot data\snapshots\macro_shadow_latest.json.gz `
  --macro-factors vix,dollar,metals,sentiment `
  --tiered-drawdown `
  --soft-drawdown-start-pct 8 `
  --hard-drawdown-stop-pct 15 `
  --drawdown-min-multiplier 0.35 `
  --event-snapshot config\event_risk_template.json `
  --loop --poll-seconds 300
```

Refresh the shadow macro snapshot at least once per trading day. If every factor is older than
five days, the overlay fails closed and blocks new entries.

For unattended shadow tracking, use the supervisor. It refreshes macro data every 12 hours and
recomputes the order-disabled portfolio every five minutes:

```powershell
python scripts\run_macro_candidate_shadow.py
```

`config/event_risk_template.json` is the point-in-time input for scheduled macro events and major
news. An event cannot affect a decision before `published_at_utc`; severity only reduces risk or
blocks entries and never creates a directional trade. Keep the template empty until a timestamped,
auditable event feed is available.

An optional higher-coverage profile adds the range module and reduces tactical risk to 0.75%.
It is frozen separately because it trades more often but had lower historical CAGR than the
default risk-controlled profile:

```powershell
python scripts\validate_frozen_strategy.py `
  --manifest config\frozen_strategy_active_20260720.json `
  --output-json data\validation\frozen_strategy_active_20260720.json `
  --output-csv data\validation\frozen_strategy_active_20260720_folds.csv
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Strategy dashboard

Explore the frozen strategy inputs, risk controls, historical returns, and prospective paper-trading status in a local dashboard:

```powershell
python -m http.server 8765
```

Then open `http://localhost:8765/dashboard/`. The dashboard reads the latest frozen manifests and validation reports directly from the repository, so rerunning validation updates the interface without copying metrics by hand.

## Automated trading terminal

Run the localhost-only trading terminal:

```powershell
python scripts\run_trading_terminal.py --port 8766
```

Open `http://127.0.0.1:8766/terminal/`. The terminal has two execution modes:

- `SIMULATION` is the default. It reads Binance USD-M mainnet market data and writes fills,
  positions, fees, and PnL only to a local simulated account. It never submits an exchange order.
  Simulated BTCUSDT quantities use the current Binance market-order step size, minimum quantity,
  and minimum notional, so sub-exchange-size target changes do not create artificial micro-fills.
  Set or reset the initial USDT balance directly in the terminal while automation is paused;
  resetting clears the local simulated positions, trades, and equity history.
- `LIVE` sends guarded BTCUSDT USD-M orders to Binance mainnet. It requires an exact UI
  confirmation plus explicit `.env` settings copied from `.env.example`.

The strategy target is scaled by account equity and capped by both `LIVE_MAX_NOTIONAL_USDT` and
`LIVE_LEVERAGE` (maximum 2). Repeated cycles use deterministic client order IDs. Live emergency
stop terminates automation, cancels BTCUSDT open orders, and sends a reduce-only market order to
flatten the BTCUSDT position. A normal pause only stops new strategy cycles and does not flatten.

The execution supervisor checks Binance server time every 30 seconds, but only recomputes and
reconciles the target once for each newly closed five-minute candle. It waits three seconds after
the candle boundary for settlement and stores the last processed signal time to prevent duplicate
execution after restarts. A newly initialized execution account does not enter a position whose
originating strategy signal predates that account; it waits for the next distinct position signal.

Use a Binance API key with Futures permission only, withdrawals disabled, and an IP restriction.
Never commit `.env` or expose the API secret in logs.
