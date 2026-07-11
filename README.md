# BTC Auto

BTCUSDT futures strategy research and paper-trading tools. The default portfolio combines
the tactical `trend` strategy with a six-hour `timeseries_trend` sleeve. Experimental event
modules are not promoted automatically.

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

An optional higher-coverage profile adds the range module and reduces tactical risk to 0.75%.
It is frozen separately because it trades more often but had lower historical CAGR than the
default risk-controlled profile:

```powershell
python scripts\validate_frozen_strategy.py `
  --manifest config\frozen_strategy_active_20260711.json `
  --output-json data\validation\frozen_strategy_active_20260711.json `
  --output-csv data\validation\frozen_strategy_active_20260711_folds.csv
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
