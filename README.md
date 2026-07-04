# BTC Auto

BTCUSDT futures strategy research and paper-trading tools. The default strategy remains
`trend`; experimental event modules and `timeseries_trend` are not promoted automatically.

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

Current tactical strategy:

```powershell
python scripts\simulate_range_swing.py --days 365 --strategy-modes trend --max-drawdown-stop-pct 0
```

Six-hour time-series trend:

```powershell
python scripts\simulate_range_swing.py --days 365 --strategy-modes timeseries_trend --max-drawdown-stop-pct 0
```

Virtual sleeves with a shared net leverage cap:

```powershell
python scripts\simulate_range_swing.py --days 365 `
  --strategy-modes trend,timeseries_trend --portfolio-mode sleeves `
  --risk-per-trade 0.02 --portfolio-leverage-cap 1.5 `
  --max-drawdown-stop-pct 0
```

Research runs disable the permanent drawdown halt and evaluate drawdown as an acceptance
metric. Paper trading keeps the 10% halt and requires `--resume-after-drawdown`.

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

## Shadow Mode

The time-series strategy has a separate shadow state and never places orders:

```powershell
python scripts\paper_trade_timeseries_trend.py --loop --poll-seconds 300
```

After a 10% drawdown halt, start a new shadow generation explicitly:

```powershell
python scripts\paper_trade_timeseries_trend.py --resume-after-drawdown
```

## Tests

```powershell
python -m unittest discover -s tests -v
```
