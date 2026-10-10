# BTC Auto

October 7 latest-strategy switch: the default four-account plan now selects
the hourly aligned SNDK candidate, with the latest BTC/MU/SKHYNIX rules retained.
Existing account migration preserves cash, fills and inventory. Startup with
`--require-existing` refuses missing ledgers; access to the original runtime is
still needed to complete the service switch. See [switch preparation and checks](docs/latest_four_account_switch_20261007.md).

October 7 continued research: 64 fixed mechanism/period/exit combinations and
three SNDK production-replay candidates yield an hourly Donchian/EMA60 paper
candidate. Its recent 30-day return is +1.1571% at normal costs and +1.0787%
at double costs; the continuous March–October replay also passes, with a thin
profit margin after removing the best winner. An isolated public-data comparison
runner has passed a one-shot market-data check. Original account defaults and
ledgers are preserved. See [rules, evidence and reproduction](docs/broad_mechanism_review_20261007.md).

October 7 strategy review: two fixed stock candidates now have reproducible
current-account replays with normal and double execution costs. Neither improves
the declared per-account return/drawdown criteria, so the four-account defaults
remain selected. Optional confirmed trend exits and directional efficiency/cost
entry checks are available for explicit paper research. Arithmetic tests no longer
inherit a user's enabled LLM gate. See [results and validation](docs/strategy_review_20261007.md).

October 6, 22:15 Beijing time: a separate six-ledger stock experiment now
compares two-closed-bar trend invalidation exits against the existing 1R/ATR
rules. Original accounts continue running. Each pair starts from matching
activation cash and inventory and shares public observations. Entries retain
the existing signals and execution risk checks, without historical profitability
admission gates. See [rules, activation and verification](docs/stock_trend_experiment_20261006.md).

October 6, 13:06 Beijing time: the joint stock paper accounts now take roughly
half off at 1R, then apply cost-aware break-even and 1.5 ATR protection. Original
exit-rule controls were cloned from the same activation balances and inventory
and share public market observations. Existing ledgers and pending signals were
preserved; prior price extrema do not create retroactive fills. See
[the rules, checks and activation](docs/stock_profit_exits_20261006.md).

October 5, 23:25 Beijing time: the user explicitly requested continued validation
in all four joint paper accounts without historical strategy admission gates.
BTC, MU, SNDK and SKHYNIX now permit new simulated entries under their existing
signals and execution risk controls. Existing balances and positions were
preserved. Earlier pause records below describe previous policy. See
[the activation and verification](docs/paper_validation_reenabled_20261005.md).

October 4 leverage update: the selected `btc_multifactor_regular_10x_20261004`
simulation generation uses a 10x ceiling across tactical/hourly sizing, portfolio
risk, execution and the terminal. Actual positions still follow the original risk
budget and volatility target. The existing account and earlier 2x reports are
preserved. See [the change and activation record](docs/leverage_ceiling_20261004.md).

October 4 data resilience update: public exchange reads can recover from a failed
TLS connection through a second verified transport. The supervisor uses an
incremental cache of complete candles and funding coverage, records strategy
judgments separately from execution, and retries a current report after an
execution-data failure. Missing current inputs are visible as unavailable.
See [implementation and activation evidence](docs/data_resilience_20261004.md).

October 4 simulation update: the `btc_multifactor_regular_20261004`
generation disables the one-time hourly trend startup. The previously opened
0.002 BTC simulated position was closed at the user's request, with normal fees
and accounting. The supervisor continues evaluating each newly closed five-minute
candle; hourly entries use the original threshold-transition signals. The existing
simulated account and old paper reports are preserved. See
[the cancellation and activation record](docs/hourly_startup_disabled_20261004.md).

The simulation terminal, execution backtest and independent observation runner default to
the strategy selected in `config/active_simulation_candidate.json`. The current selection
uses six factor groups, news/calendars, policy expectations, account risk controls and
the corrected hourly execution model. Backtests use the same selected rules and recheck
entry permissions at every execution timestamp. See
[the current strategy backtest](docs/latest_strategy_backtest_20261003.md).

October 3 reliability update: failed factor sources now retry independently with
bounded backoff and the shared exchange cooldown. Account hard-stop monitoring
continues after scheduler-clock failures when a fresh timestamped mark is available,
and records outages explicitly. The terminal exposes monitoring health. Run
`.venv/bin/python scripts/report_forward_progress.py` to capture a read-only forward
progress report with ledger checks. Fixed hourly protective-exit comparisons did
not qualify for replacement. See [details and activation evidence](docs/data_risk_reliability_20261003.md)
and [the exit comparison](docs/hourly_exit_comparison_20261003.md).

October 3 execution update: entries and additions now recheck current factor permissions,
event windows and source health immediately before execution, including retries of an old
target. Reductions and exits remain available when entries are blocked. Hourly targets use
closed signals plus the known next opening price, without unfinished hourly OHLC or volume.
The frozen signal parameters and drawdown thresholds remain unchanged. See
[repair details and activation evidence](docs/execution_integrity_fixes_20261003.md).
Run `.venv/bin/python scripts/validate_execution_integrity.py` for the local historical
cutoff regression. Previous return comparisons retain their recorded execution model.

October 1 execution update: account drawdown now constrains the executed target independently
of the shadow ledger, with an 8% soft reduction and a latched 15% flattening stop. Simulation
settles actual historical funding against its recorded inventory, archives all fills, and
blocks additions when funding data is unavailable. Legacy accounts start funding tracking at
upgrade; prior fees are not invented. Macro entry decisions are pinned on first observation.
See [implementation, execution replay and limitations](docs/execution_fixes_20261001.md).
Run `.venv/bin/python scripts/validate_execution_model.py` for comparisons using the terminal's
simulated account, including fees, funding and exchange quantity constraints. The frozen signal
parameters remain unchanged; the protected-exit alternative did not pass the comparison.

BTCUSDT futures strategy research and paper-trading tools. The selected simulation combines
the tactical `trend` strategy with a one-hour `timeseries_trend` sleeve (EMA 48/240,
0.40% spread threshold, 11% target annualized volatility). Six groups cover BTC momentum,
positioning, Fed policy, Treasury yields, FX and global risk. Automatic Fed/ECB/UN news,
official economic calendars and named-contract Fed-funds expectations provide additional
risk context. See [sources and limitations](docs/information_activation_20261002.md).
`python scripts/run_multifactor_shadow.py --once` runs the selected rules in independent
`*_shadow_*` files; omit `--once` to keep observing. It preserves the terminal's active state.

## Memory-stock swing research

The user has requested an overnight test of all three stocks. A new frozen
experiment gives MU the hourly trend-pullback arm, SNDK the hourly trend/volume
breakout, and SKHYNIX the compression-breakout arm, all with NQ confirmation
and a60% net initial-margin target. These are unvalidated hypotheses: MU and
SKHYNIX have weak or negative later-window results. Three1000USDT normal-cost
accounts each have a separate double-cost control. The original ledgers stay
under their existing protective rules. The new runner stops entries at
2026-10-06 10:00 Beijing, exits simulated inventory, and writes an automatic
report. See [the overnight protocol](docs/stock_overnight_validation_20261005.md)
and `http://127.0.0.1:8766/terminal/stock-overnight.html`.

A further fixed revision tested tighter stops, hourly trend pullbacks and
compression breakouts with NQ confirmation. None passes the existing expectancy
screen. SNDK pullbacks gain 15.21 USDT over the full double-cost path but lose
11.71 USDT under monthly admission using only earlier data. The original SNDK
trend/volume/NQ net60 rule is therefore frozen in a **separate** two-account
forward experiment, leaving the existing four-account admission unchanged.
See [the rules, results and forward protocol](docs/stock_profit_candidate_20261005.md).
Run `python scripts/run_stock_expectancy_shadow.py --once` for one observation,
or omit `--once` to keep running. Source/config hashes prevent silent changes;
new signals must close after activation. The normal/double-cost views are at
`http://127.0.0.1:8766/terminal/stock-shadow.html`.

The user has waived the 70% winning-trade requirement. Admission review now
uses net expectancy, sample size, costs and risk, retaining the 60% initial-margin
profit-target requirement. SNDK trend/volume plus NQ has a historical mean net
profit of 0.653 USDT/trade at double costs over 15 closed trades; future positive
expectancy is not established. See [the expectancy review](docs/stock_expectancy_20261005.md).

Stock research now includes timestamped NQ/ES futures, SOX, VIX, dollar and
10-year yield-index context. The fixed 216-scenario comparison reproduces the
prior price controls and checks external admission at every entry retry.
NQ confirmation improves one SNDK rule's historical win rate to 53.3% over
15 closed trades, but no declared stock/rule passes the outcome and sample
requirements. Read [the data assumptions and results](docs/stock_external_context_20261005.md).

The declared selective-entry replay now tests 1h breakouts, completed 4h trend
and volume confirmation, and subsequent retest confirmation, all with a 60%
net initial-margin profit target. Across 72 scenarios no stock/rule passes the
joint outcome and sample requirements. The paper admission configuration stays
paused. See [the actual verification results](docs/stock_selective_entries_20261005.md).

The user's earlier admission requirements were a net profit target of at least
60% of each trade's initial margin and a net winning-trade rate of at least 70%.
No previously declared candidate meets both development thresholds. All three
stock accounts now pause new simulated entries, while protecting existing
holdings and preserving their ledgers. These objectives are not guaranteed
outcomes. See [the requirements review and activation](docs/stock_user_requirements_20261005.md).

The October 5 execution refinement compared five declared activity/exit changes
per stock against the current bidirectional control. None met the stronger
development replacement gate; rejected candidates were not promoted using later
data. The active paper runner now caps new-entry bid/ask spreads at 10bp of
midpoint, while preserving admission decisions, existing inventory and exits.
This forward-only execution guard has no verified profitability effect.
See [the comparison and activation](docs/stock_execution_refinement_20261005.md).

The requested joint BTC/MU/SNDK/SKHYNIX simulation uses four independent
1,000-USDT accounts and a 10x ceiling. Its [launch plan](docs/parallel_simulation_20261005.md)
started a new four-account forward trial on October 5 at 04:42:03 UTC.
The three stock accounts are recording public Aster observations; BTC waits for
the shared Binance public-data cooldown to expire before evaluating new entries.
Open `/terminal/` and choose BTC or stocks for per-account health and equity.
The stock signal cycle was changed to completed 15-minute bars at 07:58:07 UTC
on October 5, with 30-second entry retries until the next signal close and a
three-bar (45-minute) exit cooldown. Ledgers and prior observations were preserved;
the original four-hour research results do not validate this revision. See
[the rule change and checks](docs/stock_signal_15m_activation_20261005.md).
Following the user's explicit direction correction, all three stocks allow both
long and short entries: MU uses a 15m EMA transition, SNDK a 5m breakout and
SKHYNIX a 15m EMA transition. Execution volume refreshes every 30 seconds so
delayed revisions of closed 5m bars can be observed during signal validity.
BTC retains its own multifactor profile. See [the fix](docs/stock_entry_fix_20261005.md).
The subsequent profitability review applies a per-account admission gate:
SNDK keeps its 5m bidirectional breakout at 0.25% planned trade risk and the
documented current RWA fee assumption. MU, SKHYNIX and the joint trial's BTC
pause new exposure after failed screening; observation and protective exits
continue, and existing balances/positions remain recorded. This is a limited
retrospective screen, not verified future profitability. See
[the comparison and activation](docs/strategy_screen_20261005.md).
The per-stock comparison ranked development data separately and rejected the
new SKHYNIX winner after a losing later audit. See [the choices, net-cost results
and activation evidence](docs/stock_independent_strategies_20261005.md).

The October 5 mechanism review compares eight declared rules across 288 replay
scenarios and audits 42 historical fill bars against public aggregate trades.
No variant passes the per-stock research screen; SKHYNIX long-only remains an
observation hypothesis. The terminal shows this review without changing BTC's
selected simulation. See [the mechanism findings](docs/stock_mechanism_review_20261005.md).

The October 4 follow-up replays separate stock profiles with uniform five-minute
execution and checks source gaps, costs and entry volume. A fixed prior-volume
sizing experiment reduces MU's modeled return and leaves SNDK unprofitable;
execution liquidity remains unverified. See [the follow-up findings and reproducible
research](docs/stock_swing_followup_20261004.md). The stock candidates remain research-only.

An isolated Aster/Binance Wallet candidate for `MUUSDT`, `SNDKUSDT` and
`SKHYNIXUSDT` uses completed four-hour trend breakouts, 10x isolated-margin
arithmetic, a net 120% initial-margin target and correlated account risk limits.
Public trade/mark/index/funding data, fixed-rule replay and cost/margin stress
results are recorded in [the strategy and validation report](docs/stock_swing_120_20261004.md).
The candidate remains research-only: recent target captures are absent and two
symbols have negative closed-trade contributions. It has no order-submission
path and does not change the selected BTC account or strategy.

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

Selected strategy, including execution costs, funding, current entry checks and hourly timing:

```sh
.venv/bin/python scripts/backtest_execution.py \
  --market-snapshot data/validation/current_backtest_20261003/market_extended.json.gz \
  --funding-snapshot data/validation/current_backtest_20261003/funding_extended.json \
  --start-utc 2026-09-27T02:15:00Z --end-utc 2026-10-03T03:10:00Z \
  --output data/validation/latest_strategy_backtest.json
```

The command captures the selected profile and source snapshots before replay. Missing
first-seen history remains missing; the command refuses a start before public-context
collection. Use `--cost-multiplier 2` for doubled execution fees/slippage.

Frozen price-engine research:

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
  --manifest config\frozen_strategy_active_20260809.json `
  --state-path data\paper_trading\macro_candidate_v3_state.json `
  --report-path data\paper_trading\macro_candidate_v3_report.json `
  --trades-path data\paper_trading\macro_candidate_v3_trades.csv `
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

An opt-in world-event research prototype now supports timestamped news discovery,
append-only evidence reviews, decaying risk scores, feed-health checks, and a
`--world-event-snapshot` shadow overlay. See [the workflow and limitations](docs/world_event_indicator.md).
It is disabled by default and does not modify live execution or existing shadow state.

The [market intelligence monitor](docs/market_intelligence.md) adds public spot/futures
flows, OI, long/short ratios, large trade samples, order books, Hyperliquid BTC
positions, and Bitcoin mempool observations. Run `python scripts/monitor_market_intelligence.py --loop`
and open `/terminal/intelligence.html`. An optional `--intelligence-db` shadow filter
reads only archived observations available at each entry; the monitor itself never trades.

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

Open `http://127.0.0.1:8766/terminal/` and choose one of the two account entries.
`/terminal/btc.html` shows the BTC forward account and the separate BTC execution
account with its existing controls. `/terminal/stocks.html` shows only MU/SNDK/SKHYNIX
accounts, their combined stock equity, current position details and the latest 50
simulated fills (price, quantity, fees and reason), refreshed every 10 seconds.
Exit-rule comparisons remain visible; rules and historical research are collapsed.
Trading data is read from the local runtime ledgers and is not included in Git.
Existing balances, positions and ledgers are preserved. `/terminal/parallel.html` now
offers the same two entries; old `#btc` and `#stocks` bookmarks open the corresponding
account page. The read-only
`/api/terminal/research` endpoint reads the artifacts selected by
`config/stock_research_dashboard.json`, with independent MU/SNDK/SKHYNIX capital,
normal/double cost comparisons, liquidity constraints, and forward observation status.
Missing results display an unavailable state. The research view does not start trading.
The selected historical summary, verification record, and prepared forward plan are
included in Git so a fresh checkout can display the research view. Raw market snapshots
and execution account state remain local.

BTC execution has two modes:

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

### Optional LLM entry gate

Copy the `LLM_*` settings from `.env.example` to `.env` to let an LLM review each distinct strategy
entry in real time. The default `LLM_PROVIDER=codex` runs the local Codex CLI in a temporary,
read-only working directory and reuses the existing `codex login` ChatGPT membership session; it
does not require an API key, expose the cached login to the bot, or persist the decision thread in
Codex history. `LLM_PROVIDER=openai` remains
available for API accounts. Keep `LLM_TRADE_GATE_ENABLED=false` until it has been exercised in
`SIMULATION` mode. The gate receives only point-in-time strategy, recent-price, drawdown, macro, and
event-risk features. Structured JSON output is required and the decision is cached by strategy
position ID.

The LLM is deliberately subordinate to deterministic execution controls: it cannot select a side,
increase the strategy's requested leverage, override notional caps, or block reductions and exits.
A rejection, timeout, invalid response, or API error blocks only the new/increased exposure. A
rejected reversal may still flatten the existing position. This makes the feature fail closed
without turning an unavailable model into an exit blocker.

Entry reviews distinguish the originating strategy entry, the report candle, and the current
decision cutoff. The report includes diagnostics from closed trend candles: the configured
timeframe, EMA periods and values, entry threshold, and the strategy's current holding intent.
Inputs observed after a candle opens but before the decision cutoff are valid current inputs.
For explicitly authorized forward simulation, the reviewer must identify a material current
conflict rather than veto solely on small countertrend returns, near-neutral factors, a small
historical sample, or the age of the originating signal. Model rejections remain authoritative;
the confidence requirement and deterministic execution controls still apply. Policy/context
changes invalidate older review caches. See [the review-context change](docs/btc_entry_review_context_20261011.md).

Use a Binance API key with Futures permission only, withdrawals disabled, and an IP restriction.
Never commit `.env` or expose the API secret in logs.

### 2026-09-27 accounting fix and exit research

The tiered candidate portfolio now reconstructs remaining inventory and cash from sleeve
bar-close observations before applying entry overlays. The frozen engine remains unchanged.
See [the accounting and time-separated validation report](docs/exit_validation_20260927.md).
New exit rules did not qualify for replacement; this research does not enable live orders.
Reproduce with `.venv/bin/python scripts/validate_exit_research.py` using the recorded snapshots.

### 2026-09-28 reentry candidate

An isolated research candidate adds protected exits, confirmed trend reentry, and stop-distance
risk sizing. It is available through `run_multifactor_shadow.py --research-profile
config/reentry_selected_20260928.json --once`. The 2025 holdout remains positive at normal
costs but fails double-cost stress, so the default strategy is unchanged. See the
[implementation and validation report](docs/reentry_validation_20260928.md).

### Reentry cost control

Fixed longer cooldown/breakout windows now have a separate cost-aware comparison:
`.venv/bin/python scripts/validate_cost_control.py --phase develop`.
No variant passed all preregistered development gates. The explicit exploratory command
`--phase diagnose --variant wait12_break24` evaluates the moderate slowdown without promoting it.
It reduced 2025 execution costs by 15%, but did not reduce costs in the recent window.
See the [results and limitations](docs/cost_control_20260928.md); defaults remain unchanged.

### 2026-10-02 simulation information activation

The user-selected simulation now uses all six existing factor groups with automatic
macro, positioning, news and calendar refreshes. Options, sampled orderbook depth,
exchange on-chain flows, ETF flows, and public forecast/actual records are also
collected and displayed as observations. The simulation account history is retained;
this forward trial is not evidence that the added data improves profitability.
See [sources, refresh intervals and validation](docs/information_activation_20261002.md).
