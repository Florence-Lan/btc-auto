# Expanded information in the simulation trial — 2026-10-02

The previous execution supervisor only ran the September 17 four-factor macro
candidate. The six-group research collector and its news/calendar refresh loop
were not running. Its last archived snapshot was September 28 and included a
Binance 418 error. Connecting research downloaders had not activated the inputs
in the execution account.

The user requested activation and acquisition of the missing inputs. The terminal
now selects `config/multifactor_trial_20261002.json` through
`config/active_simulation_candidate.json`. This is a prospective **simulation**,
not validated evidence of profitability or authorization for exchange orders.

## Inputs used in entry and risk decisions

The existing frozen BTC entry/exit engine is unchanged. Six groups are enabled:
BTC daily/weekly momentum, derivatives positioning, Fed policy, Treasury yields,
FX, and global risk. FRED rates, Fed assets, yields, FX, stocks and oil are refreshed
hourly; their underlying observations retain their daily/weekly publication
frequency and conservative availability delays. Binance BTC and derivatives
observations are refreshed hourly. These are entry-time direction/risk overlays,
not independent trade generators or dynamically rebalanced macro exits.

Fed/ECB/UN news, FOMC/economic calendars and named Fed-funds futures expectations
refresh every five minutes. News remains a headline risk rule. Source failures
and stale public context block new entries under the existing coverage policy.
Collected rows retain first receipt times; historical observations are not
treated as having been available before receipt. Reconstructed FRED history can
contain revisions and is not a historical point-in-time out-of-sample test.

The refresh worker runs independently of the execution scheduler so network
downloads do not block account drawdown monitoring. The Binance collector now
shares the terminal's persisted 418/429 cooldown and uses incremental overlap
instead of downloading six months again on every refresh.

## Newly collected observations

| Input | Source | Refresh | Qualification |
| --- | --- | --- | --- |
| BTC IV, DVOL and option skew | [Deribit public API](https://docs.deribit.com/) | 5 min | Nearest 30-day expiry and approximate 25-delta selection; not an interpolated 30-day/25-delta index |
| BTCUSDT book spread/depth | [Binance public depth](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data) | 5 min | Top 100 levels sampled by REST; not a complete streaming orderbook |
| Exchange inflows/outflows and net inflow | [Coin Metrics Community](https://docs.coinmetrics.io/api/v4/) | 1 hour | Daily provider-attributed exchange address flows; `flash` values may be revised |
| US spot BTC ETF net flows | [Farside Investors](https://farside.co.uk/bitcoin-etf-flow-all-data/) | 6 hours | Dollar units; missing fund cells stay missing, partial totals remain labeled partial |
| Published economic forecasts and actuals | [Forex Factory](https://www.forexfactory.com/calendar) | 1 hour | Public USD calendar records; publisher forecasts, not an independently verified consensus survey |

All five sources were successfully fetched on October 2. Initial Farside probes
returned 403, but the public-data client subsequently retrieved its table.
Trading Economics guest access returned 410. A licensed
`TRADING_ECONOMICS_API_KEY` is optional; without it Forex Factory supplies the
public forecast/actual archive. No purchase or new paid subscription was made.

These five supplemental inputs are **observation-only**: visible and continuously
archived, with no unvalidated directional weights added to the trading engine.
Data collection does not establish profitable predictive value. The economic
surprise calculator requires a forecast archived before release and uses the
first actual received after release; it does not infer past surprises from a
calendar fetched after the event. The October 2 US jobs forecast was archived
before the scheduled release. Its actual was not yet available at that receipt.

The terminal shows factor availability, public source health and supplemental
source freshness/failures. Collection success and data freshness are separate.
Snapshots preserve earlier receipts on refresh failures and expose the failure.

## Account and verification

Activation pauses the previous supervisor and preserves its files in
`data/runtime/activation_backup_20261002/`. Existing simulation equity, account
inception, fees, funding, fills and drawdown latch are retained. The account was
flat with 1,000 USDT and zero fills at activation. The new strategy starts its own
forward state, preventing the previous strategy's observations being relabeled.
The account risk limits remain 8% soft reduction, 15% hard stop, and 2x leverage.
Exchange orders remain disabled.

Validation: 169 tests passed, JavaScript syntax and whitespace checks passed.
Tests cover receipt-time lookahead, expiry, outages retaining archives, partial
ETF totals, signed exchange flows, crossed/stale depth, option expiry consistency,
the free forecast fallback, pre-release forecast matching with actual revisions,
shared Binance cooldown, matching terminal/supervisor profiles, and execution
consuming snapshots without waiting for refresh downloads.

Run the collectors independently:

```sh
.venv/bin/python scripts/download_multifactor_snapshot.py
.venv/bin/python scripts/public_context.py
.venv/bin/python scripts/supplemental_market_data.py
```

The live local terminal remains `http://127.0.0.1:8766/terminal/`. Refreshing requires
the simulation supervisor to remain running; this change does not install a
system service that survives reboot.
