# Simulation data resilience — October 4, 2026

The user requested that data connection failures should not interrupt the
five-minute simulation judgment. A transport failure must also not appear as a
strategy conclusion that trading has no profit potential.

## Implemented behavior

- Allowlisted public exchange GETs retry TLS, connection and timeout failures
  using system curl against the same endpoint. Both transports verify TLS and
  respect the shared 418/429 cooldown. Signed requests retain their existing
  behavior. No credentials are sent through the public fallback.
- Candle history is fetched incrementally into `data/runtime/market_cache`.
  Only complete, contiguous bars through the decision cutoff enter the strategy.
  A forming candle can supply its immutable opening price. Cached funding must
  have verified coverage through the required cutoff. An incomplete cache cannot
  stand in for a new missing candle or unknown funding history.
- Simulation clock resolution tries exchange time, a fresh timestamped mark,
  and then a process-local monotonic exchange anchor valid for at most 60 seconds.
  The fallback does not extend its own lifetime. Execution validates a fresh mark
  separately. Live execution does not adopt the simulation clock fallbacks.
- A completed strategy judgment is recorded before execution. A temporary
  execution failure leaves the account signal cursor unchanged, records a
  deferral, and automatically retries the same current report. Reuse requires the
  same paper generation, factor profile and closed candle, and a valid cutoff.
  Each retry rechecks current execution permissions.
- The terminal displays strategy judgment and execution checks separately.
  Missing inputs and stale status are explicit. A poll heartbeat does not change
  the recorded time of the last strategy judgment.

This handles a failed connection path or a short interruption while complete
inputs remain available. If all paths fail before the latest required candle is
received, a current market judgment is unavailable. The system records that
condition and retries; it does not invent a flat signal or a profitability claim.

The selected `btc_multifactor_regular_20261004` generation still has hourly startup
disabled. The frozen signal parameters, factor weights and account risk thresholds
are unchanged. Deployment preserves the simulation account inception, inventory,
fills, fees, funding records and risk history.

## Verification evidence

Runtime evidence is saved under
`data/runtime/data_resilience_activation_20261004/` (local, gitignored):

- `real_transport_fault_injection.json`: an injected primary requests TLS error
  recovered through real curl reads of exchange time and timestamped mark.
- `validation_report.json`: the selected strategy completed a forward judgment
  using the original paper inception and the incremental cache.
- `offline_complete_cache_proof.json`: with all network calls disabled and a
  fully received cutoff, the isolated paper run made zero network requests and
  produced exactly the same execution target as the connected run.

Automated regressions cover transport recovery and rate limits, missing/future
candles, funding coverage, clock expiry, judgment preservation, cross-generation
report rejection, stale execution prices and account-history consistency.

Funding history refreshes overlap the previous day to recover delayed publication.
Execution keeps the actual successful query cutoff and the exchange's announced
next settlement. If a crossed settlement is still unpublished, a persisted pending
boundary blocks additions until the actual event arrives. A clock that stops at
an earlier mark timestamp cannot conceal elapsed time during execution checks.

## Activation

Activated at **2026-10-04 07:48:16 Asia/Shanghai**. All **443 tests passed**;
JavaScript syntax, diff whitespace and the four historical cutoff regressions
passed. The frozen strategy manifest and engine still verify.

The old supervisor was paused before the account and paper state were backed up.
The terminal was restarted, a simulation cycle completed successfully, and the
five-minute supervisor resumed. The first judgment and independent account risk
monitor both reported healthy. `activation.json`, `status_after.json` and
`verification.json` contain the process and preservation checks.

At 07:48:35 the strategy target was flat, source coverage was 100%, and current
entry gates were allowed. No new fill occurred. Wallet balance remained
999.7187959658874 USDT; the original two fills and the inventory/funding history
prefixes were preserved. Account inception remains October 1 at
15:29:05.740853 UTC; paper inception remains October 3 at 16:36:09.931691 UTC.

The public endpoint and shared exchange limits follow
[Binance USD-M general information](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/general-info).
