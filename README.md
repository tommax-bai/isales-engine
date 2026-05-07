# isales-engine

Real-time call engine for the iSales platform. Drives a single call's state
machine, orchestrates the three-layer AI pipeline (N parallel role LLMs →
N×M parallel judge LLMs → 1 polish LLM), runs realtime modules
(filler / interruption / silence / transfer / wrap-up), persists call_record
+ pipeline_trace, and dispatches `CallEnded` to the worker.

Stage 4 of the iSales rollout (per `IMPLEMENTATION_PLAN.md`): mock providers
+ in-process `MockTelephonyClient`. Real LLM/ASR/TTS providers come in stage
5; real modem-controller IPC + USB GSM hardware in stage 6.

## Channels

| Direction | Redis key | Schema |
|---|---|---|
| consume | `engine:dial` | `DialRequest` |
| produce | `engine:worker:call-ended` | `CallEnded` |
| publish | `engine:events:campaign:{id}` | `EngineEvent` |
| subscribe | `engine:control:campaign:*` | `EngineControl` |
| dead-letter | `engine:dlq` | raw JSON |
| concurrency | `isales:concurrency:active` | `DECR` only (scheduler `INCR`s) |

## Run

```
isales-engine
```

For local development copy `.env.example` → `.env` and `source` it before
running. For production deploy as a systemd unit (`deploy/isales-engine.service`):

```
sudo cp deploy/isales-engine.service /etc/systemd/system/
sudo cp .env.example /etc/isales/engine.env  # then edit
sudo systemctl daemon-reload
sudo systemctl enable --now isales-engine
sudo journalctl -u isales-engine -f
```

## Graceful shutdown

`SIGINT` / `SIGTERM` triggers the lifespan finally block: dial / control
loops are cancelled, every in-flight session has its tasks cancelled, and
each session's `finalize_session` hook runs (writes `call_record` +
`pipeline_trace`, LPUSHes `CallEnded`, DECRs the global concurrency
counter, unregisters from `SessionManager`). Bound by
`ISALES_ENGINE_GRACEFUL_SHUTDOWN_TIMEOUT_S` (default 30s); the systemd unit's
`TimeoutStopSec=35` gives a 5-second buffer over the app timeout.

## Stage-4 verification

End-to-end with `scripts/fake_dial.py`:

```
python -m scripts.fake_dial \
  --db-url postgresql+asyncpg://isales@localhost:5432/isales \
  --redis-url redis://localhost:6379/0 \
  --campaign-id 1
```

Then check:

* `call_record` table — row exists with `status='end'`, populated transcript
* `pipeline_trace` table — at least one row per turn
* `engine:worker:call-ended` Redis list — one `CallEnded` payload per call

Required env (see `isales_engine/settings.py`):

- `ISALES_DATABASE_URL`
- `ISALES_REDIS_URL`
- `ISALES_ENGINE_LLM_PROVIDER` (default `mock`)
- `ISALES_ENGINE_ASR_PROVIDER` (default `mock`)
- `ISALES_ENGINE_TTS_PROVIDER` (default `mock`)
- `ISALES_ENGINE_TELEPHONY_MODE` (default `mock`)
- `ISALES_ENGINE_PIPELINE_DEFAULT_TIMEOUT_MS` (default `8000`)
- `ISALES_ENGINE_MAX_NO_PROGRESS_SECONDS` (default `60`)
- `ISALES_ENGINE_MOCK_CONNECT_DELAY_MS` (default `200`)
- `ISALES_ENGINE_GRACEFUL_SHUTDOWN_TIMEOUT_S` (default `30`)
- `TZ=Asia/Shanghai`

## Tests

```
pytest
```

`tests/` uses real PostgreSQL + Redis (matching the worker / scheduler test
strategy). Mock providers and `MockTelephonyClient` keep tests deterministic.
