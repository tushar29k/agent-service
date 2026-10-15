# Concurrency test results

N parallel ReAct agent runs (one agent instance per thread, fresh
checkpoint dir each) on the offline mock backends, each run consuming
one eval task round-robin. `wall_s` is the pool window (first submit →
last finish); `tasks/s` = runs / wall_s. The mock brain answers
near-instantly, so these numbers measure agent-loop overhead and
thread-safety — not model latency. `eval_miss` = run finished but failed
the suite's pass bar; `errors` = worker threads that raised.

| date | backend | threads | runs | wall_s | tasks/s | p50_ms | p99_ms | passed | eval_miss | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-15 | react | 16 | 16 | 0.34 | 47.6 | 65.1 | 308.6 | 16 | 0 | 0 |
| 2026-10-15 | native | 16 | 16 | 0.5 | 32.1 | 154.4 | 444.2 | 16 | 0 | 0 |
