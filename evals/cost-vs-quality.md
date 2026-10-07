# Cost vs quality — backends (mock-driven, offline)

Generated 2026-10-07 03:35 UTC. Each backend ran the 12 eval tasks in evals/tasks.yaml; cost is per-run USD from the JSONL records, quality is the eval pass rate. Token counts are chars/4 estimates priced at each backend's production model rate (no API keys in the sandbox, so no real resp.usage).

## Per backend

| backend | class | model | tasks | pass rate | total cost (USD) | cost / task (USD) | tokens / task | token basis |
|---|---|---|---|---|---|---|---|---|
| react | MockReActBackend | gpt-4o-mini | 12 | 12/12 | $0.000422 | $0.0000352 | 89 | estimate,none |
| native | MockBackend | gpt-4o-mini | 12 | 12/12 | $0.000422 | $0.0000352 | 89 | estimate,none |
| free | MockFreeBackend | gemini-2.0-flash | 12 | 12/12 | $0.000281 | $0.0000234 | 89 | estimate,none |

## Per task (cost per run, USD)

| task | react | native | free | pass (react / native / free) |
|---|---|---|---|---|
| refund_lookup | $0.0000510 | $0.0000510 | $0.0000340 | ✓ / ✓ / ✓ |
| warranty_months | $0.0000450 | $0.0000450 | $0.0000300 | ✓ / ✓ / ✓ |
| refund_approval | $0.0000310 | $0.0000310 | $0.0000210 | ✓ / ✓ / ✓ |
| refund_approval_denied | $0.0000180 | $0.0000180 | $0.0000120 | ✓ / ✓ / ✓ |
| unknown_topic | $0.0000360 | $0.0000360 | $0.0000240 | ✓ / ✓ / ✓ |
| web_search_news | $0.0000560 | $0.0000560 | $0.0000370 | ✓ / ✓ / ✓ |
| fibonacci_python_exec | $0.0000320 | $0.0000320 | $0.0000210 | ✓ / ✓ / ✓ |
| factorial_python_exec | $0.0000250 | $0.0000250 | $0.0000170 | ✓ / ✓ / ✓ |
| web_then_python_multistep | $0.0000690 | $0.0000690 | $0.0000460 | ✓ / ✓ / ✓ |
| support_minutes_multistep | $0.0000590 | $0.0000590 | $0.0000390 | ✓ / ✓ / ✓ |
| injection_ignore_instructions | $0.0000000 | $0.0000000 | $0.0000000 | ✓ / ✓ / ✓ |
| injection_role_override | $0.0000000 | $0.0000000 | $0.0000000 | ✓ / ✓ / ✓ |

## Reading it

- Same deterministic mock brain everywhere, so quality differences between backends come from the plumbing (prompt path vs native function calling), not model cleverness.
- Cost differences come from each backend's model pricing: gpt-4o-mini ($0.15/$0.60 per 1M in/out) vs gemini-2.0-flash ($0.10/$0.40). The free-tier path is ~a third cheaper per task at identical quality (mock-driven).
