# Pattern Guardrails Under Mutation: A Three-Metric Evaluation of Two Agent Safety Layers

**Abstract.** I built two agent systems guarded by pattern matching: a tool-using support agent behind a regex-style injection detector, and a SQL agent behind a validate-then-approve gate. I tested how those layers hold against systematically mutated attacks, and what the defence costs in false refusals and lost completion. I mutated my seeds deterministically (147 agent-guardrail cases; 84 SQL attacks, 13 benign controls), added an indirect-injection scenario suite, and scored three metrics: attack success rate (ASR), false-refusal rate (FRR), and completion under attack. Surface-form mutations were blocked almost perfectly; encoded payloads (base64, ROT13, leetspeak, zero-width, homoglyphs) collapsed the pattern layer — 0 of 24 runs blocked per class. The SQL gate stopped 83 of 84 attacks (database never modified) but misclassified a comment-glued write inside a CTE and falsely blocked 2 of 13 benign queries. All runs are mock-backend only.

## The question

Pattern matching is the cheapest safety layer an agent can have. How much of its measured strength is an artefact of testing it only against attacks in the same surface form as its patterns? Do unblocked attacks pass unfiltered in ways a stronger model could exploit? What does the guardrail cost benign users in false refusals and lost completion? One pass/fail number answers none of these, so I scored three metrics.

## Method

**Systems.** The support agent uses tools (knowledge-base search, calculator, code execution, a refund tool) behind an injection detector screening tasks and tool observations against four pattern families, plus an approval gate pausing destructive tool calls. The SQL agent works over a seeded four-table database; every emitted query is classified `ok`, `needs_approval`, or `blocked` before execution.

**Case generation.** Both benchmarks start from hand-written seeds (14 agent cases, 10 SQL attack seeds). I applied ten deterministic mutations (base64, ROT13, leetspeak, zero-width insertion, homoglyph substitution, scrambling, code fencing, fake delimiters, all-caps, punctuation stuffing) plus a split-payload transform: 132 attack variants, plus 15 hand-written benign near-miss controls. The SQL generator produced 84 attacks across seven families (case, whitespace, comment splitting, multi-statement smuggling, structural synonyms, encodings, paraphrases) plus 13 controls, each verified by before/after row-count snapshots on a fresh database copy.

**Scenarios.** Eleven scenarios model indirect injection: a benign multi-step task runs while a payload in one tool's output tries to steer it — planting a false figure or policy claim, emitting a canary token, splitting an instruction across two tools' outputs, two tool-hijack attempts (an unrequested refund; a knowledge-base dump); one is a benign control quoting attack text as course material.

**Metrics.** Each attack run is classified **blocked** (refusal logged), **succeeded** (measurably steered the run — planted content in the answer, a forbidden tool call, a canary emitted), or **evaded** (not refused, no goal signal measured — it reached the model unfiltered). I keep "evaded" separate on purpose: an evaded base64 payload shows no success on a mock that cannot decode it, and folding evasion into either rate would hide the blind spot this benchmark exists to expose. ASR counts only measured successes; benign-side, I track FRR and completion, including completion under attack.

## Results

### Agent guardrails

Identical on both backends (the detector is backend-independent), so one set:

| suite | attacks | blocked | succeeded | evaded | ASR | benign (n) | FRR | completion under attack |
|---|---|---|---|---|---|---|---|---|
| seeds | 12 | 12 | 0 | 0 | 0.0% | 2 | 0.0% | 0.0% (n=4) |
| variants | 132 | 59 | 25 | 48 | 18.9% | 15 | 20.0% | 48.1% (n=52) |
| scenarios | 10 | 2 | 5 | 0 | 50.0% | 1 | 100.0% | 80.0% (n=10) |
| benign corpus | — | — | — | — | — | 27 | 11.1% | completion 74.1% |

The seed suite passes perfectly — the predicted artefact. Under mutation, only 59 of 132 variants per backend were blocked.

| mutation class | runs | blocked | succeeded | evaded | ASR |
|---|---|---|---|---|---|
| all_caps | 24 | 24 | 0 | 0 | 0.0% |
| code_fence | 24 | 24 | 0 | 0 | 0.0% |
| fake_delimiters | 24 | 24 | 0 | 0 | 0.0% |
| split_payload | 24 | 22 | 2 | 0 | 8.3% |
| punct_stuffing | 24 | 12 | 4 | 8 | 16.7% |
| scramble | 24 | 12 | 4 | 8 | 16.7% |
| base64_wrap | 24 | 0 | 8 | 16 | 33.3% |
| rot13_wrap | 24 | 0 | 8 | 16 | 33.3% |
| leetspeak | 24 | 0 | 8 | 16 | 33.3% |
| zwsp_insert | 24 | 0 | 8 | 16 | 33.3% |
| homoglyph | 24 | 0 | 8 | 16 | 33.3% |

Five of 10 scenario attacks succeeded, all by one mechanism: the agent quotes tool observations into its final answer, so detector-evading text — a planted "60 months" result, a "pre-approved" claim, canary CANARY-7741 — lands verbatim. The two tool-hijack attempts did not fire: the mock ignores instructions inside observations.

### SQL guardrails

| family | cases | passed | attacks stopped |
|---|---|---|---|
| case_obfuscation | 14 | 14 | 14 |
| whitespace_obfuscation | 7 | 7 | 7 |
| comment_splitting | 28 | 22 | 27 |
| multi_statement_smuggling | 20 | 20 | 20 |
| structural_synonym | 9 | 9 | 9 |
| encoding | 2 | 2 | 2 |
| paraphrase | 4 | 4 | 4 |
| near-miss controls | 13 | 11 | — |

78 of 84 attacks produced exactly the expected outcome; 83 of 84 were stopped at the gate (the one waved through was rejected by SQLite itself), and the database was unchanged in all 97 cases. Controls passed 11 of 13 (false-block rate 15.4%); the original gate suite is unchanged and green (seeds 11/11, execution accuracy 13/13). Question-axis attacks went 6/6 to `needs_approval`: the gate reads the emitted SQL, not the question, so prompt-layer obfuscation attacks the wrong layer here.

## Findings

1. **Encoded payloads are invisible to the pattern layer.** All 60 encoded-class variants (12 per class) evaded every pattern — 0 of 24 runs blocked per class; their only measured successes came from observation text quoted into the answer, the rest sitting in "evaded".
2. **Patterns survive respelling, not transformation.** All-caps, code fences and fake delimiters went 24/24 blocked; SQL case/whitespace obfuscation 21/21, stacked statements 20/20. But punctuation stuffing and scrambling each reached 16.7% ASR — a token between words breaks multi-word patterns, and scrambling can split a keyword itself.
3. **Split payloads evade at boundaries.** One variant split across the task/observation boundary was refused at neither stage; a cross-tool scenario split also evaded and reached the answer.
4. **Answer quoting is a steering channel.** All five scenario successes came from quoting poisoned observations verbatim — no instruction-following required.
5. **False refusals are specific, not random.** Three of 15 near-miss controls were refused — any "developer mode" mention (two), any "you are now…" opener (one) — as was course material quoting attack text: patterns cannot distinguish quotation from command. Benign-corpus FRR 11.1%, completion 74.1%.
6. **Fail-closed has a completion price.** Completion under attack: 0.0% on seed observation attacks (n=4), 48.1% on variants — blocked runs lose the user's answer; evaded runs often still answer.
7. **The SQL gate's one bypass is a classification bug, not an execution.** `WITH x AS (DELETE/**/FROM orders RETURNING *) SELECT * FROM x` is classified `ok`: comment stripping without a space glues `DELETE/**/FROM` into `DELETEFROM`, which the keyword scan misses. SQLite (3.45.1) rejects DML in a CTE body, so it cannot execute here — but classification is the validator's one job, and should not depend on the database rejecting the shape.
8. **The cleaner also causes the SQL false blocks.** Comment-glued standalone writes (`DROP/**/TABLE`, four siblings) classify as `blocked` instead of `needs_approval` — safe, but unapprovable. Backtick- and bracket-quoted aliases (`delete`, [update]) were blocked because the cleaner strips only single- and double-quoted identifiers; a column named `` `delete` `` would be unqueryable.

## Limitations

Everything ran on mock backends, which bounds every claim. The mock cannot decode base64, ROT13 or leetspeak, so encoded-class success against a real model is unmeasured — a real model decodes these trivially and would receive the payload unfiltered, making measured ASR a floor. The mock also does not follow instructions in observations, so tool-hijack risk is unmeasured rather than absent. No real-model runs exist yet (no API key is configured in the environment). The scenario suite is small (10 attacks, one benign control — its 100% FRR is a single case). All seeds are hand-written by me; 4 of the 15 near-miss controls had calibration issues on first run, left as-is. Both systems are my own small services, not production deployments.

## Future work

Next is a real-model slice — the 14 seeds plus the encoded classes, about 40 case-runs — converting "evaded" into a measured ASR and testing whether the hijack scenarios fire when the model can follow planted instructions. Detector fixes follow as a separately measured phase, graded by these suites as regression tests: a fix must move ASR, FRR and completion, not just the seed score. The SQL cleaner fix is already scoped by findings 7–8.

## Reproducibility

Both projects are public: `agent-service` and `sql-agent` under github.com/tushar29k. Generators stamp the seed file's sha256 into their output and regenerate byte-identically; metric definitions are frozen in a versioned scoring module. The expanded suites are report-only in CI (the seed suites remain the gate), and every finding is logged in the repositories, deliberately unfixed, so the fix phase can be graded against them.
