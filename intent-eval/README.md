# Intent prompt evaluation score

Evaluation of `src/services/intent/prompts.py` against 1000 real,
naturalistic on-chain analytics questions (`questions/c_batch_*.txt`, full
results in `results/c_results_*.jsonl`).

## Round 6 — full 1000-question run

```
class:  analytical 885, explanatory 69, conversational 46, editorial 0
status: complete 572, clarify 313   (analytical only)
gap:true: 70 (7.0%)
```

## Round 7 — gap fixes, verified

Re-ran all 70 round-6 gaps against the round-7 prompt
(`questions/gap_reval7_*.jsonl`, `results/gap_reval7_results_*.jsonl`):

```
resolved:      30 / 70  (43%)
still_present: 40 / 70  (57%)
new_issue:      0 / 70
```

No fresh full 1000-question run yet since round 7 — the numbers above are
gap-only; ask for a full re-run to get updated overall rates.
