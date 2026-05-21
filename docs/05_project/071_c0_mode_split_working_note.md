# 071 — C0 Instant/Thinking Mode Split Working Note

Status: working observation  
Scope: C0 smoke tests and live C0_instant full_v1 campaign  
Role: hypothesis note, not final OBS result

## Observation

The C0 smoke tests show a sharp macrostate split between `C0_instant` and `C0_thinking` at the same initial band (`r=0.10`, `alpha=0.03`, 8 seeds).

`C0_instant` is freeze-active:
- `piF_mean` is nonzero across the smoke rows.
- `corr0` and `best_corr` are finite.
- early `full_v1` rows continue to show finite freeze activity.

`C0_thinking` is freeze-inactive in the smoke test:
- `piF_mean = 0.0` for 8 / 8 rows.
- `piF_tail = 0.0` for 8 / 8 rows.
- freeze-derived correlation fields are undefined / NaN.

## Working interpretation

This suggests that the bare baseline is mode-conditioned rather than singular.

A current working hypothesis is that `Instant` behaves more like a continuation-open / exploratory response ecology, while `Thinking` behaves more like a closure-oriented / problem-solving response ecology.

Under this hypothesis, the freeze macrostate may index continuation-open recursive dynamics rather than semantic quality or model capability.

## Relation to Cp2

Cp2 `full_v2` is also freeze-inactive across the full grid. Therefore, Cp2 and `C0_thinking` may share a freeze-suppression signature despite different provenance:

- `C0_thinking`: mode-induced closure.
- `Cp2`: prompt/operator-induced closure.

This remains provisional until full `C0_instant` and `C0_thinking` campaigns are completed and validated.

## Guardrail

This note records a working hypothesis only. It does not claim that Instant only explores or Thinking only solves. It does not establish causal mechanism. Full-grid validation and downstream comparisons are required before an OBS-level result.
