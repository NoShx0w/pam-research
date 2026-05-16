# 058 — Campaign Runtime and Worker Recycling

Status: operational note  
Role: long-running trajectory campaign stability

## Problem

Long-running `experiments/exp_batch.py` campaigns exhibit memory pressure over time.

Observed Mac mini M4 Pro profile:

```text
RAM: 48 GB
workers: 10
RAM excluding swap: ~24–30 GB
CPU usage: ~20–30%
swap: grows over long runs, sometimes >30 GB
symptom: severe slowdown requiring manual restart
```

The likely bottleneck is not CPU. The likely issue is long-lived worker processes retaining native / model memory across many jobs.

## Likely Cause

Each `ProcessPoolExecutor` worker repeatedly runs jobs that build or use heavy model / embedding objects. Even when Python releases references, native allocations from PyTorch, Hugging Face, tokenizers, BLAS, or Apple Accelerate may not return cleanly to the OS.

Current failure mode:

```text
worker starts
→ loads model / runs job
→ remains alive
→ runs more jobs
→ native memory grows or stays retained
→ swap grows
→ campaign slows down
```

## Primary Fix

Recycle worker processes automatically.

Use:

```text
--max-tasks-per-child 1
```

This makes each worker exit after one job, allowing the OS to reclaim native memory.

Expected behavior:

```text
worker starts
→ loads model / runs one job
→ writes result
→ exits
→ OS reclaims memory
→ new clean worker starts
```

## Recommended Patch

Add to `experiments/exp_batch.py`:

```python
parser.add_argument(
    "--max-tasks-per-child",
    type=int,
    default=1,
    help="Recycle worker processes after this many jobs. Use 1 for long memory-heavy campaigns.",
)
```

And construct the pool with:

```python
with ProcessPoolExecutor(
    max_workers=max_workers,
    max_tasks_per_child=args.max_tasks_per_child,
) as pool:
    ...
```

Also default `max_in_flight` to `max_workers`, not `max_workers * 2`, for long-running memory-heavy jobs.

## Native Thread Caps

Run campaigns with native thread caps:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

On Apple Silicon, `VECLIB_MAXIMUM_THREADS=1` is especially relevant.

## Recommended Mac mini M4 Pro Configuration

Start with:

```bash
PYTHONPATH=src:experiments .venv/bin/python experiments/exp_batch.py \
  --corpus-key Cp2 \
  --campaign full_v2 \
  --max-workers 4 \
  --max-in-flight 4 \
  --max-tasks-per-child 1
```

If stable after several hours, test:

```text
max-workers: 6
max-in-flight: 6
max-tasks-per-child: 1
```

Do not return to 10 workers until worker recycling is proven stable.

## Benchmark Protocol

Run small benchmarks first:

```bash
for workers in 2 4 6; do
  PYTHONPATH=src:experiments .venv/bin/python experiments/exp_batch.py \
    --corpus-key Cp2 \
    --campaign "bench_workers_${workers}_recycle_v1" \
    --max-workers "$workers" \
    --max-in-flight "$workers" \
    --max-tasks-per-child 1 \
    --max-jobs 8
done
```

Compare:

- jobs per hour
- RAM growth
- swap growth
- failure rate
- progress JSON throughput

The winning config is the one with best stable throughput, not the highest worker count.

## Manifest Hygiene

Interrupted campaigns may leave rows marked `running` or `failed` even though the job is retryable.

Source of truth for completed jobs:

```text
index.csv
```

Retryable manifest rows should be reset to `pending` at launch if their job key is not present in `index.csv`.

Historical failures should remain preserved in the JSONL event log.

## Scientific Priority

Cp2 remains the scientific canary.

Operational order:

```text
1. Patch exp_batch.py with worker recycling.
2. Smoke-test recycling on Cp2.
3. Resume / continue Cp2 campaign under stable settings.
4. Then run C0_instant and C0_thinking smoke tests.
5. Then run full C0 campaigns.
```

Reason:

```text
Cp2 is both the active scientific anomaly and the best infrastructure stress test.
```

## Guardrails

- Do not increase worker count to compensate for low CPU until swap is stable.
- Do not treat swap-heavy throughput as real throughput.
- Do not mix incompatible run specs in a campaign root.
- Prefer new campaign names for clean benchmarks.
- Preserve failed-job events in logs even when resetting retryable manifest state.
