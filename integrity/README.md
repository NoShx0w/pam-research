# Integrity

This directory owns validation, recovery, and artifact-chain integrity tooling for the PAM Observatory.

Integrity utilities are responsible for tasks such as:

- scanning for missing trajectory artifacts
- backfilling missing trajectories
- validating trajectory schema and completeness
- checking that artifact chains remain scientifically usable

These tools are separate from `experiments/` because they do not generate new scientific analyses.  
They verify and restore the observatory’s file-first substrate.

## Current tools

- `scan_missing_trajectories.py`
- `backfill_trajectories.py`
- `validate_trajectories.py`

## Notes

Where compatibility wrappers still exist under `experiments/`, the canonical implementation lives here.