# Observatory Interface

This section covers interface-facing components of the PAM Observatory.

## Status

The terminal observatory interface is no longer the canonical architectural center of the repository.

The canonical runtime now lives in:

- `src/pam/pipeline/runner.py`
- `scripts/run_full_pipeline.sh`

Interface-facing material may still be documented here when needed, especially for:

- historical TUI workflows
- artifact inspection interfaces
- future monitoring or observatory-facing tools

For the implemented repository architecture, see:

- [`../architecture.md`](../architecture.md)