# Observatory Corpora

This directory contains the externalized corpus payloads used by the PAM instrument.

## Purpose

Corpora are stored outside the canonical Python implementation so that:

- corpus content is reproducible and inspectable
- data does not remain embedded in code
- canonical modules can treat corpora as external inputs
- future corpus revisions can be tracked explicitly

## Layout

- `registry.json` — maps corpus names to corpus files
- `C.json` — base corpus
- `Cp.json` — corpus variant `Cp`
- `Cp2.json` — corpus variant `Cp2`
- `Cp3.json` — corpus variant `Cp3`
- `Cp4.json` — corpus variant `Cp4`

Each corpus file contains a JSON list of text samples.

## Canonical access

Corpora should be accessed through:

- `src/pam/corpora.py`

That module is the canonical loader/interface layer for corpus access.

## Notes

- Corpus files are treated as data artifacts, not implementation code.
- Changes to corpus contents should be made explicitly and reviewed carefully.
- Corpus naming is preserved for compatibility with existing experiments.Placeholder for future canonical corpus artifacts.
