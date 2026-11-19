# AGENTS.md

## Design rule

`ilamb3` is being formalized as a logical system that follows the rules of probability theory.
Derive each design choice from those rules where they decide it, and label every other choice as a convention, a modeling choice, or a value judgment.
Read [docs/design/probabilistic-framework.md](docs/design/probabilistic-framework.md) before changing how errors are normalized, decomposed, aggregated, or scored.
The settled decisions are ADRs in [docs/adr/](docs/adr/README.md); record a new one when a decision is hard to reverse, and supersede an ADR rather than edit its decision.

- **Aggregate-then-score.** Combine squared errors over cells, times, components, and datasets, then apply the score function once, so the ranking of models does not depend on the score function.
- **Add like with like.** A sum of errors takes terms in the same units, divided by one normalizer, and orthogonal to each other; report any other metric as a diagnostic.

## Language

[CONTEXT.md](CONTEXT.md) holds the canonical terms; use them in code, docs, and PR text, and add a term there when it settles.
User-facing definitions are in [docs/reference/glossary.md](docs/reference/glossary.md); keep the two consistent.

## Checks

- Setup and tests: [docs/contribute/develop.md](docs/contribute/develop.md).
- Every number in the design doc comes from `docs/design/experiments/verify_*.py`.
  After changing a claim, or the scoring code a claim describes, rerun them and update the doc from their output.

## Writing

- Docs are MyST Markdown with semantic line breaks: one sentence or clause per line.
  `docs/design/` holds proposals and stays out of the table of contents.
- Cite sources you have read, with a DOI when one exists, and keep PDFs out of the repository.
- On GitHub, write each paragraph of an issue or PR body on one line, and write upstream references in full (`rubisco-sfa/ilamb3#217`) so they resolve from forks.
