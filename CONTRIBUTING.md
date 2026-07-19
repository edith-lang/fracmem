# Contributing to fracmem

Thanks for considering a contribution. This is a small, focused library, so the bar for changes is mostly "does it stay correct and well-tested," not process overhead.

## Setup

```bash
git clone https://github.com/edith-lang/fracmem.git
cd fracmem
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

(If `.[dev]` isn't defined yet, just `pip install -e . pytest` — the project doesn't currently split out a dev extras group.)

## Running tests

```bash
pytest tests/ -v
```

All new code should come with a test. If you're fixing a bug, add a test that fails without your fix and passes with it — this project's own development history (see `docs/theory.pdf` and `docs/code_walkthrough.pdf`) leaned hard on catching bugs via verification rather than by inspection, and contributions should keep that standard.

## What's especially welcome

- **Real hardware validation reports** on new platforms (this library was developed and checked on desktop Python and a Raspberry Pi; reports from other microcontrollers/SBCs are valuable).
- **Extending the certificate** (`docs/theory.pdf`, Theorem 4.1) to cover configurations not yet analyzed.
- **Additional fractional orders / sample rates** exercised in the test suite.
- **Bug reports with a minimal reproduction** — a short script showing `alpha`, `h`, `L`, `p`, and the unexpected output is far more useful than a description alone.

## What to avoid

- Don't re-fit the decay rates (`lambda`) from data anywhere in the core library — this is a deliberate design decision with a mathematical justification (see `docs/theory.pdf`, §10), not an oversight. If you think you have a case where it helps, please open an issue with data before sending a PR that changes it.
- Don't add heavyweight dependencies. The library's whole value proposition includes being embeddable and lightweight; it currently depends on nothing but `numpy` and `scipy`.

## Pull requests

1. Fork, branch, make your change with tests.
2. `pytest tests/ -v` must pass.
3. Open a PR describing *why*, not just *what* — especially for anything touching the core fitting logic, since the reasoning behind the current design is usually non-obvious and documented in `docs/theory.pdf`.

## Code of conduct

Be respectful, be specific, assume good faith. Technical disagreements should be resolved with evidence (a test, a benchmark, a counterexample) rather than assertion.
