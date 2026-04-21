# Contributing to biolab-runners

This is a private repository. All contributions must be authorized.

## Commit Format

Use [Conventional Commits](https://www.conventionalcommits.org/). See [`.gitmessage`](.gitmessage) for the template.

```bash
git config commit.template .gitmessage
```

## Pre-commit Checklist

Run before every commit:

```bash
make validate
```

This executes: `ruff` (format + lint check) → `pyright` (type check) → `complexipy` (cognitive complexity) → `pytest` (tests).

## Pull Request Guidelines

- One logical change per PR
- Title under 70 characters
- Include test plan in PR description
- All CI checks must pass
- No secrets, API keys, or `.env` files

## Code Style

- Google-style docstrings (enforced by ruff)
- Type annotations on all public functions (enforced by pyright basic)
- Max cyclomatic complexity: 10
- Max cognitive complexity: 15

## Adding a New Runner

1. Create `biolab_runners/new_runner/` with `__init__.py`, `config.py`, `runner.py`, `utils.py`
2. Define config + result dataclasses
3. Implement runner class with `run()`, `dry_run`, idempotency, logging
4. Add tests in `tests/` using mocks (no real GPU/CLI deps)
5. Add optional extras in `pyproject.toml`
6. Export from `__init__.py`
