# Contributing to Mantis

Thanks for your interest in contributing to Mantis.

## Getting started

1. Check the [issue tracker](https://github.com/mallochio/mantis/issues) to see if your idea is already being discussed. Open a new issue if it isn't.
2. Fork the repository and create a feature branch from `main`.
3. Make your changes, add tests when applicable, and keep commits focused.
4. Run the development checks listed below.
5. Open a pull request with a clear description of what changed and why.

## Branching workflow

The `main` branch is protected. Direct pushes to `main` are not allowed — all changes must come through a pull request.

## Development checks

Run the same checks that CI runs:

```bash
uv run ruff check .
uv run mypy apps/api scripts --exclude outputs
uv run pytest tests -q
uv run coverage report --fail-under=85
```

If you add new dependencies, update `pyproject.toml` and run `uv sync --all-extras --locked`.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0 (see `LICENSE`).
