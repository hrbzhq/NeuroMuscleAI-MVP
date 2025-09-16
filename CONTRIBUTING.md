# Contributing to NeuroMuscleAI-MVP

Thank you for your interest in contributing! This document describes the basic workflow, coding style, and checks we expect contributors to run before opening a pull request.

## Development setup

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## pre-commit

We use `pre-commit` to enforce formatting and basic checks (Black, isort, flake8, yamllint, etc.). Run:

```powershell
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Fix any reported issues locally before pushing.

## Pull requests

- Create a feature branch from `master`.
- Keep PRs small and focused. Explain the motivation and include testing steps.
- Ensure all tests pass locally: `pytest -q`.
- Squash or rebase commits to keep history tidy if requested by maintainers.

## Code style

- Use Black for formatting. We follow the default Black settings.
- Use isort to sort imports.
- Lint with flake8 and address warnings before creating a PR.

## Tests

Add unit tests under `tests/`. Run the test suite with:

```powershell
pytest -q
```

## Large files and history

We avoid committing large binary files directly into the repository. If you must add large artifacts, prefer using releases or external storage. For emergency transfers (e.g. network-restricted environments), see `tools/network_fallback/` for creating `repo.bundle` and patches.

Thanks for contributing!
# Contributing to NeuroMuscleAI-MVP

Thanks for your interest in contributing! This document outlines a minimal workflow for contributors.

1. Fork the repository and clone your fork.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and run tests locally.
4. Commit changes with clear messages and push your branch.
5. Open a pull request describing the change.

Guidelines:
- Keep changes focused and well-documented.
- Add tests for new functionality where practical.
- Keep the coding style consistent (PEP8).

Contact: beginningstone (see repository owner on GitHub).
