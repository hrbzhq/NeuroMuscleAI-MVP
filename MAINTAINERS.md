Project maintenance policy
-------------------------

This file outlines the maintenance automation and how external help (bot/assistant) can support the project.

Automations included:
- Dependabot: weekly dependency update PRs (`.github/dependabot.yml`).
- CI: lightweight lint and smoke tests (`.github/workflows/ci.yml`).
- Pages deploy: publish `docs/` to `gh-pages` via Actions (`.github/workflows/deploy_pages.yml`).

How assistant can help:
- Open PRs for bug fixes, refactors, tests and documentation updates.
- Respond to issues and triage (labeling, assigning, opening PRs).
- Keep translations in sync and expand i18n coverage.

Security & permissions:
- The assistant will never push secrets or request credentials. Anything requiring elevated permissions will be proposed and left for human approval.
