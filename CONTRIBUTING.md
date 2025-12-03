# Contributing to Federated Meta Learning

Thanks for your interest in contributing! This project follows a lightweight, collaborative workflow — please follow the guidelines below to make contributions easy to review and merge.

## Start Here
- Read the project `docs/GUIDE.md` for high-level goals and current phase status.
- Search existing issues before creating a new one.

## Reporting Issues
When opening an issue, include:
- **Title**: short summary
- **Description**: clear reproduction steps, expected vs actual behavior
- **Environment**: OS, Python version, installed packages (if relevant)
- **Logs / Tracebacks**: paste or attach minimal logs
- **Data / Code**: minimal reproduction script or notebook cells when possible

## Branch Naming
Use clear branch names:
- `issue-<number>` (for bug fixes tied to an issue)
- `feature/<short-description>`
- `fix/<short-description>`

## Pull Request Checklist
Before opening a PR:
- Target `main` (or the specified target branch).
- Reference the related issue(s) in the PR description.
- Keep changes focused — one logical change per PR.
- Add a short description of the change and testing steps.
- Include tests or a short reproduction script for bug fixes when feasible.
- Run basic sanity checks:
  - `python -m compileall src`
  - run relevant unit tests (if present): `pytest` (optional)

## Code Style & Quality
- Follow the repository's existing style and conventions.
- Prefer clear, descriptive names and small functions.
- Avoid excessive comments — keep code self-explanatory; document complex logic with a short comment or docstring.
- Type hints are encouraged where they help readability.

## Commit Messages
Use concise, informative commit messages. Suggested prefixes:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `test:` tests or test fixes
Include `#<issue-number>` in the message when relevant.

## Review Process
- Maintainers will review PRs and leave comments. Please address feedback and update the PR.
- Keep commits tidy; rebase or squash as requested by reviewers.

## Tests
- Add unit tests for bug fixes or new behavior when practical.
- If you cannot add tests (e.g., data or environment constraints), provide a minimal reproduction script and clear verification steps in the PR description.

## License
This repository is licensed under the GNU Affero General Public License v3 (AGPL-3.0). By contributing you agree that your contributions will be licensed under the same terms. Do not submit third-party code that conflicts with this license.

## Contact
- Open an issue or ping `@Sahilbhatane` on PRs for attention.

Thank you — your contributions help improve this project!
