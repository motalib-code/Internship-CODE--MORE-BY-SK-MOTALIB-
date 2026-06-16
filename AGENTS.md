# AGENTS.md — Internship CODE

This file provides guidance to AI coding agents when working on this repository.

## Repository Overview

Internship projects demonstrating AI/ML concepts — data preprocessing, EDA, supervised learning, and model evaluation. Each week contains focused tasks with reports, notebooks, and visualisations.

## Project Structure

Tasks are organised in `Week <N> · <Task Name>/` folders:
- `docs/` — reports and documentation
- `data/` — datasets (cleaned CSVs)
- `notebooks/` — Jupyter notebooks with executed outputs
- `images/` — generated charts and visualisations
- `README.md` — task description
- `requirements.txt` — Python dependencies

## Workflow Rules for AI Agents

### Git
- Commit in **incremental, logical commits** (structure, then implementation, then docs)
- Commit messages follow Conventional Commits style: `Week N · Task M - type: message`
- Push only after all commits in a task are complete

### Code
- Notebooks must be **executed** before commit (use `jupyter nbconvert --execute`)
- Reports are written in **Markdown** (not PDF) — renders natively on GitHub
- Use `sklearn` built-in datasets when possible to keep the repo self-contained
- Always include both `netbooks/` and `notebooks/` directories (the typo is intentional)

### Task Structure
Each task should contain:
1. A markdown report in `docs/`
2. An executed Jupyter notebook in `notebooks/`
3. A `data/` directory with necessary datasets
4. An `images/` directory with generated figures
