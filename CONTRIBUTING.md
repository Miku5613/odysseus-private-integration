# Contributing to Odysseus Private Integration

Thanks for taking an interest in this repository.

This repository is my personal/private version of Odysseus. It contains local integrations, experiments, runtime changes, Windows-focused workflows, and project-specific behavior that may not be suitable for the public upstream project.

## Public Contributions

For general bug fixes, features, documentation improvements, or changes intended for the wider Odysseus community, contribute to the original public repository:

https://github.com/pewdiepie-archdaemon/odysseus

Please follow the contribution rules, branch model, issue process, and pull-request requirements defined by the upstream project.

Public pull requests are not accepted in this repository. Pull requests are limited to invited collaborators. Public Issues remain open for fork-specific bug reports, questions, and maintenance concerns.

Changes submitted by collaborators should normally be limited to this fork's own integrations, private workflow, compatibility fixes, or maintenance needs.

## Repository Purpose

This fork exists to maintain a stable personal version of Odysseus with custom integrations and local workflow changes.

Current fork-specific areas may include:

- OpenCode runtime integration
- local model and provider compatibility
- MCP integrations
- Obsidian and personal knowledge workflows
- Windows-native development and testing
- upstream synchronization and compatibility patches
- private runtime hardening and reliability work

The goal is not to replace the upstream project or create a competing public distribution.

## Branch Model

- **`Stable-Runtime`** — the default and primary branch. Changes merged here should be tested and suitable for normal use.
- Temporary feature, repair, audit, test, and upstream-integration branches may be created as needed.
- Pull requests to this repository should target `Stable-Runtime` unless a maintainer note says otherwise.

Upstream changes are normally reviewed on a temporary integration branch before being merged into `Stable-Runtime`.

## Before You Start

Before proposing a change to this fork:

- Public users should open Issues for fork-specific reports and questions; pull requests are reserved for invited collaborators.
- Confirm that the change is specific to this repository.
- For broadly useful public improvements, prefer contributing upstream.
- Keep changes focused and easy to review.
- Avoid unrelated formatting, cleanup, or large structural rewrites.
- Explain any effect on local integrations, Windows behavior, model providers, MCP tools, or OpenCode runtime behavior.
- Do not include private data, credentials, local paths containing personal information, or generated user content.

## Setup

This fork is primarily developed and tested on Windows 10/11 with Python 3.11.

Example setup:

```powershell
git clone https://github.com/Miku5613/odysseus-private-integration.git
Set-Location "odysseus-private-integration"

py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 7000
```

Docker support is retained where relevant, but native Windows workflows are actively used in this fork.

Use `.env.example` as a template. Never commit a live `.env` file.

## Running Checks

Run the smallest relevant checks for your change.

Examples:

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m compileall -q app.py routes src services
```

For JavaScript files:

```powershell
node --check "static\js\<file-you-changed>.js"
```

For Docker-related changes:

```powershell
docker compose config
docker compose up -d --build
docker compose logs --tail=120 odysseus
```

Mention which checks were run. If a check could not be run, state that clearly.

## Pull Requests

Pull requests to this repository are limited to invited collaborators. Public users should open an Issue for fork-specific bug reports, questions, or maintenance concerns. General-purpose code contributions should be submitted to the original Odysseus repository.

A useful pull request should include:

- a clear explanation of the problem or change;
- the files or systems affected;
- automated test results;
- manual verification steps where relevant;
- screenshots or short recordings for UI changes;
- any effect on upstream compatibility.

Keep pull requests focused. Large mixed changes are difficult to validate safely.

## UI and Visual Changes

For changes affecting buttons, icons, fonts, colors, spacing, layout, CSS, HTML, SVG, or DOM-rendering JavaScript:

- run the app locally;
- inspect the change in the browser;
- attach a screenshot or recording;
- reuse existing CSS variables and components;
- avoid parallel styling systems;
- preserve the existing visual language unless the change intentionally updates it.

## Code Conventions

Prefer existing constants, helpers, adapters, and configuration paths over new hardcoded values.

In particular:

- use named path constants from `src/constants.py`;
- do not hardcode writable source-tree paths;
- do not hardcode `http://localhost:7000` when an existing helper should be used;
- reuse existing ports, limits, model lists, and configuration helpers where available;
- preserve Windows compatibility unless the change is explicitly platform-specific.

Commit messages should be clear and descriptive. Conventional Commit style is encouraged but not mandatory for this personal fork.

Examples:

```text
fix(opencode): preserve MCP tools on text-only turns
test(runtime): cover low-context image handling
docs(security): document fork-specific reporting
```

## Security and Privacy

Do not post or commit:

- API keys or access tokens;
- private logs;
- personal documents;
- uploaded files;
- local databases;
- model-provider credentials;
- password hashes;
- authentication or session files;
- private IP addresses;
- screenshots containing sensitive information.

For security concerns, follow [SECURITY.md](SECURITY.md).
