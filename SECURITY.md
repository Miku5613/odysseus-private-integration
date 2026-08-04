# Security Policy

This repository is my personal/private version of Odysseus.

It contains custom local integrations, runtime changes, MCP connections, model-provider configuration, OpenCode support, and personal workflow features. These additions may grant models access to local tools, files, services, or private data.

This repository should not be treated as a hardened public cloud service.

## Original Project

General security issues affecting the public Odysseus project should be reported to the original repository:

https://github.com/pewdiepie-archdaemon/odysseus

Use the upstream project's security policy and private reporting process for vulnerabilities that are not specific to this fork.

Fork-specific vulnerabilities may be reported through this repository's private GitHub security reporting features when available.

## Supported Version

Security fixes for this fork are handled on the default branch:

- **`Stable-Runtime`**

No formal long-term support or release-version policy is currently provided.

## Deployment Guidance

- Do not expose Odysseus directly to the public internet without authentication and network protection.
- Keep `AUTH_ENABLED=true` for any network-accessible deployment.
- Keep `LOCALHOST_BYPASS=false` outside local development.
- Set `SECURE_COOKIES=true` when using HTTPS through a trusted reverse proxy or private access gateway.
- Use HTTPS when exposing the app beyond localhost.
- Prefer a trusted reverse proxy, VPN, Tailscale, Cloudflare Access, or another private access layer.
- Keep ChromaDB, SearXNG, ntfy, Ollama, vLLM, llama.cpp, databases, and raw model-provider APIs internal-only.
- Disable open signup unless new accounts are intentionally allowed.
- Use strong passwords for administrator accounts and enable multi-factor authentication where supported.
- Remove test and demo users from serious deployments.

Common internal-only ports may include:

- Odysseus: `7000`
- SearXNG: `8080`
- ntfy: `8091`
- ChromaDB: `8100`
- Ollama: `11434`
- local model-provider APIs: commonly `8000-8020`

## Privileged Features

Treat the following as privileged functionality:

- shell execution;
- Python execution;
- file read and write tools;
- MCP servers;
- OpenCode runtime tools;
- email and calendar access;
- Obsidian or vault access;
- application API tools;
- task, skill, and memory management;
- model-serving controls;
- authentication and token management;
- settings changes;
- local database access.

Do not assume a tool call is safe simply because the model is running locally.

Review:

- tool permissions;
- MCP server configuration;
- allowed workspace roots;
- writable directories;
- model-provider endpoints;
- authentication settings;
- exposed ports;
- session and token storage.

## Sensitive Files and Data

Protect and never publish:

- `.env`;
- `data/`;
- `logs/`;
- uploads;
- generated media;
- backups;
- authentication and session files;
- local databases;
- API keys;
- webhook secrets;
- model-provider tokens;
- password hashes;
- personal documents;
- exported conversations;
- private project files.

Rotate any key, token, or secret that appears in logs, screenshots, demos, shared chats, or repository history.

## Before Publishing This Fork

Before pushing publicly, review the working tree and scan tracked files for common secrets.

```powershell
git status --short

git check-ignore -v `
    ".env" `
    "data/auth.json" `
    "data/app.db" `
    "logs/compound.log" `
    "odysseus.db"

git grep -n -I -E `
    "(sk-[A-Za-z0-9_-]{20,}|xox[baprs]-|AIza[0-9A-Za-z_-]{20,}|Bearer [A-Za-z0-9._~+/-]{20,})" `
    -- . `
    ":(exclude)static/lib/**" `
    ":(exclude)package-lock.json"
```

These checks are useful but are not a complete secret audit.

Only commit files intentionally meant to be public, such as:

- `.env.example`;
- documentation;
- source code;
- tests;
- static assets;
- public configuration examples.

Do not commit live configuration, runtime data, local databases, generated content, logs, backups, authentication files, tokens, or personal documents.

## Fork-Specific Scope

This fork may differ significantly from upstream in areas involving:

- OpenCode runtime behavior;
- model-provider compatibility;
- context handling;
- tool calling;
- MCP integrations;
- Obsidian and local knowledge access;
- Windows-native execution;
- local automation;
- personal data workflows.

A vulnerability caused by one of these fork-specific changes should be reported here rather than upstream unless the same issue also affects the original project.

## Reporting a Vulnerability

For fork-specific issues:

1. Use this repository's private GitHub vulnerability-reporting or Security Advisory feature when available.
2. Do not publish exploit details, credentials, personal information, or working proof-of-concept attacks in a public issue.
3. If private reporting is unavailable, open a minimal public issue requesting a private contact channel without disclosing sensitive details.

For issues affecting the original public Odysseus project, report them through the upstream repository's security process.
