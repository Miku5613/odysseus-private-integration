# Acknowledgments

> This file is inherited from the original Odysseus project and maintained in
> this personal fork as dependencies and integrations change.

Odysseus stands on the shoulders of a lot of open-source work. This file
credits the projects whose code, assets, or designs are included in or
adapted by this repository, and notes their licenses.

If you believe something here is misattributed or missing, please open an
issue — it will be corrected promptly.

---

## Adapted / borrowed code

Portions of this project were adapted from other open-source repositories.
Their original authors retain copyright over the adapted portions, under the
licenses noted below.

The sources below are under permissive licenses such as MIT and Apache-2.0,
which permit this use as long as their original copyright and license notices
are preserved. The full license texts are kept in
[`licenses/`](licenses/).

- **[opencode](https://github.com/anomalyco/opencode)** — open-source AI coding
  agent, originally maintained at
  [opencode-ai/opencode](https://github.com/opencode-ai/opencode).
  Copyright © the opencode authors. **MIT License.** Adapted for agent-loop,
  tool-execution, and UI concepts.
- **[llmfit](https://github.com/AlexsJones/llmfit)** by **Alex Jones** — the
  engine behind the Cookbook's model download, serving, and “What Fits?”
  features. Copyright © Alex Jones. **MIT License.** Adapted in
  `services/hwfit/` for hardware detection, quantization-aware fit scoring,
  and the model catalog, as well as related Cookbook routes, front-end files,
  and scripts.
- **[Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)** by
  **Alibaba-NLP / Tongyi Lab** — the multi-step deep-research agent pipeline.
  Copyright © Alibaba-NLP / Tongyi Lab. **Apache-2.0.** Adapted for
  Odysseus's Deep Research feature in `services/research/`,
  `src/research_handler.py`, `routes/research_routes.py`, and
  `services/search/`. The full license text is available at
  [`licenses/DeepResearch-Apache-2.0.txt`](licenses/DeepResearch-Apache-2.0.txt).

---

## Bundled via Docker Compose

These services are pulled as images by the project's Docker Compose
configuration and run alongside Odysseus. They are composed with the
application rather than maintained as part of its source code.

| Service | Image | Purpose | License |
|---|---|---|---|
| [SearXNG](https://github.com/searxng/searxng) | `searxng/searxng:2026.5.31-7159b8aed` or the tag specified in the current Compose file | Default metasearch backend | AGPL-3.0 |
| [ChromaDB](https://github.com/chroma-core/chroma) | `chromadb/chroma:latest` | Vector store for memory and retrieval-augmented generation | Apache-2.0 |
| [ntfy](https://github.com/binwiederhier/ntfy) | `binwiederhier/ntfy` | Push notifications and self-hosted reminders | Apache-2.0 / GPL-2.0 |

---

## Bundled front-end libraries

The following libraries are vendored in `static/lib/` and served directly by
the application.

| Library | Purpose | License |
|---|---|---|
| [highlight.js](https://github.com/highlightjs/highlight.js) v11.9.0 | Code syntax highlighting | BSD-3-Clause |
| [SheetJS / xlsx](https://github.com/SheetJS/sheetjs) (`xlsx.full.min.js`) | Spreadsheet `.xlsx` reading and writing | Apache-2.0 |
| [docx](https://github.com/dolanmiu/docx) (`docx.umd.min.js`) | Generate `.docx` documents | MIT |
| [mammoth.js](https://github.com/mwilliamson/mammoth.js) | Convert `.docx` files to HTML | BSD-2-Clause |
| [html2pdf.js](https://github.com/eKoopmans/html2pdf.js) | HTML-to-PDF export; bundles jsPDF and html2canvas | MIT |
| [jsPDF](https://github.com/parallax/jsPDF) | PDF generation; bundled through html2pdf.js | MIT |
| [html2canvas](https://github.com/niklasvh/html2canvas) | DOM-to-canvas rasterization; bundled through html2pdf.js | MIT |
| [node-qrcode](https://github.com/soldair/node-qrcode) (`qrcode.min.js`) | QR-code rendering for two-factor authentication setup | MIT |

---

## Front-end libraries loaded at runtime

The following libraries are referenced from services such as
`cdn.jsdelivr.net` or `cdnjs.cloudflare.com` at runtime and are not vendored
in this repository.

| Library | Purpose | License |
|---|---|---|
| [KaTeX](https://github.com/KaTeX/KaTeX) 0.16.22 | Mathematical typesetting | MIT |
| [Mermaid](https://github.com/mermaid-js/mermaid) 11 | Diagrams generated from text | MIT |
| [Pyodide](https://github.com/pyodide/pyodide) 0.27.5 | In-browser Python runtime | MPL-2.0 |
| [PDFObject](https://github.com/pipwerks/PDFObject) 2.1.1 | Inline PDF embedding | MIT |

---

## Fonts

The following fonts are bundled in `static/fonts/`.

| Font | License | Author |
|---|---|---|
| [Fira Code](https://github.com/tonsky/FiraCode) | SIL Open Font License 1.1 | Nikita Prokopov and contributors |
| [Inter](https://github.com/rsms/inter) | SIL Open Font License 1.1 | Rasmus Andersson |
| [GohuFont](https://font.gohu.org/) (`fonts/custom/GohuFont.ttf`) | WTFPL | Hugo Chargois |
| [OpenDyslexic](https://opendyslexic.org/) (`fonts/OpenDyslexic-{Regular,Bold}.woff2`) | SIL Open Font License 1.1 | Abbie Gonzalez |

The OpenDyslexic license text is available at
[`licenses/OpenDyslexic-OFL.txt`](licenses/OpenDyslexic-OFL.txt).

---

## Python dependencies

The following packages are included through `requirements.txt`,
`requirements-optional.txt`, or related dependency configuration.

| Package | License |
|---|---|
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| python-multipart | Apache-2.0 |
| python-dotenv | BSD-3-Clause |
| HTTPX | BSD-3-Clause |
| Pydantic / pydantic-settings | MIT |
| SQLAlchemy | MIT |
| pypdf | BSD-3-Clause |
| BeautifulSoup4 | MIT |
| charset-normalizer | MIT |
| NumPy | BSD-3-Clause |
| ChromaDB / chromadb-client | Apache-2.0 |
| fastembed | Apache-2.0 |
| youtube-transcript-api | MIT |
| markdown | BSD-3-Clause |
| icalendar | BSD-2-Clause |
| caldav | GPL-3.0-or-later OR Apache-2.0; used under Apache-2.0 |
| cryptography | Apache-2.0 / BSD-3-Clause |
| bcrypt | Apache-2.0 |
| MCP — Model Context Protocol SDK | MIT |
| pyotp | MIT |
| qrcode\[pil] | BSD-3-Clause |
| croniter | MIT |
| pytest / pytest-asyncio | MIT / Apache-2.0 |
| duckduckgo-search, optional | MIT |
| markitdown, optional Office and EPUB text extraction | MIT |
| PyMuPDF, optional PDF form-filling support | AGPL-3.0 |

Dependency versions and the packages included by default may change over time.
The repository's current dependency files remain the authoritative source for
what is installed.

---

## Companion services

Odysseus can communicate with the following projects through local processes,
command-line tools, network services, or APIs. They are not necessarily
distributed as part of this repository, and their respective licenses continue
to apply to those projects.

- [Ollama](https://github.com/ollama/ollama) — local model serving, MIT.
- [Radicale](https://github.com/Kozea/Radicale) — CardDAV and CalDAV server,
  GPL-3.0.
- [Dovecot](https://www.dovecot.org/) — IMAP server.
- [isync / mbsync](https://isync.sourceforge.io/) — IMAP mailbox
  synchronization, GPL-2.0.
- [tmux](https://github.com/tmux/tmux) — terminal multiplexer used by some
  Linux and macOS Cookbook workflows, ISC.
- [OpenSSH](https://www.openssh.com/) — tools such as `ssh`, `ssh-keygen`, and
  `ssh-copy-id`, used when connecting to or managing remote model servers;
  BSD-style permissive license.
- Model and API providers such as Anthropic, OpenAI, Google Gemini, and
  DuckDuckGo remain governed by their respective licenses and terms of
  service.

---

## License-compatibility notes

This repository is distributed under the **GNU Affero General Public License
version 3**.

The dependencies and adapted components listed above use licenses that are
compatible with their inclusion in or use alongside this AGPL-3.0 project,
subject to their respective copyright, notice, attribution, redistribution,
and source-availability requirements.

- PDF text extraction uses **pypdf**, which is licensed under BSD-3-Clause.
- Encoding detection uses **charset-normalizer**, which is licensed under MIT.
- **PyMuPDF** is an optional dependency used by the PDF form-filling feature,
  including `src/pdf_forms.py` and the relevant document routes. It is loaded
  only when that feature is used and is licensed under AGPL-3.0. Its inclusion
  does not change this repository's existing AGPL-3.0 license, but deployments
  must continue to comply with applicable source-availability obligations.
  Artifex also offers a separate commercial PyMuPDF license.
- The Python **caldav** library is dual-licensed under
  GPL-3.0-or-later or Apache-2.0. Odysseus uses it under Apache-2.0.
- **markitdown** is an optional MIT-licensed dependency used for local Office
  and EPUB text extraction through `src/markitdown_runtime.py`. The core
  application can run without it.
- Optional or external services may carry additional obligations of their own.
  Consult their original license files and documentation before redistributing
  them or deploying modified versions.

This acknowledgment file is informational and does not replace the complete
license texts included in this repository.

---

## Fork attribution

This repository is a personal modified version of the original
[Odysseus project](https://github.com/pewdiepie-archdaemon/odysseus).

The original authors and contributors retain copyright in their respective
work. Fork-specific modifications are maintained by **Devjit Das**.

This repository is not the official Odysseus project. General-purpose
contributions and upstream security reports should be directed to the original
repository. Fork-specific information is available in
[`NOTICE.md`](NOTICE.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), and
[`SECURITY.md`](SECURITY.md).

---

## Thanks to

Much of Odysseus has been developed with assistance from AI models as well as
human contributors, testers, and open-source maintainers.

Acknowledgment is given to:

- **gpt-oss-120b**
- **Qwen3-235B**
- **DeepSeek V3.1**
- **DeepSeek V4 Pro**
- **DeepSeek V4 Flash**
- **Claude**, by Anthropic
- **Codex**, by OpenAI
- Friends and collaborators who helped test, review, and debug the project
- The maintainers and contributors of every open-source project listed above
