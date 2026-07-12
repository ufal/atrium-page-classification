<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" title="Python Version"></a>
  <a href="https://huggingface.co/ufal/vit-historical-page"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-vit--historical--page-yellow.svg" title="Hugging Face Model"></a>
  <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/github/license/ufal/atrium-page-classification" title="MIT License"></a>
  <a href="https://atrium-research.eu/"><img src="https://img.shields.io/badge/funded%20by-ATRIUM-8A2BE2.svg" title="ATRIUM Project"></a>
</p>

---

# ATRIUM Page Classification - Agent Skill 🤖🪧

### Goal: let coding agents classify historical document pages via a server-client skill

This branch (`agent-skill`) packages the **ATRIUM Page Classification API service** together
with a **Skill for coding agents** (Claude Code, Codex, Gemini/Antigravity). The design
follows a strict server-client split:

- **Server** 🖥️ - the FastAPI service in [`service/`](service/) runs the fine-tuned
  ViT / RegNetY / EffNetV2 models (Docker or local venv, CPU or GPU).
- **Client** 🪶 - [`scripts/atrium_classify.py`](scripts/atrium_classify.py), a
  **zero-dependency** stdlib-only script that agents call directly.
- **Skill contract** 📜 - [`SKILL.md`](SKILL.md) tells the agent when and how to use it:
  category semantics, model selection, ambiguity handling, and error playbooks.

For training, evaluation, data preparation, and full project documentation, see the
[`test`](https://github.com/ufal/atrium-page-classification/tree/test) branch - this
branch intentionally carries only what the skill needs.

### Table of contents 📑

  * [Quick start 🚀](#quick-start-)
  * [Skill installation 🔧](#skill-installation-)
    + [Claude Code](#claude-code)
    + [Codex](#codex)
    + [Google Antigravity](#google-antigravity)
  * [Server setup 🖥️](#server-setup-)
  * [Client usage 🪶](#client-usage-)
  * [Remote server / LINDAT 🌐](#remote-server--lindat-)
  * [Contacts 📧](#contacts-)

----

## Quick start 🚀

```bash
git clone -b agent-skill https://github.com/ufal/atrium-page-classification.git
cd atrium-page-classification

bash scripts/serve.sh                          # start the server (Docker or venv)
python3 scripts/atrium_classify.py page.png    # classify a page
```

> [!NOTE]
> The first server start downloads model weights from the HF 😊 hub
> (`ufal/vit-historical-page`, ~0.2-1.2 GB per revision) - be patient. ⏳

## Skill installation 🔧

### Claude Code

Clone this branch into your personal skills directory:

```bash
git clone -b agent-skill https://github.com/ufal/atrium-page-classification.git \
    ~/.claude/skills/atrium-page-classification
```

Restart Claude Code - the skill is available as `/atrium-page-classification` and is
selected automatically for page-classification requests. For a project-local install,
clone into `.claude/skills/atrium-page-classification` inside the target repository.

### Codex

```bash
git clone -b agent-skill https://github.com/ufal/atrium-page-classification.git \
    ~/.codex/skills/atrium-page-classification
```

The skill is detected automatically in the next Codex session.

### Google Antigravity

Clone the branch into your project and point `AGENTS.md` at it:

```
Use the ATRIUM page classification skill from
`atrium-page-classification/SKILL.md` for classifying historical document pages.
Start the server with `bash atrium-page-classification/scripts/serve.sh`, then run
`python3 atrium-page-classification/scripts/atrium_classify.py [FILES...]`.
```

## Server setup 🖥️

The server exposes three endpoints (see [`service/README.md`](service/README.md) for
details): `GET /info`, `POST /predict_image`, `POST /predict_document`.

```bash
bash scripts/serve.sh          # auto: Docker CPU if available, else local uvicorn
bash scripts/serve.sh --gpu    # Docker with GPU (docker-compose.gpu.yml)
bash scripts/serve.sh --local  # force local uvicorn via setup/setup_api_service.sh
```

The script is idempotent and health-waits on `/info`. Port defaults to `8000`
(`ATRIUM_PC_PORT` to change).

## Client usage 🪶

For batch work without the HTTP layer, the bundled inference-only `run.py` drives the
same models directly (this is also what the Docker image's default command wraps):

```bash
python3 run.py --hf -rev v4.3                 # download a model (no inference)
python3 run.py -f page.png --hf               # single image, top-N to stdout
python3 run.py -d scans/ --hf                 # directory → CSV in result/tables/
python3 run.py -d scans/ --best --hf          # best-5 ensemble, averaged CSV
```

Training, evaluation, and dataset tooling are intentionally **not** on this branch -
see the [`test`](https://github.com/ufal/atrium-page-classification/tree/test) branch.

Output columns: `FILE, PAGE, RANK, LABEL, SCORE`. The 11 categories 🪧 and their routing
semantics are documented in [`SKILL.md`](SKILL.md#categories-).

## Remote server / LINDAT 🌐

The client is location-agnostic: point it at any deployment with `--base-url` or

```bash
export ATRIUM_PC_URL="https://<hosted-instance>/atrium-pc"
```

A hosted LINDAT instance is planned; once available, the environment variable is the
only change needed - the skill contract and client stay identical.

## Contacts 📧

**For support write to:** lutsai.k@gmail.com responsible for the
[GitHub repository](https://github.com/ufal/atrium-page-classification)

### Acknowledgements 🙏

- **Developed by** UFAL, Charles University 👥
- **Funded by** [ATRIUM](https://atrium-page-classification) 💰
- **Related services**: the skill pattern follows
  [uwebasr-skill](https://github.com/honzas83/uwebasr-skill) by our Pilsen colleagues
  (LINDAT UWebASR).
