---
name: add-paper
description: Add a paper, technical report, or tool to the awesome-on-policy-distillation list (a curated collection of on-policy distillation (OPD) methods for LLMs). Use this skill whenever the user provides an arxiv URL, paper link, or asks to add a paper/report/tool to the list. Also trigger when the user pastes a URL and says something like "add this", "include this paper", or "put this in the list".
---

# Add a Paper to Awesome On-Policy Distillation

This skill adds one entry to `README.md` — the only file it edits. It operationalizes `CONTRIBUTING.md` for an agent. Treat the repo's own files as the source of truth: `CONTRIBUTING.md` owns the inclusion criteria and entry format, and the `README.md` headers own the section structure. If this skill ever disagrees with those files, follow the files and tell the user the skill is stale.

## Workflow

### 1. Check for duplicates first

Before fetching or evaluating anything, search `README.md` for the candidate: its arxiv ID, the normalized `arxiv.org/abs/<id>` URL, the title, and any likely short name. If it is already listed, cite the existing entry (section and line) and stop — do not add a duplicate.

### 2. Read the source

Fetch the URL to get the title, year, and what the work contributes. For arxiv, fetch `https://arxiv.org/abs/<id>`; for a GitHub repo or HuggingFace space, fetch that page instead. Read the abstract. When the section fit or the one-line description is not obvious from the abstract, read the method section — `CONTRIBUTING.md` requires it for borderline inclusion calls. Use an arxiv-reader-style tool if your environment has one; otherwise read the PDF or source directly.

### 3. Confirm it qualifies

Verify the work passes the inclusion criteria in `CONTRIBUTING.md` — student rollouts central to the learning signal, or work that directly enables OPD deployment — and that it does not hit an exclusion (near-policy is not on-policy, self-play is not distillation, survey classification or reframing alone, or a domain application with no transferable method). If fit is unclear, tell the user why and ask before adding.

### 4. Choose the section from the live structure

Do not rely on a hardcoded section list — the README is reorganized periodically, and a baked-in map goes stale. Read the current structure with `grep -n '^## ' README.md` and `grep -n '^### ' README.md`, then place the entry under the subsection that fits. Routing heuristics:

- Survey, position paper, blog post, essay, or walkthrough → **Surveys and Essays**.
- New OPD method where student rollouts are the central contribution and it would be cited as a defining work → **Core OPD Papers** (deliberately selective; when unsure, prefer Adjacent).
- Cross-tokenizer enablers, mismatch mitigation, preference / reward-guided / hybrid RL+KD, self-play and iterative bootstrapping, or historical precursors → **Adjacent and Enabling Work**.
- Agent, multimodal, speech, embodied, or speculative-decoding (draft-model) extensions → **Domain Extensions**.
- Production system using OPD as a post-training stage → **Technical Reports and Industrial Recipes** (table format).
- Training framework or implementation / code repo → **Frameworks and Implementations** (Training Frameworks is a table; Implementations is bullets).

If you cannot place it, default to **Adjacent and Enabling Work** — the core section is intentionally selective.

### 5. Write the entry in the house format

Follow the entry format in `CONTRIBUTING.md`. The rules that break most often if missed:

- Bullet entries: `- [Full Paper Title](url) *(Year)* — One-line description.` The year is **italic**.
- If there is an associated code release, append the badge: `[![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](repo-url)`.
- Implementation repos under `### Implementations` **omit the year**: `- [Repo Name](url) — Description.`
- Tables (Technical Reports, Training Frameworks): add one row matching the existing columns exactly — do not invent headers.
- Description: one sentence, **hard cap 22 words** (well-written entries average 12–16), stating **one mechanism phrase plus one differentiator**. Forbidden: equation symbols, Greek letters, benchmark numbers, hyperparameters, model/dataset enumerations, ablation lists, percentage-point gains, and any second mechanism clause introduced by a semicolon or "and." Run the deletion test before committing: at each comma, semicolon, or "and," try deleting the trailing clause; if the entry still tells a scanning reader why to click, that clause was bloat.

### 6. Place within the subsection

Insert near existing entries of the same topic or year without reordering anything. If unsure, append to the end of the subsection. Never reorder existing entries as part of an add.

## Taxonomy tables

Update the `## Taxonomy` tables (`By Teacher Type`, `By Primary Goal`) only for Core OPD papers or a paper that clearly extends an existing row. Read the current rows first, then add the short name as a markdown link reusing the paper's URL — `[ShortName](url)` — comma-separated within the row. New rows are rare. Skip taxonomy updates for surveys, essays, frameworks, implementations, and technical reports.

## Start Here reading path

The `## Start Here` reading path changes rarely. Only suggest adding a paper if it is the best introduction to a topic the path currently underrepresents — and if you do, tell the user explicitly so they can confirm.

## Table of contents

If you create a new subsection (rare), add it to the `## Contents` list at the top.

## Entry-count badge

The `Entries-N` badge is auto-maintained by the maintainer's local tooling. Leave it alone — never edit the badge number by hand — and it is refreshed automatically.

## Output

After editing, summarize:

- The duplicate-check result.
- Which inclusion criterion the work passed, and any exclusion you ruled out.
- The section and subsection you chose, and why.
- The exact entry text you added.
- Whether you updated the taxonomy or reading path, and why or why not.

If anything was ambiguous (e.g., the work could fit multiple sections), explain your reasoning so the user can adjust.
