# Git Tutorials

A practical guide to Git and GitHub for students — from zero to confident collaboration.

---

## What is this tutorial?

This tutorial teaches you how to use **Git** (a version control system) and **GitHub** (a cloud platform for storing and sharing code). You will learn how to track changes in your work, back it up online, and collaborate with others — skills used in every professional software team.

No prior experience is required.

---

## How to use this tutorial

Each part is a standalone document. Start from Part 1 if you are new to Git, or jump to any section you need.

| Part | File | What you will learn |
|------|------|---------------------|
| 1 | [Part 1: Concepts](part_1_concepts.md) | Key Git terminology and mental models |
| 2 | [Part 2: Setup](part_2_setup.md) | Installing Git and connecting to GitHub |
| 3 | [Part 3: Solo Workflow](part_3_solo_workflow.md) | The everyday commands for saving and syncing your work |
| 4 | [Part 4: Common Situation](part_4_common_situations.md) | Handling real-world problems: multiple machines, merge conflicts, and .gitignore |
| 5 | [Part 5: Collaboration](part_5_collaboration.md) | Working with others using branches and pull requests |

---

## Prerequisites

- A computer running macOS or Windows
- A free [GitHub account](https://github.com)
- No programming experience required

---

## Part Summaries

### Part 1 — Concepts
Before touching any commands, you need the right mental model. This part explains what Git actually is, why it exists, and defines every key term (repository, commit, branch, merge, and more) with everyday analogies.

### Part 2 — Setup
A one-time setup guide. You will install Git, configure your identity, and authenticate with GitHub so your machine can push and pull from the cloud.

### Part 3 — Solo Workflow
The core daily loop: make changes → stage → commit → push. You will also learn how to pull updates and check your repository status. By the end of this part, you can fully manage a personal project.

### Part 4 — Common Situations
Real problems you will encounter: switching between machines, divergent branches, merge conflicts, and ignoring files you never want to track. Each situation is explained with the exact commands to resolve it.

### Part 5 — Collaboration
How to work with others without overwriting each other's work. This part covers branches, merging, and pull requests on GitHub — the foundation of team-based development.

---

## Quick Reference

If you already know the basics and just need a command reminder:

```bash
git status                        # see what has changed
git add <file>                    # stage a file
git commit -m "your message"      # save a snapshot
git push origin main              # upload to GitHub
git pull origin main              # download from GitHub
git log --oneline                 # view commit history
```
