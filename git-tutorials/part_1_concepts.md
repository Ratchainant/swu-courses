# Part 1 — Concepts

Before running any commands, you need the right mental model. This part explains **what Git is**, **why it exists**, and **what every key term means** — with everyday analogies so the ideas stick.

---

## What is Git?

**Git** is a **version control system** — a tool that tracks every change you make to your files over time.

Think of it like this:

> Imagine writing a research paper. Every time you make a major edit, you save a copy:
> - `essay.docx`
> - `essay_v2.docx`
> - `essay_final.docx`
> - `essay_final_REAL.docx`
> - `essay_final_REAL_v2.docx`
>
> This gets messy fast. Git replaces this chaos with a clean, automatic history of every version — with timestamps, descriptions, and the ability to go back to any point.

Git works **locally on your computer**. It does not require the internet.

---

## What is GitHub?

**GitHub** is a website that hosts your Git repositories online.

| | Git | GitHub |
|---|---|---|
| What it is | Software on your computer | A website (cloud service) |
| What it does | Tracks history of your files locally | Stores your repository online so you can access it anywhere and share with others |
| Requires internet? | No | Yes |

**Analogy:** Git is like writing in a notebook. GitHub is like scanning your notebook and storing it in Google Drive so you can access it from any device.

---

## Core Concepts

### Repository (Repo)

A **repository** is a project folder that Git is tracking. It contains all your files plus the full history of every change ever made.

> **Analogy:** A repository is like a project binder — but one that remembers every version of every page you've ever put in it.

There are two types:
- **Local repository** — the copy on your own computer
- **Remote repository** — the copy stored on GitHub

---

### Commit

A **commit** is a saved snapshot of your project at a specific point in time. Every commit has:
- A **unique ID** (a short code like `a3f92c1`)
- A **timestamp**
- A **message** you write to describe what changed
- The **author's name**

> **Analogy:** A commit is like pressing a checkpoint in a video game. If something goes wrong later, you can always load back to any checkpoint.

---

### Staging Area (Index)

Before you commit, you must **stage** the files you want to include. The staging area is a preparation zone between your working files and the commit.

> **Analogy:** Think of packing a box to ship. Your room is full of items (working directory). You pick specific items and put them in the box (staging area). Only when you seal the box does it become a permanent record (commit).

This lets you commit only part of your changes at once, giving you fine-grained control.

```
Working Directory  →  Staging Area  →  Commit (history)
  (your edits)         (git add)        (git commit)
```

---

### Working Directory

The **working directory** is simply the folder on your computer where your files live — what you see in your file explorer. Any edits you make to files happen here first, before staging or committing.

---

### Branch

A **branch** is an independent line of development — a parallel version of your project that diverges from the main line.

> **Analogy:** Imagine writing a short story. You want to try two different endings without ruining the original. You make a copy, experiment on it, and if you like it, merge it back. Branches work exactly this way.

Branches let you:
- Work on a new feature without breaking the working version
- Try risky experiments safely
- Have multiple people work on different things at the same time

---

### Main / Master

**main** (or **master** in older repos) is the default branch — the "official" published version of your project.

> **Analogy:** Think of `main` as the published edition of a book. Branches are the draft manuscripts being revised before they make it into the next edition.

New repositories on GitHub default to a branch called `main`.

---

### HEAD

**HEAD** is a pointer that tells Git where you currently are in the history.

> **Analogy:** HEAD is a bookmark in a book. It shows you which page (commit or branch) you're currently on.

Normally HEAD points to the latest commit of your current branch. When you switch branches, HEAD moves with you.

---

### Clone

**Cloning** means downloading a full copy of a remote repository (from GitHub) to your local machine — including all files and the entire history.

> **Analogy:** Cloning is like downloading a complete project ZIP, but smarter — it also brings the full version history and knows where it came from, so you can sync changes later.

You only clone once. After that, you use `pull` to get updates.

---

### Fork

A **fork** is your personal copy of someone else's repository on GitHub. It lives under your GitHub account.

> **Analogy:** Forking is like photocopying a classmate's notebook. You get your own copy to write in, and your changes don't affect their original.

Used when you want to contribute to a project you don't own, or experiment with someone else's code.

---

## Sync Operations

### Push

**Push** uploads your local commits to the remote repository on GitHub.

> **Analogy:** Push is like syncing your phone photos to the cloud. Your local work gets backed up and becomes accessible from anywhere.

```
Local repo  ──push──▶  Remote repo (GitHub)
```

---

### Pull

**Pull** downloads the latest commits from GitHub and merges them into your local branch.

> **Analogy:** Pull is checking your email and downloading new messages. If someone pushed changes to GitHub, pulling gets those changes onto your machine.

```
Remote repo (GitHub)  ──pull──▶  Local repo
```

---

### Fetch

**Fetch** downloads new changes from GitHub but does **not** merge them into your current branch. It lets you inspect what's new before integrating.

> **Analogy:** Fetch is like previewing email subject lines before opening them. You can see what arrived without it affecting your inbox yet.

`git pull` = `git fetch` + `git merge` combined.

---

### Merge

**Merge** combines the changes from two branches into one.

> **Analogy:** Two students each edited a different chapter of a shared document. Merging combines both their edits into a single complete document.

Git handles most merges automatically. If two people edited the **same line** differently, Git can't decide — this creates a **merge conflict** (see below).

---

### Merge Conflict

A **merge conflict** happens when two commits change the same part of the same file in incompatible ways, and Git needs a human to decide which version to keep.

> **Analogy:** Two editors both rewrote the same paragraph of a book differently. The publisher has to read both versions and decide which one (or what combination) to use.

Git marks the conflicting sections in the file so you can see both versions and choose.

---

### Rebase

**Rebase** moves your commits to start from a different point in history, replaying them on top of another branch. The result is a cleaner, linear history.

> **Analogy:** Imagine your work diary and your colleague's are slightly out of order. Rebase rewrites your entries so they appear after all of theirs — cleaner to read, but the dates get adjusted.

> ⚠️ Rebase rewrites history. Avoid rebasing commits that have already been pushed to GitHub and shared with others.

---

## Collaboration Terms

### Pull Request (PR)

A **pull request** is a proposal to merge your branch into another branch (usually `main`), submitted through GitHub. It lets teammates review your changes, leave comments, and approve before anything is merged.

> **Analogy:** A pull request is like submitting a draft report to your professor for feedback before it becomes the final submission.

Despite the name, a pull request is not the same as `git pull` — it is a GitHub feature for code review and discussion.

---

### Origin

**Origin** is the default name Git gives to the remote repository your local repo was cloned from (usually your GitHub repo). It's just a shortcut name so you don't have to type the full URL every time.

```bash
git push origin main   # push to the remote named "origin", branch "main"
```

---

### .gitignore

A **.gitignore** file is a list of files and folders that Git should completely ignore — never track, never commit.

> **Analogy:** It's a "do not pack" list when moving house. Some things (system files, passwords, build outputs) should never go into the box.

Common entries:
```
.DS_Store          # macOS system file
node_modules/      # JavaScript dependencies (large, auto-generated)
.env               # environment variables and secrets
```

---

### Tag

A **tag** is a permanent, named label attached to a specific commit — usually used to mark release versions.

> **Analogy:** A tag is like stamping "FINAL SUBMISSION" on a specific printed copy of your report. It doesn't change the document, just marks it as significant.

```bash
git tag v1.0
```

---

### Stash

**Stash** temporarily saves your uncommitted changes without committing them, so you can switch tasks and come back later.

> **Analogy:** Stash is like putting your current work in a drawer so your desk is clean for something urgent. When you're done, you open the drawer and pick up exactly where you left off.

```bash
git stash        # save current changes
git stash pop    # restore them later
```

---

## Summary

| Term | One-line definition |
|------|---------------------|
| **Repository** | A project folder tracked by Git, with full history |
| **Commit** | A saved snapshot of your project at a point in time |
| **Staging area** | A preparation zone for choosing what goes into the next commit |
| **Working directory** | The actual files on your computer that you edit |
| **Branch** | A parallel version of your project for safe experimentation |
| **Main** | The default official branch |
| **HEAD** | A pointer to your current position in history |
| **Clone** | Download a full remote repo to your machine |
| **Fork** | Your personal copy of someone else's GitHub repo |
| **Push** | Upload local commits to GitHub |
| **Pull** | Download remote commits and merge them locally |
| **Fetch** | Download remote changes without merging |
| **Merge** | Combine two branches into one |
| **Merge conflict** | A clash that requires a human to resolve |
| **Rebase** | Replay commits on top of another branch for cleaner history |
| **Pull Request** | A GitHub proposal to merge a branch, with review |
| **Origin** | The default name for the remote repository |
| **.gitignore** | A file listing paths Git should never track |
| **Tag** | A named label on a specific commit, used for versioning |
| **Stash** | Temporarily shelve uncommitted changes |

---

**Next:** [Part 2 — Setup](part_2_setup.md)
