# Part 3 — Solo Workflow

This part covers the **everyday commands** you will use when working on a personal project. By the end, you can create a repository, track your changes, and sync them with GitHub.

---

## The Core Loop

Every working session with Git follows the same pattern:

```
1. Pull latest changes from GitHub       (git pull)
2. Edit your files
3. Stage the changes you want to save    (git add)
4. Commit with a descriptive message     (git commit)
5. Push your commits to GitHub           (git push)
```

This loop keeps your local machine and GitHub in sync.

---

## Starting a Project

You have two ways to start: create a new repo from scratch, or download an existing one from GitHub.

### Option A: Create a new repository (starting fresh)

**1. Create the repo on GitHub first**

1. Log in to GitHub
2. Click the **+** icon → **New repository**
3. Give it a name (e.g., `my-project`)
4. Choose **Public** or **Private**
5. Check **Add a README file**
6. Click **Create repository**

**2. Clone it to your machine**

```bash
git clone https://github.com/your-username/my-project.git
cd my-project
```

Now you have a local copy connected to GitHub.

---

### Option B: Clone an existing repository

If the repo already exists on GitHub (your own or someone else's):

```bash
git clone https://github.com/username/repository-name.git
cd repository-name
```

---

## Checking Status

Before doing anything, always check the current state of your repository:

```bash
git status
```

This shows you:
- Which files have been **modified** but not staged
- Which files are **staged** and ready to commit
- Which files are **untracked** (new files Git doesn't know about yet)

**Example output:**

```
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  modified:   notes.md

Untracked files:
  new_file.txt
```

> **Habit:** Run `git status` before and after every action. It keeps you oriented.

---

## Staging Changes

After editing files, you need to **stage** them before committing.

### Stage a specific file

```bash
git add filename.md
```

### Stage multiple files

```bash
git add file1.md file2.md
```

### Stage all changed files at once

```bash
git add .
```

> The `.` means "everything in the current directory and below."

### Check what is staged

```bash
git status
```

Staged files appear under **"Changes to be committed"** in green.

---

## Committing

A commit saves a permanent snapshot of everything in the staging area.

```bash
git commit -m "your message here"
```

### Writing good commit messages

A good message completes the sentence: *"This commit will..."*

| Good | Bad |
|------|-----|
| `Add notation breakdown for Markov property` | `update` |
| `Fix typo in introduction` | `fix` |
| `Add Part 3 to git tutorial` | `stuff` |
| `Remove .DS_Store from tracking` | `changes` |

> **Rule of thumb:** Be specific enough that someone (including future you) can understand what changed without opening the files.

---

## Viewing History

See a log of all past commits:

```bash
git log
```

For a compact one-line view:

```bash
git log --oneline
```

**Example output:**

```
a3f92c1 Add notation breakdown for Markov property
ed4f1a3 Add week 1 RL and MDP notes
b72c310 Initial commit
```

Each line shows the **short commit ID** and the **commit message**.

---

## Pushing to GitHub

Upload your local commits to GitHub:

```bash
git push origin main
```

- `origin` — the remote repository (GitHub)
- `main` — the branch you are pushing to

After pushing, your changes are visible on GitHub.

---

## Pulling from GitHub

Download and merge the latest changes from GitHub into your local branch:

```bash
git pull origin main
```

> **Always pull before you start working**, especially if you work on more than one machine or share a repo with others. This prevents conflicts.

---

## Seeing What Changed

Before staging, you can inspect exactly what lines were added or removed:

```bash
git diff
```

This shows the **unstaged** changes. Lines starting with `+` were added; lines starting with `-` were removed.

To see changes that are **already staged**:

```bash
git diff --staged
```

---

## Undoing Things

### Unstage a file (remove from staging area, keep the edit)

```bash
git restore --staged filename.md
```

### Discard edits in a file (revert to last commit)

```bash
git restore filename.md
```

> ⚠️ This permanently discards your uncommitted changes to that file. There is no undo.

### Amend the last commit message (before pushing)

```bash
git commit --amend -m "corrected message"
```

> Only do this if you have **not yet pushed** the commit to GitHub.

---

## A Complete Worked Example

You are working on a class assignment saved in a GitHub repository.

```bash
# 1. Start your session — get the latest version
git pull origin main

# 2. Edit your file (e.g., add a new section to assignment.md)
# ... make your changes in VS Code ...

# 3. Check what changed
git status

# 4. Stage the file
git add assignment.md

# 5. Commit with a clear message
git commit -m "Add analysis section to assignment 1"

# 6. Push to GitHub
git push origin main
```

Your work is now saved locally, committed to history, and backed up on GitHub.

---

## Summary

| Command | What it does |
|---------|--------------|
| `git clone <url>` | Download a repo from GitHub to your machine |
| `git status` | Show the state of your working directory and staging area |
| `git add <file>` | Stage a specific file for the next commit |
| `git add .` | Stage all changed files |
| `git commit -m "msg"` | Save a snapshot with a message |
| `git log --oneline` | View commit history in compact form |
| `git push origin main` | Upload commits to GitHub |
| `git pull origin main` | Download and merge commits from GitHub |
| `git diff` | Show unstaged changes line by line |
| `git diff --staged` | Show staged changes line by line |
| `git restore --staged <file>` | Unstage a file |
| `git restore <file>` | Discard uncommitted edits in a file |

---

**Previous:** [Part 2 — Setup](part_2_setup.md) | **Next:** [Part 4 — Common Situations](part_4_common_situations.md)
