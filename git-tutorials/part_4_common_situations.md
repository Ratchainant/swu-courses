# Part 4 — Common Situations

This part covers real problems you will encounter — not just in theory, but in actual day-to-day work. Each situation explains **what happened**, **why it happened**, and **exactly how to fix it**.

---

## Situation 1: Working Across Multiple Machines

**Scenario:** You worked on your laptop at home and pushed your changes to GitHub. Now you're at university using a different computer. The local copy there is out of date.

**What to do:** Always pull before you start working on a different machine.

```bash
git pull origin main
```

This downloads the latest commits from GitHub and merges them into your local branch.

> **Rule:** Treat GitHub as the single source of truth. Every session on any machine should start with `git pull`.

---

## Situation 2: Push Rejected — Remote Has Changes You Don't Have

**Error message:**
```
! [rejected]        main -> main (fetch first)
error: failed to push some refs to 'https://github.com/...'
hint: Updates were rejected because the remote contains work that you do
hint: not have locally.
```

**What happened:** Someone (or you, on another machine) pushed new commits to GitHub after your last pull. Your local branch and GitHub have diverged — they each have commits the other doesn't know about.

**Fix:**

```bash
git pull --no-rebase origin main
git push origin main
```

The `--no-rebase` flag tells Git to merge the remote changes into your local branch (rather than requiring you to specify a strategy manually). After the merge, you can push.

> **Why does this happen?** Git refuses to push because it would overwrite the commits on GitHub. Pulling first brings your local copy up to date, then the push succeeds cleanly.

---

## Situation 3: Divergent Branches — Git Asks How to Reconcile

**Error message:**
```
hint: You have divergent branches and need to specify how to reconcile them.
hint:   git config pull.rebase false  # merge
hint:   git config pull.rebase true   # rebase
hint:   git config pull.ff only       # fast-forward only
fatal: Need to specify how to reconcile divergent branches.
```

**What happened:** Your local branch and the remote branch have both moved forward independently — each has commits the other doesn't have. Git doesn't know your preferred strategy for combining them.

**Fix (recommended for beginners):**

Set a global default so Git never asks this again:

```bash
git config --global pull.rebase false
```

Then pull normally:

```bash
git pull origin main
```

This tells Git to always use **merge** when pulling — the simplest and safest strategy.

**What the three options mean:**

| Option | What it does | When to use |
|--------|--------------|-------------|
| `pull.rebase false` | Creates a merge commit combining both histories | **Recommended for beginners** — safe and clear |
| `pull.rebase true` | Replays your commits on top of the remote commits — cleaner history but more complex | Experienced users who prefer linear history |
| `pull.ff only` | Only pulls if no merge is needed (fast-forward) — fails if branches have diverged | Strict teams where divergence should be an explicit error |

---

## Situation 4: Stuck in Vim During a Merge

**What happened:** You ran `git pull` and Git opened a text editor to ask you to write a merge commit message. The editor that opened is **vim**, which has unintuitive controls.

**Your screen looks like:**
```
Merge branch 'main' of https://github.com/...
# Please enter a commit message to explain why this merge is necessary,
~
~
```

**How to exit and save:**

1. Press `Escape` (ensure you are not in insert mode)
2. Type `:wq`
3. Press `Enter`

The merge completes and your terminal returns to normal.

**To avoid this in the future**, switch to a friendlier editor (see Part 2 — Step 5), or use `--no-rebase` with a pre-configured `pull.rebase false` setting so Git uses a default message automatically.

---

## Situation 5: Merge Conflict

**What happened:** You and someone else (or you on two machines) edited the **same lines** of the same file. Git merged what it could but couldn't decide which version of those lines to keep — it needs you to decide.

**Git will tell you:**
```
Auto-merging notes.md
CONFLICT (content): Merge conflict in notes.md
Automatic merge failed; fix conflicts and then commit the result.
```

**Inside the conflicted file, Git marks the clash like this:**

```
<<<<<<< HEAD
This is the version from your local branch.
=======
This is the version from the remote branch (GitHub).
>>>>>>> origin/main
```

**How to resolve it:**

1. Open the conflicted file in VS Code
2. Find all the `<<<<<<<`, `=======`, `>>>>>>>` markers
3. Edit the file to keep the version you want (or combine both)
4. Delete all the conflict markers
5. Save the file
6. Stage and commit:

```bash
git add notes.md
git commit -m "Resolve merge conflict in notes.md"
```

**VS Code tip:** VS Code highlights conflict markers with buttons to **Accept Current Change**, **Accept Incoming Change**, or **Accept Both Changes** — making it easy to resolve without editing manually.

---

## Situation 6: Ignoring Files with .gitignore

**What happened:** `git status` shows files you never want to commit — like `.DS_Store` (a macOS system file), build outputs, or files containing passwords.

**Fix:** Create a `.gitignore` file in the root of your repository:

```bash
touch .gitignore
```

Open it and add one pattern per line:

```
# macOS system files
.DS_Store
.AppleDouble

# Editor files
.vscode/
*.swp

# Python
__pycache__/
*.pyc
.env

# Node.js
node_modules/
```

Git will now completely ignore any file matching these patterns.

**If Git is already tracking a file you want to ignore:**

```bash
git rm --cached filename
```

This removes it from Git's tracking without deleting the actual file. Then add it to `.gitignore` and commit.

> **Commit your `.gitignore` file** — push it to GitHub so the same rules apply on every machine.

---

## Situation 7: You Committed to the Wrong Branch

**What happened:** You made commits on `main` but they should have been on a feature branch.

**Fix (if you have not pushed yet):**

1. Create the correct branch pointing to your current commits:

```bash
git branch feature/my-work
```

2. Reset `main` back to where it should be (the last commit before your mistake):

```bash
git reset --hard origin/main
```

3. Switch to the new branch:

```bash
git checkout feature/my-work
```

Your commits are now on the correct branch and `main` is clean.

> ⚠️ `git reset --hard` discards uncommitted changes permanently. Only run it if your commits are safely on the new branch first.

---

## Situation 8: You Want to See What Changed Before Pulling

Sometimes you want to check what's on GitHub before merging it into your work.

```bash
git fetch origin
git log HEAD..origin/main --oneline
```

This shows all commits on GitHub that you don't have locally — without changing any of your files. Once you're comfortable, run `git pull` to apply them.

---

## Summary

| Situation | Command(s) to use |
|-----------|-------------------|
| Start work on a new machine | `git pull origin main` |
| Push rejected (remote has new commits) | `git pull --no-rebase origin main` → `git push origin main` |
| Divergent branches (Git asks for strategy) | `git config --global pull.rebase false` |
| Stuck in vim | `Escape` → `:wq` → `Enter` |
| Merge conflict | Edit file, remove markers → `git add` → `git commit` |
| Ignore a file | Add to `.gitignore` → `git rm --cached <file>` if already tracked |
| Committed to wrong branch | `git branch <new>` → `git reset --hard origin/main` → `git checkout <new>` |
| Preview remote changes before pulling | `git fetch origin` → `git log HEAD..origin/main --oneline` |

---

**Previous:** [Part 3 — Solo Workflow](part_3_solo_workflow.md) | **Next:** [Part 5 — Collaboration](part_5_collaboration.md)
