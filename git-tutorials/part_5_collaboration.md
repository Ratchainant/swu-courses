# Part 5 — Collaboration

This part covers how to work with others on the same repository without overwriting each other's work. The foundation is **branches** and **pull requests**.

---

## Why You Need a Different Workflow

In the solo workflow, everyone commits directly to `main`. This works for one person, but breaks down with a team:

- Two people editing the same file at the same time creates constant conflicts
- A half-finished feature gets pushed to `main` and breaks everyone's work
- There's no way to review someone's changes before they go live

The solution is a **branch-based workflow**: each person works on their own branch and only merges into `main` after review.

---

## The Collaboration Loop

```
1. Pull the latest main
2. Create a new branch for your work
3. Make commits on your branch
4. Push your branch to GitHub
5. Open a Pull Request on GitHub
6. Team reviews and approves
7. Merge the branch into main
8. Delete the branch
```

---

## Branches

### Create a new branch

```bash
git checkout -b feature/my-feature
```

This creates a new branch called `feature/my-feature` **and** switches to it immediately.

> **Naming convention:** Use short, descriptive names with hyphens.
> - `feature/add-login-page`
> - `fix/typo-in-readme`
> - `docs/update-setup-guide`

### Check which branch you are on

```bash
git branch
```

The current branch is marked with `*`.

```
* feature/my-feature
  main
```

### Switch to a different branch

```bash
git checkout main
```

Or using the newer syntax:

```bash
git switch main
```

### See all branches (including remote)

```bash
git branch -a
```

---

## The Feature Branch Workflow — Step by Step

### Step 1: Start from the latest main

```bash
git checkout main
git pull origin main
```

Always make sure your `main` is up to date before branching off it.

### Step 2: Create your branch

```bash
git checkout -b feature/add-summary-table
```

### Step 3: Do your work and commit

```bash
# edit files...
git add .
git commit -m "Add summary table to week 1 notes"
```

Make as many commits as you need on your branch. They are isolated from `main`.

### Step 4: Push your branch to GitHub

```bash
git push origin feature/add-summary-table
```

The first time you push a new branch, Git may ask you to set the upstream:

```bash
git push --set-upstream origin feature/add-summary-table
```

Or use the shorthand:

```bash
git push -u origin feature/add-summary-table
```

After this, plain `git push` works for subsequent pushes on the same branch.

### Step 5: Open a Pull Request on GitHub

1. Go to your repository on GitHub
2. You will see a yellow banner: **"Your branch had recent pushes — Compare & pull request"** — click it
3. Or go to the **Pull requests** tab → **New pull request**
4. Set:
   - **base:** `main` (the branch you want to merge INTO)
   - **compare:** `feature/add-summary-table` (your branch)
5. Write a clear title and description of what your changes do
6. Click **Create pull request**

### Step 6: Review and Approve

Team members can now:
- Read through your changes line by line
- Leave comments on specific lines
- Request changes before approving
- Approve the pull request

As the author, you can push more commits to the same branch — the PR updates automatically.

### Step 7: Merge the Pull Request

Once approved, click **Merge pull request** on GitHub, then **Confirm merge**.

GitHub gives you three merge options:

| Option | What it does | When to use |
|--------|--------------|-------------|
| **Create a merge commit** | Keeps all commits and adds a merge commit | Default — preserves full history |
| **Squash and merge** | Combines all your branch commits into one | When your branch has many small "WIP" commits |
| **Rebase and merge** | Replays commits linearly without a merge commit | Teams that prefer clean linear history |

For most student projects, **Create a merge commit** is fine.

### Step 8: Clean up the branch

After merging, delete the branch — it has served its purpose.

**On GitHub:** Click **Delete branch** after merging (GitHub shows this button automatically).

**Locally:**

```bash
git checkout main
git pull origin main
git branch -d feature/add-summary-table
```

---

## Keeping Your Branch Up to Date

If `main` moves forward while you are working on your branch (because teammates merged their PRs), you should update your branch to include those new changes.

```bash
git checkout feature/my-feature
git merge main
```

This merges the latest `main` into your branch. Resolve any conflicts (see Part 4 — Situation 5), then push.

---

## Forking and Contributing to Others' Repositories

If you want to contribute to a repository you **do not own** (e.g., a classmate's project or an open-source project):

### Step 1: Fork the repository

On GitHub, click **Fork** (top right of any repository page). This creates a copy under your account.

### Step 2: Clone your fork

```bash
git clone https://github.com/YOUR-USERNAME/repository-name.git
cd repository-name
```

### Step 3: Add the original repo as "upstream"

```bash
git remote add upstream https://github.com/ORIGINAL-OWNER/repository-name.git
```

Now you have two remotes:
- `origin` — your fork on GitHub
- `upstream` — the original repository

### Step 4: Work on a branch, push to your fork, open a PR

Work normally on a branch, push to `origin`, and open a pull request from your fork to the original repository on GitHub.

### Step 5: Keep your fork in sync with the original

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

---

## Reviewing a Pull Request Locally

Sometimes you need to check out a teammate's branch locally to test it:

```bash
git fetch origin
git checkout their-branch-name
```

You are now on their branch. Test, review, then switch back:

```bash
git checkout main
```

---

## Common Collaboration Rules

Good teams agree on conventions upfront. Here are practices used in most professional environments:

| Rule | Why |
|------|-----|
| Never commit directly to `main` | Keeps `main` always stable and deployable |
| Always work on a branch | Isolates your work from others |
| Write descriptive PR titles | Makes the history readable |
| Keep PRs small and focused | Easier to review; faster to merge |
| Pull from `main` before branching | Reduces conflicts |
| Delete branches after merging | Keeps the repo tidy |
| Review before merging | Catches bugs and improves code quality |

---

## Summary

| Command | What it does |
|---------|--------------|
| `git checkout -b <branch>` | Create and switch to a new branch |
| `git checkout <branch>` | Switch to an existing branch |
| `git branch` | List local branches |
| `git branch -a` | List all branches including remote |
| `git push -u origin <branch>` | Push a new branch to GitHub and set upstream |
| `git merge main` | Merge latest main into your current branch |
| `git branch -d <branch>` | Delete a local branch (after merging) |
| `git remote add upstream <url>` | Add the original repo as a remote (for forks) |
| `git fetch upstream` | Download changes from the original repo |

---

## Full Workflow at a Glance

```
git checkout main
git pull origin main
git checkout -b feature/your-feature

# ... do your work ...

git add .
git commit -m "Describe what you did"
git push -u origin feature/your-feature

# Open Pull Request on GitHub
# Get it reviewed and approved
# Merge on GitHub

git checkout main
git pull origin main
git branch -d feature/your-feature
```

---

**Previous:** [Part 4 — Common Situations](part_4_common_situations.md) | [Back to README](README.md)
