# Part 2 — Setup

This is a one-time setup. Once complete, you will be ready to use Git and GitHub on your machine.

---

## Step 1: Create a GitHub Account

If you don't have one yet:

1. Go to [https://github.com](https://github.com)
2. Click **Sign up**
3. Choose a username, enter your **personal email**, and set a password
4. Verify your email address

> **Use your personal email, not your university email.**
> Your GitHub account belongs to you — not to your university or any future employer. People change workplaces, graduate, and lose access to institutional emails. Using a personal email ensures you always own and can access your account.

> **Tip:** Choose a professional username — you will likely share your GitHub profile with employers or instructors.

---

## Step 1.5: Apply for GitHub Education Benefits

As a SWU student, you can apply for the **GitHub Student Developer Pack** — a free bundle of professional tools normally worth hundreds of dollars, including GitHub Pro (unlimited private repositories, advanced code review tools, and more).

### What you get
- **GitHub Pro** for free while you are a student
- Free access to tools like GitHub Copilot, JetBrains IDEs, Namecheap domains, and more
- See the full list at [https://education.github.com/pack](https://education.github.com/pack)

### How to apply

1. Log in to your GitHub account
2. Go to [https://education.github.com/students](https://education.github.com/students)
3. Click **Get student benefits**
4. When asked to verify your academic status, add your SWU student email (`yourname@g.swu.ac.th`) as a secondary email on your GitHub account:
   - GitHub → **Settings** → **Emails** → **Add email address**
   - Verify it via the confirmation email sent to your SWU inbox
5. Return to the application and select your SWU email as proof of enrollment
6. Enter your school name: **Srinakharinwirot University**
7. Submit — approval usually takes a few hours to a few days

> Your personal email remains your primary login. The SWU email is added only as proof of student status and can be removed after graduation.

---

## Step 2: Install Git

### macOS

Open **Terminal** and run:

```bash
git --version
```

- If a version number appears (e.g., `git version 2.39.0`), Git is already installed. Skip to Step 3.
- If a prompt appears asking to install developer tools, click **Install** and follow the instructions.

Alternatively, install via [Homebrew](https://brew.sh):

```bash
brew install git
```

### Windows

1. Download the installer from [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. Run the installer — the default options are fine for most users
3. Open **Git Bash** (installed alongside Git) to run all git commands

> On Windows, use **Git Bash** instead of Command Prompt or PowerShell for the best compatibility with commands in this tutorial.

---

## Step 3: Configure Your Identity

Git records your name and email on every commit you make. Set them once:

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

Use the same email address you registered on GitHub.

**Verify your configuration:**

```bash
git config --list
```

You should see `user.name` and `user.email` in the output.

---

## Step 4: Set the Default Branch Name

Modern GitHub uses `main` as the default branch name. Configure Git to match:

```bash
git config --global init.defaultBranch main
```

---

## Step 5: Set Your Default Editor (Optional)

When Git needs you to write a message (e.g., during a merge), it opens a text editor. The default is **vim**, which can be confusing for beginners.

To switch to a more familiar editor:

**VS Code:**
```bash
git config --global core.editor "code --wait"
```

**nano** (simpler terminal editor):
```bash
git config --global core.editor nano
```

> If you leave this as vim and ever get stuck in it, press `Escape`, then type `:wq` and press `Enter` to save and exit.

---

## Step 6: Authenticate with GitHub

When you push or pull, GitHub needs to verify it's actually you. The recommended method is **HTTPS with a Personal Access Token (PAT)**.

### Create a Personal Access Token

1. Log in to GitHub
2. Click your profile picture → **Settings**
3. Scroll down to **Developer settings** (bottom of the left sidebar)
4. Click **Personal access tokens** → **Tokens (classic)**
5. Click **Generate new token (classic)**
6. Give it a name (e.g., `my-laptop`), set an expiration, and check the **repo** scope
7. Click **Generate token**
8. **Copy the token immediately** — GitHub will never show it again

> **Treat your token like a password.** Never share it or commit it to a repository.

### Use the Token When Prompted

The first time you run `git push` or `git pull`, Git will ask for your GitHub credentials:

```
Username: your-github-username
Password: paste-your-token-here   ← use the token, NOT your GitHub password
```

### Save Credentials (so you're not asked every time)

**macOS** — use the built-in credential helper:
```bash
git config --global credential.helper osxkeychain
```

**Windows** — this is set automatically during installation.

After entering your token once, macOS Keychain will remember it.

---

## Step 7: Verify Everything Works

Test the full setup by cloning a public repository:

```bash
git clone https://github.com/Ratchainant/swu-courses.git
```

If the files download without errors, your setup is complete.

---

## Summary

| Step | What you did |
|------|--------------|
| 1 | Created a GitHub account with your personal email |
| 1.5 | Applied for GitHub Student Developer Pack using your SWU email |
| 2 | Installed Git on your machine |
| 3 | Set your name and email for commits |
| 4 | Set `main` as the default branch name |
| 5 | Set your preferred text editor |
| 6 | Created a Personal Access Token and saved credentials |
| 7 | Verified the setup with a test clone |

---

**Previous:** [Part 1 — Concepts](part_1_concepts.md) | **Next:** [Part 3 — Solo Workflow](part_3_solo_workflow.md)
