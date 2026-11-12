# GitHub publishing from the host terminal

This development sandbox cannot resolve github.com and masks the root .git metadata as read-only. No remote push has succeeded. A local commit bundle is available at dist/humansense-v0.1.bundle.

First inspect the normal terminal (not the restricted development shell):

```bash
cd ~/Desktop/catch-your-mind
git status --short --branch
git ls-remote https://github.com/NomaDamas/catch-your-mind.git HEAD
```

If HTTPS prompts for credentials, use your configured Git credential manager or GitHub CLI `gh auth login` followed by `gh auth setup-git` if gh is installed. GitHub account passwords do not authenticate Git HTTPS operations. Never paste a token or password into an assistant conversation. For SSH, configure your own GitHub SSH key and use your authenticated SSH remote separately.

To publish the delivered bundle while preserving existing remote history, run:

```bash
bash scripts/push-github.sh
```

Set `HUMANSENSE_BASE_BRANCH=feat/humansense-20261112-154922` to base a follow-up delivery on the previously published branch. If its tree matches an earlier bundle snapshot, the script applies only subsequent changes. Unknown existing-file conflicts stop without overwriting.

The script clones the bundle and the remote into a new temporary directory, creates a uniquely named feature branch, applies the snapshot patch without overwriting conflicts, commits it and pushes that branch. It does not change this checkout or force-push main. Authentication is handled by Git in your terminal. The temporary directory is printed and retained for review. If the remote has conflicting source files, it stops and pushes nothing; resolve those in the printed checkout before continuing. An empty remote can also be used.

The bundle represents the release snapshot, not subsequent unbundled edits. Always use the freshly regenerated bundle from the latest delivery. Opening or merging a pull request is separate from a successful branch push.
