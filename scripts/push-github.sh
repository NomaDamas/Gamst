#!/usr/bin/env bash
# Run in a normal host terminal. Never force-push or alter the current checkout.
set -euo pipefail
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
bundle_path="$project_dir/dist/humansense-v0.1.bundle"
remote_url='https://github.com/NomaDamas/catch-your-mind.git'
branch_name="feat/humansense-$(date +%Y%m%d-%H%M%S)"
if ! test -f "$bundle_path"; then
  printf '%s\n' "Missing bundle: $bundle_path" >&2
  exit 1
fi
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/humansense-publish.XXXXXX")
printf 'Preparing a separate checkout at %s\n' "$work_dir"
git clone "$bundle_path" "$work_dir/source"
if test -n "${HUMANSENSE_BASE_BRANCH:-}"; then
    git clone --branch "$HUMANSENSE_BASE_BRANCH" "$remote_url" "$work_dir/remote"
else
    git clone "$remote_url" "$work_dir/remote"
fi
git -C "$work_dir/remote" switch -c "$branch_name"
base_commit=""
if git -C "$work_dir/remote" rev-parse --verify HEAD >/dev/null 2>&1; then
    remote_tree=$(git -C "$work_dir/remote" rev-parse 'HEAD^{tree}')
    while IFS= read -r candidate; do
        if test "$(git -C "$work_dir/source" rev-parse "$candidate^{tree}")" = "$remote_tree"; then
            base_commit="$candidate"
            break
        fi
    done < <(git -C "$work_dir/source" rev-list HEAD)
fi
if test -n "$base_commit"; then
    git -C "$work_dir/source" diff --binary "$base_commit" HEAD > "$work_dir/implementation.patch"
else
    empty_tree=$(git -C "$work_dir/source" hash-object -t tree /dev/null)
    git -C "$work_dir/source" diff --binary "$empty_tree" HEAD > "$work_dir/implementation.patch"
fi
if ! test -s "$work_dir/implementation.patch"; then
    printf '%s\n' 'Remote already matches this delivery snapshot. Nothing to push.'
    exit 0
fi
# Root bundle commit supplies additions; existing-file conflicts are not overwritten.
if ! git -C "$work_dir/remote" apply --index --3way "$work_dir/implementation.patch"; then
  printf '\nPatch conflicts need review in %s. Nothing was pushed.\n' "$work_dir/remote" >&2
  exit 1
fi
author_name=$(git -C "$work_dir/remote" config user.name || true)
author_email=$(git -C "$work_dir/remote" config user.email || true)
git -C "$work_dir/remote" -c "user.name=${author_name:-HumanSense Development}" \
  -c "user.email=${author_email:-humansense@localhost}" \
  commit -m 'Implement HumanSense local SDK, MCP and real-time demo'
git -C "$work_dir/remote" push -u origin "$branch_name"
printf '\nPushed branch: %s\nReview: https://github.com/NomaDamas/catch-your-mind/pull/new/%s\n' "$branch_name" "$branch_name"
