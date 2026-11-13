#!/usr/bin/env bash
# Run in a normal host terminal. Never force-push or alter the current checkout.
set -euo pipefail
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
bundle_path="$project_dir/dist/humansense-v0.1.bundle"
remote_url='https://github.com/NomaDamas/Gamst.git'
branch_name="feat/humansense-$(date +%Y%m%d-%H%M%S)"
if ! test -f "$bundle_path"; then
  printf '%s\n' "Missing bundle: $bundle_path" >&2
  exit 1
fi
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/humansense-publish.XXXXXX")
printf 'Preparing a separate checkout at %s\n' "$work_dir"
git clone "$bundle_path" "$work_dir/source"
git clone "$remote_url" "$work_dir/remote"
git -C "$work_dir/remote" switch -c "$branch_name"
empty_tree=$(git -C "$work_dir/source" hash-object -t tree /dev/null)
git -C "$work_dir/source" diff --binary "$empty_tree" HEAD > "$work_dir/implementation.patch"
# Root bundle commit supplies additions; existing-file conflicts are not overwritten.
if ! git -C "$work_dir/remote" apply --index --3way "$work_dir/implementation.patch"; then
  printf '\nPatch conflicts need review in %s. Nothing was pushed.\n' "$work_dir/remote" >&2
  exit 1
fi
author_name=$(git -C "$work_dir/remote" config user.name || true)
author_email=$(git -C "$work_dir/remote" config user.email || true)
git -C "$work_dir/remote" -c "user.name=${author_name:-gamst Development}" \
  -c "user.email=${author_email:-humansense@localhost}" \
  commit -m 'Implement gamst local SDK, MCP and real-time demo'
git -C "$work_dir/remote" push -u origin "$branch_name"
printf '\nPushed branch: %s\nReview: https://github.com/NomaDamas/Gamst/pull/new/%s\n' "$branch_name" "$branch_name"
