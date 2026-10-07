#!/usr/bin/env bash
# Run on the destination cluster. Default: inspect a non-destructive copy.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf 'Usage: bash %s REMOTE:BUNDLE_FOLDER /absolute/destination READY_SHA256 [--execute]\n' "$0" >&2
  exit 2
fi
source_bundle=$1
destination_bundle=$2
ready_sha256=$3
transfer_mode=${4:---dry-run}
[[ "$source_bundle" =~ ^[A-Za-z0-9_-]+:.+ && "$destination_bundle" == /* && "$destination_bundle" != / ]] || {
  printf 'Use a configured rclone remote and a dedicated absolute destination.\n' >&2; exit 2;
}
[[ "$ready_sha256" =~ ^[0-9a-f]{64}$ ]] || { printf 'Provide the source BUNDLE_READY SHA256.\n' >&2; exit 2; }
[[ "$transfer_mode" == --dry-run || "$transfer_mode" == --execute ]] || exit 2
command -v rclone >/dev/null
copy_args=(copy "$source_bundle" "$destination_bundle" --immutable --transfers 4 --checkers 8)
if [[ "$transfer_mode" == --dry-run ]]; then
  rclone "${copy_args[@]}" --dry-run
  printf 'Dry run only. Expected BUNDLE_READY SHA256: %s\n' "$ready_sha256"
else
  rclone "${copy_args[@]}"
  printf '%s  %s\n' "$ready_sha256" "$destination_bundle/BUNDLE_READY.json" | sha256sum --check --status
  printf 'Copy finished and receipt hash matches. Full file validation must pass in the verify job before smoke.\n'
fi
