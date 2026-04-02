#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${1:-$ROOT_DIR/api/app/session_artifacts}"

if [ -d "$ARTIFACT_DIR" ]; then
  rm -rf "$ARTIFACT_DIR"
  printf "Removed session artifacts at %s\n" "$ARTIFACT_DIR"
else
  printf "No session artifacts found at %s\n" "$ARTIFACT_DIR"
fi

mkdir -p "$ARTIFACT_DIR"
printf "Created empty session artifacts directory at %s\n" "$ARTIFACT_DIR"

if git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  tracked_artifacts="$(
    git -C "$ROOT_DIR" ls-files -- "api/app/session_artifacts/*"
  )"

  if [ -n "$tracked_artifacts" ]; then
    git -C "$ROOT_DIR" rm -r --cached --quiet -- api/app/session_artifacts
    printf "Removed tracked session artifacts from git index.\n"
  fi
fi
