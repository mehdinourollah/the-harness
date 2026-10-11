#!/usr/bin/env bash
#
# The Harness installer.
#
#   curl -LsSf https://raw.githubusercontent.com/mehdinourollah/the-harness/main/install.sh | sh
#
# Installs into ~/.the-harness/ with its own virtualenv, then links the
# `harness` command into ~/.local/bin. Override with:
#
#   HARNESS_VERSION=v0.1.0   a tag/branch to install (default: main)
#   HARNESS_HOME=~/.the-harness
#   HARNESS_BIN=~/.local/bin

set -euo pipefail

REPO="mehdinourollah/the-harness"
VERSION="${HARNESS_VERSION:-main}"
HARNESS_HOME="${HARNESS_HOME:-$HOME/.the-harness}"
HARNESS_BIN="${HARNESS_BIN:-$HOME/.local/bin}"

err() { printf 'error: %s\n' "$1" >&2; exit 1; }
info() { printf '  %s\n' "$1"; }

# --- pick a python -----------------------------------------------------------
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
      PYTHON="$candidate"
      break
    fi
  fi
done
[ -n "$PYTHON" ] || err "Python 3.11+ is required but was not found on PATH"

curl_bin=""
for candidate in curl wget; do
  command -v "$candidate" >/dev/null 2>&1 && { curl_bin="$candidate"; break; }
done
[ -n "$curl_bin" ] || err "curl or wget is required"

# --- fetch the source --------------------------------------------------------
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

archive="$tmp/the-harness.tar.gz"
url="https://codeload.github.com/$REPO/tar.gz/refs/heads/$VERSION"
# A tag works too; fall back to the tag path if the branch path 404s.
if ! "$curl_bin" -fsSL "$url" -o "$archive" 2>/dev/null; then
  url="https://codeload.github.com/$REPO/tar.gz/refs/tags/$VERSION"
  "$curl_bin" -fsSL "$url" -o "$archive" || err "could not download $VERSION from $REPO"
fi

tar -xzf "$archive" -C "$tmp"
src="$(find "$tmp" -maxdepth 1 -type d -name 'the-harness-*' | head -n1)"
[ -n "$src" ] || err "unexpected archive layout"

# --- install -----------------------------------------------------------------
printf 'Installing The Harness (%s)\n' "$VERSION"
"$PYTHON" -m venv "$HARNESS_HOME/venv"
"$HARNESS_HOME/venv/bin/pip" install --quiet --upgrade pip
"$HARNESS_HOME/venv/bin/pip" install --quiet "$src"

mkdir -p "$HARNESS_BIN"
ln -sf "$HARNESS_HOME/venv/bin/harness" "$HARNESS_BIN/harness"

printf '\nInstalled: %s\n' "$HARNESS_BIN/harness"
case ":$PATH:" in
  *":$HARNESS_BIN:"*) ;;
  *) info "Add $HARNESS_BIN to your PATH:"
     info "  export PATH=\"$HARNESS_BIN:\$PATH\"" ;;
esac
info "Try: harness --help"
