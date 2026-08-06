#!/usr/bin/env bash
# Install actionlint (https://github.com/rhysd/actionlint) - the standard
# GitHub Actions workflow linter. Catches YAML syntax errors, context
# availability issues (like hashFiles() at job-level if:), and known
# runner label typos BEFORE GitHub's renderer reports a generic
# "workflow file issue" error.
#
# One-time install:
#   bash ~/repos/opencode-config/templates/gha-lint/install-actionlint.sh
#
# Add to ~/.bashrc for persistence:
#   export PATH="$HOME/.local/bin:$PATH"
#
# Lint every workflow before commit:
#   actionlint
#
# Initialize a project-level config:
#   actionlint -init-config

set -euo pipefail

VERSION="${ACTIONLINT_VERSION:-1.7.12}"
INSTALL_DIR="${ACTIONLINT_INSTALL_DIR:-$HOME/.local/bin}"

if [ -x "$INSTALL_DIR/actionlint" ]; then
  echo "actionlint already installed at $INSTALL_DIR/actionlint"
  actionlint --version
  exit 0
fi

mkdir -p "$INSTALL_DIR"

# actionlint releases a single static Go binary per platform
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)  GOARCH=amd64 ;;
  aarch64) GOARCH=arm64 ;;
  *)
    echo "Unsupported architecture: $ARCH"
    exit 1
    ;;
esac

URL="https://github.com/rhysd/actionlint/releases/download/v${VERSION}/actionlint_${VERSION}_linux_${GOARCH}.tar.gz"

echo "Downloading actionlint v${VERSION} for linux/${GOARCH}..."
TMP="$(mktemp -d)"
trap "rm -rf $TMP" EXIT

if ! curl -fsSL --retry 3 --max-time 60 "$URL" -o "$TMP/actionlint.tar.gz"; then
  echo "Failed to download $URL"
  exit 1
fi

# Verify SHA256 against the published checksums (skip if curl fails — not
# critical since actionlint is a read-only static binary)
CHECKSUMS_URL="https://github.com/rhysd/actionlint/releases/download/v${VERSION}/actionlint_${VERSION}_checksums.txt"
EXPECTED_SHA=""
ACTUAL_SHA="$(sha256sum "$TMP/actionlint.tar.gz" | awk '{print $1}')"
if curl -fsSL --max-time 30 "$CHECKSUMS_URL" -o "$TMP/checksums.txt" 2>/dev/null; then
  EXPECTED_SHA="$(grep "actionlint_${VERSION}_linux_${GOARCH}.tar.gz" "$TMP/checksums.txt" | awk '{print $1}')"
  if [ -n "$EXPECTED_SHA" ] && [ "$EXPECTED_SHA" != "$ACTUAL_SHA" ]; then
    echo "SHA256 mismatch:"
    echo "  expected: $EXPECTED_SHA"
    echo "  actual:   $ACTUAL_SHA"
    exit 1
  fi
  echo "SHA256 verified."
else
  echo "WARNING: could not fetch checksums; skipping SHA256 verification"
fi

tar -xzf "$TMP/actionlint.tar.gz" -C "$TMP"
install -m 0755 "$TMP/actionlint" "$INSTALL_DIR/actionlint"

echo ""
echo "Installed actionlint at $INSTALL_DIR/actionlint"
"$INSTALL_DIR/actionlint" --version
echo ""
echo "Add to your shell rc for persistence:"
echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
