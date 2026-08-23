#!/usr/bin/env bash
# Fail if anything this repository WOULD PUBLISH identifies the authors.
#
# The engine/ tree is copied from a working repository that is not
# anonymous, and result files are copied from a working directory whose
# absolute paths contain a username, so a re-sync can silently
# reintroduce a name, an institutional email, a home directory, a
# cluster name, or a link back to the hosting account. Run this before
# every push while the paper is under double-blind review.
#
# Scope is tracked files PLUS untracked-but-not-ignored files. Scanning
# only `git ls-files` misses a file that is about to be added for the
# first time, which is exactly the one most likely to carry a stray
# absolute path. Build output under target/ and __pycache__/ embeds the
# build path but is gitignored and never published, so it is excluded:
# scanning it produces thousands of false alarms and trains you to
# ignore the check. This script excludes itself, since it has to
# contain the strings it looks for.
set -uo pipefail
cd "$(dirname "$0")"

# Personal names, institutional mail, the cluster, home and scratch
# directories, and any GitHub URL naming a person or an organisation we
# control. github.com/rust-lang and the like are upstream registries and
# name nobody, so the owner is matched explicitly rather than wholesale.
PATTERNS='jding|jiacheng|xiaofei|memphis\.edu|itiger|graphuofm'
PATTERNS="$PATTERNS"'|github\.com/(jding|graphuofm)'
PATTERNS="$PATTERNS"'|/home/(jding|[a-z]*ding)|/project/[a-z0-9]+|/tmp/claude'

# /home/bruce is the container WORKDIR in engine/Dockerfile: a path
# inside the image, not anyone's home directory.
ALLOW='engine/Dockerfile:[0-9]+:WORKDIR /home/bruce'

SCAN=$(mktemp)
trap 'rm -f "$SCAN"' EXIT
{ git ls-files; git ls-files --others --exclude-standard; } \
  | grep -v '^check_anonymity\.sh$' | sort -u > "$SCAN"
n_files=$(wc -l < "$SCAN")
if [ "$n_files" -eq 0 ]; then
  echo "ANONYMITY CHECK FAILED -- listed no files to scan; refusing to"
  echo "report success. Is this a git repository?"
  exit 1
fi

fail=0
hits=$(tr '\n' '\0' < "$SCAN" \
       | xargs -0 grep -rniIE "$PATTERNS" -- 2>/dev/null \
       | grep -vE "$ALLOW" || true)
if [ -n "$hits" ]; then
  echo "ANONYMITY CHECK FAILED -- file contents would identify the authors:"
  echo "$hits"
  fail=1
fi

paths=$(grep -iE 'jding|jiacheng|xiaofei|graphuofm' "$SCAN" || true)
if [ -n "$paths" ]; then
  echo "ANONYMITY CHECK FAILED -- file names:"; echo "$paths"; fail=1
fi

authors=$(git log --format='%an <%ae>%n%cn <%ce>' | sort -u \
          | grep -viE 'facetfold|noreply' || true)
if [ -n "$authors" ]; then
  echo "ANONYMITY CHECK FAILED -- commit metadata:"; echo "$authors"; fail=1
fi

[ "$fail" -ne 0 ] && exit 1
echo "anonymity check passed: no author name, institutional email, home"
echo "directory, cluster name, hosting account, or identifying commit"
echo "metadata in any tracked or newly added file. ($n_files scanned.)"
