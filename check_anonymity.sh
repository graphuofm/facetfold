#!/usr/bin/env bash
# Fail if anything this repository WOULD PUBLISH identifies the authors.
#
# The engine/ tree is copied from a working repository that is not
# anonymous, so re-syncing it can silently reintroduce a name, an
# institutional email, a home directory, a cluster name, or a link back
# to the hosting account. Run this before every push while the paper is
# under double-blind review.
#
# Scope is `git ls-files`: build output under target/ and __pycache__/
# embeds the absolute build path, but it is gitignored and never leaves
# this machine. Scanning it produces thousands of false alarms and
# trains you to ignore the check. This script excludes itself, since it
# has to contain the very strings it looks for.
set -uo pipefail
cd "$(dirname "$0")"

# Personal names, institutional mail, the cluster, home and scratch
# directories, and any GitHub URL naming a person or an organisation we
# control. github.com/rust-lang and the like are upstream registries and
# name nobody, so the owner is matched explicitly rather than wholesale.
PATTERNS='jding|jiacheng|xiaofei|memphis\.edu|itiger|graphuofm'
PATTERNS="$PATTERNS"'|github\.com/(jding|graphuofm)'
PATTERNS="$PATTERNS"'|/home/(jding|[a-z]*ding)|/project/[a-z0-9]+|/tmp/claude'

fail=0
hits=$(git ls-files -z | grep -zv '^check_anonymity\.sh$' \
       | xargs -0 grep -rniIE "$PATTERNS" -- 2>/dev/null || true)
if [ -n "$hits" ]; then
  echo "ANONYMITY CHECK FAILED -- file contents would identify the authors:"
  echo "$hits"
  fail=1
fi

paths=$(git ls-files | grep -iE 'jding|jiacheng|xiaofei|graphuofm' || true)
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
echo "metadata in any tracked file. ($(git ls-files | wc -l) files scanned.)"
