#!/usr/bin/env bash
# Fail if anything this repository WOULD PUBLISH identifies the authors.
#
# The engine/ tree is copied from a working repository that is not
# anonymous, so re-syncing it can silently reintroduce a name, an
# institutional email, a home directory, or a cluster name. Run this
# before every push while the paper is under double-blind review.
#
# Scope is deliberately `git ls-files`: build output under target/ and
# __pycache__/ embeds the absolute build path, but it is gitignored and
# never leaves this machine. Scanning it produces thousands of false
# alarms and trains you to ignore the check.
set -uo pipefail
cd "$(dirname "$0")"

PATTERNS='jding|jiacheng|xiaofei|memphis\.edu|itiger|/home/[a-z]+|/project/[a-z0-9]+|/tmp/claude'
fail=0

# /home/bruce is the container WORKDIR in engine/Dockerfile: a path
# inside the image, not anyone's home directory.
ALLOW='engine/Dockerfile:[0-9]+:WORKDIR /home/bruce'
hits=$(git ls-files -z \
       | xargs -0 grep -rniIE "$PATTERNS" -- 2>/dev/null \
       | grep -vE "$ALLOW" || true)
if [ -n "$hits" ]; then
  echo "ANONYMITY CHECK FAILED -- file contents would identify the authors:"
  echo "$hits"
  fail=1
fi

paths=$(git ls-files | grep -iE 'jding|jiacheng|xiaofei' || true)
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
echo "directory, cluster name, or identifying commit metadata in any"
echo "tracked file. ($(git ls-files | wc -l) files scanned.)"
