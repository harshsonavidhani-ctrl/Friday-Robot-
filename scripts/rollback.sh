#!/usr/bin/env bash
# rollback.sh — put Friday back to the last state that was known to work.
#
# This is the floor under every risky change, and especially under Friday
# editing her own code (see mcp_server/tools/dev.py). If anything goes wrong,
# ONE COMMAND puts the robot back:
#
#     ./scripts/rollback.sh                  # back to the known-good tag
#     ./scripts/rollback.sh --mark           # declare NOW to be known-good
#     ./scripts/rollback.sh --to <ref>       # back to a specific commit
#     ./scripts/rollback.sh --list           # what the known-good points are
#
# NOTHING IS EVER THROWN AWAY. A hard reset would lose whatever was in the
# working tree, including the broken change you probably want to look at
# afterwards, so the current state is always committed to a rescue branch
# first. Rolling back is therefore reversible too.
#
# ⚠ SCOPE. This touches Friday's git repository and Friday's own systemd user
# units. It does not touch, and must never be extended to touch, anything in
# /etc, /boot, or any unit this project did not create. See CLAUDE.md Part 1.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TAG="known-good"
UNITS=(friday-mcp friday-dashboard friday-face)

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

mark_known_good() {
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "refusing to mark a dirty tree as known-good — commit first:" >&2
        git status --short >&2
        exit 1
    fi
    local ref; ref="$(git rev-parse --short HEAD)"
    local when; when="$(date +%Y%m%d-%H%M)"
    # Keep the old one under a dated name rather than silently overwriting it,
    # so there is more than one floor to stand on.
    if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
        local prev; prev="$(git rev-parse --short "$TAG")"
        git tag "known-good-$when" "$TAG" 2>/dev/null || true
        echo "previous known-good ($prev) kept as known-good-$when"
    fi
    git tag -f -a "$TAG" -m "Known-good: verified on the real robot $(date -Iseconds)" HEAD
    echo "known-good now points at $ref: $(git log -1 --format=%s)"
}

list_points() {
    printf '%-22s %-10s %s\n' TAG COMMIT SUBJECT
    for t in $(git tag -l 'known-good*' | sort -r); do
        # ^{commit} dereferences the tag OBJECT to the commit it points at --
        # without it an annotated tag reports its own sha, which matches
        # nothing in the log and makes the list quietly useless.
        printf '%-22s %-10s %s\n' "$t" "$(git rev-parse --short "$t^{commit}")" \
               "$(git log -1 --format=%s "$t")"
    done
}

TARGET="$TAG"
case "${1:-}" in
    -h|--help) usage ;;
    --mark)    mark_known_good; exit 0 ;;
    --list)    list_points; exit 0 ;;
    --to)      TARGET="${2:?--to needs a commit or tag}" ;;
    "")        ;;
    *)         echo "unknown option $1" >&2; exit 2 ;;
esac

git rev-parse -q --verify "$TARGET^{commit}" >/dev/null || {
    echo "no such commit or tag: $TARGET" >&2; exit 1; }

say "Rolling back to $TARGET ($(git rev-parse --short "$TARGET"): $(git log -1 --format=%s "$TARGET"))"
echo "  from $(git rev-parse --short HEAD): $(git log -1 --format=%s)"

# 1. Preserve everything currently here, committed or not, on a rescue branch.
RESCUE="rescue/$(date +%Y%m%d-%H%M%S)"
say "1. Saving the current state to $RESCUE"
git add -A
if [[ -n "$(git status --porcelain)" ]]; then
    git stash push -u -q -m "rollback rescue" || true
    git branch "$RESCUE" HEAD
    git stash pop -q || true
    git add -A
    git commit -q -m "WIP saved by rollback.sh before returning to $TARGET" || true
    git branch -f "$RESCUE" HEAD
else
    git branch "$RESCUE" HEAD
fi
echo "  saved — recover it later with: git checkout $RESCUE"

# 2. Stop Friday before swapping the code under her.
say "2. Stopping Friday's services"
for u in "${UNITS[@]}"; do
    systemctl --user stop "$u.service" 2>/dev/null && echo "  stopped $u" || true
done

# 3. The rollback itself.
say "3. Restoring the code"
git checkout -q "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)" 2>/dev/null || true
git reset --hard -q "$TARGET"
find . -name "__pycache__" -type d -not -path "./venv/*" -not -path "./tools/*" \
     -exec rm -rf {} + 2>/dev/null || true
echo "  now at $(git rev-parse --short HEAD): $(git log -1 --format=%s)"

# 4. Bring her back.
say "4. Restarting Friday's services"
for u in "${UNITS[@]}"; do
    systemctl --user start "$u.service" 2>/dev/null && echo "  started $u" || true
done
sleep 5

# 5. The checks CLAUDE.md requires after EVERY change.
say "5. Verifying the machine is still reachable"
ok=0
systemctl is-active --quiet ssh && echo "  ssh:   active" || { echo "  ssh:   DOWN"; ok=1; }
ss -ltn 2>/dev/null | grep -q ':22 ' && echo "  ssh:   listening on 22" || { echo "  ssh:   NOT LISTENING"; ok=1; }
if command -v nmcli >/dev/null; then
    state="$(nmcli -t -f STATE general 2>/dev/null || true)"
    [[ "$state" == connected* ]] && echo "  wifi:  $state" || { echo "  wifi:  $state"; ok=1; }
fi
systemctl is-active --quiet systemd-logind && echo "  login: logind active" || { echo "  login: LOGIND DOWN"; ok=1; }
for u in "${UNITS[@]}"; do
    printf '  %-18s %s\n' "$u" "$(systemctl --user is-active "$u.service" 2>/dev/null)"
done

if [[ $ok -ne 0 ]]; then
    say "⚠ SOMETHING IS WRONG WITH THE MACHINE ITSELF — tell Harsh now."
    exit 1
fi
say "Rolled back cleanly. The state you left is on branch $RESCUE."
