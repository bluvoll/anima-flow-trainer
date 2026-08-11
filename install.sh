#!/usr/bin/env bash
# Anima trainer installer -- Linux.
#
#   ./install.sh              install into ./venv and write ./start-gui.sh
#   ./install.sh --recreate   delete an existing ./venv first
#
# Creates `venv/` (not `.venv/`) deliberately: the repo's own development environment is `.venv`,
# managed by uv against uv.lock, and an installed copy must not silently take it over.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV="venv"
# The diffusers commit this trainer was verified against. Anima support only exists on main, so
# there is no release to pin to -- but floating HEAD means an install can break without a single
# local change. Matches uv.lock; bump both together after re-running the parity gates.
DIFFUSERS_REF="50e7158093710f9c1b4ea9ff100137a91c9228f3"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$GREEN" "$OFF" "$*"; }
warn() { printf '%s!!!%s %s\n' "$YELLOW" "$OFF" "$*"; }
die()  { printf '%sxxx%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

for arg in "$@"; do
    case "$arg" in
        --recreate) RECREATE=1 ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) die "unknown option: $arg (try --help)" ;;
    esac
done

# ---------------------------------------------------------------- find an interpreter
# 3.11 first: every measurement in the README was taken on it. 3.12 resolves cleanly (77 packages,
# torch ships cp312 wheels) but has not been run, so it is a fallback rather than a peer. 3.13 is
# excluded -- torch has wheels, sdnq and triton are untested there.
usable() {
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)' \
        >/dev/null 2>&1
}

PYTHON=""
for candidate in python3.11 python3.12 python3 python; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if usable "$candidate"; then PYTHON="$(command -v "$candidate")"; break; fi
done

# pyenv keeps its interpreters off PATH unless selected, so a machine with 3.11 installed can look
# like it has none. Ask pyenv directly before giving up.
if [ -z "$PYTHON" ] && command -v pyenv >/dev/null 2>&1; then
    for ver in $(pyenv versions --bare 2>/dev/null | grep -E '^3\.(11|12)\.' | sort -V -r); do
        cand="$(pyenv root)/versions/$ver/bin/python"
        if [ -x "$cand" ] && usable "$cand"; then PYTHON="$cand"; break; fi
    done
fi

if [ -z "$PYTHON" ]; then
    echo
    die "no Python 3.11 or 3.12 found.
     Install one and re-run:
       Fedora/RHEL    sudo dnf install python3.11
       Debian/Ubuntu  sudo apt install python3.11 python3.11-venv
       Arch           sudo pacman -S python311      (AUR)
       any distro     pyenv install 3.11"
fi

PY_VER="$("$PYTHON" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
say "Python $PY_VER  ${DIM}$PYTHON${OFF}"
case "$PY_VER" in
    3.12.*) warn "3.12 resolves but has never been run end to end; 3.11 is the tested version." ;;
esac

# ---------------------------------------------------------------- venv
if [ -d "$VENV" ] && [ "${RECREATE:-0}" = "1" ]; then
    say "removing existing $VENV/"
    rm -rf "$VENV"
fi
if [ -d "$VENV" ]; then
    say "reusing existing $VENV/  ${DIM}(--recreate to start clean)${OFF}"
else
    say "creating $VENV/"
    "$PYTHON" -m venv "$VENV" \
        || die "could not create the venv. On Debian/Ubuntu you may need: sudo apt install python3-venv"
fi
VPY="$PWD/$VENV/bin/python"
[ -x "$VPY" ] || die "venv looks broken: no $VPY"

# ---------------------------------------------------------------- dependencies
if command -v uv >/dev/null 2>&1; then
    # uv honours uv.lock, so this reproduces the exact resolved set including the git diffusers and
    # the cu128 torch index. UV_PROJECT_ENVIRONMENT points it at venv/ instead of its default .venv.
    say "installing with uv ${DIM}(from uv.lock)${OFF}"
    UV_PROJECT_ENVIRONMENT="$VENV" uv sync --extra gui --extra ot \
        || die "uv sync failed"
else
    say "installing with pip ${DIM}(uv not found)${OFF}"
    "$VPY" -m pip install --upgrade pip setuptools wheel >/dev/null

    # `requirements.txt` is `uv export` of uv.lock, so this path is as reproducible as the uv one:
    # it carries the exact `torch==2.10.0+cu128`, the pinned diffusers commit, and the gui/ot
    # extras. Resolving fresh from pyproject instead drifted -- measured transformers 5.15.0
    # against the locked 5.14.1, plus pillow and numpy -- which is exactly the silent difference
    # between "it works on my machine" and a bug report nobody can reproduce.
    #
    # --extra-index-url, not --index-url: the +cu128 wheels live on PyTorch's index while
    # everything else comes from PyPI, and --index-url would replace PyPI rather than add to it.
    if [ -f requirements.txt ]; then
        say "  from requirements.txt ${DIM}(pinned, matches uv.lock)${OFF}"
        "$VPY" -m pip install -r requirements.txt --extra-index-url "$TORCH_INDEX" \
            || die "pip install failed"
    else
        warn "requirements.txt missing -- resolving fresh, which may not match uv.lock"
        # Order matters here. torch must come from the cu128 index first: `pip install -e .` would
        # otherwise pull the default PyPI build, which on Linux is the CUDA 12.6 one. And Anima
        # support exists only on diffusers main -- a PyPI release satisfies the unpinned
        # `diffusers` requirement, imports fine, then fails at model load.
        say "  torch (cu128)"
        "$VPY" -m pip install "torch==2.10.0" torchvision --index-url "$TORCH_INDEX" \
            || die "torch install failed"
        say "  diffusers (git @ ${DIFFUSERS_REF:0:9})"
        "$VPY" -m pip install \
            "diffusers @ git+https://github.com/huggingface/diffusers@$DIFFUSERS_REF" \
            || die "diffusers install failed"
        say "  anima-trainer and the rest"
        "$VPY" -m pip install -e ".[gui,ot]" || die "install failed"
    fi
fi

# ---------------------------------------------------------------- launcher
say "writing start-gui.sh"
cat > start-gui.sh <<'LAUNCHER'
#!/usr/bin/env bash
# Start the Anima trainer GUI. Generated by install.sh -- safe to delete and regenerate.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./venv/bin/python -m anima.gui "$@"
LAUNCHER
chmod +x start-gui.sh

# ---------------------------------------------------------------- report
echo
say "checking the install"
"$VPY" anima/tools/check_install.py --require-gui

echo
say "done.  Start the GUI with:  ./start-gui.sh"
echo "    ${DIM}or the CLI:  ./venv/bin/python -m anima.training.train configs/<your>.toml${OFF}"
